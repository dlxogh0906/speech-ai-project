# -*- coding: utf-8 -*-
"""국적 모델 여러 개를 한 프로세스에 올려 같은 파형을 모두에 넣는다 — 모델 비교 데모용 (2026-10-27).

형식
  torch  finetune_full_encoder.py 가 쓴 체크포인트(전체학습 · LoRA 합침 · 학습 안 함 · 운율 실험)와 KD 학생(kd_student.save_student).
         그 스크립트의 --eval_only 와 같은 경로: bf16 autocast 로 forward_pooled → (x−mu)/sd → 선형 헤드(fp32).
         운율 실험(xh_cfg)은 프레임 출력을 xh 모듈에 넣는다. 운율 궤적이 필요한 모델은 pros_fn(파형) → (T,3) 을 받아야 돈다.
         mel 융합(mf_cfg, results/45)은 프레임 출력과 파형을 mf 모듈에 넣는다(평가처럼 autocast 밖 fp32).
  2b     xlsr2b_finetune.py 의 부분 체크포인트(encoder_partial) — 9월 2B 실험. ckpt 가 "" 면 2B 동결 + 0704 헤드(보고용).
  onnx   KD 인코더 ONNX(encoder_kd_l17.onnx) + head_np.npz — 배포 형식 그대로 CPU 에서 (OnnxKD).

GPU 메모리를 줄이려고 인코더의 nn.Linear 가중치만 bf16 으로 둔다. autocast 가 어차피 bf16 으로 바꿔 곱하는 부분이라
행렬곱 결과는 fp32 가중치 + autocast 와 같다. 층 정규화 · 합성곱(CNN · 위치 합성곱) · 헤드는 fp32 그대로다.
가중치가 똑같은 인코더(학습 안 함 2B 와 9월 보고용 2B 처럼 사전학습 그대로인 것)는 한 번만 올리고 한 번만 돌린다.
입력 길이는 호출하는 쪽이 정한다(평가는 앞 12초).
"""
import copy
import importlib.util
import threading
import time

import numpy as np

_MODS = {}


def _script(name, rel):
    """저장소 안 스크립트를 모듈로 읽는다(tta_full_encoder.py 와 같은 방식). load_backbone · make_xhead · head_from_0704 를 쓴다."""
    if name not in _MODS:
        from shared.paths import NAT_TASK
        spec = importlib.util.spec_from_file_location(name, str(NAT_TASK / rel))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODS[name] = mod
    return _MODS[name]


def _ff():
    return _script("finetune_full_encoder", "scripts/train/finetune_full_encoder.py")


def _xb():
    return _script("xlsr2b_finetune", "experiments/xlsr2b_finetune.py")


def _fingerprint(state):
    """인코더 가중치 지문 — 모든 텐서의 합과 절댓값 합. 일부 텐서만 보면 안 된다: LoRA 는 Linear 가중치만 바꾸고
    bias · 층 정규화 · CNN 은 출발점 그대로라, 그런 텐서만 뽑히면 다른 모델을 같은 인코더로 묶는다(실제로 그랬다)."""
    return tuple((k, float(v.double().sum()), float(v.double().abs().sum())) for k, v in sorted(state.items()))


class TorchZoo:
    """torch 형식 모델 묶음. add_* 로 올리고 logits(wav) 로 전부 돌린다. 한 번에 요청 하나(lock)."""

    def __init__(self, device=None, pros_fn=None):
        import torch
        self.torch = torch
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.pros_fn = pros_fn
        self._tmpl = {}          # (xlsr 경로, 층) → CPU fp32 인코더 (복사 원본)
        self.groups = []         # 인코더 하나와 그 인코더를 쓰는 모델들
        self.models = {}         # id → 헤드 · 정규화 · 라벨 · 그룹
        self.lock = threading.Lock()

    def _template(self, xlsr, layer):
        k = (str(xlsr), int(layer))
        if k not in self._tmpl:
            enc = _ff().load_backbone(xlsr, layer, "cpu").eval()
            # weight_norm(위치 합성곱)이 계산해 둔 weight 는 그래프 중간 텐서라 deepcopy 가 거부한다. forward 때마다 다시 계산하니 떼어 둔다.
            for m in enc.modules():
                for name, v in list(vars(m).items()):
                    if isinstance(v, self.torch.Tensor) and not v.is_leaf:
                        setattr(m, name, v.detach())
            self._tmpl[k] = enc
        return self._tmpl[k]

    def drop_templates(self):
        """다 올린 뒤 CPU 복사 원본을 버린다."""
        self._tmpl.clear()

    def _group(self, xlsr, layer, state, strict):
        torch = self.torch
        enc = copy.deepcopy(self._template(xlsr, layer))
        if state is not None:
            missing, unexpected = enc.load_state_dict(state, strict=strict)
            if unexpected:
                raise ValueError("체크포인트에 인코더에 없는 키가 있다: %s" % unexpected[:3])
        fp = _fingerprint(enc.state_dict())
        for g in self.groups:
            if g["fp"] == fp and g["layer"] == int(layer):
                return g
        for m in enc.modules():
            if isinstance(m, torch.nn.Linear):
                m.to(torch.bfloat16)
        for p in enc.parameters():
            p.requires_grad_(False)
        g = {"enc": enc.to(self.dev).eval(), "layer": int(layer), "fp": fp, "ids": []}
        self.groups.append(g)
        return g

    def _register(self, mid, g, head_w, head_b, mu, sd, labels, xh=None, mu2=None, sd2=None, pros=False, mf=None):
        torch = self.torch
        head = torch.nn.Linear(head_w.shape[1], head_w.shape[0])
        head.load_state_dict({"weight": torch.as_tensor(head_w).float(), "bias": torch.as_tensor(head_b).float()})
        t = lambda a: None if a is None else torch.as_tensor(np.asarray(a, np.float32)).to(self.dev)
        self.models[mid] = {"group": g, "head": head.to(self.dev).eval(), "mu": t(mu), "sd": t(sd), "labels": list(labels),
                            "xh": xh, "mu2": t(mu2), "sd2": t(sd2), "pros": pros, "mf": mf}
        g["ids"].append(mid)

    def add_ckpt(self, mid, path):
        """finetune_full_encoder.py · kd_student 형식 체크포인트."""
        from shared.paths import MODEL_ROOT
        ff = _ff()
        ck = self.torch.load(path, map_location="cpu")
        xlsr = ck.get("xlsr") or str(MODEL_ROOT / ff.BACKBONES[ck.get("backbone", "300m")][0])
        g = self._group(xlsr, ck["layer"], ck["encoder"], strict=True)
        xh, pros = None, False
        if ck.get("xh_cfg"):
            c = ck["xh_cfg"]
            pros = bool(c["pros"])
            if pros and self.pros_fn is None:
                raise RuntimeError("운율 궤적 모델인데 pros_fn 이 없다")
            xh = ff.make_xhead(len(np.asarray(ck["mu"]).reshape(-1)), c["pool"], c["pros"])
            xh.load_state_dict(ck["xh"])
            xh = xh.to(self.dev).eval()
        mf = None
        if ck.get("mf_cfg"):
            c = ck["mf_cfg"]
            mf = ff.make_melfuse(len(np.asarray(ck["mu"]).reshape(-1)), c["mode"], c["d"], c["layers"])
            mf.load_state_dict(ck["mf"])
            mf = mf.to(self.dev).eval()
        self._register(mid, g, ck["head"]["weight"], ck["head"]["bias"], np.asarray(ck["mu"]).reshape(-1),
                       np.asarray(ck["sd"]).reshape(-1), ck["labels"], xh, ck.get("mu2"), ck.get("sd2"), pros, mf)

    def add_2b_partial(self, mid, path, class_order):
        """xlsr2b_finetune.py 형식. path 가 "" 면 2B 동결 + 0704 헤드(class_order 순서로 재배열)."""
        xb = _xb()
        if path:
            ck = self.torch.load(path, map_location="cpu")
            g = self._group(xb.XLSR2B, xb.LAYER, ck["encoder_partial"], strict=False)
            W, b = ck["head"]["weight"].numpy(), ck["head"]["bias"].numpy()
            mu, sd, labels = np.asarray(ck["mu"], np.float32), np.asarray(ck["sd"], np.float32), [str(x) for x in ck["labels"]]
        else:
            g = self._group(xb.XLSR2B, xb.LAYER, None, strict=True)
            W, b, mu, sd = xb.head_from_0704(list(class_order))
            labels = list(class_order)
        self._register(mid, g, W, b, mu.reshape(-1), sd.reshape(-1), labels)

    def logits(self, wav):
        """wav: float32 1차원(16 kHz) → ({id: 로짓 np.float64}, {id: 그 인코더 그룹의 ms})."""
        torch = self.torch
        from country_classifier.kd_student import forward_pooled
        amp = dict(device_type=self.dev.type, dtype=torch.bfloat16, enabled=self.dev.type == "cuda")
        out, ms = {}, {}
        with self.lock, torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(wav, np.float32))[None].to(self.dev)
            pm = torch.zeros_like(x, dtype=torch.bool)
            ptrack = None
            for g in self.groups:
                t0 = time.perf_counter()
                enc, layer = g["enc"], g["layer"]
                plain = [i for i in g["ids"] if self.models[i]["xh"] is None and self.models[i]["mf"] is None]
                fancy = [i for i in g["ids"] if i not in plain]
                if plain:
                    with torch.autocast(**amp):
                        pooled = forward_pooled(enc, x, pm, layer)
                    for i in plain:
                        m = self.models[i]
                        out[i] = m["head"]((pooled - m["mu"]) / m["sd"]).double().cpu().numpy()[0]
                if fancy:
                    with torch.autocast(**amp):
                        res = enc.extract_features(source=x, padding_mask=pm, mask=False, layer=layer)
                    h = torch.nan_to_num(res["layer_results"][layer][0].transpose(0, 1).float())
                    fp = res.get("padding_mask")
                    for i in fancy:
                        m = self.models[i]
                        if m["mf"] is not None:
                            out[i] = m["head"](m["mf"](h, fp, m["mu"], m["sd"], x, pm)).double().cpu().numpy()[0]
                            continue
                        pt = None
                        if m["pros"]:
                            if ptrack is None:
                                ptrack = np.asarray(self.pros_fn(wav), np.float32)
                            arr = np.zeros((1, h.shape[1], 3), np.float32)
                            n = min(h.shape[1], len(ptrack))
                            arr[0, :n] = ptrack[:n]
                            pt = torch.from_numpy(arr).to(self.dev)
                        z = m["head"](m["xh"](h, fp, m["mu"], m["sd"], m["mu2"], m["sd2"], pt))
                        out[i] = z.double().cpu().numpy()[0]
                if self.dev.type == "cuda":
                    torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1000
                for i in g["ids"]:
                    ms[i] = dt
        return out, ms


class OnnxKD:
    """KD 인코더 ONNX + numpy 헤드 — nationality-kd-demo 의 Encoder 와 같은 계산(CPU)."""

    def __init__(self, model_dir, threads=2):
        import os
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(os.path.join(model_dir, "encoder_kd_l17.onnx"), so, providers=["CPUExecutionProvider"])
        h = np.load(os.path.join(model_dir, "head_np.npz"), allow_pickle=True)
        self.W, self.b, self.mu, self.sd = h["W"], h["b"], h["mu"], h["sd"]
        self.labels = [str(x) for x in h["labels"]]

    def logits(self, wav):
        x = np.clip(np.nan_to_num(self.sess.run(["pooled"], {"source": np.ascontiguousarray(wav, np.float32)[None, :]})[0]), -50, 50)
        return (((x - self.mu) / self.sd) @ self.W.T + self.b)[0].astype(np.float64)


def softmax(z):
    z = np.asarray(z, np.float64)
    p = np.exp(z - z.max())
    return p / p.sum()
