# -*- coding: utf-8 -*-
"""문장에 구애받지 않는 발음 진단 — 음소 판별기(phone detector) 모델 정의 + 서빙 파이프라인.

받아쓴 뒤 비교하지 않는다. 원어민만으로 학습한 음향 기준 모델로 **표준발음을 음성에 강제정렬**하고, 자리마다 뽑은
음향 증거(`mdd.phone_feats`)를 작은 트랜스포머가 보고 "오발음인가 · 무엇으로 냈나"를 낸다. 입력이 음소 단위 증거뿐이라
문장을 외울 수 없다 → 학습에 없던 문장도 같은 정확도로 진단한다(results/15).

    det = PhoneDetector(anchor_ckpt, [detector.pt, ...], dict_dir)          # 판별기 여러 개면 확률을 평균(앙상블)
    out = det.diagnose(wave_16k_float32, "아무 문장이나", l1="Vietnamese")

모델 정의(`build_model`)는 학습 스크립트(`scripts/train/train_phone_detector.py`)와 공유한다 — 가중치 키가 같아야 한다.
"""
import argparse
import os

import numpy as np

from mdd import phone_feats as PF


# ─────────────────────────── 모델 ───────────────────────────
def build_model(torch, nn, d_small, d_emb, has_mid, n_l1, V, a):
    class Det(nn.Module):
        def __init__(self):
            super().__init__()
            D = a.dim
            self.small = nn.Sequential(nn.Linear(d_small, 192), nn.GELU(), nn.Linear(192, D))
            self.use_emb = not a.no_emb
            if self.use_emb:
                self.pl = nn.Sequential(nn.LayerNorm(d_emb), nn.Dropout(a.dropout), nn.Linear(d_emb, D))
                self.pm = nn.Sequential(nn.LayerNorm(d_emb), nn.Dropout(a.dropout), nn.Linear(d_emb, D)) if has_mid else None
            self.ph = None if a.no_phone else nn.Embedding(V, D)
            self.l1 = None if a.no_l1 else nn.Embedding(n_l1, D)
            self.norm = nn.LayerNorm(D); self.drop = nn.Dropout(a.dropout)
            if a.layers:
                lay = nn.TransformerEncoderLayer(D, a.heads, D * 2, a.dropout, activation="gelu", batch_first=True, norm_first=True)
                self.enc = nn.TransformerEncoder(lay, a.layers)
                self.conv = nn.Conv1d(D, D, 5, padding=2, groups=D)          # 절대 위치 대신 국소 순서만 준다(문항 암기 억제)
            else:
                self.enc = None
            self.out = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(), nn.Dropout(a.dropout))
            self.h_err = nn.Linear(D, 1); self.h_real = nn.Linear(D, V + 1)
            self.h_all = nn.Linear(D, V + 1)      # 전 자리 '실제 발음' 예측 — 표현 학습용 정규화(검출 F1 에 도움), 출력으로는 안 쓴다

        def forward(self, small, cid, el, em, l1, mask):
            x = self.small(small)
            if self.use_emb:
                x = x + self.pl(el)
                if self.pm is not None and em is not None:
                    x = x + self.pm(em)
            if self.ph is not None:
                x = x + self.ph(cid)
            if self.l1 is not None:
                x = x + self.l1(l1)[:, None, :]
            x = self.drop(self.norm(x))
            if self.enc is not None:
                x = x + self.conv((x * mask[..., None]).transpose(1, 2)).transpose(1, 2)
                x = self.enc(x, src_key_padding_mask=~mask)
            h = self.out(x)
            self.last_all = self.h_all(h)
            return self.h_err(h).squeeze(-1), self.h_real(h)
    return Det()


def cond_logits(logits, cid):
    """오발음이라는 조건에서의 실제 발음 분포 — 표준음 자신과 특수기호(pad·eos·unk)는 후보에서 뺀다. 마지막 클래스 = 삭제."""
    out = logits.clone()
    out[..., 1:4] = float("-inf")
    out.scatter_(-1, cid.unsqueeze(-1), float("-inf"))
    out[..., 0] = float("-inf")                                 # blank 자리는 안 쓴다(삭제는 클래스 V)
    return out


def small_features(scal, sp, sn, cid, stats_m, stats_s):
    """판별기의 '작은' 입력 — 학습(`make_batch`)과 같은 식. scal[L,7] · sp/sn[L,V] · cid[L] (numpy 또는 torch)."""
    canon_lp = sp[range(len(cid)), cid][:, None]
    cat = np.concatenate if isinstance(sp, np.ndarray) else (lambda xs, ax: __import__("torch").cat(xs, ax))
    return cat([(scal - stats_m) / stats_s, sp / 10.0, sn / 10.0, (sp - canon_lp) / 10.0], -1)


# ─────────────────────────── 서빙 ───────────────────────────
class PhoneDetector:
    def __init__(self, anchor_ckpt, detector_paths, dict_dir, device=None, threshold=None, anchor_onnx=None, onnx_threads=4):
        """anchor_onnx 를 주면 기준 모델을 ONNX Runtime(CPU)으로 돌린다 — `export_onnx_anchor.py` 산출물(lprobs · h_last · h_mid).
        그때는 fairseq 체크포인트를 올리지 않는다(메모리 1/3). 판별기는 작아서(2.3M) torch CPU 로 돈다."""
        from shared.fairseq_guard import ensure_fairseq
        ensure_fairseq()
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torchaudio
        self.torch, self.F, self.torchaudio = torch, F, torchaudio
        self.sess = None
        if anchor_onnx:
            import onnxruntime as ort
            from fairseq.data import Dictionary
            dev = torch.device("cpu")
            so = ort.SessionOptions(); so.intra_op_num_threads = onnx_threads; so.inter_op_num_threads = 1
            self.sess = ort.InferenceSession(anchor_onnx, so, providers=["CPUExecutionProvider"])
            self.d = Dictionary.load(os.path.join(dict_dir, "dict.ltr.txt"))
            self.blank, self.normalize, self.anchor = 0, True, None
            self.hidden = int(self.sess.get_outputs()[1].shape[-1])
        else:
            from fairseq import checkpoint_utils
            dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([anchor_ckpt], arg_overrides={"data": dict_dir})
            m = models[0].eval()
            self.anchor = m.to(dev).half() if dev.type == "cuda" else m.float()
            self.d = task.target_dictionary
            self.blank = self.d.index(task.blank_symbol) if hasattr(task, "blank_symbol") else self.d.bos()
            self.normalize = cfg.task.normalize
            self.hidden = self.anchor.w2v_encoder.proj.weight.shape[1]
        self.dev = dev
        self.dets, ths = [], []
        for p in detector_paths:
            ck = torch.load(p, map_location=dev)
            a = argparse.Namespace(**ck["args"])
            V = len(ck["symbols"])
            d_small = PF.N_SCAL + 3 * V
            C = self.hidden
            net = build_model(torch, nn, d_small, 0 if a.no_emb else C, not a.no_emb, len(ck["l1_vocab"]), V, a).to(dev)
            net.load_state_dict(ck["model"]); net.eval()
            self.dets.append({"net": net, "m": torch.tensor(ck["stats"]["m"], device=dev), "s": torch.tensor(ck["stats"]["s"], device=dev),
                              "l1": ck["l1_vocab"], "use_emb": not a.no_emb})
            ths.append(ck["threshold"]); self.symbols = ck["symbols"]
        self.threshold = float(threshold if threshold is not None else np.mean(ths))
        self.mid_layer = 16

    def canonical(self, sentence):
        from mdd import pipeline as MD
        if "ri" not in MD.S:
            from g2p import readRules, ver_info
            MD.S["ri"], MD.S["ro"] = readRules(ver_info[0], MD.RULEBOOK)
        return list(MD.canonical(sentence)[0])

    def diagnose(self, wave, sentence=None, l1=None, canon=None):
        """wave: 16 kHz float32 · sentence 또는 canon(자모 리스트). 반환 dict(자리별 p_err · 지적 여부 · 들린 소리 상위 3 · 시간)."""
        torch, F = self.torch, self.F
        canon = canon or self.canonical(sentence)
        y = [self.d.index(s) for s in canon]
        if not y or self.d.unk() in y:
            raise ValueError("표준발음에 사전 밖 기호가 있습니다")
        with torch.inference_mode():
            if self.sess is not None:                           # ONNX: 입력 규약 = 정규화한 파형 [1, samples]
                w = np.asarray(wave, np.float32); x = ((w - w.mean()) / np.sqrt(w.var() + 1e-5))[None].astype(np.float32)
                lp_, h_, hm_ = self.sess.run(None, {"source": x})
                lp_all, h, hm = torch.from_numpy(lp_), torch.from_numpy(h_), torch.from_numpy(hm_); T = lp_all.shape[1]
            else:
                lp_all, h, hm, Ts = PF.forward_anchor(self.anchor, [np.asarray(wave, np.float32)], self.normalize, self.mid_layer, torch, F)
                T = Ts[0]
            o = PF.align_featurize(lp_all[0, :T].float(), h[0, :T], hm[0, :T], y, self.blank, torch, self.torchaudio)
            if o is None:
                return {"align_ok": False, "canonical": canon, "phones": []}
            cid = torch.tensor(y, device=self.dev)
            rt = lambda x: x.half().float()                     # 학습 특징은 fp16 으로 저장됐다 — 같은 반올림을 거친다
            sp, sn = rt(o["lpp_spike"]), rt(o["lpp_span"])
            ps, rl = [], []
            for D in self.dets:
                small = small_features(o["scal"], sp, sn, cid, D["m"], D["s"])
                li = torch.tensor([D["l1"].get(l1 or "?", 0)], device=self.dev)
                mask = torch.ones(1, len(y), dtype=torch.bool, device=self.dev)
                el = rt(o["emb_last"])[None] if D["use_emb"] else None
                em = rt(o["emb_mid"])[None] if D["use_emb"] else None
                lo, lr_ = D["net"](small[None].float(), cid[None], el, em, li, mask)
                ps.append(torch.sigmoid(lo[0].float())); rl.append(torch.softmax(cond_logits(lr_[0].float(), cid), -1))
            p = torch.stack(ps).mean(0); r = torch.stack(rl).mean(0)
            top = r.topk(3, -1)
            # "원어민 귀의 증거" — 원어민만으로 학습한 기준 모델이 그 자리에서 들은 소리(스파이크 프레임 사후확률, blank·특수기호 제외 후 정규화)
            ear = o["lpp_spike"].float().exp(); ear[:, :self.d.nspecial] = 0
            ear = ear / ear.sum(-1, keepdim=True).clamp_min(1e-8); etop = ear.topk(3, -1)
        V = len(self.symbols)
        name = lambda k: "∅" if k == V else self.symbols[k]
        span = (o["span"].float() * PF.FRAME_SEC).tolist()
        phones = [{"i": j, "jamo": canon[j], "p_err": round(float(p[j]), 4), "flag": bool(p[j] >= self.threshold),
                   "heard_top3": [[name(int(k)), round(float(v), 3)] for k, v in zip(top.indices[j], top.values[j])],
                   "native_ear": [[name(int(k)), round(float(v), 3)] for k, v in zip(etop.indices[j], etop.values[j])],
                   "start": round(span[j][0], 3), "end": round(span[j][1], 3)} for j in range(len(y))]
        return {"align_ok": True, "canonical": canon, "threshold": self.threshold, "n_flag": sum(x["flag"] for x in phones),
                "duration_sec": round(len(wave) / 16000, 3), "phones": phones}
