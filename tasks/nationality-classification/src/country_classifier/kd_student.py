"""KD 학생 모델(XLS-R 300m ≤17층 + 마스크 평균풀링 + 선형 헤드) 공용 부품.

experiments/kd_encoder_*.py 가 각자 인라인으로 갖고 있던 구성과 같다:
    300m → layer17 출력 → 마스크 평균풀링 → nan_to_num · clip ±50 → (x−mu)/sd → Linear(1024, 41)
체크포인트 형식도 같다: encoder(state_dict) · head{weight,bias} · mu · sd · labels · layer.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import csv

import numpy as np

LOADER_HDR = ["audio_path", "json_path", "speaker_id", "label_name", "label_id", "duration_sec"]


def build_student(xlsr300, ckpt, device):
    """300m 을 ≤layer 로 자르고 ckpt(없으면 원본)를 얹는다. (enc, head, mu, sd, labels, layer)"""
    import torch
    import torch.nn as nn
    from shared.fairseq_guard import ensure_fairseq
    ensure_fairseq()
    from fairseq import checkpoint_utils

    ck = torch.load(ckpt, map_location="cpu")
    layer = int(ck["layer"])
    models, _ = checkpoint_utils.load_model_ensemble([str(xlsr300)])
    enc = models[0]
    del enc.encoder.layers[layer + 1:]
    # 학습 중 layerdrop 이 켜져 있으면 layer_results 인덱스가 밀린다 — 조용히 틀리지 않게 막는다
    assert enc.encoder.layerdrop == 0, f"layerdrop={enc.encoder.layerdrop}"
    enc.load_state_dict(ck["encoder"])
    labels = [str(x) for x in ck["labels"]]
    head = nn.Linear(1024, len(labels))
    head.load_state_dict({k: torch.as_tensor(v) for k, v in ck["head"].items()})
    mu = torch.as_tensor(np.asarray(ck["mu"], np.float32)).to(device)
    sd = torch.as_tensor(np.asarray(ck["sd"], np.float32)).to(device)
    return enc.to(device), head.to(device), mu, sd, labels, layer


def forward_pooled(enc, wav, pm, layer):
    """(B,S) 파형 → (B,1024) 마스크 평균 (fp32, clip ±50). autocast 는 호출하는 쪽에서."""
    import torch
    res = enc.extract_features(source=wav, padding_mask=pm, mask=False, layer=layer)
    x = res["layer_results"][layer][0].transpose(0, 1)             # (B,T,C)
    fp = res.get("padding_mask")
    if fp is None:
        pooled = x.mean(1)
    else:
        v = (~fp).to(x.dtype).unsqueeze(-1)
        pooled = (x * v).sum(1) / v.sum(1).clamp_min(1.0)
    return torch.nan_to_num(pooled.float()).clamp(-50, 50)


def save_student(path, enc, head, mu, sd, labels, layer, **extra):
    import torch
    torch.save({"encoder": enc.state_dict(), "head": head.state_dict(),
                "mu": mu.detach().cpu().numpy(), "sd": sd.detach().cpu().numpy(),
                "labels": list(labels), "layer": layer, **extra}, path)


def write_loader_csv(rows, dst):
    """행번호를 speaker_id 칸에 실어 AudioCountryDataset 호환 CSV 로 (읽기 실패로 빠진 행 복원용)."""
    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(LOADER_HDR)
        for i, r in enumerate(rows):
            w.writerow([r["audio_path"], "", str(i), r.get("label_en", ""), 0, r.get("duration_sec", "")])
    return dst


class LengthBucketBatchSampler:
    """길이가 비슷한 발화끼리 배치 — 패딩 낭비를 줄이고, 단건 추론(패딩 없음)과 조건을 가깝게 한다.

    배치 크기는 (배치 안 최대 길이 × 개수) ≤ max_batch_sec, 개수 ≤ max_batch 로 정한다.
    shuffle 이면 길이에 ±jitter 초 잡음을 섞어 정렬하고 배치 순서를 섞는다(epoch 마다 다름).
    """

    def __init__(self, durations, max_sec, max_batch_sec, max_batch, shuffle, seed=0, jitter=0.5):
        self.d = np.minimum(np.asarray(durations, np.float64), max_sec)
        self.max_batch_sec, self.max_batch = max_batch_sec, max_batch
        self.shuffle, self.seed, self.jitter = shuffle, seed, jitter
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _batches(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        key = self.d + (rng.uniform(-self.jitter, self.jitter, len(self.d)) if self.shuffle else 0)
        out, cur, cur_max = [], [], 0.0
        for i in np.argsort(key, kind="stable"):
            m = max(cur_max, self.d[i])
            if cur and (m * (len(cur) + 1) > self.max_batch_sec or len(cur) >= self.max_batch):
                out.append(cur)
                cur, m = [], self.d[i]
            cur.append(int(i))
            cur_max = m
        if cur:
            out.append(cur)
        if self.shuffle:
            rng.shuffle(out)
        return out

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return len(self._batches())
