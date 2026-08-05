# -*- coding: utf-8 -*-
"""인코더 KD 체크포인트 정밀 평가 — 도메인별 · clean(교사-train 화자 제외) 분해.

헤드 KD 와 같은 잣대로 비교하기 위해:
- 오염 마스크 = 교사(0702) 인덱스 재구성(043 제외) 후 assign seed42 의 train 화자
- 도메인 = 505 / 041 / 043 / korean
- 043·꼬리 클래스에서 교사 무지가 학생 표현으로 새어 들어갔는지가 관전 포인트

    $PY .../kd_encoder_eval.py --ckpt outputs/kd_encoder/run1/encoder_kd_best.pt
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), str(_r / "tasks/nationality-classification/experiments"),
                 *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import NAT_TASK

import argparse, csv, json, os, time
import numpy as np

ENC_DIR = NAT_TASK / "outputs" / "kd_encoder"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ENC_DIR / "run1" / "encoder_kd_best.pt"))
    ap.add_argument("--xlsr300", default=os.path.expanduser("~/models/xlsr2_300m.pt"))
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--out", default="")
    A = ap.parse_args()
    out = A.out or str(_pl.Path(A.ckpt).parent / "eval_detail.json")

    import torch, torch.nn as nn
    from shared.fairseq_guard import ensure_fairseq
    ensure_fairseq()
    from fairseq import checkpoint_utils
    from country_classifier.dataset import build_dataloader
    import kd_2b_to_300m as K
    from country_classifier.make_speaker_split import assign_speakers

    # ── test 행 + 도메인 + 오염 마스크 ──
    rows = list(csv.DictReader(open(ENC_DIR / "test.csv", encoding="utf-8")))
    ck = torch.load(A.ckpt, map_location="cpu")
    classes = list(ck["labels"]); c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[r["label_en"]] for r in rows], np.int64)
    dom = np.array([r["domain"] for r in rows], dtype=object)
    spk = np.array([r["speaker"] for r in rows], dtype=object)

    print("[오염 마스크] 교사 인덱스 재구성", flush=True)
    _, _, _, _, sps_i, lab_i, dom_i = K.build_index()
    t_items = [{"speaker_id": s, "label_name": l, "num_frames": 0}
               for s, l, d in zip(sps_i, lab_i, dom_i) if d != "asia043"]
    t_assign = assign_speakers(t_items, [8, 1, 1], seed=42)
    t_train = {s for s, a in t_assign.items() if a == "train"}
    contam = np.array([s in t_train for s in spk])
    print(f"    test {len(rows):,}행 · 오염 {contam.sum():,} ({contam.mean():.1%})", flush=True)

    # ── 모델 복원 ──
    models, _ = checkpoint_utils.load_model_ensemble([A.xlsr300])
    enc = models[0]
    LAYER = int(ck["layer"])
    del enc.encoder.layers[LAYER + 1:]
    enc.load_state_dict(ck["encoder"])
    head = nn.Linear(1024, len(classes)); head.load_state_dict(ck["head"])
    mu = torch.from_numpy(np.asarray(ck["mu"], np.float32)).cuda()
    sd = torch.from_numpy(np.asarray(ck["sd"], np.float32)).cuda()
    enc.cuda().eval(); head.cuda().eval()

    # ── 전량 forward ──
    tmp = str(ENC_DIR / "_evald_tmp.csv")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audio_path", "json_path", "speaker_id", "label_name", "label_id", "duration_sec"])
        for i, r in enumerate(rows):
            w.writerow([r["audio_path"], "", str(i), r["label_en"], 0, r["duration_sec"]])
    ld = build_dataloader(tmp, 16000, A.batch_size, shuffle=False, seed=0,
                          max_duration_sec=12.0, num_workers=8)
    h1 = np.zeros(len(rows), bool); h5 = np.zeros(len(rows), bool); seen = np.zeros(len(rows), bool)
    t0 = time.time(); done = 0
    with torch.no_grad():
        for bi, bt in enumerate(ld):
            if bt is None: continue
            idx = np.array([int(s) for s in bt["speaker_ids"]])
            wav = bt["waveforms"].cuda(); pm = bt["padding_mask"].cuda()
            with torch.autocast("cuda"):
                res = enc.extract_features(source=wav, padding_mask=pm, mask=False)
                x = res["layer_results"][LAYER][0].transpose(0, 1)
                fp = res.get("padding_mask")
                if fp is None: pooled = x.mean(1)
                else:
                    v = (~fp).to(x.dtype).unsqueeze(-1)
                    pooled = (x * v).sum(1) / v.sum(1).clamp_min(1.0)
            pooled = torch.nan_to_num(pooled.float()).clamp(-50, 50)
            L = head((pooled - mu) / sd).cpu().numpy()
            t5 = np.argpartition(-L, 4, 1)[:, :5]
            h1[idx] = L.argmax(1) == y[idx]
            h5[idx] = [a in r for a, r in zip(y[idx], t5)]
            seen[idx] = True
            done += len(idx)
            if bi % 200 == 0:
                sp_ = done / max(time.time() - t0, 1e-9)
                print(f"  {done:,}/{len(rows):,} ({sp_:.0f} utt/s · 잔여 ~{(len(rows)-done)/max(sp_,1e-9)/60:.0f}분)", flush=True)

    def block(mask, name):
        m = mask & seen
        if not m.sum(): return None
        pres = sorted(set(y[m].tolist()))
        return {"n": int(m.sum()), "top1": round(float(h1[m].mean()), 4),
                "top5": round(float(h5[m].mean()), 4),
                "macro5": round(float(np.mean([h5[m & (y == c)].mean() for c in pres])), 4)}

    res = {"ckpt": A.ckpt, "overall": block(np.ones(len(rows), bool), "all"),
           "clean": block(~contam, "clean"), "by_domain": {}, "by_domain_clean": {}}
    for dd, nm in [("asia", "505"), ("euro", "041"), ("asia043", "043"), ("korean", "korean")]:
        res["by_domain"][nm] = block(dom == dd, nm)
        res["by_domain_clean"][nm] = block((dom == dd) & ~contam, nm)
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
