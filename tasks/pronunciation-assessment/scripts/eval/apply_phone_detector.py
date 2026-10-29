# -*- coding: utf-8 -*-
"""학습된 음소 판별기를 특징 split 에 적용한다 → 자리별 오발음 확률 · 실제 발음 상위 3개.

`extract_phone_feats.py` 가 만든 split 폴더에 `p_err.npy`(float32) · `real_top3.npy`(int16, 마지막 클래스 = 삭제)를 쓴다.
정렬에 실패한 발화의 자리는 p_err = -1 로 남긴다(호출 쪽이 구분할 수 있게).
모델·전처리는 `train_phone_detector.py` 의 것을 그대로 불러 쓴다(복제하지 않는다).

    $PY tasks/pronunciation-assessment/scripts/eval/apply_phone_detector.py --detector <run>/detector.pt --feats <dir> --split pool
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import REPO_ROOT

import argparse, importlib.util, json, os, time

import numpy as np


def load_trainmod():
    p = REPO_ROOT / "tasks" / "pronunciation-assessment" / "scripts" / "train" / "train_phone_detector.py"
    spec = importlib.util.spec_from_file_location("train_phone_detector", str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--suffix", default="", help="출력 파일 접미사(p_err<suffix>.npy) — 여러 판별기를 같은 split 에 적용할 때 덮어쓰지 않게")
    a = ap.parse_args()

    import torch
    import torch.nn as nn
    TM = load_trainmod()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.detector, map_location=dev)
    args = argparse.Namespace(**ck["args"])
    use_emb = not args.no_emb
    S = TM.Split(os.path.join(a.feats, a.split), use_emb)
    S.set_l1(ck["l1_vocab"])
    stats = {k: np.asarray(v, np.float32) for k, v in ck["stats"].items()}
    V = S.lps.shape[1]
    model = TM.build_model(torch, nn, S.scal.shape[1] + 3 * V, S.el.shape[1] if use_emb else 0,
                           use_emb and S.em is not None, len(ck["l1_vocab"]), V, args).to(dev)
    model.load_state_dict(ck["model"]); model.eval()

    t0 = time.time()
    p = np.full(len(S.y), -1.0, np.float32); top3 = np.full((len(S.y), 3), -1, np.int16)
    ks_ok = np.where(S.ok)[0]; ks_ok = ks_ok[np.argsort(S.len[ks_ok])]
    with torch.no_grad():
        for i in range(0, len(ks_ok), a.bs):
            ks = ks_ok[i:i + a.bs]
            small, cid, el, em, l1, mask, _, _ = TM.make_batch(S, ks, dev, torch, stats, use_emb)
            lo, lr_ = model(small, cid, el, em, l1, mask)
            pr = torch.sigmoid(lo).float().cpu().numpy()
            rr = TM.cond_logits(lr_.float(), cid).topk(3, -1).indices.cpu().numpy()
            for b, k in enumerate(ks):
                n = int(S.len[k]); p[S.off[k]:S.off[k + 1]] = pr[b, :n]; top3[S.off[k]:S.off[k + 1]] = rr[b, :n]
    od = os.path.join(a.feats, a.split)
    np.save(os.path.join(od, f"p_err{a.suffix}.npy"), p); np.save(os.path.join(od, f"real_top3{a.suffix}.npy"), top3)
    json.dump({"detector": os.path.abspath(a.detector), "threshold": ck["threshold"], "n_sites": int(len(p)),
               "flag_rate_at_threshold": float((p >= ck["threshold"]).mean()), "sec": round(time.time() - t0, 1)},
              open(os.path.join(od, f"p_err{a.suffix}.meta.json"), "w"), ensure_ascii=False, indent=1)
    print(f"[완료] {od}/p_err{a.suffix}.npy · 자리 {len(p):,} · 임계값 {ck['threshold']:.3f} 에서 지적률 {(p >= ck['threshold']).mean():.2%} · {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
