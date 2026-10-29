# -*- coding: utf-8 -*-
"""한국인 대표 오류 짝별 **구분력(AUC)** — 모델이 "표준대로 / 한국인식"을 얼마나 잘 가르는가(문턱과 무관).

L2-ARCTIC 한국인 4명(사람 주석 · CC BY-NC · 평가 전용)의 표준 음소열(주석)을 그대로 쓰고, 대표 오류 자리마다
    LLR = log P(… 한국인식 s …) − log P(… 표준 c …)            (CTC 우도, 나머지 소리는 주석 표준대로 고정)
을 구해, 사람이 c→s 로 들은 자리(양성)와 표준대로 들은 자리(음성)를 얼마나 잘 가르는지 AUC 로 잰다.
받아쓰기(greedy)의 헛지적이 많을 때 "구분을 못 하는가 / 기준선만 치우쳤는가"를 가른다.

    $PY .../en_pair_auc_l2.py --model-dir <체크포인트> [--model-dir <다른 체크포인트> …]
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT

import argparse, collections, json, os

import numpy as np


def auc(pos, neg):
    if not pos or not neg:
        return None
    from sklearn.metrics import roc_auc_score
    return round(float(roc_auc_score([1] * len(pos) + [0] * len(neg), list(pos) + list(neg))), 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", dest="model_dirs", action="append", required=True)
    ap.add_argument("--l2arctic", default=os.path.join(str(DATA_ROOT), "english", "l2arctic_v5"))
    ap.add_argument("--out", default=None, help="결과 JSON")
    a = ap.parse_args()
    import soundfile as sf
    import torch, torchaudio
    from mdd_en.audio import trim_edges
    from mdd_en.gop import _ctc_loglik
    from mdd_en.l1_ko_errors import strip
    from mdd_en.l2arctic import annotation_words, load_speakers
    from mdd_en.pseudo import candidates
    from mdd_en.recognizer import PhoneRecognizer

    rows = load_speakers(a.l2arctic)
    wavs = []
    for r in rows:
        x, sr = sf.read(r["wav"], dtype="float32", always_2d=True)
        x = torchaudio.functional.resample(torch.from_numpy(x.mean(1)), sr, 16000).numpy()
        wavs.append(trim_edges(x))
    report = {}
    for md in a.model_dirs:
        rec = PhoneRecognizer(md, "arpa")
        idx = dict((p, i) for i, p in enumerate(rec.names))
        to_ids = lambda ph: [idx[p] for p in ph]
        llr = collections.defaultdict(lambda: ([], []))          # 규칙 → (양성 LLR, 음성 LLR)
        for r, x in zip(rows, wavs):
            ws = annotation_words(r)
            canon = [strip(p) for w in ws for p in w["ph"]]
            act = [y for w in ws for y in w["act"]]
            cands = [c for c in candidates(ws) if c[3] not in (None, "+AH")]
            if not cands:
                continue
            with torch.no_grad():
                post, _ = rec.phone_post(rec.log_probs(x))
            seqs = [to_ids(canon)] + [to_ids(canon[:p] + [alt] + canon[p + 1:]) for p, _, _, alt in cands]
            ll = _ctc_loglik(post.float(), seqs)
            if not np.isfinite(ll[0]) or ll[0] < -1e3:
                continue
            for (p, name, c, alt), v in zip(cands, ll[1:]):
                d = float(v - ll[0])
                if act[p] == alt:
                    llr[name][0].append(d)
                elif act[p] == c:
                    llr[name][1].append(d)
        res, allp, alln = {}, [], []
        for name, (pos, neg) in sorted(llr.items(), key=lambda kv: -len(kv[1][0])):
            res[name] = {"auc": auc(pos, neg), "n_pos": len(pos), "n_neg": len(neg),
                         "greedy_like_fa": round(float(np.mean(np.array(neg) > 0)), 3) if neg else None,
                         "greedy_like_recall": round(float(np.mean(np.array(pos) > 0)), 3) if pos else None}
            allp += pos; alln += neg
        res["_all"] = {"auc": auc(allp, alln), "n_pos": len(allp), "n_neg": len(alln)}
        report[md] = res
        print("== %s\n   전체 AUC %s (양성 %d · 음성 %d)" % (md, res["_all"]["auc"], len(allp), len(alln)))
        print("   " + " · ".join("%s AUC %s (%d/%d, LLR>0 재현 %s 헛 %s)" % (k, v["auc"], v["n_pos"], v["n_neg"],
                                  v["greedy_like_recall"], v["greedy_like_fa"])
                                  for k, v in res.items() if k != "_all" and v["n_pos"] >= 10), flush=True)
    if a.out:
        json.dump(report, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
