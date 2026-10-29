# -*- coding: utf-8 -*-
"""잠근 외부 검증 — AIHub 037 SPK(말하기 평가, PRN 에 없는 화자)에서 예전·새 문장 점수가 사람 점수를 얼마나 따라가나.

SPK 사람 점수는 말하기 전반 1개(평가자 3명 평균, 약 1~9)라 낭독 발음 점수와 같은 잣대가 아니다.
그래서 절대 수치보다 **예전(speechocean762 기준) 대 새(037 기준) 비교**로 읽는다. 학습·모델 선택에는 쓰지 않았다.

입력: `en_extract_037_feats.py --list $MDD_OUTPUTS/english/aihub037_spk/test.jsonl --carry rating,task` 결과
      + 새 점수기 폴더(sent_scorer_ko.joblib)

    $PY .../en_eval_score_spk037.py --feats $MDD_OUTPUTS/english/score037/synth-d-spk \\
        --scorer-dir $MDD_OUTPUTS/english/scorer/synth-d-ko037
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, collections, glob, json, os

import numpy as np


def pcc(a, b):
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--scorer-dir", dest="scorer_dir", required=True)
    a = ap.parse_args()
    from mdd_en.sent_score import KoreanSentenceScorer
    ko = KoreanSentenceScorer(os.path.join(a.scorer_dir, "sent_scorer_ko.joblib"))
    R = []
    for p in sorted(glob.glob(os.path.join(a.feats, "feats.*.jsonl"))):
        R += [json.loads(l) for l in open(p, encoding="utf-8")]
    y = np.array([r["rating"] for r in R], float)
    spk = [r["spk"] for r in R]
    keys = ("overall", "pronunciation", "fluency", "rhythm")
    old = dict((k, np.array([r["api"][k] or 0 for r in R], float)) for k in keys)
    pred = [ko.predict_feats(r["feats"])[0] for r in R]
    new = dict((k, np.array([p[k] or 0 for p in pred], float)) for k in keys)
    idx = collections.defaultdict(list)
    for i, s in enumerate(spk):
        idx[s].append(i)
    g = [np.array(v) for v in idx.values()]
    sp = lambda v: pcc([v[i].mean() for i in g], [y[i].mean() for i in g])
    lines = ["## 잠근 외부 검증 — 037 SPK 말하기 평가 (PRN 에 없는 화자 %d명 · 발화 %d)" % (len(g), len(R)), "",
             "사람 점수 = 말하기 전반 1개(평가자 3명 평균). 낭독 발음 점수와 같은 잣대가 아니므로 예전 대 새 비교로 읽는다.", "",
             "| 칸 | 발화: 예전 | 발화: 새 | 화자 평균: 예전 | 화자 평균: 새 |", "|---|---:|---:|---:|---:|"]
    rep = {"n_utts": len(R), "n_spk": len(g), "pcc": {}}
    for k in keys:
        m = {"utt_old": pcc(old[k], y), "utt_new": pcc(new[k], y), "spk_old": sp(old[k]), "spk_new": sp(new[k])}
        rep["pcc"][k] = dict((x, round(v, 4)) for x, v in m.items())
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f |" % (k, m["utt_old"], m["utt_new"], m["spk_old"], m["spk_new"]))
    lines += ["", "과제별 overall 발화 상관(예전 → 새):", ""]
    for t in sorted(set(r["task"] for r in R)):
        sel = np.array([r["task"] == t for r in R])
        if sel.sum() >= 100:
            lines.append("- %s (%d발화): %.3f → %.3f" % (t, sel.sum(), pcc(old["overall"][sel], y[sel]), pcc(new["overall"][sel], y[sel])))
    json.dump(rep, open(os.path.join(a.scorer_dir, "sent_scorer_ko_spk.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(a.scorer_dir, "sent_scorer_ko_spk.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
