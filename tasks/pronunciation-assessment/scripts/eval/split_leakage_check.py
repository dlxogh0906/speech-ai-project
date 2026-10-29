# -*- coding: utf-8 -*-
"""매니페스트 사이의 **화자 누출 · 문장 누출**을 센다 — 평가 조건을 문서에 쓰기 전에 돌릴 것.

왜 필요한가: `manifest_foreign_clean_v2/test_unseen` 은 예전 레시피에서 문항 30종을 빼서 만든
split 이라 이름에 "unseen" 이 붙어 있다. 그러나 Stage 2 (`manifest_full_tag_replay10`) 는
**모든 문항을 학습에 쓴다.** 그 모델에 이 split 을 쓰면 "안 본 문항" 이 아니다.
이 스크립트로 조건을 사실 확인한 뒤 표 제목을 정한다.

문항 식별: 메타의 `norm_prompt`(있으면) → 없으면 stem 의 항목코드(RS001 등).
화자 식별: stem 앞 5자리.

    $PY .../split_leakage_check.py --train $MDD_OUTPUTS/manifest_full_tag_replay10:train \\
        --test $MDD_OUTPUTS/manifest_full_tag_replay10:test \\
             $MDD_OUTPUTS/manifest_foreign_clean_v2:test_unseen
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import argparse, json, os, re

ITEM = re.compile(r"-([A-Z]{2}\d{3})-")


def read(spec):
    mdir, split = spec.rsplit(":", 1)
    spk, item, prompt, n = set(), set(), set(), 0
    with open(os.path.join(mdir, "%s.tsv" % split), encoding="utf-8") as f:
        next(f)
        for line in f:
            stem = os.path.basename(line.split("\t")[0])[:-4]
            spk.add(stem[:5])
            m = ITEM.search(stem)
            if m:
                item.add(m.group(1))
            n += 1
    mp = os.path.join(mdir, "%s.meta.jsonl" % split)
    if os.path.exists(mp):
        for line in open(mp, encoding="utf-8"):
            d = json.loads(line)
            if d.get("norm_prompt"):
                prompt.add(d["norm_prompt"])
    return {"spec": spec, "utts": n, "speakers": spk, "items": item, "prompts": prompt}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="<매니페스트경로>:<split>")
    ap.add_argument("--test", required=True, nargs="+")
    ap.add_argument("--out", default=None, help="JSON 으로 저장할 경로(선택)")
    a = ap.parse_args()

    tr = read(a.train)
    print("[train] %s · 발화 %d · 화자 %d · 문항코드 %d · 문장 %d"
          % (tr["spec"], tr["utts"], len(tr["speakers"]), len(tr["items"]), len(tr["prompts"])))
    out = {"train": {k: (len(v) if isinstance(v, set) else v) for k, v in tr.items()}, "tests": []}
    for spec in a.test:
        te = read(spec)
        sp = tr["speakers"] & te["speakers"]
        it = tr["items"] & te["items"]
        pr = tr["prompts"] & te["prompts"]
        verdict = ("speaker-disjoint" if not sp else "화자 누출 %d명" % len(sp))
        verdict += " · " + ("sentence-disjoint" if not it else "문항 누출 %d/%d" % (len(it), len(te["items"])))
        print("[test ] %s · 발화 %d · 화자 %d(겹침 %d) · 문항 %d(겹침 %d) · 문장겹침 %d → %s"
              % (te["spec"], te["utts"], len(te["speakers"]), len(sp), len(te["items"]), len(it), len(pr), verdict))
        out["tests"].append({"spec": spec, "utts": te["utts"], "speakers": len(te["speakers"]),
                             "speaker_overlap": len(sp), "items": len(te["items"]), "item_overlap": len(it),
                             "prompt_overlap": len(pr), "verdict": verdict,
                             "overlapping_items": sorted(it)[:50]})
    if a.out:
        json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
        print("[saved] %s" % a.out)


if __name__ == "__main__":
    main()
