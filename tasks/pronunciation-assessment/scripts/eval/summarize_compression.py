# -*- coding: utf-8 -*-
"""경량화 변형 결과 모아 보기 — 채점 JSON 들을 한 표로.

읽는 파일 (--eval_dir 안, 변형 이름 v 기준)
    tag71_{set}_{v}.json        mdd_eval_tag71.py      — weighted F1 · 오류 F1 · P/R
    fixedref_{set}_{v}.json     mdd_eval_fixedref.py   — 고정 정답지 weighted F1 · 오류 F1
    cmp_{set}_{ref}_vs_{v}.json eval_score_backend.py compare — 점수 3종 보존(기준 대비)
    {set}_feats_{v}.jsonl.meta.json                    — 모델 파일 크기
없는 파일은 빈칸으로 둔다. 기준(--ref) 대비 차이(Δ)를 같이 적는다.

    $PY tasks/pronunciation-assessment/scripts/eval/summarize_compression.py \\
        --eval_dir /data/speech-ai/out/compress/mdd/eval --set sub10k \\
        --ref torch_gpu --variants torch_gpu,ort_fp32,ort_dyn_matmul_pc,ort_dyn_matmul --out summary_sub10k.md
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, json, os


def load(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def fmt(x, nd=4, sign=False):
    if x is None:
        return ""
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--set", default="sub10k")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--out", required=True, help=".md (같은 이름 .json 도 쓴다)")
    a = ap.parse_args()
    E, S = a.eval_dir, a.set
    names = a.variants.split(",")

    rows = {}
    for v in names:
        t = load(os.path.join(E, f"tag71_{S}_{v}.json"))
        f = load(os.path.join(E, f"fixedref_{S}_{v}.json"))
        c = load(os.path.join(E, f"cmp_{S}_{a.ref}_vs_{v}.json"))
        m = load(os.path.join(E, f"{S}_feats_{v}.jsonl.meta.json"))
        rows[v] = {
            "model_mib": (m["model_bytes"] / 2**20) if m and "model_bytes" in m else None,
            "tag71_f1w": t["jamo"]["f1_weighted"] if t else None, "tag71_err_f1": t["jamo"]["err_f1"] if t else None,
            "tag71_p": t["jamo"]["precision"] if t else None, "tag71_r": t["jamo"]["recall"] if t else None,
            "fixed_f1w": f["jamo"]["f1_weighted"] if f else None, "fixed_err_f1": f["jamo"]["err_f1"] if f else None,
            "pred_equal_vs_ref": c["pred_equal_rate"] if c else None,
            "score": {d: {"pred_pcc": s["pred_pcc_a_b"], "mean_abs": s["pred_mean_abs_diff"],
                          "pcc_spk": s["pcc_spk_b"], "d_pcc_spk": s["d_pcc_spk"]}
                      for d, s in (c["scores"].items() if c else [])},
        }
    ref = rows.get(a.ref, {})

    def d(v, k):
        return None if rows[v].get(k) is None or ref.get(k) is None else rows[v][k] - ref[k]

    lines = [f"### MDD 정확도 — `{S}` (기준 `{a.ref}`)", "",
             "| 변형 | 모델 MiB | weighted F1 | Δ | 오류 F1 | Δ | P / R | 고정정답 F1 | 고정정답 오류 F1 | Δ |",
             "|---|--:|--:|--:|--:|--:|---|--:|--:|--:|"]
    for v in names:
        r = rows[v]
        pr = f"{fmt(r['tag71_p'])} / {fmt(r['tag71_r'])}" if r["tag71_p"] is not None else ""
        lines.append(f"| {v} | {fmt(r['model_mib'], 0)} | {fmt(r['tag71_f1w'])} | {fmt(d(v, 'tag71_f1w'), sign=True)} | "
                     f"{fmt(r['tag71_err_f1'])} | {fmt(d(v, 'tag71_err_f1'), sign=True)} | {pr} | "
                     f"{fmt(r['fixed_f1w'])} | {fmt(r['fixed_err_f1'])} | {fmt(d(v, 'fixed_err_f1'), sign=True)} |")
    lines += ["", f"### 점수 3종 보존 — 기준 `{a.ref}` 특징 → 같은 회귀기", "",
              "| 변형 | 인식 동일률 | 발음숙달 점수상관 · 평균|Δ| · 화자PCC Δ | 이해가능 | 유창성 |", "|---|--:|---|---|---|"]
    for v in names:
        r = rows[v]
        if not r["score"]:
            lines.append(f"| {v} | {fmt(r['pred_equal_vs_ref'])} | | | |")
            continue
        cells = []
        for dim in ("pron", "comp", "flu"):
            s = r["score"].get(dim)
            cells.append(f"{fmt(s['pred_pcc'])} · {fmt(s['mean_abs'])} · {fmt(s['d_pcc_spk'], sign=True)}" if s else "")
        lines.append(f"| {v} | {fmt(r['pred_equal_vs_ref'])} | " + " | ".join(cells) + " |")
    md = "\n".join(lines) + "\n"
    with open(a.out, "w", encoding="utf-8") as fo:
        fo.write(md)
    with open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8") as fo:
        json.dump(rows, fo, ensure_ascii=False, indent=1)
    print(md)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
