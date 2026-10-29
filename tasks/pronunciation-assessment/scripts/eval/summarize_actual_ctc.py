# -*- coding: utf-8 -*-
"""OLD / E1 / E2 / E2-F 결과표 — eval_actual_ctc.py 의 metrics.json 과 매니페스트 stats 를 모은다.

    $PY .../summarize_actual_ctc.py --exp $MDD_OUTPUTS/experiments --out tasks/pronunciation-assessment/results/10_actual_ctc_e1_e2.md

모델별 기대 파일 (없으면 표에 '-')
  <exp>/<model_dir>/<prefix>_test_seen.metrics.json · _test_unseen.metrics.json · _native_dev.metrics.json
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import argparse, json, os, time

MODELS = [("OLD", "old_reference", "old", None),
          ("E1", "e1_foreign_clean", "e1", "manifest_foreign_clean_v2"),
          ("E2", "e2_native20", "e2", "manifest_mix_f80_n20"),
          ("E2-F", "e2f_foreign_adapt", "e2f", "manifest_mix_f80_n20")]


def load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def g(d, *ks):
    for k in ks:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def fmt(v, pct=False):
    if v is None:
        return "-"
    if isinstance(v, (int,)) and not pct:
        return f"{v:,}"
    return f"{v * 100:.2f}%" if pct else f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=str(MDD_OUTPUTS / "experiments"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=None,
                    help='"이름|실험폴더|파일접두어|매니페스트(-면 없음)|학습 설명(선택)" — 기본은 OLD/E1/E2/E2-F')
    a = ap.parse_args()
    notes = {}
    if a.models:
        MODELS = []
        for spec in a.models:
            f = (spec.split("|") + [""] * 5)[:5]
            MODELS.append((f[0], f[1], f[2], None if f[3] in ("", "-") else f[3]))
            if f[4]:
                notes[f[0]] = f[4]
    else:
        MODELS = globals()["MODELS"]
    M = {}
    for name, d, pre, mani in MODELS:
        base = os.path.join(a.exp, d)
        M[name] = {"seen": load(f"{base}/{pre}_test_seen.metrics.json"), "unseen": load(f"{base}/{pre}_test_unseen.metrics.json"),
                   "native": load(f"{base}/{pre}_native_dev.metrics.json"), "run": load(f"{base}/run_info.json")}
        if name in notes:
            M[name]["train"] = {"note": notes[name]}
        elif name == "E2-F":
            M[name]["train"] = {"note": "15k 스텝 E2 혼합 데이터 → 5k 스텝 E1 외국인 데이터"}
        elif mani:
            st = load(os.path.join(MDD_OUTPUTS, mani, "stats.json")) or {}
            if "splits" in st:
                t = st["splits"]["train"]
                M[name]["train"] = {"utt": t["utterances"], "hours": t["hours"], "texts": t["prompts"], "speakers": t["speakers"], "ratio": "100 / 0"}
            else:
                fs = load(os.path.join(MDD_OUTPUTS, "manifest_foreign_clean_v2", "stats.json"))["splits"]["train"]
                M[name]["train"] = {"utt": st["foreign"]["utterances"] + st["native"]["utterances"],
                                    "hours": round(st["foreign"]["hours"] + st["native"]["hours"], 2),
                                    "texts": st["foreign"]["unique_texts"] + st["native"]["unique_texts"],
                                    "speakers": f"{fs['speakers']:,} + 원어민(ID 없음)",
                                    "ratio": f"{(1 - st['native_hour_ratio']) * 100:.0f} / {st['native_hour_ratio'] * 100:.0f} (시간)"}
        else:
            M[name]["train"] = {"note": "기존 서비스 모델 tag_all_native (재학습 안 함)"}
    rows = [
        ("Foreign Seen JER", lambda m: g(m["seen"], "jer", "jer"), True),
        ("Foreign Unseen JER", lambda m: g(m["unseen"], "jer", "jer"), True),
        ("  Substitution", lambda m: g(m["unseen"], "jer", "sub"), True),
        ("  Deletion", lambda m: g(m["unseen"], "jer", "del"), True),
        ("  Insertion", lambda m: g(m["unseen"], "jer", "ins"), True),
        ("Reversion Rate (unseen)", lambda m: g(m["unseen"], "sites", "reversion"), True),
        ("Preservation Rate (unseen)", lambda m: g(m["unseen"], "sites", "preservation"), True),
        ("Other Error Rate (unseen)", lambda m: g(m["unseen"], "sites", "other"), True),
        ("Normal-site Error Rate (unseen)", lambda m: g(m["unseen"], "sites", "normal_site_error"), True),
        ("MDD Precision (unseen)", lambda m: g(m["unseen"], "mdd", "precision"), False),
        ("MDD Recall (unseen)", lambda m: g(m["unseen"], "mdd", "recall"), False),
        ("MDD F1 (unseen)", lambda m: g(m["unseen"], "mdd", "f1"), False),
        ("FAR (unseen)", lambda m: g(m["unseen"], "mdd", "far"), True),
        ("FRR (unseen)", lambda m: g(m["unseen"], "mdd", "frr"), True),
        ("Strict-text-unseen JER", lambda m: g(m["unseen"], "strict_text_unseen", "jer", "jer"), True),
        ("Reversion (seen)", lambda m: g(m["seen"], "sites", "reversion"), True),
        ("Existing Seen Weighted F1", lambda m: g(m["seen"], "legacy", "f1_weighted"), False),
        ("Native JER (Kspon dev)", lambda m: g(m["native"], "jer", "jer"), True),
    ]
    names = [n for n, *_ in MODELS]
    L = ["# 실제 발음 CTC — OLD / E1 / E2 / E2-F", "", f"생성 {time.strftime('%Y-%m-%d %H:%M')} · greedy CTC (LM 없음)", "",
         "| Metric | " + " | ".join(names) + " |", "|---|" + "--:|" * len(names)]
    for label, fn, pct in rows:
        L.append(f"| {label} | " + " | ".join(fmt(fn(M[n]), pct) for n in names) + " |")
    for label, key in (("Train utterances", "utt"), ("Train hours", "hours"), ("Unique texts", "texts"),
                       ("Unique speakers", "speakers"), ("Foreign/native ratio", "ratio")):
        L.append(f"| {label} | " + " | ".join(str(M[n]["train"].get(key, M[n]["train"].get("note", "-"))) if isinstance(M[n]["train"].get(key), (str, type(None))) else f"{M[n]['train'][key]:,}" for n in names) + " |")
    L += ["", "## 읽는 법", "",
          "- JER = (S+D+I)/N, 기준 = TAG 실제 발음. 자리 지표는 LAB 표준발음 축(치환·삭제형 오발음) — 삽입형은 JER 에만 들어간다.",
          "- 보존 + 되돌림 + other = 100% (오발음 자리). FAR = 되돌림률, FRR = 정상 자리에서 표준과 다르게 낸 비율.",
          "- ⚠️ OLD 는 287문항 전체로 학습됐다 — OLD 에게 test_unseen 문항은 **본 문장**이다. OLD 의 unseen 수치는 암기 이득이 섞인 참고값.",
          "- Strict-text-unseen = test_unseen 중 원어민 학습 문장에도 없는 문항만 (E2·E2-F 는 원어민 문장 1개 '배고파요' 제외).",
          "- Native JER 은 sanity check 이며 모델 선택 기준이 아니다."]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(M, open(os.path.splitext(a.out)[0] + ".json", "w"), ensure_ascii=False, indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
