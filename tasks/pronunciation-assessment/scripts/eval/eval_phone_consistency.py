"""Frozen validation selection and fresh paired evaluation of phone consistency."""

import sys as _sys
import pathlib as _pl
_root = _pl.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    if _root.parent == _root:
        raise RuntimeError("Repository root not found")
    _root = _root.parent
_sys.path[:0] = [str(_root), *(str(p) for p in sorted(_root.glob("tasks/*/src"))),
                str(_pl.Path(__file__).resolve().parents[1] / "train")]

import argparse
import json
import numpy as np

import eval_noise_training as evaluation
from summarize_noise_preprocess import paired_deltas
from train_noise_pilot import digest, write_json

PRIMARY = ("babble10", "babble5")
raw_conditions = evaluation.conditions


def configure(plan):
    evaluation.CONDITIONS = plan["eval_conditions"]
    def filtered(wav, row, seed, pool, unseen=False):
        for name, signal in raw_conditions(wav, row, seed, pool, unseen):
            if name in plan["eval_conditions"]:
                yield name, signal
    evaluation.conditions = filtered


def safety(changes, policy):
    clean = changes["clean"]
    return (clean["api_recall"] >= -policy["clean_recall_preservation_loss_max_pp"]
        and clean["preservation"] >= -policy["clean_recall_preservation_loss_max_pp"]
        and clean["api_frr"] <= policy["clean_frr_increase_max_pp"]
        and clean["jer"] <= policy["clean_jer_increase_max_pp"]
        and all(changes[c]["api_frr"] <= policy["primary_frr_increase_max_pp"] for c in PRIMARY))


def select(out, plan):
    methods = ["baseline"] + ["targeted_%d" % s for s in plan["checkpoints"]]
    rows = evaluation.read_results(out, "valid", methods)
    values = evaluation.metrics(rows)
    candidates = []
    for step in plan["checkpoints"]:
        method = "targeted_%d" % step
        changes = {c: {k: 100 * (v[method][k] - v["baseline"][k])
                      for k in ("api_recall", "api_frr", "preservation", "jer")}
                   for c, v in values.items()}
        recall = float(np.mean([changes[c]["api_recall"] for c in PRIMARY]))
        preservation = float(np.mean([changes[c]["preservation"] for c in PRIMARY]))
        safe = safety(changes, plan["selection_policy"])
        candidates.append({"step": step, "method": method, "safe": safe,
            "eligible": safe and preservation >= 0, "primary_recall_delta_pp": recall,
            "primary_preservation_delta_pp": preservation, "changes": changes})
    eligible = [v for v in candidates if v["eligible"]]
    chosen = max(eligible, key=lambda v: (v["primary_recall_delta_pp"],
                 v["primary_preservation_delta_pp"], -v["step"])) if eligible else candidates[-1]
    result = {"chosen": chosen, "candidates": candidates, "selection_split": "valid only",
        "validation_supports_use": bool(eligible and chosen["primary_recall_delta_pp"] > 0
                                          and chosen["primary_preservation_delta_pp"] > 0),
        "test_retuning": False, "plan_sha256": digest(out / "plan.json")}
    if (out / "selection.json").exists():
        if json.loads((out / "selection.json").read_text()) != result:
            raise RuntimeError("Frozen selection changed")
    else:
        write_json(out / "selection.json", result)
    write_json(out / "valid/summary.json", values)
    print("[frozen selection] " + json.dumps(chosen, ensure_ascii=False), flush=True)


def report(out, plan, bootstraps):
    frozen = json.loads((out / "selection.json").read_text())
    methods = ["baseline", "paired_control", "targeted"]
    rows = evaluation.read_results(out, "test", methods)
    values = evaluation.metrics(rows)
    baseline = sorted([r for r in rows if r["method"] == "baseline" and r["condition"] == "clean"], key=lambda r: r["k"])
    strata = [[i for i, r in enumerate(baseline) if r["cohort"] == name] for name in ("normal", "error")]
    rng = np.random.RandomState(plan["seed"] + 5)
    draws = np.concatenate([rng.choice(s, (bootstraps, len(s)), replace=True) for s in strata if s], axis=1)
    comparisons = {}
    for name, reference, candidate in (("targeted_vs_baseline", "baseline", "targeted"),
        ("targeted_vs_paired_control", "paired_control", "targeted"),
        ("paired_control_vs_baseline", "baseline", "paired_control")):
        comparisons[name] = {"primary": paired_deltas(evaluation.pooled(rows, reference, PRIMARY),
            evaluation.pooled(rows, candidate, PRIMARY), draws),
            "by_condition": {c: paired_deltas(evaluation.pooled(rows, reference, [c]),
                evaluation.pooled(rows, candidate, [c]), draws) for c in plan["eval_conditions"]}}
    primary = comparisons["targeted_vs_baseline"]["primary"]
    control = comparisons["targeted_vs_paired_control"]["primary"]
    changes = {c: {k: v["delta_pp"] for k, v in data.items()}
               for c, data in comparisons["targeted_vs_baseline"]["by_condition"].items()}
    safe = safety(changes, plan["selection_policy"])
    confirmed = all(primary[k]["ci95_pp"][0] > 0 for k in ("api_recall", "preservation"))
    contribution = all(control[k]["ci95_pp"][0] > 0 for k in ("api_recall", "preservation"))
    if not frozen["validation_supports_use"] or not safe:
        verdict, message = "not_supported", "이번 설정은 채택 기준을 충족하지 못했다."
    elif confirmed and contribution:
        verdict, message = "supported_on_synthetic_holdout", "새 합성 소음 시험에서 검출/보존 개선과 보조 학습 목표의 기여를 확인했다."
    else:
        verdict, message = "inconclusive", "개선 방향은 있으나 핵심 성능 또는 보조 학습 목표의 독립 효과가 확실하지 않다."
    decision = {"verdict": verdict, "message": message, "selected_step": frozen["chosen"]["step"],
        "validation_supports_use": frozen["validation_supports_use"], "test_safety": safe,
        "core_gain_confirmed": confirmed, "auxiliary_contribution_confirmed": contribution,
        "comparisons": comparisons, "bootstraps": bootstraps, "production_changed": False,
        "uncertainty_scope": "paired utterance bootstrap, stratified normal/error; not speaker clustered",
        "calibration": "original fixed production phone calibration; raw recognition also reported"}
    write_json(out / "decision.json", decision)
    write_json(out / "test/summary.json", values)
    lines = ["오발음 보존 + 원본/소음 일관성 학습 — 통제 실험", "", "판단: " + message,
        "검증 표본 {}개, 새 시험 표본 {}개. 선택 {} updates. 시험 후 재선택 없음.".format(
            len(plan["valid_selection"]), len(plan["test_selection"]), frozen["chosen"]["step"]),
        "학습 정답은 실제 발음 .ltr. 원본/소음 CTC + 실제 오류 음소 peak CE + 정답과 일치하는 교사의 KL.",
        "대조군은 같은 원본/소음 입력으로 CTC만 학습. 초기 모델/seed/배치/학습률/업데이트 수 동일.",
        "원본 모델이 잘못 인식한 음소는 일관성 교사로 사용하지 않았다.", "",
        "| 조건 | 모델 | API 검출 | 정상자리 헛지적 | 오발음 보존 | 원시 검출 | 정상으로 되돌림 | JER |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    names = {"baseline": "기존", "paired_control": "동일 입력 CTC", "targeted": "오발음 보존 학습"}
    for c in plan["eval_conditions"]:
        for m in methods:
            v = values[c][m]
            lines.append("| %s | %s | %s |" % (c, names[m], " | ".join("%.2f%%" % (100 * v[k])
                for k in ("api_recall", "api_frr", "preservation", "raw_recall", "reversion", "jer"))))
    lines += ["", "주 조건 = 웅성거림 10/5dB; paired bootstrap 95% 구간"]
    for name, title in (("targeted_vs_baseline", "보존 학습 − 기존"),
                        ("targeted_vs_paired_control", "보존 학습 − 동일 입력 CTC")):
        lines.append(title)
        for k, v in comparisons[name]["primary"].items():
            lines.append("  %s %+.2f%%p [%+.2f, %+.2f]" % (k, v["delta_pp"], *v["ci95_pp"]))
    lines += ["", "한계와 검증 범위",
        "- 합성 소음, 기존 학습 문항의 별도 녹음. 실제 부스/앱 녹음은 사용하지 않았다.",
        "- 이전 전처리/믹싱/소음 학습 평가 녹음과 10-08 표본을 제외했다. 이전 시험을 재튜닝하지 않았다.",
        "- 정렬 시점은 원본 모델 강제정렬 추정이다. 사람이 표시한 음소 시작/끝 시점은 아니다.",
        "- 삭제 오류는 직접 peak CE가 없고 CTC로만 학습한다. 보조 목표 두 개의 개별 기여는 분리하지 않았다.",
        "- 모든 입력을 평가. 거절되는 어려운 녹음을 평가에서 제외하지 않았다.",
        "- 화자 메타를 읽지 않아 화자 중복을 재검사하지 않았다. 신뢰구간은 발화 단위다.",
        "- 한 seed/600 updates 이하의 탐색 실험. 운영 보정표를 고정했으며 배포 적합성 시험은 별도다.",
        "- 운영 API/모델/보정표 변경 및 재시작 없음."]
    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[decision] " + verdict + ": " + message, flush=True)
    print(json.dumps(primary), flush=True)


def verify(out, plan):
    frozen = json.loads((out / "selection.json").read_text())
    step = frozen["chosen"]["step"]
    metas = [json.loads((out / "checkpoints" / m / ("step_%d" % step) / "experiment.json").read_text())
             for m in ("targeted", "paired_control")]
    for key in ("exposure_sha256", "paired_examples_seen", "micro_batches", "skipped_updates"):
        if metas[0][key] != metas[1][key]:
            raise RuntimeError("Unmatched training: " + key)
    if metas[0]["skipped_updates"]:
        raise RuntimeError("Skipped updates require control audit")
    for key in ("input_hashes", "fixed_asset_hashes", "production_hashes", "source_hashes"):
        for path, expected in plan[key].items():
            p = _pl.Path(path)
            if not p.is_absolute():
                p = _root / p
            if digest(p) != expected:
                raise RuntimeError("Changed protected input/source: " + path)
    total = 0
    for split, methods in (("valid", ["baseline"] + ["targeted_%d" % s for s in plan["checkpoints"]]),
                           ("test", ["baseline", "paired_control", "targeted"])):
        rows = evaluation.read_results(out, split, methods)
        expected = len(plan[split + "_selection"]) * len(plan["eval_conditions"]) * len(methods)
        if len(rows) != expected:
            raise RuntimeError("Evaluation matrix incomplete")
        total += len(rows)
    write_json(out / "verification.json", {"status": "complete", "evaluation_records": total,
        "training_matched": True, "selected_step": step, "all_input_asset_source_hashes_unchanged": True,
        "production_changed": False, "production_restarted": False})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("valid", "test", "report", "smoke"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstraps", type=int, default=3000)
    ap.add_argument("--smoke-ckpt")
    args = ap.parse_args()
    out = _pl.Path(args.out)
    plan = json.loads((out / "plan.json").read_text())
    configure(plan)
    if args.action == "valid":
        evaluation.evaluate_method(out, plan, "valid", "baseline", plan["ckpt"])
        for step in plan["checkpoints"]:
            evaluation.evaluate_method(out, plan, "valid", "targeted_%d" % step,
                out / "checkpoints/targeted" / ("step_%d" % step))
        select(out, plan)
    elif args.action == "test":
        step = json.loads((out / "selection.json").read_text())["chosen"]["step"]
        for method, ckpt in (("baseline", plan["ckpt"]),
            ("paired_control", out / "checkpoints/paired_control" / ("step_%d" % step)),
            ("targeted", out / "checkpoints/targeted" / ("step_%d" % step))):
            evaluation.evaluate_method(out, plan, "test", method, ckpt)
        report(out, plan, args.bootstraps)
        verify(out, plan)
    elif args.action == "smoke":
        evaluation.evaluate_method(out, plan, "valid", "smoke", args.smoke_ckpt or plan["ckpt"], limit=2)
    else:
        report(out, plan, args.bootstraps)
        verify(out, plan)


if __name__ == "__main__":
    main()
