"""Select a noise-trained checkpoint on valid; run a frozen paired holdout.

Compare original, noise continuation and matched clean continuation. Preserve
actual-phone labels and reproduce the current production error flag rule.
"""

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
import collections
import gc
import json
import random
import time

import numpy as np

import compare_noise_mixing as mixing
import compare_noise_preprocess as previous
from summarize_noise_preprocess import aggregate, paired_deltas
from train_noise_pilot import digest, noise_paths, write_json
from mdd.noise_augment import reverberate

PRIMARY = ("babble10", "babble5")
CONDITIONS = [c[0] for c in previous.CONDITIONS] + ["reverb", "reverb_babble5"]


def conditions(wav, row, seed, pool, unseen=False):
    for name, noisy in mixing.noisy_conditions(wav, row, seed, pool):
        if not unseen or name in ("clean", "babble10", "babble5"):
            yield name, noisy
    if not unseen:
        reverb = reverberate(wav, seed * 7919 + row, 0.35)
        yield "reverb", reverb
        babble = previous.make_babble(random.Random(seed * 1000003 + row), pool, len(wav))
        yield "reverb_babble5", previous.mix(reverb, babble, 5)


def evaluate_method(out, plan, split, method, ckpt, limit=0):
    import torch
    from mdd.gop_feats_v2 import HFDecoder, forward_lprobs
    from mdd.speechsuper import SpeechSuperFormatter
    from mdd.vad import trim

    folder = out / split
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (method + ".jsonl")
    complete = folder / (method + ".complete.json")
    if complete.exists():
        info = json.loads(complete.read_text())
        if info["plan_sha256"] != digest(out / "plan.json") or info["ckpt"] != str(ckpt):
            raise RuntimeError("Stale evaluation output")
        return
    if path.exists():
        raise RuntimeError("Partial evaluation output exists: " + str(path))
    source = plan["unseen_manifest"] if split == "unseen" else plan["manifest"]
    source_split = "test" if split == "unseen" else split
    root, rows, tags, canons = mixing.load_split(source, source_split)
    selection = plan[split + "_selection"]
    if limit:
        selection = selection[:limit]
    pool = noise_paths(plan, "valid" if split == "valid" else "test")
    seed = plan["seed"] + (1 if split == "valid" else 2)
    decoder = HFDecoder(str(ckpt))
    formatter = SpeechSuperFormatter(plan["norms"], calib_path=plan["calib"])
    if formatter.cal is None:
        raise RuntimeError("Production phone calibration missing")
    forward_lprobs(decoder, [np.zeros(2 * previous.SR, np.float32)])
    started = time.monotonic()
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for k, (i, cohort) in enumerate(selection):
            wav = previous.read_wav(root / rows[i][0])
            for condition, noisy in conditions(wav, i, seed, pool, split == "unseen"):
                t0 = time.perf_counter()
                processed, vinfo = trim(noisy, previous.SR)
                gate = previous.audio_check(noisy)
                preprocessing_ms = (time.perf_counter() - t0) * 1000
                result = mixing.infer(decoder, [processed], canons[i].split(), tags[i].split(), formatter, 1)[0]
                result.update(k=k, cohort=cohort, condition=condition, method=method,
                              original_check_pass=previous.audio_check(wav)["pass"],
                              post_denoise_check_pass=gate["pass"], silero_check_pass=None,
                              silero_no_speech=False, prep_ms=preprocessing_ms,
                              total_ms=preprocessing_ms + result["model_ms"],
                              original_sec=len(wav) / previous.SR, processed_sec=len(processed) / previous.SR,
                              cut_sec=vinfo.get("cut_sec", 0))
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                count += 1
            stream.flush()
            if k == 0 or (k + 1) % 25 == 0 or k + 1 == len(selection):
                elapsed = time.monotonic() - started
                print("[eval %s %s] %d/%d %.1fs eta=%.1fs" % (
                    split, method, k + 1, len(selection), elapsed,
                    elapsed / (k + 1) * (len(selection) - k - 1)), flush=True)
    condition_names = ["clean", "babble10", "babble5"] if split == "unseen" else CONDITIONS
    if count != len(selection) * len(condition_names):
        raise RuntimeError("Incomplete matrix")
    write_json(complete, {"method": method, "split": split, "ckpt": str(ckpt),
               "n_utts": len(selection), "n_records": count,
               "conditions": condition_names, "plan_sha256": digest(out / "plan.json"),
               "ckpt_hashes": {p.name: digest(p) for p in _pl.Path(ckpt).glob("*.safetensors")},
               "calib_sha256": digest(plan["calib"]), "norms_sha256": digest(plan["norms"]),
               "elapsed_sec": time.monotonic() - started,
               "batch_size": 1, "label_reference": "unchanged human actual-phone .ltr",
               "all_inputs_scored_including_gate_rejects": True})
    del decoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def read_results(out, split, methods):
    result = []
    for method in methods:
        folder = out / split
        info = json.loads((folder / (method + ".complete.json")).read_text())
        rows = [json.loads(line) for line in (folder / (method + ".jsonl")).read_text().splitlines()]
        if len(rows) != info["n_records"]:
            raise RuntimeError("Incomplete result file")
        identities = {(r["k"], r["condition"]) for r in rows}
        if len(identities) != len(rows):
            raise RuntimeError("Duplicate result identities")
        result += rows
    truth = {}
    for r in result:
        current = (r["n_err"], r["n_ok"], r["jer_den"], r["cohort"])
        if truth.setdefault(r["k"], current) != current:
            raise RuntimeError("Truth changed across paired conditions/models")
    return result


def metrics(rows):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["condition"], r["method"])].append(r)
    result = {}
    for (condition, method), group in groups.items():
        result.setdefault(condition, {})[method] = aggregate(group)
    return result


def selection(out, plan):
    methods = ["baseline"] + ["noise_%d" % s for s in plan["checkpoints"]]
    rows = read_results(out, "valid", methods)
    values = metrics(rows)
    policy = plan["selection_policy"]
    candidates = []
    for step in plan["checkpoints"]:
        method = "noise_%d" % step
        changes = {condition: {metric: 100 * (v[method][metric] - v["baseline"][metric])
                   for metric in ("api_recall", "api_frr", "preservation", "jer")}
                   for condition, v in values.items()}
        clean = changes["clean"]
        safe = (clean["api_recall"] >= -policy["clean_recall_preservation_loss_max_pp"]
                and clean["preservation"] >= -policy["clean_recall_preservation_loss_max_pp"]
                and clean["api_frr"] <= policy["clean_frr_increase_max_pp"]
                and clean["jer"] <= policy["clean_jer_increase_max_pp"]
                and all(changes[c]["api_frr"] <= policy["primary_frr_increase_max_pp"] for c in PRIMARY))
        candidates.append({"step": step, "method": method, "safe": safe,
                           "primary_preservation": float(np.mean([values[c][method]["preservation"] for c in PRIMARY])),
                           "primary_recall_delta_pp": float(np.mean([changes[c]["api_recall"] for c in PRIMARY])),
                           "primary_preservation_delta_pp": float(np.mean([changes[c]["preservation"] for c in PRIMARY])),
                           "changes": changes})
    eligible = [c for c in candidates if c["safe"]]
    chosen = max(eligible, key=lambda c: (c["primary_preservation"], -c["step"])) if eligible else candidates[-1]
    result = {"candidates": candidates, "chosen": chosen,
              "validation_supports_use": bool(eligible and chosen["primary_preservation_delta_pp"] > 0
                                              and chosen["primary_recall_delta_pp"] > 0),
              "selection_split": "valid only", "policy": policy,
              "test_retuning": False, "plan_sha256": digest(out / "plan.json")}
    if (out / "selection.json").exists():
        if json.loads((out / "selection.json").read_text()) != result:
            raise RuntimeError("Refusing to change frozen selection")
    else:
        write_json(out / "selection.json", result)
    write_json(out / "valid/summary.json", values)
    print("[frozen selection] step=%d safe=%s preservation_delta=%+.2fpp recall_delta=%+.2fpp" %
          (chosen["step"], chosen["safe"], chosen["primary_preservation_delta_pp"],
           chosen["primary_recall_delta_pp"]), flush=True)
    return result


def pooled(rows, method, names):
    grouped = collections.defaultdict(list)
    for row in rows:
        if row["method"] == method and row["condition"] in names:
            grouped[row["k"]].append(row)
    result = []
    for k in sorted(grouped):
        records = grouped[k]
        if {r["condition"] for r in records} != set(names):
            raise RuntimeError("Incomplete paired conditions")
        result.append({"k": k, **{key: sum(r[key] for r in records) for key in
                       ("api_tp", "n_err", "api_fp", "n_ok", "preserved", "jer_num", "jer_den")}})
    return result


def report(out, plan, bootstraps):
    frozen = json.loads((out / "selection.json").read_text())
    methods = ["baseline", "clean_control", "noise_trained"]
    rows = read_results(out, "test", methods)
    unseen = read_results(out, "unseen", methods)
    values, unseen_values = metrics(rows), metrics(unseen)
    clean = sorted([r for r in rows if r["method"] == "baseline" and r["condition"] == "clean"], key=lambda r: r["k"])
    rng = np.random.RandomState(plan["seed"] + 5)
    strata = [[i for i, r in enumerate(clean) if r["cohort"] == cohort] for cohort in ("normal", "error")]
    draws = np.concatenate([rng.choice(s, (bootstraps, len(s)), replace=True) for s in strata if s], axis=1)
    comparisons = {}
    for name, reference, candidate in (("noise_vs_baseline", "baseline", "noise_trained"),
                                        ("noise_vs_clean_control", "clean_control", "noise_trained"),
                                        ("clean_control_vs_baseline", "baseline", "clean_control")):
        comparisons[name] = {"primary": paired_deltas(pooled(rows, reference, PRIMARY),
                                                     pooled(rows, candidate, PRIMARY), draws),
                             "by_condition": {c: paired_deltas(pooled(rows, reference, [c]),
                                                               pooled(rows, candidate, [c]), draws) for c in CONDITIONS}}
    primary = comparisons["noise_vs_baseline"]["primary"]
    control = comparisons["noise_vs_clean_control"]["primary"]
    clean_changes = comparisons["noise_vs_baseline"]["by_condition"]["clean"]
    safe = (clean_changes["api_recall"]["delta_pp"] >= -2
            and clean_changes["preservation"]["delta_pp"] >= -2
            and clean_changes["api_frr"]["delta_pp"] <= 0.3
            and clean_changes["jer"]["delta_pp"] <= 1
            and all(comparisons["noise_vs_baseline"]["by_condition"][c]["api_frr"]["delta_pp"] <= 0.5 for c in PRIMARY))
    core_confirmed = all(primary[name]["ci95_pp"][0] > 0 for name in ("preservation", "api_recall"))
    noise_contribution = control["preservation"]["ci95_pp"][0] > 0
    if not frozen["validation_supports_use"] or not safe:
        verdict = "not_supported"
        message = "이번 소규모 학습 설정은 적용 기준을 충족하지 못했다."
    elif core_confirmed and noise_contribution:
        verdict = "supported_on_synthetic_holdout"
        message = "합성 소음에서 음소 오류 검출/보존 개선을 확인했다. 실제 녹음과 새 모델 보정표 검증은 별도로 필요하다."
    else:
        verdict = "inconclusive"
        message = "개선 방향은 있으나 핵심 검출 개선 또는 소음 학습의 독립 효과가 아직 확실하지 않다."
    decision = {"verdict": verdict, "message": message, "selected_step": frozen["chosen"]["step"],
                "validation_supports_use": frozen["validation_supports_use"], "test_safety": safe,
                "core_gain_confirmed": core_confirmed, "noise_contribution_confirmed": noise_contribution,
                "comparisons": comparisons, "bootstraps": bootstraps, "production_changed": False,
                "uncertainty_scope": "paired utterance bootstrap stratified normal/error; not speaker-clustered",
                "calibration": "original phone calibration fixed for all models; raw metrics also reported"}
    write_json(out / "decision.json", decision)
    write_json(out / "test/summary.json", values)
    write_json(out / "unseen/summary.json", unseen_values)
    labels = {"baseline": "기존 모델", "clean_control": "깨끗한 음성 추가 학습", "noise_trained": "소음 증강 학습"}
    noise_training = json.loads((out / "checkpoints/noise/complete.json").read_text())
    lines = ["발음평가 소음 증강 학습 — 소규모 통제 실험", "", "판단: " + message,
             "선정 학습 풀: 실제 발음 TAG {:,}개(정상60%/오발음40%), 최대 {} updates.".format(
                 len(plan["train_indices"]), plan["training"]["max_updates"]),
             "실제 학습 입력: {:,}개. 소음 모델 학습 {:.1f}초.".format(
                 noise_training["training_examples_seen"], noise_training["elapsed_sec"]),
             "후보 선택 %d개, 최종 시험 %d개, 미관측 문단 참고 %d개." %
             (len(plan["valid_selection"]), len(plan["test_selection"]), len(plan["unseen_selection"])),
             "선택: %d update. 선택과 비교 기준은 시험 전에 고정했다." % frozen["chosen"]["step"],
             "소음 모델은 깨끗한 음성50%, 증강50%. CTC 정답은 실제 발음 .ltr 그대로다.",
             "대조 모델은 같은 초기 모델/음성/배치/학습률/업데이트 수로 깨끗한 음성만 추가 학습했다.", "",
             "| 조건 | 방법 | API 검출 | 정상자리 헛지적 | 오발음 보존 | 원시 검출 | 원시 헛지적 | JER |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        for method in methods:
            v = values[condition][method]
            lines.append("| %s | %s | %s |" % (condition, labels[method], " | ".join(
                "%.2f%%" % (100 * v[name]) for name in
                ("api_recall", "api_frr", "preservation", "raw_recall", "raw_frr", "jer"))))
    lines += ["", "주 조건=웅성거림10/5dB, paired bootstrap95% 구간"]
    for name, title in (("noise_vs_baseline", "소음학습 − 기존"),
                        ("noise_vs_clean_control", "소음학습 − 깨끗한 추가학습")):
        lines.append(title)
        for key in ("api_recall", "api_frr", "preservation", "jer"):
            v = comparisons[name]["primary"][key]
            lines.append("  %s %+.2f%%p [%+.2f, %+.2f]" % (key, v["delta_pp"], *v["ci95_pp"]))
    lines += ["", "처음 보는 문단 참고 평가 — 전체 파형/전체 실제 발음 라벨 사용",
              "대부분 운영 API의20초 제한보다 길다. 짧은 새 문장의 서비스 성능으로 일반화하지 않는다.",
              "| 조건 | 방법 | 오발음 보존 | 원시 헛지적 | JER |", "|---|---|---:|---:|---:|"]
    for condition, mapping in unseen_values.items():
        for method in methods:
            v = mapping[method]
            lines.append("| %s | %s | %.2f%% | %.2f%% | %.2f%% |" % (
                condition, labels[method], 100 * v["preservation"], 100 * v["raw_frr"], 100 * v["jer"]))
    lines += ["", "해석 범위",
              "- 합성6인 웅성거림·저주파 소음·가상 잔향이다. 실제 부스/사용자 녹음은 분석하지 않았다.",
              "- TAG 시험은 학습에 있던 문항의 별도 녹음이다. 정상/오발음 균형 표본이므로 실제 오류 빈도를 추정하지 않는다.",
              "- 기존 점수 보정표를 고정했다. 새 체크포인트는 배포 전 독립 보정/검증이 필요하다.",
              "- API 검출률과 실제 오발음 보존은 다르다. 주변 말소리 때문에 검출만 올라간 경우를 개선으로 세지 않는다.",
              "- 전체 입력을 평가했으며 재녹음 거절로 어려운 사례를 제외하지 않았다.",
              "- 발화/소음 파일 경로가 학습·선택·시험 사이 겹치지 않는다. 화자 메타데이터를 읽지 않아 화자 겹침은 재검사하지 않았다.",
              "- 한 seed, 짧은 추가 학습, 원어민 replay 없음. 실패는 소음 증강 전체가 효과 없다는 뜻이 아니다.",
              "- 생산 API·체크포인트·보정표 변경/서버 재시작은 하지 않았다."]
    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[decision] " + verdict + ": " + message, flush=True)
    print(json.dumps(primary, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("valid", "test", "report", "smoke"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstraps", type=int, default=3000)
    ap.add_argument("--smoke-ckpt")
    args = ap.parse_args()
    out = _pl.Path(args.out)
    plan = json.loads((out / "plan.json").read_text())
    if args.action == "valid":
        evaluate_method(out, plan, "valid", "baseline", plan["ckpt"])
        for step in plan["checkpoints"]:
            evaluate_method(out, plan, "valid", "noise_%d" % step, out / "checkpoints/noise" / ("step_%d" % step))
        selection(out, plan)
    elif args.action == "test":
        frozen = json.loads((out / "selection.json").read_text())
        step = frozen["chosen"]["step"]
        for method, ckpt in (("baseline", plan["ckpt"]),
                              ("clean_control", out / "checkpoints/clean" / ("step_%d" % step)),
                              ("noise_trained", out / "checkpoints/noise" / ("step_%d" % step))):
            for split in ("test", "unseen"):
                evaluate_method(out, plan, split, method, ckpt)
        report(out, plan, args.bootstraps)
    elif args.action == "smoke":
        evaluate_method(out, plan, "valid", "smoke", args.smoke_ckpt or plan["ckpt"], limit=2)
    else:
        report(out, plan, args.bootstraps)


if __name__ == "__main__":
    main()
