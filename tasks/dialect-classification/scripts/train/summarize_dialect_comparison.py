"""Produce model and regional comparison tables from completed experiment reports."""
import argparse
import csv
import datetime
import html
import json
from pathlib import Path

REGIONS = ["Gangwon", "Gyeongsang", "Jeolla", "Jeju", "Chungcheong"]
REGIONS_KO = ["강원도", "경상도", "전라도", "제주도", "충청도"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data/speech-ai/out/dialect/compare_20261011")
    parser.add_argument("--reference", default="/data/speech-ai/out/dialect/20261010/xlsr2b/report.json")
    args = parser.parse_args()
    root = Path(args.root)
    sources = [("xlsr2b", "XLS-R 2B", Path(args.reference))]
    sources.extend((name, label, root / name / "report.json") for name, label in [
        ("w2vbert2", "w2v-BERT 2.0"), ("wavlm_large", "WavLM Large"), ("eat_large", "EAT Large")])
    rows, full, regional = [], {}, []
    for key, label, path in sources:
        if not path.exists():
            rows.append({"model": label, "model_id": key, "status": "pending"})
            continue
        report = json.loads(path.read_text())
        full[key] = report
        selected = report["validation_variants"][report["selected_variant"]]
        test = report["test_single_window"]
        triple = report["test_three_utterance_windows_mean_logits"]
        timing_path = root / key / "benchmark.json"
        timing = json.loads(timing_path.read_text()) if timing_path.exists() else report.get("benchmark_batch1", {})
        resources = report.get("resources", {})
        if key == "xlsr2b" and not resources:
            config_path = path.parent / "config.json"
            config = json.loads(config_path.read_text()) if config_path.exists() else {}
            resources = {"extraction_total_seconds": sum(json.loads(p.read_text())["elapsed_seconds"] for p in path.parent.glob("features/*.done.json")),
                         "batch_size": config.get("batch_size")}
        row = {"model": label, "model_id": key, "status": "complete", "selected_variant": report["selected_variant"],
               "person_disjoint_verified": report["data_summary"].get("person_disjoint_verified", False),
               "evaluation_status": "person_split_verified" if report["data_summary"].get("person_disjoint_verified", False) else "historical_recording_split_baseline",
               "parameters": timing.get("parameters"), "validation_top1": selected["valid_top1"], "validation_macro_f1": selected["valid_macro_f1"],
               "test_top1": test["top1"], "test_macro_f1": test["macro_f1"], "test_n": test["n"], "test_failed": test["failed"],
               "test_three_window_top1": triple["top1"], "test_three_window_macro_f1": triple["macro_f1"],
               "batch1_encoder_p50_ms": timing.get("p50_ms"), "batch1_encoder_p95_ms": timing.get("p95_ms"),
               "batch1_encoder_peak_allocated_gib": timing.get("peak_allocated_gib"),
               "extraction_total_hours": resources.get("extraction_total_seconds", 0) / 3600,
               "extraction_batch_size": resources.get("batch_size"),
               "extraction_peak_allocated_gib": resources.get("extraction_peak_allocated_gib"),
               "report": str(path)}
        for region, ko in zip(REGIONS, REGIONS_KO):
            r = test["per_region"][region]
            t = triple["per_region"][region]
            row[region + "_top1"] = r["recall_top1"]
            regional.append({"model": label, "model_id": key, "region": ko, "region_id": region,
                             "test_n": r["n"], "test_top1": r["recall_top1"], "test_precision": r["precision"], "test_f1": r["f1"],
                             "three_window_n": t["n"], "three_window_top1": t["recall_top1"], "three_window_f1": t["f1"]})
        rows.append(row)
    completed = [r for r in rows if r["status"] == "complete"]
    summaries = [full[r["model_id"]]["data_summary"] for r in completed]
    if summaries and any(s["counts"] != summaries[0]["counts"] or s["recordings"] != summaries[0]["recordings"] for s in summaries):
        raise RuntimeError("Comparison does not use identical data split counts")
    if summaries and any(s.get("manifest_sha256") != summaries[0].get("manifest_sha256") for s in summaries):
        raise RuntimeError("Comparison does not use identical manifest rows")
    benchmark_hashes = set()
    for row in completed:
        p = root / row["model_id"] / "benchmark.json"
        if p.exists():
            benchmark_hashes.add(json.loads(p.read_text())["row_id_sha256"])
    if len(benchmark_hashes) > 1:
        raise RuntimeError("Latency benchmarks do not use identical validation rows")
    verified = [r for r in completed if r["person_disjoint_verified"]]
    completed_verified = bool(completed) and len(verified) == len(completed)
    winner = max(verified, key=lambda row: (row["validation_macro_f1"], row["validation_top1"])) if verified else None
    audit_path = root / "speaker_audit.json"
    speaker_audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    payload = {"generated_at": datetime.datetime.now().astimezone().isoformat(), "complete": len(completed) == len(rows),
               "selection_rule": "Select the overall model by validation macro F1, then validation top1; report test metrics without tuning on test",
               "selected_model_by_validation": winner["model_id"] if winner else None,
               "person_disjoint_evaluation_complete": len(verified) == len(rows),
               "speaker_audit": str(audit_path) if speaker_audit else None,
               "data_summary": summaries[0] if summaries else None,
               "evaluation_status": "person_split_verified" if completed_verified else ("pending" if not completed else "Unseen-person performance is unverified. Historical recording-split metrics require reevaluation."),
               "models": rows, "per_region": regional,
               "protocol": {"encoder_frozen": True, "classifier": "Person and region weighted linear head for verified person splits; historical baseline used recording weights. Seed42, AdamW lr0.001, max100 epochs, patience12",
                            "splits": "Provider global speaker IDs, connected recording-participant groups" if verified else "Historical recording-disjoint split only",
                            "audio": "Existing annotated 3-12s 16kHz mono clips",
                            "feature_selection": "Two intermediate/final speech layers and concatenation; EAT validates CLS and masked mean separately",
                            "latency": "Batch1 audio-read, model-native preprocessing and encoder features, 120 identical validation clips; excludes classifier head and HTTP",
                            "precision": "FP32 weights and FP16 CUDA autocast"},
               "limitations": ["Person independence is established according to documented source speaker IDs; identity aliases cannot be ruled out independently." if verified else "Recording split does not establish disjoint people because speaker IDs are local to a recording.",
                               "Official Training/Validation recordings are resplit by provider person IDs; not an untouched official benchmark." if verified else "Jeolla uses official Validation recordings resplit for this experiment; not an untouched official benchmark.",
                               "Three-utterance-window aggregation is not the NPU production three-part segmentation.",
                               "EAT has six validation feature variants; speech encoders have three.",
                               "Extraction times use each configured batch size, unlike the common batch1 latency benchmark."]}
    (root / "comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    for name, values in [("comparison.csv", rows), ("per_region.csv", regional)]:
        columns = list(dict.fromkeys(k for r in values for k in r))
        with (root / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(values)
    def pct(value):
        return "—" if value is None else "%.2f%%" % (value * 100)
    def num(value):
        return "—" if value is None else "%.2f" % value
    body = ["<!doctype html><html lang='ko'><meta charset='utf-8'><title>방언 모델 비교</title>",
            "<style>body{font:16px system-ui;margin:36px;max-width:1300px;color:#182230}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{padding:12px;border-bottom:1px solid #dce2ea;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#f1f5f9}h2{margin-top:32px}p,li{line-height:1.6}</style>",
            "<h1>방언 분류 모델 비교</h1>", "<p>" + html.escape(payload["generated_at"]) + " · 완료 " + str(len(completed)) + "/4</p>",
            ("<p>완료된 모델은 원본 화자 ID를 기준으로 분리한 결과입니다. " + ("모델 비교가 진행 중입니다." if len(completed) < len(rows) else "모델 비교를 완료했습니다.") + "</p>") if completed_verified else ("<p>데이터 준비 및 모델 학습을 기다리고 있습니다.</p>" if not completed else "<p><strong>화자 분리가 확인되지 않은 기존 수치는 녹음 분할 기준의 참고 결과입니다. 모델 선정은 보류합니다.</strong></p>"),
            "<p>동일한 데이터 분할·선형 분류기 학습 조건. 전체 모델 선택은 검증 Macro F1 기준.</p>",
            "<table><tr><th>모델</th><th>파라미터</th><th>검증 Top1</th><th>테스트 Top1</th><th>Macro F1</th><th>3구간 합산</th><th>인코더 p50(ms)</th><th>배치1 최대 VRAM(GiB)</th></tr>"]
    for r in rows:
        params = "—" if r.get("parameters") is None else "%.1fM" % (r["parameters"] / 1e6)
        body.append("<tr>" + "".join("<td>" + html.escape(str(v)) + "</td>" for v in [r["model"], params, pct(r.get("validation_top1")), pct(r.get("test_top1")), num(r.get("test_macro_f1")), pct(r.get("test_three_window_top1")), num(r.get("batch1_encoder_p50_ms")), num(r.get("batch1_encoder_peak_allocated_gib"))]) + "</tr>")
    body.append("</table><h2>지역별 테스트 Top1</h2><table><tr><th>모델</th>" + "".join("<th>" + v + "</th>" for v in REGIONS_KO) + "</tr>")
    for r in rows:
        body.append("<tr><td>" + html.escape(r["model"]) + "</td>" + "".join("<td>" + pct(r.get(v + "_top1")) + "</td>" for v in REGIONS) + "</tr>")
    scope = "원본의 화자 고유 ID를 기준으로 사람을 분리했습니다. 동일인에게 여러 ID가 부여된 경우까지 별도로 확인한 결과는 아닙니다." if verified else "학습과 다른 녹음으로 평가했지만, 사람까지 분리된 성능은 확인되지 않았습니다."
    body.append("</table><h2>해석 범위</h2><ul><li>" + scope + "</li><li>새 데이터는 중·노년층의 질문 응답 발화이며, 모든 연령의 방언 성능을 의미하지 않습니다.</li><li>공식 데이터 분할을 이 실험의 분할로 재구성했습니다.</li><li>3구간 합산은 발화 구간 3개를 합친 실험이며 NPU 운영 분할 규칙과 다릅니다.</li><li>지연은 음성 읽기·전처리·인코더 특징 추출이며 API 응답 지연이 아닙니다.</li></ul></html>")
    (root / "comparison.html").write_text("\n".join(body), encoding="utf-8")
    print(json.dumps({"complete": payload["complete"], "models": rows, "selected_model_by_validation": payload["selected_model_by_validation"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
