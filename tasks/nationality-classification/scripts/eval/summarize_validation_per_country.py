# -*- coding: utf-8 -*-
"""Summarize cached validation predictions by country and recording task."""

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np


COUNTRY_KO = {
    "Bangladesh": "방글라데시", "Brazil": "브라질", "Bulgaria": "불가리아", "Cambodia": "캄보디아",
    "Canada": "캐나다", "Chile": "칠레", "China": "중국", "Colombia": "콜롬비아",
    "Egypt": "이집트", "Ethiopia": "에티오피아", "France": "프랑스", "Germany": "독일",
    "Hong Kong": "홍콩", "India": "인도", "Indonesia": "인도네시아", "Iran": "이란",
    "Italy": "이탈리아", "Japan": "일본", "Kazakhstan": "카자흐스탄", "Korea": "한국",
    "Kyrgyzstan": "키르기스스탄", "Laos": "라오스", "Malaysia": "말레이시아",
    "Mongolia": "몽골", "Myanmar": "미얀마", "Nepal": "네팔", "Pakistan": "파키스탄",
    "Peru": "페루", "Philippines": "필리핀", "Poland": "폴란드", "Romania": "루마니아",
    "Russia": "러시아", "Rwanda": "르완다", "Singapore": "싱가포르", "South Korea": "한국",
    "Spain": "스페인", "Sri Lanka": "스리랑카", "Taiwan": "대만",
    "Tajikistan": "타지키스탄", "Thailand": "태국", "Turkey": "튀르키예",
    "Turkmenistan": "투르크메니스탄", "Ukraine": "우크라이나",
    "United Kingdom": "영국", "United States": "미국", "Uzbekistan": "우즈베키스탄",
    "Vietnam": "베트남", "Zambia": "잠비아",
}


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evaluation = json.loads((args.eval_dir / "report.json").read_text())
    manifest = Path(evaluation["test"])
    if manifest.name != "valid.csv":
        raise ValueError("Expected the actual valid.csv manifest, not the test split")
    rows = read_rows(manifest)
    train = read_rows(manifest.with_name("train.csv"))
    test = read_rows(manifest.with_name("test.csv"))
    name = next(iter(evaluation["models"]))
    model = Path(evaluation["models"][name])
    with np.load(model / "head_np.npz", allow_pickle=True) as head:
        labels = [str(label) for label in head["labels"]]
    with np.load(args.eval_dir / "logits.npz", allow_pickle=False) as predictions:
        logits, ok = predictions["L_" + name], predictions["ok"]
    if logits.shape != (len(rows), len(labels)) or len(ok) != len(rows):
        raise ValueError("Prediction cache does not match the validation manifest")
    c2i = {country: index for index, country in enumerate(labels)}
    y = np.array([c2i.get(row["label_en"], -1) for row in rows])
    if np.any(y < 0):
        raise ValueError("A validation label is absent from the model's class list")
    domains = np.array([row["domain"] for row in rows])
    speakers = np.array([row["speaker"] for row in rows])
    word = np.array([row["task"] == "RW" for row in rows])
    rank = np.argsort(-logits, axis=1)
    pred = rank[:, 0]
    hit1 = (pred == y) & ok
    hit5 = np.any(rank[:, :5] == y[:, None], axis=1) & ok
    web = np.array([domain.startswith("web") for domain in domains])
    valid_speakers = set(speakers)
    train_speakers = set(row["speaker"] for row in train)
    test_speakers = set(row["speaker"] for row in test)
    overlaps = {"train_valid": len(train_speakers & valid_speakers),
                "valid_test": len(valid_speakers & test_speakers),
                "train_test": len(train_speakers & test_speakers)}
    train_counts = collections.Counter(row["label_en"] for row in train)
    train_spk = collections.defaultdict(set)
    for row in train:
        train_spk[row["label_en"]].add(row["speaker"])

    def metrics(mask):
        total = int(mask.sum())
        if not total:
            return None
        good = mask & ok
        count = int(good.sum())
        if not count:
            return None
        classes = np.unique(y[good])
        return {
            "n": count, "manifest_n": total, "decoded": count, "failed": total - count,
            "speakers": len(set(speakers[good])), "manifest_speakers": len(set(speakers[mask])),
            "correct_top1": int(hit1[mask].sum()), "correct_top5": int(hit5[mask].sum()),
            "top1": float(hit1[good].mean()), "top5": float(hit5[good].mean()),
            "macro_top1_present_countries": float(np.mean([hit1[good & (y == country)].mean() for country in classes])),
            "macro_top5_present_countries": float(np.mean([hit5[good & (y == country)].mean() for country in classes])),
        }

    country_results = []
    for country, index in c2i.items():
        mask = y == index
        correct = int(hit1[mask].sum())
        predicted = int(((pred == index) & ok).sum())
        precision = correct / predicted if predicted else None
        main = metrics(mask)
        errors = collections.Counter(labels[int(other)] for other in pred[mask & ok & ~hit1])
        country_results.append({
            "country": country, "country_ko": COUNTRY_KO.get(country, country),
            "train_utterances": train_counts[country], "train_speakers": len(train_spk[country]),
            "all_validation": main, "web_all": metrics(mask & web),
            "web_sentence": metrics(mask & web & ~word),
            "web_word": metrics(mask & web & word), "p505": metrics(mask & (domains == "p505")),
            "predicted_as_country": predicted, "precision": precision,
            "top_errors": [{"country": other, "country_ko": COUNTRY_KO.get(other, other), "n": n}
                           for other, n in errors.most_common(5)],
            "classified_as_philippines": int(((pred == c2i["Philippines"]) & mask & ok).sum()),
        })
    country_results.sort(key=lambda item: -(item["all_validation"]["top1"]
                                           if item["all_validation"] else -1))
    overview_masks = {"all_validation": np.ones(len(rows), dtype=bool),
                      "foreign_validation": y != c2i["Korea"], "web_all": web,
                      "web_sentence": web & ~word, "web_word": web & word,
                      "p505": domains == "p505", "korean": y == c2i["Korea"]}
    decode_issues = []
    for index in np.flatnonzero(~ok):
        row = rows[int(index)]
        result = subprocess.run([shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-i", row["audio_path"],
                                 "-f", "f32le", "-ac", "1", "-ar", "16000", "-t", "12", "pipe:1"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        seconds = len(result.stdout) / 4 / 16000 if result.returncode == 0 else None
        decode_issues.append({"index": int(index), "audio_path": row["audio_path"], "country": row["label_en"],
                              "seconds": seconds, "reason": "under_0.5_seconds" if seconds is not None else "decode_error",
                              "error": result.stderr.decode(errors="replace")[-300:] if result.returncode else None})
    report = {
        "split": "validation", "manifest": str(manifest), "manifest_sha256": digest(manifest),
        "model": str(model), "model_name": name, "model_classes": len(labels),
        "head_sha256": digest(model / "head_np.npz"),
        "encoder_sha256": digest(model / "encoder_kd_l17.onnx"),
        "method": "Original run_zeroth ONNX on GPU; first 12 seconds per utterance; no silence normalization, Korea gate or multi-part aggregation. Unavailable audio is reported separately and excluded from model accuracy.",
        "speaker_overlap": overlaps, "speaker_disjoint_by_manifest_id": not any(overlaps.values()),
        "decode_issues": decode_issues,
        "overview": {key: metrics(mask) for key, mask in overview_masks.items()},
        "countries_without_validation": [item["country"] for item in country_results if item["all_validation"] is None],
        "per_country": country_results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    fields = ["country", "country_ko", "train_speakers", "train_utterances", "valid_speakers", "valid_utterances",
              "top1", "top5", "precision", "sentence_utterances", "sentence_speakers", "sentence_top1", "sentence_top5",
              "word_utterances", "word_top1", "p505_utterances", "p505_top1", "top_errors"]
    with (args.out / "per_country.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in country_results:
            all_metrics = item["all_validation"] or {}
            sent = item["web_sentence"] or {}
            words = item["web_word"] or {}
            p505 = item["p505"] or {}
            writer.writerow({"country": item["country"], "country_ko": item["country_ko"],
                             "train_speakers": item["train_speakers"], "train_utterances": item["train_utterances"],
                             "valid_speakers": all_metrics.get("speakers", 0), "valid_utterances": all_metrics.get("n", 0),
                             "top1": all_metrics.get("top1"), "top5": all_metrics.get("top5"), "precision": item["precision"],
                             "sentence_utterances": sent.get("n", 0), "sentence_speakers": sent.get("speakers", 0),
                             "sentence_top1": sent.get("top1"), "sentence_top5": sent.get("top5"),
                             "word_utterances": words.get("n", 0), "word_top1": words.get("top1"),
                             "p505_utterances": p505.get("n", 0), "p505_top1": p505.get("top1"),
                             "top_errors": json.dumps(item["top_errors"], ensure_ascii=False)})
    print(json.dumps({key: value for key, value in report.items() if key != "per_country"}, ensure_ascii=False, indent=2))
    print("country\ttrain_spk\tvalid_spk\tvalid_n\ttop1%\ttop5%\tsentence_n\tsentence_top1%\tsentence_top5%\tword_n\tword_top1%\tp505_n\tp505_top1%\tas_Philippines\terrors")
    for item in country_results:
        groups = [item["all_validation"] or {}, item["web_sentence"] or {},
                  item["web_word"] or {}, item["p505"] or {}]
        all_metrics, sent, words, p505 = groups
        def percent(group, key):
            return "-" if key not in group else "{:.1f}".format(group[key] * 100)
        print("\t".join(str(value) for value in [
            item["country_ko"], item["train_speakers"], all_metrics.get("speakers", 0), all_metrics.get("n", 0),
            percent(all_metrics, "top1"), percent(all_metrics, "top5"), sent.get("n", 0), percent(sent, "top1"),
            percent(sent, "top5"), words.get("n", 0), percent(words, "top1"), p505.get("n", 0),
            percent(p505, "top1"), item["classified_as_philippines"],
            ", ".join("{}:{}".format(error["country_ko"], error["n"]) for error in item["top_errors"])]))


if __name__ == "__main__":
    main()
