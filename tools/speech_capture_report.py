"""Summarize complete speech API captures without exposing audio or credentials."""
import argparse
import collections
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="capture root containing nationality/ and pronunciation/")
    parser.add_argument("--day", help="YYYYMMDD (default: all days)")
    parser.add_argument("--include-ops", action="store_true", help="include npu-ops-test verification requests")
    parser.add_argument("--label", help="filter final nationality prediction, e.g. Philippines")
    args = parser.parse_args()
    counts, statuses, labels = collections.Counter(), collections.Counter(), collections.Counter()
    size, file_count, incomplete, parse_errors = 0, 0, 0, []
    for manifest in sorted(args.root.glob("*/%s/captures.jsonl" % (args.day or "*"))):
        with manifest.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                try:
                    record = json.loads(line)
                except ValueError:
                    parse_errors.append({"manifest": str(manifest), "line": number})
                    continue
                if not args.include_ops and record.get("headers", {}).get("user-agent") == "npu-ops-test":
                    continue
                top = record.get("result", {}).get("top5", [])
                label = top[0].get("label") if top else None
                if args.label and label != args.label:
                    continue
                key = "%s %s %s" % (record["service"], record["method"], record["path"])
                counts[key] += 1
                statuses[str(record.get("status", "unknown"))] += 1
                if label:
                    labels[label] += 1
                for audio in record.get("files", []):
                    file_count += 1
                    size += audio["bytes"]
                incomplete += record.get("state") not in ("complete", "exception", "disconnected")
    # Requests interrupted by process termination may have request.json but no
    # completed JSONL row; keep these visible to the analyst.
    interrupted = []
    for path in args.root.glob("*/%s/*/request.json" % (args.day or "*")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if record.get("state") in ("receiving", "processing"):
            interrupted.append({"rid": record["rid"], "state": record["state"], "record": str(path)})
    print(json.dumps({"requests": dict(counts), "http_status": dict(statuses), "nationality_top1": dict(labels),
                      "original_files": file_count, "original_bytes": size, "incomplete_rows": incomplete,
                      "interrupted_requests": interrupted, "manifest_parse_errors": parse_errors}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
