"""Audit real blinded responses without treating missing ratings as agreement."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import argparse
import collections
import hashlib
import itertools
import json

CATEGORIES = ("s", "ss", "other", "uncertain", "inaudible")


def agreement(pairs):
    n = len(pairs)
    if not n:
        return {"n": 0, "agreement": None, "kappa": None}
    left, right = collections.Counter(a for a, b in pairs), collections.Counter(b for a, b in pairs)
    observed = sum(a == b for a, b in pairs) / n
    expected = sum(left[k] * right[k] for k in CATEGORIES) / (n * n)
    return {"n": n, "agreement": observed,
            "kappa": (observed - expected) / (1 - expected) if expected < 1 else None}


def summarize(packet, responses):
    private = packet / "items.private.jsonl"
    items = {r["id"]: r for r in map(json.loads, private.read_text(encoding="utf-8").splitlines())}
    trials = json.loads((packet / "trials.private.json").read_text())
    sha = hashlib.sha256(private.read_bytes()).hexdigest()
    labels, first, repeat, pending = {}, {}, {}, {}
    for who, ts in trials.items():
        allowed = {t["trial"]: t["item"] for t in ts}
        path = responses / ("labels_" + who + ".jsonl")
        latest = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("packet_sha256") != sha or row.get("reviewer") != who or row.get("trial") not in allowed:
                    raise ValueError("response provenance mismatch: " + str(path))
                if row.get("item") != allowed[row["trial"]] or row.get("actor_type") != "human_self_declared":
                    raise ValueError("response identity/type mismatch")
                if row.get("choice") not in CATEGORIES or type(row.get("confidence")) is not int or row["confidence"] not in (1, 2, 3):
                    raise ValueError("invalid response content")
                if row.get("native_korean") is not True:
                    raise ValueError("native-listener declaration absent")
                latest[row["trial"]] = row
        labels[who] = latest
        first[who], pairs = {}, []
        first_trial = {}
        for trial in ts:
            item = trial["item"]
            if item not in first_trial:
                first_trial[item] = trial["trial"]
                if trial["trial"] in latest:
                    first[who][item] = latest[trial["trial"]]
            elif first_trial[item] in latest and trial["trial"] in latest:
                pairs.append((latest[first_trial[item]]["choice"], latest[trial["trial"]]["choice"]))
        repeat[who] = agreement(pairs)
        pending[who] = len(ts) - len(latest)
    records = []
    for item_id, item in items.items():
        rs = [first[who][item_id] for who in first if item_id in first[who]]
        choices = collections.Counter(r["choice"] for r in rs)
        consensus = None
        if len(rs) >= 2 and len(choices) == 1 and all(r["confidence"] >= 2 for r in rs):
            chosen = next(iter(choices))
            consensus = {"s": "ㅅ", "ss": "ㅆ"}.get(chosen)
        status = "awaiting_two_listeners" if len(rs) < 2 else "phonetic_consensus" if consensus else "uncertain_or_disagreement"
        records.append({"id": item_id, "ratings": len(rs), "choices": dict(choices), "status": status,
                        "consensus": consensus, "existing_TAG": item["actual"],
                        "TAG_agrees": consensus == item["actual"] if consensus is not None else None,
                        "context_audio_used": sum(r["plays"].get("context", 0) > 0 for r in rs)})
    between = {}
    for a, b in itertools.combinations(first, 2):
        common = first[a].keys() & first[b].keys()
        between[a + "_" + b] = agreement([(first[a][k]["choice"], first[b][k]["choice"]) for k in common])
    enough = sum(r["ratings"] >= 2 for r in records)
    return {"packet_sha256": sha, "items": len(items),
            "human_responses_by_reviewer": {w: len(v) for w, v in labels.items()},
            "pending_trials_by_reviewer": pending,
            "items_with_at_least_two_first_pass_ratings": enough,
            "all_items_have_two_ratings": enough == len(items) and len(items) > 0,
            "between_listener_agreement": between, "within_listener_repeat_agreement": repeat,
            "status_counts": dict(collections.Counter(r["status"] for r in records)),
            "TAG_comparable_consensus_items": sum(r["consensus"] is not None for r in records),
            "TAG_disagreements": [r["id"] for r in records if r["TAG_agrees"] is False],
            "limitations": ["Listener identity and native status are self-declared; different links do not prove different persons.",
                            "Repeated trials assess consistency and are not counted as additional independent listeners.",
                            "A missing response is not a correct response or a phonetic judgement.",
                            "Two agreeing confident first-pass ratings are the minimum; disagreement/uncertainty is retained."],
            "records": records}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--responses", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        ap.error("refusing to overwrite an earlier response audit")
    report = summarize(a.packet, a.responses)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("records", "limitations")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
