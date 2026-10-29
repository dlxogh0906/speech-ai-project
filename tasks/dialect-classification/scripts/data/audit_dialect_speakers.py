"""Audit recording splits against original, cross-recording speaker metadata.

Local IDs and demographic matches do not prove real-person independence.
Names are never printed or saved. Code-shaped source identifiers are hashed.
"""
import argparse
import csv
import datetime
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

PAIRS = [("train", "valid"), ("train", "test"), ("valid", "test")]
REGIONS = {"118": "Gangwon", "119": "Gyeongsang", "120": "Jeolla", "121": "Jeju", "122": "Chungcheong"}
CODE = re.compile(r"[A-Z]{2}\d{4,}")
PROFILE = ["age", "sex", "birthplace", "principal_residence", "current_residence", "occupation", "education"]


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", default="/data/speech-ai/out/dialect/20261010/manifests")
    parser.add_argument("--archives", default="/data/speech-ai/corpus/dialect_10pct_20261010/archives")
    parser.add_argument("--out", default="/data/speech-ai/out/dialect/compare_20261011/speaker_audit.json")
    args = parser.parse_args()
    records = defaultdict(set)
    speakers = defaultdict(set)
    counts = Counter()
    code_records = defaultdict(lambda: defaultdict(set))
    manifest_hashes = {}
    for split in ["train", "valid", "test"]:
        path = Path(args.manifests) / (split + ".csv")
        manifest_hashes[split] = digest(path.read_text())
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key, sid = row["record_key"], row["speaker_key"].rsplit(":", 1)[-1]
                records[split].add(key)
                speakers[key].add(sid)
                counts[split] += 1
                if CODE.fullmatch(sid):
                    code_records[key.split(":", 1)[0] + ":" + sid][split].add(key)
    wanted = set().union(*records.values())
    record_splits = {key: split for split, keys in records.items() for key in keys}
    found = set()
    stats = {ds: Counter() for ds in REGIONS}
    fields = {ds: Counter() for ds in REGIONS}
    code_profiles = defaultdict(set)
    candidate_records = defaultdict(lambda: defaultdict(set))
    candidate_mapping = []
    for ds in REGIONS:
        for archive in sorted((Path(args.archives) / ds).glob("*/*.zip")):
            with zipfile.ZipFile(archive) as bundle:
                for info in bundle.infolist():
                    key = ds + ":" + Path(info.filename).stem
                    if not info.filename.lower().endswith(".json") or key not in wanted or key in found:
                        continue
                    record = json.loads(bundle.read(info).decode("utf-8-sig"))
                    if str(record["id"]) != key.split(":", 1)[1]:
                        raise RuntimeError("Source JSON ID differs from filename: " + key)
                    found.add(key)
                    stats[ds]["source_recordings"] += 1
                    for speaker in record.get("speaker", []):
                        sid = str(speaker.get("id", ""))
                        if sid not in speakers[key]:
                            continue
                        fields[ds].update(speaker.keys())
                        stats[ds]["source_speaker_occurrences"] += 1
                        stats[ds]["local_numeric_id_occurrences"] += int(sid.isdigit())
                        name = str(speaker.get("name", "")).strip()
                        stats[ds]["missing_or_null_name_occurrences"] += int(name.lower() in ("", "null", "none", "nan"))
                        profile = json.dumps({k: speaker.get(k) for k in PROFILE}, ensure_ascii=False, sort_keys=True)
                        if CODE.fullmatch(sid):
                            code_profiles[ds + ":" + sid].add(digest(profile))
                        source_code = sid if CODE.fullmatch(sid) else name if CODE.fullmatch(name) else None
                        if source_code:
                            candidate = ds + ":" + source_code
                            candidate_records[candidate][record_splits[key]].add(key)
                            code_profiles[candidate].add(digest(profile))
                            candidate_mapping.append({"speaker_key": key + ":" + sid, "candidate_person_id": digest(candidate),
                                                      "source_field": "speaker.id" if CODE.fullmatch(sid) else "speaker.name", "provider_identity_confirmed": False})
                        stats[ds]["code_shaped_id_occurrences"] += int(bool(CODE.fullmatch(sid)))
                        stats[ds]["code_shaped_name_occurrences"] += int(bool(CODE.fullmatch(name)))
            print("[source audit] " + ds + " matched=" + str(stats[ds]["source_recordings"]), flush=True)
    regional = {}
    overlaps = []
    for ds, region in REGIONS.items():
        codes = {key: folds for key, folds in code_records.items() if key.startswith(ds + ":")}
        per_split = {s: {code for code, folds in codes.items() if s in folds} for s in records}
        candidates = {key: folds for key, folds in candidate_records.items() if key.startswith(ds + ":")}
        candidate_splits = {s: {code for code, folds in candidates.items() if s in folds} for s in records}
        regional[region] = {"dataset": ds, "source_statistics": dict(stats[ds]), "speaker_fields": dict(fields[ds]),
                            "distinct_code_shaped_ids": {s: len(v) for s, v in per_split.items()},
                            "shared_code_shaped_ids": {a + "_" + b: len(per_split[a] & per_split[b]) for a, b in PAIRS},
                            "distinct_candidate_person_codes": {s: len(v) for s, v in candidate_splits.items()},
                            "shared_candidate_person_codes": {a + "_" + b: len(candidate_splits[a] & candidate_splits[b]) for a, b in PAIRS},
                            "candidate_codes_with_multiple_demographic_profiles": sum(len(code_profiles[c]) > 1 for c in candidates),
                            "codes_with_multiple_demographic_profiles": sum(len(code_profiles[c]) > 1 for c in codes)}
        for code, folds in candidates.items():
            if len(folds) > 1:
                overlaps.append({"dataset": ds, "source_code_sha256": digest(code), "recordings": {s: sorted(v) for s, v in folds.items()},
                                 "demographic_profile_count": len(code_profiles[code])})
    payload = {"generated_at": datetime.datetime.now().astimezone().isoformat(), "person_disjoint_verified": False,
               "evaluation_status": "Cross-recording speaker-code overlap detected; unseen-person performance is not established.",
               "manifest_sha256": manifest_hashes, "windows": dict(counts), "recordings": {s: len(v) for s, v in records.items()},
               "recording_overlap": {a + "_" + b: len(records[a] & records[b]) for a, b in PAIRS},
               "source_metadata_missing_recordings": sorted(wanted - found), "per_region": regional, "overlapping_source_codes": overlaps,
               "limitations": ["Recording-scoped speaker keys hide reuse of source speaker IDs across recordings.",
                               "Local IDs 1/2 and null names cannot establish real-person identity.",
                               "Matching demographics do not establish identity; they are only consistency evidence for code-shaped IDs.",
                               "Provider-confirmed anonymized person IDs are needed for a verified person-disjoint split.",
                               "Names are not saved; source speaker codes are saved as SHA256 hashes only."]}
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mapping_path = destination.with_name("candidate_person_mapping.csv")
    with mapping_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["speaker_key", "candidate_person_id", "source_field", "provider_identity_confirmed"])
        writer.writeheader()
        writer.writerows(candidate_mapping)
    payload["partial_candidate_mapping"] = str(mapping_path)
    payload["candidate_mapping_is_provider_verified"] = False
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out": str(destination), "person_disjoint_verified": False, "per_region": regional}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
