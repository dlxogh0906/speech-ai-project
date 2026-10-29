"""Compare frozen LAB targets with the current text-derived pronunciation.

Disagreement is a review flag, not a corrected human label. All optimal edit
paths are considered; ambiguous or deleted sites are never assigned a target.
No waveform, reference label, model or production setting is changed.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import re


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def site_status(reference, target, position, all_optimal_mapping):
    if not 0 <= position < len(reference):
        raise ValueError("reference position out of range")
    opts = all_optimal_mapping(reference, target)[position]
    mapped = next(iter(opts)) if len(opts) == 1 else None
    status = ("ambiguous" if len(opts) != 1 else "deleted" if mapped is None
              else "agrees" if reference[position] == target[mapped] else "disagrees")
    return {"status": status, "possible_target_positions": sorted(q for q in opts if q is not None),
            "deletion_possible": None in opts, "unique_target_position": mapped,
            "unique_target_phone": target[mapped] if mapped is not None else None,
            "target_next_phone": target[mapped + 1] if mapped is not None and mapped + 1 < len(target) else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--evaluation", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, help="frozen evaluation baseline.jsonl, including rejected inputs")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("refusing to overwrite an audit")
    from mdd.l1prior import s0_kog2p_wrapper as wrapper
    from mdd.jamo_to_kog2p import phones_to_jamo
    from mdd import canonical_fix
    from shared.paths import G2P_TOOLS

    mapper_path = Path(__file__).with_name("eval_sibilant_directional.py")
    spec = importlib.util.spec_from_file_location("canonical_audit_mapper", str(mapper_path))
    mapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mapper)
    prep_path = Path(__file__).parents[1] / "data/prepare_sibilant_evidence.py"
    spec = importlib.util.spec_from_file_location("canonical_audit_packet_reader", str(prep_path))
    packet_reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packet_reader)
    manifest = args.packet / "items.private.jsonl"
    items = [json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    required = {r["uid"] for r in items}
    rows = {}
    with args.evaluation.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["uid"] in required:
                if row["uid"] in rows:
                    raise ValueError("duplicate evaluation uid")
                rows[row["uid"]] = row
    if set(rows) != required:
        raise ValueError("evaluation missing packet uids")
    cached = {}
    if args.baseline:
        with args.baseline.open() as stream:
            for line in stream:
                row = json.loads(line)
                uid = row["row"]["uid"]
                if uid in required:
                    if uid in cached:
                        raise ValueError("duplicate baseline uid")
                    cached[uid] = row
        if set(cached) != required:
            raise ValueError("baseline missing packet uids")

    def app_g2p(text):
        # Same normalization and vowel convention as pronunciation-demo.canonical.
        clean = re.sub(r"\s+", " ", re.sub(r"[^0-9A-Za-z가-힣\s]", " ", text or "")).strip()
        out = []
        for phone in phones_to_jamo(wrapper.text_to_phones(clean)):
            out.extend({"ㅙ": ["ㅗ", "ㅐ"], "ㅞ": ["ㅜ", "ㅔ"]}.get(phone, [phone]))
        return out

    builder = canonical_fix.CanonicalBuilder(app_g2p)
    records = []
    for item in items:
        row = rows[item["uid"]]
        ref = row["canonical"]
        q = item["q"]
        if row["prompt"] != item["prompt"] or ref[q] != item["canon"]:
            raise ValueError("frozen packet/evaluation mismatch")
        target = phones_to_jamo(wrapper.text_to_phones(row["prompt"]))
        if not target:
            raise ValueError("empty G2P output")
        result = site_status(ref, target, q, mapper.all_optimal_mapping)
        fixed = builder.fixed(row["prompt"])
        fixed_result = site_status(ref, fixed, q, mapper.all_optimal_mapping)
        final_result = {"status": "not_supplied"}
        final_tokens = None
        if args.baseline:
            cached_row = cached[item["uid"]]
            if cached_row["row"]["canonical"] != ref or cached_row["row"]["prompt"] != row["prompt"]:
                raise ValueError("baseline reference or prompt differs")
            final_tokens = (cached_row.get("raw") or {}).get("canon")
            if final_tokens:
                final_result = site_status(ref, final_tokens, q, mapper.all_optimal_mapping)
            else:
                final_result = {"status": "unavailable", "rejected": cached_row.get("rejected")}
        tag_path = Path(item["tag_file"])
        if digest(tag_path) != item["tag_sha256"]:
            raise ValueError("original TAG fingerprint differs")
        words = [w for w in packet_reader.tiers(tag_path)["words"] if w[2]]
        word_positions = [i for i, w in enumerate(words) if list(w) == item["tag_word"]]
        if len(word_positions) != 1:
            raise ValueError("target word cannot be uniquely located in original TAG")
        wi = word_positions[0]
        nearby = words[max(0, wi - 3):wi + 3]
        near_text = " ".join(w[2] for w in nearby)
        records.append({"id": item["id"], "uid": item["uid"], "item": item["item"],
            "prompt": row["prompt"], "word": item["tag_word"][2], "selection_role": item["selection_role"],
            "existing_LAB_canonical": ref[q], "existing_TAG_actual": item["actual"],
            "LAB_position": q, "LAB_tokens": ref, "current_G2P_tokens": target,
            "original_TAG_word_context": nearby,
            "original_TAG_word_position": wi,
            "G2P_for_TAG_word_context": phones_to_jamo(wrapper.text_to_phones(near_text)),
            "whole_sequence_agrees": ref == target, "comparison": result,
            "deterministically_fixed_tokens": fixed,
            "after_existing_canonical_fix": fixed_result,
            "cached_final_tokens": final_tokens, "cached_final_comparison": final_result,
            "following_vowel_agrees": result["target_next_phone"] == item["following_vowel"]
                if result["unique_target_position"] is not None else None})
    files = [manifest, args.evaluation, mapper_path, prep_path, Path(__file__), Path(wrapper.__file__),
             ROOT / "tasks/pronunciation-assessment/src/mdd/jamo_to_kog2p.py",
             Path(canonical_fix.__file__), ROOT / "deploy/pronunciation-demo/app.py",
             Path(G2P_TOOLS) / "KoG2P/g2p.py", Path(G2P_TOOLS) / "KoG2P/rulebook.txt"]
    if args.baseline:
        files.append(args.baseline)
    report = {"status": "canonical_consistency_audit_not_adjudicated_ground_truth",
        "items": len(items), "evaluation_uids": len(rows),
        "source_recordings": len({r["source_wav"] for r in items}),
        "site_status_counts": dict(Counter(r["comparison"]["status"] for r in records)),
        "existing_canonical_fix_status_counts": dict(Counter(r["after_existing_canonical_fix"]["status"] for r in records)),
        "cached_final_status_counts": dict(Counter(r["cached_final_comparison"]["status"] for r in records)),
        "whole_sequence_agrees": sum(r["whole_sequence_agrees"] for r in records),
        "limitations": ["G2P output is an automatic reference candidate, not human adjudication.",
            "Raw KoG2P disagreements may already be handled by the existing deterministic fixes or audio-conditioned allowed variants.",
            "Cached final outputs are from a frozen offline evaluation, not a live deployment attestation; rejected inputs stay unavailable.",
            "Disagreement may reflect omitted left context in a cropped prompt, an optional pronunciation, or a label/G2P problem.",
            "Matching at a site does not validate either the recorded pronunciation or its acoustic location.",
            "The existing LAB and TAG labels remain unchanged.",
            "All optimal edit paths are used. Ambiguous or deleted sites remain unresolved.",
            "This convenience sample does not estimate corpus-wide label error prevalence."],
        "provenance": {str(path): digest(path) for path in files}, "records": records}
    args.out.mkdir(parents=True)
    (args.out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (args.out / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: v for k, v in report.items() if k not in ("provenance", "records")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
