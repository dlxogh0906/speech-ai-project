"""Index existing Seoul Corpus s0/ss + vowel labels for acoustic inspection.

This reads a local label archive only. It neither downloads audio nor treats
published phoneme edges as first-voicing/aspiration landmarks or learner gold.
Duplicate tier names are intentional in this corpus and must be preserved.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import argparse
import bisect
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
import re
import tarfile
import zipfile

VOWELS = frozenset("aa ee ii xx vv uu oo ye ya yv yu yo wi we wa wv xi".split())
EXPECTED_TIERS = ["phoneme", "pWord.prono.", "pWord.prono.", "utt.prono.",
                  "pWord.ortho.", "pWord.ortho.", "utt.ortho."]
INTERVAL = re.compile(r'xmin = ([-\d.eE]+)\s*xmax = ([-\d.eE]+)\s*text = "((?:[^"]|"")*)"')


def parse_textgrid(data):
    """Read validated long-format IntervalTiers, preserving tier ordinals."""
    text = data.decode("utf-16" if data[:2] in (b"\xfe\xff", b"\xff\xfe") else "utf-8-sig")
    blocks = re.split(r"(?m)^    item \[\d+\]:\s*$", text)
    count = re.search(r"(?m)^size = (\d+)\s*$", blocks[0])
    if count is None or int(count.group(1)) != len(blocks) - 1:
        raise ValueError("tier count mismatch or unsupported TextGrid format")
    tiers = []
    for ordinal, block in enumerate(blocks[1:], 1):
        if 'class = "IntervalTier"' not in block:
            raise ValueError("expected IntervalTier")
        name = re.search(r'name = "((?:[^"]|"")*)"', block)
        expected = re.search(r"intervals: size = (\d+)", block)
        entries = [(float(lo), float(hi), label.replace('""', '"'))
                   for lo, hi, label in INTERVAL.findall(block)]
        if name is None or expected is None or len(entries) != int(expected.group(1)):
            raise ValueError("interval count mismatch")
        for j, (lo, hi, _) in enumerate(entries):
            if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo < hi):
                raise ValueError("invalid interval endpoints")
            if j and abs(lo - entries[j - 1][1]) > 1e-7:
                raise ValueError("noncontiguous intervals")
        tiers.append({"ordinal": ordinal, "name": name.group(1).replace('""', '"'),
                      "intervals": entries})
    return tiers


def containing(entries, time):
    """Return the one interval containing a point, without nearest guessing."""
    j = bisect.bisect_right([r[0] for r in entries], time) - 1
    if 0 <= j < len(entries) and entries[j][0] <= time < entries[j][1]:
        return j, entries[j]
    return None, None


def collect_cv(tiers, recording, margin=.25):
    if [t["name"] for t in tiers] != EXPECTED_TIERS:
        raise ValueError("unexpected Seoul Corpus tier order")
    m = re.fullmatch(r"(s\d{2})([mf])(\d{2})([mf])(\d+)", recording)
    if not m:
        raise ValueError("unexpected recording identifier")
    phones = tiers[0]["intervals"]
    rows, rejected = [], Counter()
    for i, (lo, hi, phone) in enumerate(phones):
        if phone not in ("s0", "ss"):
            continue
        if i + 1 >= len(phones) or phones[i + 1][2] not in VOWELS:
            rejected["no_immediately_following_labeled_vowel"] += 1
            continue
        vlo, vhi, vowel = phones[i + 1]
        clip_lo, clip_hi = max(0., lo - margin), min(phones[-1][1], vhi + margin)
        # Trim context at marked nonspeech/private regions. Dropping every
        # adjacent silence would systematically remove utterance onsets.
        requested_context = [clip_lo, clip_hi]
        bad = [(a, b) for a, b, p in phones if a < clip_hi and b > clip_lo
               and (not p.strip() or p.startswith("<"))]
        if any(a < vhi and b > lo for a, b in bad):
            rejected["marked_nonspeech_or_private_in_target"] += 1
            continue
        for a, b in bad:
            if b <= lo:
                clip_lo = max(clip_lo, b)
            elif a >= vhi:
                clip_hi = min(clip_hi, a)
        wi, word = containing(tiers[1]["intervals"], (lo + hi) / 2)
        _, roman = containing(tiers[2]["intervals"], (lo + hi) / 2)
        _, ortho = containing(tiers[4]["intervals"], (lo + hi) / 2)
        _, utterance = containing(tiers[3]["intervals"], (lo + hi) / 2)
        if any(v is None for v in (word, roman, ortho, utterance)):
            rejected["missing_containing_word_or_utterance"] += 1
            continue
        if any("<" in v[2] for v in (word, roman, ortho)):
            rejected["marked_containing_word"] += 1
            continue
        rows.append({
            "id": "%s_p%05d" % (recording, i + 1), "recording": recording,
            "speaker": m.group(1), "speaker_sex_metadata": m.group(2),
            "speaker_age_metadata": int(m.group(3)), "interviewer_sex_metadata": m.group(4),
            "part": int(m.group(5)), "phone_interval_ordinal": i + 1,
            "phone": phone, "following_vowel": vowel,
            "previous_phone": phones[i - 1][2] if i else None,
            "label_consonant_interval": [lo, hi], "label_vowel_interval": [vlo, vhi],
            "context_interval": [clip_lo, clip_hi], "word_ordinal": wi + 1,
            "context_trimmed_at_marked_interval": requested_context != [clip_lo, clip_hi],
            "word_pronounced_hangul": word[2], "word_pronounced_roman": roman[2],
            "word_orthographic_hangul": ortho[2], "word_interval": list(word[:2]),
            "at_labeled_word_start": abs(lo - word[0]) < 1e-7,
            "at_labeled_utterance_start": abs(lo - utterance[0]) < 1e-7,
            "cv_within_labeled_word": vhi <= word[1] + 1e-7,
            "label_source": "published_human_corrected_corpus_annotation",
            "acoustic_landmarks_status": "not_independently_annotated",
        })
    return rows, rejected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, required=True, help="local OpenSLR 113 label.tgz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--context-margin", type=float, default=.25)
    a = ap.parse_args()
    if a.out.exists():
        ap.error("refusing to overwrite a reference index")
    if not math.isfinite(a.context_margin) or a.context_margin < 0:
        ap.error("context margin must be finite and nonnegative")
    a.out.mkdir(parents=True)
    rows, files, rejected, failures = [], [], Counter(), []
    with tarfile.open(str(a.labels), "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".zip")]
        if len(members) != 1:
            raise ValueError("expected one label ZIP inside the tar")
        with zipfile.ZipFile(io.BytesIO(tf.extractfile(members[0]).read())) as z:
            names = sorted(n for n in z.namelist()
                           if n.endswith(".TextGrid") and not n.startswith("__MACOSX/"))
            for name in names:
                data = z.read(name)
                try:
                    tiers = parse_textgrid(data)
                    found, skipped = collect_cv(tiers, Path(name).stem, a.context_margin)
                    rows.extend(found)
                    rejected.update(skipped)
                    files.append({"member": name, "sha256": hashlib.sha256(data).hexdigest(),
                                  "phoneme_intervals": len(tiers[0]["intervals"]),
                                  "cv_candidates": len(found)})
                except (ValueError, UnicodeError) as exc:
                    failures.append({"member": name, "error": str(exc)})
    # Report the source counts, including omitted conditions, before looking
    # at acoustic outcomes. No learned threshold or score is produced here.
    by_vowel, by_speaker = defaultdict(Counter), defaultdict(Counter)
    for row in rows:
        by_vowel[row["following_vowel"]][row["phone"]] += 1
        by_speaker[row["speaker"]][row["phone"]] += 1
    report = {
        "source": "https://openslr.org/113/", "license": "CC BY-NC 2.0",
        "archive_sha256": hashlib.sha256(a.labels.read_bytes()).hexdigest(),
        "source_files": len(names), "parsed_files": len(files), "failures": failures,
        "phoneme_intervals": sum(f["phoneme_intervals"] for f in files),
        "context_margin_seconds": a.context_margin, "candidates": len(rows),
        "speakers": len(by_speaker), "counts_by_following_vowel": dict(by_vowel),
        "counts_by_speaker": dict(by_speaker), "excluded_counts": dict(rejected),
        "files": files,
        "limitations": ["Metadata index only; audio has not been verified for every file.",
                        "Published phone boundaries are not new acoustic event annotations.",
                        "Actual pronunciation transcription is not a canonical target or learner error label.",
                        "Spontaneous native speech is not a matched learner assessment benchmark.",
                        "Merged vowel labels ee and we do not distinguish all orthographic vowels.",
                        "No training or service integration is performed."],
    }
    (a.out / "candidates.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (a.out / "inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("parsed_files", "failures", "candidates", "speakers", "counts_by_following_vowel", "excluded_counts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
