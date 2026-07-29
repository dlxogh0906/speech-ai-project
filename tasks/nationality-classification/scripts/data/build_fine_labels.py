"""Build a basename -> fine-grained label (motherTongue) map from the labeling JSONs.

The AIHub labeling JSONs live in the `라벨링데이터/*.zip` archives (one JSON per wav).
The fine-grained class is `skill_info.motherTongue` (e.g. '영어', '베트남어', ...). We
optionally normalise it through configs/fine_grained_label_alias_*.json (typo fixes /
selected merges) WITHOUT touching the raw data.

Output: $NATIONALITY_DATA/fine_labels.csv  (basename, speaker_id, mother_tongue, country)
plus a printed label distribution. Streams the zips (no extraction of 1M JSONs).
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, NAT_DATA, NAT_TASK


import argparse
import csv
import json
import zipfile
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path

DEFAULT_AUDIO_ROOT = (
    f"{DATA_ROOT}/aihub/foreign_korean_speech/"
    "131.인공지능_학습을_위한_외국인_한국어_발화_음성_데이터/01.데이터_new_20220719"
)


def load_alias(path):
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("aliases", {}) if isinstance(data, dict) else {}


def decode_json(raw):
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return json.loads(raw.decode(enc))
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio_root", default=DEFAULT_AUDIO_ROOT)
    ap.add_argument("--out", default=f"{NAT_DATA}/fine_labels.csv")
    ap.add_argument("--alias_config",
                    default=str(NAT_TASK / "configs/fine_grained_label_alias_merged6.json"))
    args = ap.parse_args()

    alias = load_alias(args.alias_config)
    print(f"alias entries: {len(alias)}")

    label_zips = sorted(glob(f"{args.audio_root}/*/라벨링데이터/*.zip"))
    print(f"labeling zips: {len(label_zips)}")

    rows = []
    dist = Counter()
    spk_labels = defaultdict(set)
    n_missing = 0
    for z in label_zips:
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                if not info.filename.lower().endswith(".json"):
                    continue
                data = decode_json(zf.read(info))
                if data is None:
                    n_missing += 1
                    continue
                fn = data.get("fileName") or Path(info.filename).with_suffix(".wav").name
                skill = data.get("skill_info", {})
                mt = (skill.get("motherTongue") or "").strip()
                mt = alias.get(mt, mt)
                spk = data.get("SpeakerID") or data.get("file_info", {}).get("speakerID", "")
                country = data.get("residence_info", {}).get("country", "")
                if not mt:
                    n_missing += 1
                    continue
                rows.append((fn, spk, mt, country))
                dist[mt] += 1
                spk_labels[spk].add(mt)
        print(f"  done {Path(z).parent.parent.name}/{Path(z).name}: total rows so far {len(rows):,}")

    out = Path(args.out)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["basename", "speaker_id", "mother_tongue", "country"])
        w.writerows(rows)

    multi = {s: ls for s, ls in spk_labels.items() if len(ls) > 1}
    print(f"\n=== fine labels -> {out} ===")
    print(f"utts: {len(rows):,}  missing/skipped: {n_missing}")
    print(f"distinct mother_tongue classes: {len(dist)}")
    print(f"distinct speakers: {len(spk_labels):,}  speakers w/ >1 label: {len(multi)}")
    print("\nclass distribution (sorted):")
    for lab, c in dist.most_common():
        print(f"  {c:>8,}  {lab}")
    Path(str(out) + ".dist.json").write_text(
        json.dumps({"classes": len(dist), "distribution": dict(dist.most_common())},
                   ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
