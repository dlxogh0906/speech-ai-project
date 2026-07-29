"""European(041) labels -> existing 40-class manifest (no new classes).

Reads each label JSON's SpeakerMetadata.nationality, maps to one of the existing
40 country classes, DROPS anything not in the 40, prefixes speaker ids with 'EU_'
(avoid clash with Asian ids), resolves the matching wav, and writes a CSV manifest
(same columns as the Asian pipeline) for feature extraction.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, NAT_DATA


import argparse
import csv
import json
import os
from collections import Counter

from country_classifier import baseline_common as bc

ROOT = (f"{DATA_ROOT}/aihub/european_korean_speech/041.교육용_유럽어_모국어_사용자의_한국어_음성_데이터"
        "/01-1.정식개방데이터/Training")
LBL = ROOT + "/02.라벨링데이터/extracted"
AUD = ROOT + "/01.원천데이터/extracted"
LABEL_SUBDIRS = ["TL_Speech_lab", "TL_pronunciation_lab"]

# existing 40 classes (Korean names)
K40 = ["중국", "일본", "베트남", "태국", "몽골", "러시아", "미얀마", "인도네시아", "미국", "홍콩",
       "우즈베키스탄", "말레이시아", "튀르키예", "네팔", "필리핀", "캄보디아", "영국", "페루", "독일",
       "키르기스스탄", "카자흐스탄", "에티오피아", "루마니아", "르완다", "인도", "타지키스탄",
       "우크라이나", "투르크메니스탄", "이란", "이탈리아", "불가리아", "잠비아", "콜롬비아", "프랑스",
       "브라질", "파키스탄", "싱가포르", "칠레", "방글라데시", "스페인"]
C2I = {k: i for i, k in enumerate(sorted(K40))}

# English nationality (041) -> Korean 40-class name. Anything absent -> dropped.
NAT2K = {
    "Spain": "스페인", "Russian Federation": "러시아", "Turkey": "튀르키예", "France": "프랑스",
    "Italy": "이탈리아", "Romania": "루마니아", "Germany": "독일", "Uzbekistan": "우즈베키스탄",
    "Bulgaria": "불가리아", "Brazil": "브라질", "Kazakhstan": "카자흐스탄", "Colombia": "콜롬비아",
    "Ukraine": "우크라이나", "Peru": "페루", "Turkmenistan": "투르크메니스탄",
    "Kyrgyzstan": "키르기스스탄", "Chile": "칠레", "United States": "미국",
    "United Kingdom": "영국", "Tajikistan": "타지키스탄", "India": "인도", "Iran": "이란",
    "Pakistan": "파키스탄", "Bangladesh": "방글라데시", "Singapore": "싱가포르",
    "Ethiopia": "에티오피아", "Rwanda": "르완다", "Zambia": "잠비아",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{NAT_DATA}/euro_country40_manifest.csv")
    args = ap.parse_args()

    print("[audio] indexing wavs ...", flush=True)
    idx = {}
    for dp, _, fs in os.walk(AUD):
        for f in fs:
            if f.endswith(".wav"):
                idx[f[:-4]] = os.path.join(dp, f)
    print(f"[audio] {len(idx):,} wavs indexed", flush=True)

    nat_cache = {}          # uid -> nationality (per-speaker, read once)
    rows = []
    kept, dropped, no_audio = Counter(), Counter(), 0
    for sub in LABEL_SUBDIRS:
        d = os.path.join(LBL, sub)
        with os.scandir(d) as it:
            for e in it:
                if not e.name.endswith(".json"):
                    continue
                base = e.name[:-5]
                uid = base.split("-")[0]
                if uid not in nat_cache:
                    try:
                        nat_cache[uid] = json.load(open(e.path, encoding="utf-8"))[
                            "SpeakerMetadata"]["nationality"]
                    except Exception:
                        nat_cache[uid] = "?"
                k = NAT2K.get(nat_cache[uid])
                if not k:
                    dropped[nat_cache[uid]] += 1
                    continue
                wav = idx.get(base)
                if not wav:
                    no_audio += 1
                    continue
                rows.append({"audio_path": wav, "json_path": e.path,
                             "speaker_id": "EU_" + uid, "label_name": k,
                             "label_id": C2I[k], "duration_sec": ""})
                kept[k] += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bc.CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\n[done] {len(rows):,} utts -> {args.out}")
    print(f"  kept {sum(kept.values()):,} / dropped(non-40) {sum(dropped.values()):,} / no_audio {no_audio:,}")
    print("  per-class kept (utts):")
    for k, n in kept.most_common():
        print(f"    {k:<10} {n:>8,}")


if __name__ == "__main__":
    main()
