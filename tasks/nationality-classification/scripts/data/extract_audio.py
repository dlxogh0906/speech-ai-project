"""Extract the AIHub foreign-Korean-speech zips into the layout the manifests expect.

On this server the corpus lives (still zipped) under, e.g.:
    $SPEECH_AI_DATA/aihub/foreign_korean_speech/131.../01.데이터_new_20220719/{1.Training,2.Validation}/
        원천데이터/VS_1._베트남어.zip
        원천데이터_230113_add/TS_6._기타_add.zip

The manifests, however, reference paths like:
    원천데이터/VS_1. 베트남어/1. 한국일반/<utt>.wav
    원천데이터_230113_add/TS_6. 기타_add/1. 한국일반/<utt>.wav

So each `<NAME>.zip` must be extracted into a sibling folder named `<NAME normalized>`,
where normalization replaces the first "._" with ". " (e.g. VS_1._베트남어 -> "VS_1. 베트남어",
TS_6._기타_add -> "TS_6. 기타_add"). This script does exactly that.

Examples:
    # Validation only (smaller; ~38GB zipped), wav only:
    python scripts/extract_audio.py --split 2.Validation --only-ext .wav

    # A single language zip, to verify the pipeline cheaply:
    python scripts/extract_audio.py --split 2.Validation --zip-glob 'VS_2._영어.zip' --only-ext .wav

    # Dry run (list what would happen):
    python scripts/extract_audio.py --split 2.Validation --dry-run
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]


import argparse
import sys
import zipfile
from pathlib import Path

from country_classifier.baseline_common import DEFAULT_AUDIO_ROOT


def normalize_zip_stem(stem):
    """VS_1._베트남어 -> 'VS_1. 베트남어' ; TS_6._기타_add -> 'TS_6. 기타_add'."""
    return stem.replace("._", ". ", 1)


def fixed_member_name(info):
    """Recover the real (Korean) member name regardless of interpreter.

    These AIHub zips store CP949 names without the UTF-8 flag. Python's zipfile then
    decodes them with cp437 -> mojibake like '1. ╟╤▒╣└╧╣▌' (seen on py3.8/3.10; py3.13
    happens to decode differently). The manifests expect proper Korean ('1. 한국일반'),
    so we re-decode: cp437 bytes -> cp949. If that fails the name was already correct.
    """
    if info.flag_bits & 0x800:  # UTF-8 flag set -> already correct
        return info.filename
    try:
        return info.filename.encode("cp437").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def find_zips(split_dir, zip_glob):
    zips = []
    for sub in ("원천데이터", "원천데이터_230113_add"):
        d = split_dir / sub
        if d.is_dir():
            zips.extend(sorted(d.glob(zip_glob)))
    return zips


def extract_one(zip_path, only_ext, dry_run, overwrite):
    target = zip_path.parent / normalize_zip_stem(zip_path.stem)
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        if only_ext:
            infos = [i for i in infos if i.filename.lower().endswith(only_ext.lower())]
        n_total = len(infos)
        n_done = 0
        n_skip = 0
        for info in infos:
            name = fixed_member_name(info)            # proper Korean for the path on disk
            if name.endswith("/"):
                continue
            dest = target / name
            if dest.exists() and not overwrite:
                n_skip += 1
                continue
            if dry_run:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, dest.open("wb") as out:  # read via original entry
                out.write(src.read())
            n_done += 1
    return target, n_total, n_done, n_skip


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio_root", default=DEFAULT_AUDIO_ROOT,
                    help="base dir holding 1.Training / 2.Validation")
    ap.add_argument("--split", default="2.Validation",
                    choices=["1.Training", "2.Validation"],
                    help="which split's zips to extract")
    ap.add_argument("--zip-glob", default="*.zip",
                    help="glob to select zips (e.g. 'VS_2._영어.zip')")
    ap.add_argument("--only-ext", default=".wav",
                    help="extract only members with this extension (\"\" for all)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    split_dir = Path(args.audio_root) / args.split
    if not split_dir.is_dir():
        sys.exit(f"[error] split dir not found: {split_dir}")

    zips = find_zips(split_dir, args.zip_glob)
    if not zips:
        sys.exit(f"[error] no zips matched {args.zip_glob} under {split_dir}")

    print(f"split_dir = {split_dir}")
    print(f"matched {len(zips)} zip(s); only_ext={args.only_ext!r} dry_run={args.dry_run}")
    grand_done = grand_skip = 0
    for z in zips:
        target, n_total, n_done, n_skip = extract_one(
            z, args.only_ext, args.dry_run, args.overwrite
        )
        grand_done += n_done
        grand_skip += n_skip
        print(f"  {z.name} -> {target.name}/  members(filtered)={n_total} "
              f"written={n_done} skipped_existing={n_skip}")
    print(f"DONE: written={grand_done} skipped_existing={grand_skip}")


if __name__ == "__main__":
    main()
