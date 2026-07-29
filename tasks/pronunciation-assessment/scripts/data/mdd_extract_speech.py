# -*- coding: utf-8 -*-
"""040/042 Speech zip에서 records.jsonl이 참조하는 wav만 선별 추출 (용량 최소화).

records의 file 필드 4번째 토큰(언어코드)으로 대상 데이터셋을 거르고,
그 basename의 wav만 zip에서 꺼내 out 디렉토리에 평면 저장. 멱등(이미 있으면 skip).
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_ANALYSIS

import argparse, glob, json, os, zipfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Speech zip이 있는 다운로드 디렉토리")
    ap.add_argument("--records", default=f"{MDD_ANALYSIS}/040_041_042_043_both/records.jsonl")
    ap.add_argument("--langcodes", required=True, help="쉼표구분 언어코드 (예: EN  또는  ZH,JA)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    codes = set(a.langcodes.split(","))
    os.makedirs(a.out, exist_ok=True)

    # 1) 이 데이터셋(언어코드)이 필요로 하는 basename stem
    need = set()
    for line in open(a.records, encoding="utf-8"):
        f = json.loads(line).get("file", "")
        p = f.split("-")
        if len(p) > 3 and p[3] in codes:
            need.add(f[:-5] if f.endswith(".json") else f)
    print(f"[records] 언어 {a.langcodes}: 필요 wav {len(need):,}건", flush=True)

    # 2) Speech zip 순회하며 대상 wav만 추출 (resume)
    zips = sorted(glob.glob(os.path.join(a.root, "**", "*Speech_sound*.zip"), recursive=True))
    print(f"[zip] Speech zip {len(zips)}개 발견", flush=True)
    total_new = 0
    for zp in zips:
        try:
            z = zipfile.ZipFile(zp)
        except Exception as e:
            print(f"[skip] 손상 {os.path.basename(zp)}: {e}", flush=True)
            continue
        members = [n for n in z.namelist()
                   if n.endswith(".wav") and os.path.basename(n)[:-4] in need]
        new = 0
        for n in members:
            dst = os.path.join(a.out, os.path.basename(n)[:-4] + ".wav")
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                continue
            with z.open(n) as src, open(dst, "wb") as out:
                while True:
                    b = src.read(1 << 20)
                    if not b:
                        break
                    out.write(b)
            new += 1
        z.close()
        total_new += new
        print(f"[unzip] {os.path.basename(zp)}: 대상 {len(members):,}, 신규 {new:,}", flush=True)

    have = {f[:-4] for f in os.listdir(a.out) if f.endswith(".wav")}
    resolved = len(need & have)
    print(f"\n[done] 신규추출 {total_new:,} | records 연결 {resolved:,}/{len(need):,} "
          f"| 미연결 {len(need) - resolved:,}", flush=True)


if __name__ == "__main__":
    main()
