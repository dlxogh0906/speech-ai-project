# -*- coding: utf-8 -*-
"""AIHub zip 을 한글 파일명이 깨지지 않게 푼다 (040~043 발음평가 zip · 037 한국인 영어 zip).

AIHub zip 은 파일명을 CP949 로 저장하면서 UTF-8 플래그를 안 켠다. 그래서 plain unzip 이나
파이썬 zipfile 기본 동작으로 풀면 폴더명·파일명이 깨진다(cp437 로 해석됨). 여기서는
cp437 바이트 → cp949 로 다시 해석한다. UTF-8 플래그가 켜진 항목은 그대로 쓴다.

각 zip 은 옆에 zip 이름과 같은 폴더로 푼다.
    .../02.라벨링데이터/TL_pronunciation_tag.zip  →  .../02.라벨링데이터/TL_pronunciation_tag/
zip 안의 모든 항목이 그 이름의 최상위 폴더 하나에 들어 있으면 한 겹을 벗긴다(이중 중첩 방지).
앞에 붙은 "/" 는 떼고(041 VL_pronunciation_lab.zip 이 그렇다), ".." 가 든 경로는 거부한다.

멱등: 크기가 같은 파일이 이미 있으면 건너뛴다. 전부 성공한 zip 은 대상 폴더에
`.extracted_ok`(항목 수·바이트)를 남기고, 다음 실행에서 zip 째로 건너뛴다.

--delete-zip: 완료 표시가 있고 **풀린 파일이 zip 의 모든 항목과 이름·크기까지 맞을 때만** 그 zip 을 지운다
(CRC 는 풀 때 zipfile 이 검사했다). 하나라도 어긋나면 지우지 않고 ⚠️ 로 남긴다.

    $PY tasks/pronunciation-assessment/scripts/data/extract_aihub_zips.py --root $SPEECH_AI_DATA/aihub --dry-run
    $PY tasks/pronunciation-assessment/scripts/data/extract_aihub_zips.py --root $SPEECH_AI_DATA/aihub --glob '*pronunciation_l*.zip'
    $PY tasks/pronunciation-assessment/scripts/data/extract_aihub_zips.py --root $SPEECH_AI_DATA/aihub --workers 4
    $PY tasks/pronunciation-assessment/scripts/data/extract_aihub_zips.py \
        --root "$SPEECH_AI_DATA/aihub/037.교육용_한국인의_영어_음성_데이터" --glob '*.zip' --delete-zip
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, json, os, shutil, time, zipfile
from multiprocessing import Pool
from pathlib import Path, PurePosixPath

MARKER = ".extracted_ok"


def member_name(info):
    """zip 항목의 실제 한글 이름. (이름, cp949 로 고쳤는지)"""
    if info.flag_bits & 0x800:
        return info.filename, False
    try:
        fixed = info.filename.encode("cp437").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename, False
    return fixed, fixed != info.filename


def safe_parts(name):
    """'/a/b.wav' → ('a','b.wav'). 빈 조각·'.' 은 버리고 '..' 는 거부."""
    parts = [p for p in PurePosixPath(name.replace("\\", "/")).parts if p not in ("/", "", ".")]
    if any(p == ".." for p in parts):
        raise ValueError(f"안전하지 않은 경로: {name!r}")
    return parts


def plan(zip_path):
    """(대상폴더, [(info, 상대경로 parts)], 이름을 고친 항목 수)"""
    target = zip_path.with_suffix("")
    with zipfile.ZipFile(zip_path) as zf:
        rows, n_fixed = [], 0
        for info in zf.infolist():
            name, fixed = member_name(info)
            n_fixed += fixed
            if info.is_dir():
                continue
            rows.append((info, safe_parts(name)))
    tops = {parts[0] for _, parts in rows if len(parts) > 1}
    if rows and len(tops) == 1 and all(len(p) > 1 for _, p in rows) and tops == {target.name}:
        rows = [(i, p[1:]) for i, p in rows]
    return target, rows, n_fixed


def extract(job):
    """zip 1개를 푼다. 예외는 삼켜서 status='error' 로 돌려준다 — 한 zip 이 깨져도 나머지는 계속 푼다."""
    try:
        return _extract(job)
    except Exception as e:
        return {"zip": str(job[0]), "target": str(job[0].with_suffix("")), "members": 0, "bytes": 0,
                "name_fixed": 0, "sample": [], "written": 0, "skipped": 0,
                "status": "error", "error": f"{type(e).__name__}: {e}"}


def verify(target, rows):
    """풀린 파일이 zip 항목과 이름·크기까지 다 맞는가. → 어긋난 항목 수"""
    bad = 0
    for info, parts in rows:
        dst = target.joinpath(*parts)
        if not dst.is_file() or dst.stat().st_size != info.file_size:
            bad += 1
    return bad


def _extract(job):
    zip_path, dry_run, delete_zip = job
    t0 = time.time()
    target, rows, n_fixed = plan(zip_path)
    total_bytes = sum(i.file_size for i, _ in rows)
    head = [str(PurePosixPath(*p)) for _, p in rows[:3]]
    res = {"zip": str(zip_path), "target": str(target), "members": len(rows), "bytes": total_bytes,
           "name_fixed": n_fixed, "sample": head, "written": 0, "skipped": 0, "status": "dry-run"}
    if dry_run:
        return res
    marker = target / MARKER
    if marker.exists():
        res["status"] = "already"
        return _maybe_delete(res, zip_path, target, rows, delete_zip)
    with zipfile.ZipFile(zip_path) as zf:
        for info, parts in rows:
            dst = target.joinpath(*parts)
            if dst.exists() and dst.stat().st_size == info.file_size:
                res["skipped"] += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            with zf.open(info) as src, open(tmp, "wb") as out:   # CRC 는 zipfile 이 읽으면서 검사한다
                shutil.copyfileobj(src, out, 1 << 20)
            os.replace(tmp, dst)
            res["written"] += 1
    marker.write_text(json.dumps({"zip": zip_path.name, "members": len(rows), "bytes": total_bytes,
                                  "time": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False))
    res["status"] = "ok"
    res["sec"] = round(time.time() - t0, 1)
    return _maybe_delete(res, zip_path, target, rows, delete_zip)


def _maybe_delete(res, zip_path, target, rows, delete_zip):
    if not delete_zip:
        return res
    bad = verify(target, rows)
    if bad:
        res["status"] = "error"
        res["error"] = "풀린 파일 %d개가 zip 과 다르다 — zip 을 지우지 않았다" % bad
        return res
    zip_path.unlink()
    res["deleted"] = True
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="zip 을 재귀로 찾을 폴더 (보통 $SPEECH_AI_DATA/aihub)")
    ap.add_argument("--glob", default="*pronunciation*.zip", help="zip 이름 패턴")
    ap.add_argument("--workers", type=int, default=4, help="동시에 푸는 zip 수")
    ap.add_argument("--dry-run", action="store_true", help="풀지 않고 항목 수·용량·이름 샘플만 본다")
    ap.add_argument("--delete-zip", dest="delete_zip", action="store_true",
                    help="다 풀리고 이름·크기 검증이 끝난 zip 만 지운다")
    a = ap.parse_args()

    root = Path(a.root)
    if not root.is_dir():
        raise SystemExit(f"폴더가 없다: {root}")
    zips = sorted(p for p in root.rglob(a.glob) if p.is_file())
    if not zips:
        raise SystemExit(f"{root} 아래에 {a.glob} 에 맞는 zip 이 없다")

    need = sum(sum(i.file_size for i in zipfile.ZipFile(z).infolist()) for z in zips)
    free = shutil.disk_usage(root).free
    print(f"zip {len(zips)}개 · 풀면 {need / 2**30:.1f} GiB · 여유 {free / 2**30:.1f} GiB", flush=True)
    if not a.dry_run and need > free:
        raise SystemExit("디스크가 모자란다 (이미 풀린 파일은 계산에 안 뺐다 — 필요하면 --glob 로 나눠 풀 것)")

    fail = 0
    with Pool(max(1, min(a.workers, len(zips)))) as pool:
        for r in pool.imap_unordered(extract, [(z, a.dry_run, a.delete_zip and not a.dry_run) for z in zips]):
            print(f"[{r['status']}] {Path(r['zip']).relative_to(root)} → {Path(r['target']).name}/  "
                  f"항목 {r['members']:,} · {r['bytes'] / 2**30:.2f} GiB · 이름수정 {r['name_fixed']:,} · "
                  f"새로씀 {r['written']:,} · 건너뜀 {r['skipped']:,}  {r.get('sec', '')}"
                  f"{'  · zip 지움' if r.get('deleted') else ''}", flush=True)
            print(f"        예: {r['sample']}" if r["status"] != "error" else f"        ⚠️ {r['error']}", flush=True)
            fail |= r["status"] == "error"
    if fail:
        print("실패한 zip 이 있다 — 위 ⚠️ 줄을 볼 것. 받다 만 zip 이면 다시 받은 뒤 재실행하면 이어서 푼다", flush=True)
    return int(fail)


if __name__ == "__main__":
    raise SystemExit(main())
