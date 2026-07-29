"""Tiny async load tester. Fires N requests at concurrency C to <url>/predict.
Usage: python loadtest.py <url> <concurrency> <total> <wav>"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)

import asyncio, time, sys, httpx

URL, C, N, WAV = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
raw = open(WAV, "rb").read()


async def one(client, sem):
    async with sem:
        t0 = time.time()
        try:
            r = await client.post(f"{URL}/predict",
                                  files={"audio": ("a.wav", raw, "audio/wav")}, timeout=40)
            return (r.status_code == 200), time.time() - t0
        except Exception:
            return False, time.time() - t0


async def main():
    sem = asyncio.Semaphore(C)
    async with httpx.AsyncClient() as client:
        t0 = time.time()
        res = await asyncio.gather(*[one(client, sem) for _ in range(N)])
        wall = time.time() - t0
    lats = sorted(d for ok, d in res if ok)
    errs = sum(1 for ok, _ in res if not ok)
    p = lambda q: lats[min(len(lats) - 1, int(q * len(lats)))] if lats else 0
    print(f"conc={C:3d} N={N:3d} | ok={len(lats):3d} err/timeout={errs:2d} | "
          f"wall={wall:5.1f}s  thru={len(lats)/wall:5.1f} req/s | "
          f"p50={p(.5):5.1f}s p95={p(.95):5.1f}s max={(max(lats) if lats else 0):5.1f}s")


asyncio.run(main())
