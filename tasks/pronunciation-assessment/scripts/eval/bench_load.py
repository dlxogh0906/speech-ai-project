# -*- coding: utf-8 -*-
"""동시 요청 부하 테스트 — 살아 있는 API 에 던져 큐잉·꼬리지연·실패를 본다.

`bench_latency.py` 는 1건 지연을, `bench_batch.py` 는 배치 상한을 쟀다. 이건 실제
HTTP 경로로 동시에 던졌을 때 **무엇이 먼저 무너지는가**를 본다. 서버가 인코더 호출을
lock 으로 직렬화하므로 처리량은 안 오르고 대기만 쌓이는 게 예상 동작이다 —
그 예상이 맞는지, 어디서 타임아웃·에러가 나는지 확인한다.

    $PY tasks/pronunciation-assessment/scripts/eval/bench_load.py --url http://localhost:8914 --levels 1,2,4,8,16,32
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import json, time, argparse, statistics
from concurrent.futures import ThreadPoolExecutor


def main():
    ap = argparse.ArgumentParser(description="동시 요청 부하 테스트")
    ap.add_argument("--url", default="http://localhost:8914")
    ap.add_argument("--levels", default="1,2,4,8,16,32")
    ap.add_argument("--per_level", type=int, default=48, help="레벨마다 보낼 총 요청 수")
    ap.add_argument("--sample", type=int, default=0, help="사용할 내장 샘플 번호")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--out", default=f"{MDD_OUTPUTS}/bench_load.json")
    A = ap.parse_args()

    # requests 는 fairseq 환경에 없다. 표준 라이브러리만 쓴다.
    import urllib.request, urllib.error, uuid

    def post_form(url, fields, timeout):
        b = uuid.uuid4().hex
        body = b"".join(
            f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
            for k, v in fields.items()) + f"--{b}--\r\n".encode()
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": f"multipart/form-data; boundary={b}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()

    def one(i):
        t0 = time.perf_counter()
        try:
            st, _ = post_form(f"{A.url}/diagnose_sample",
                              {"i": A.sample},        # session_id 없음 — 세션 누적을 피한다
                              A.timeout)
            return (time.perf_counter() - t0, st == 200, st)
        except urllib.error.HTTPError as e:
            return (time.perf_counter() - t0, False, e.code)
        except Exception as e:
            return (time.perf_counter() - t0, False, type(e).__name__)

    with urllib.request.urlopen(f"{A.url}/health", timeout=30) as r:
        h = json.loads(r.read())
    print(f"대상 {A.url} · {h.get('ckpt')}\n")
    print(f"{'동시':>5}{'요청':>6}{'성공':>6}{'p50_ms':>9}{'p95_ms':>9}{'최대_ms':>9}{'초당건수':>10}")

    rows = []
    for C in [int(x) for x in A.levels.split(",")]:
        n = max(C, A.per_level)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=C) as ex:
            res = list(ex.map(one, range(n)))
        wall = time.perf_counter() - t0
        lat = sorted(x[0] * 1000 for x in res)
        ok = sum(1 for x in res if x[1])
        bad = {}
        for x in res:
            if not x[1]:
                bad[str(x[2])] = bad.get(str(x[2]), 0) + 1
        p = lambda q: lat[min(len(lat) - 1, int(len(lat) * q))]
        row = {"동시": C, "요청": n, "성공": ok,
               "p50_ms": round(p(0.50), 1), "p95_ms": round(p(0.95), 1),
               "최대_ms": round(lat[-1], 1), "초당건수": round(n / wall, 2)}
        if bad:
            row["실패"] = bad
        rows.append(row)
        print(f"{C:>5}{n:>6}{ok:>6}{row['p50_ms']:>9.1f}{row['p95_ms']:>9.1f}"
              f"{row['최대_ms']:>9.1f}{row['초당건수']:>10.2f}" + (f"   실패 {bad}" if bad else ""))

    json.dump({"_설명": "살아 있는 API 에 동시 요청. 서버가 인코더를 lock 으로 직렬화한다.",
               "url": A.url, "ckpt": h.get("ckpt"), "측정": rows},
              open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n→ {A.out}")


if __name__ == "__main__":
    main()
