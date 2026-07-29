#!/usr/bin/env python3
"""동접(동시 사용자) 부하 테스트 — XLS-R GPU vs CPU 데모 서버.

동시성 레벨을 단계적으로 올리며(1→2→4→…) 각 레벨에서 여러 요청을 동시에 쏘고,
클라이언트 체감 시간(wall)과 서버가 보고한 큐 대기(wait_ms)/연산(compute_ms)을 수집한다.

사용:
  python load_test.py --wav <파일> --mode both   # /predict_both (GPU+CPU 동시)
  python load_test.py --wav <파일> --mode gpu    # /predict device=gpu
  python load_test.py --wav <파일> --mode cpu    # /predict device=cpu
  옵션: --levels 1,2,4,8,16  --per-level 8  --url http://localhost:8777
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]


import argparse
import json
import statistics
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor


def post_multipart(url, wav_bytes, fields):
    boundary = uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
             f"filename=\"load_test.wav\"\r\nContent-Type: audio/wav\r\n\r\n").encode()
    body += wav_bytes + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read())
        return {"ok": True, "wall_s": time.perf_counter() - t0, "resp": data}
    except Exception as e:
        return {"ok": False, "wall_s": time.perf_counter() - t0, "error": str(e)[:200]}


def pct(vals, p):
    if not vals:
        return float("nan")
    vals = sorted(vals)
    idx = min(len(vals) - 1, max(0, round(p / 100 * (len(vals) - 1))))
    return vals[idx]


def run_level(url, wav_bytes, fields, concurrency, per_level):
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(lambda _: post_multipart(url, wav_bytes, fields), range(per_level)))
    elapsed = time.perf_counter() - t_start
    ok = [r for r in results if r["ok"]]
    errs = [r for r in results if not r["ok"]]
    walls = [r["wall_s"] for r in ok]

    def server_metric(key_path):
        vals = []
        for r in ok:
            node = r["resp"]
            try:
                for k in key_path:
                    node = node[k]
                vals.append(node)
            except (KeyError, TypeError):
                pass
        return vals

    row = {
        "concurrency": concurrency,
        "requests": per_level,
        "ok": len(ok),
        "errors": len(errs),
        "throughput_rps": round(len(ok) / elapsed, 2) if elapsed > 0 else 0,
        "wall_p50_s": round(pct(walls, 50), 2),
        "wall_p95_s": round(pct(walls, 95), 2),
        "wall_max_s": round(pct(walls, 100), 2),
    }
    for dev in ("gpu", "cpu"):
        w = server_metric([dev, "wait_ms"]) or server_metric(["timing", "wait_ms"])
        c = server_metric([dev, "compute_ms"]) or server_metric(["timing", "encoder_ms"])
        if w:
            row[f"{dev}_wait_p95_ms"] = round(pct(w, 95), 0)
        if c:
            row[f"{dev}_compute_p50_ms"] = round(pct(c, 50), 0)
    if errs:
        row["first_error"] = errs[0]["error"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8777")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--mode", choices=("both", "gpu", "cpu"), default="both")
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--per-level", type=int, default=8)
    ap.add_argument("--truncate", type=float, default=0.0)
    args = ap.parse_args()

    wav_bytes = open(args.wav, "rb").read()
    if args.mode == "both":
        url = args.url + "/predict_both"
        fields = {"truncate_sec": args.truncate}
    else:
        url = args.url + "/predict"
        fields = {"truncate_sec": args.truncate, "runs": 1, "device": args.mode}

    levels = [int(x) for x in args.levels.split(",")]
    print(f"# mode={args.mode} url={url} wav={args.wav} per_level={args.per_level}")
    rows = []
    for lv in levels:
        row = run_level(url, wav_bytes, fields, lv, max(args.per_level, lv))
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    print("\n=== summary ===")
    hdr = ["concurrency", "ok", "errors", "throughput_rps", "wall_p50_s", "wall_p95_s"]
    extra = [k for k in ("gpu_wait_p95_ms", "cpu_wait_p95_ms", "cpu_compute_p50_ms", "gpu_compute_p50_ms")
             if any(k in r for r in rows)]
    print(" | ".join(f"{h:>18}" for h in hdr + extra))
    for r in rows:
        print(" | ".join(f"{str(r.get(h, '-')):>18}" for h in hdr + extra))


if __name__ == "__main__":
    main()
