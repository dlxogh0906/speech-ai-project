#!/usr/bin/env python3
"""Send the same audio to the CPU (8000) and GPU (8001) demo instances and compare.

Usage:  python compare.py <audio.wav> [more.wav ...]
"""
import sys, time, json, urllib.request, mimetypes

ENDPOINTS = {"CPU :8000": "http://localhost:8000/predict",
             "GPU :8001": "http://localhost:8001/predict"}


def post_audio(url, path):
    with open(path, "rb") as f:
        data = f.read()
    boundary = "----compareboundary"
    ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="audio"; filename="{path.split("/")[-1]}"\r\n'.encode(),
        f"Content-Type: {ctype}\r\n\r\n".encode(),
        data, b"\r\n", f"--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    out["_wall_sec"] = round(time.time() - t0, 2)
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    for path in sys.argv[1:]:
        print(f"\n=== {path} ===")
        res = {}
        for name, url in ENDPOINTS.items():
            try:
                res[name] = post_audio(url, path)
            except Exception as e:
                print(f"  {name}: ERROR {e}"); res[name] = None
        # side-by-side
        print(f"  {'':10} {'device':7} {'서버추론':>8} {'왕복(체감)':>10}  top-1")
        for name, r in res.items():
            if not r:
                continue
            t1 = r["top"][0]
            print(f"  {name:10} {r['device']:7} {r['took_sec']:>7.2f}s {r['_wall_sec']:>9.2f}s  "
                  f"{t1['flag']} {t1['nationality']} ({t1['prob']*100:.1f}%)")
        # speedup + agreement
        c, g = res.get("CPU :8000"), res.get("GPU :8001")
        if c and g:
            sp = c["took_sec"] / max(g["took_sec"], 1e-9)
            agree = [x["label"] for x in c["top"]] == [x["label"] for x in g["top"]]
            print(f"  -> GPU가 서버추론 기준 {sp:.1f}배 빠름 | top-5 동일: {'예' if agree else '아니오'}")


if __name__ == "__main__":
    main()
