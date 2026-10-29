# -*- coding: utf-8 -*-
"""발음 평가 API **동시 요청 부하 측정** — 같은 녹음을 여러 클라이언트가 동시에 보내 지연·처리량을 잰다.

운영 서버(8932)에 쓰지 말 것 — 측정용 인스턴스를 다른 포트에 띄워서 잰다(CLAUDE.md: 포트 재사용 금지).
토큰은 환경변수 MDD_API_TOKEN 에서 읽어 헤더로만 보낸다(출력하지 않는다).

    $PY .../load_test_pronunciation_api.py --url http://127.0.0.1:8942/sent.eval.kr --wav a.wav --text 시계 \\
        --concurrency 1 4 8 16 --per-client 10

    # 발음 챌린지 단어 판정 서버(deploy/pronunciation-challenge) — 필드가 word · audio · device
    $PY .../load_test_pronunciation_api.py --mode judge --url http://127.0.0.1:8946/judge --device npu \\
        --wav word_2s.wav --text 소리
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, json, os, threading, time, urllib.request, uuid
import numpy as np


def post(url, text, audio, token, mode="sent", device=""):
    b = uuid.uuid4().hex
    fields = {"refText": text} if mode == "sent" else {"word": text, "device": device}
    head = "".join("--" + b + "\r\nContent-Disposition: form-data; name=\"" + k + "\"\r\n\r\n" + v + "\r\n"
                   for k, v in fields.items())
    body = (head + "--" + b + "\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"a.wav\"\r\n"
            "Content-Type: audio/wav\r\n\r\n").encode("utf-8") + audio + ("\r\n--" + b + "--\r\n").encode()
    h = {"Content-Type": "multipart/form-data; boundary=" + b}
    if token:
        h["Authorization"] = "Bearer " + token
    s = time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(url, data=body, headers=h), timeout=120) as r:
        d = json.loads(r.read())
    if mode == "judge":
        if "error" in d or d.get("result") == "NO_SPEECH":
            raise RuntimeError("판정 실패: %s" % (d.get("error") or "NO_SPEECH"))
        return time.perf_counter() - s, d.get("server_ms")
    return time.perf_counter() - s, (d.get("_meta") or {}).get("ms")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 8, 16])
    ap.add_argument("--per-client", dest="per_client", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", choices=["sent", "judge"], default="sent",
                    help="sent: 발음 API(refText) · judge: 챌린지 단어 판정 서버(word·device)")
    ap.add_argument("--device", default="", help="judge 모드에서 보낼 장치(cuda · cpu · npu). 비우면 서버 기본값")
    a = ap.parse_args()
    audio = open(a.wav, "rb").read()
    tok = os.environ.get("MDD_API_TOKEN")
    for _ in range(3):
        post(a.url, a.text, audio, tok, a.mode, a.device)    # 준비
    res = {}
    print("%4s | %9s %9s %9s | %10s | 서버 계산 중앙값" % ("동시", "p50 ms", "p95 ms", "최대 ms", "처리 건/초"))
    for c in a.concurrency:
        lat, srv = [], []
        lock = threading.Lock()
        def worker():
            for _ in range(a.per_client):
                t, m = post(a.url, a.text, audio, tok, a.mode, a.device)
                with lock:
                    lat.append(t); srv.append(m or 0)
        th = [threading.Thread(target=worker) for _ in range(c)]
        s = time.perf_counter()
        for t in th: t.start()
        for t in th: t.join()
        wall = time.perf_counter() - s
        r = {"p50_ms": round(1000 * float(np.median(lat)), 1), "p95_ms": round(1000 * float(np.percentile(lat, 95)), 1),
             "max_ms": round(1000 * max(lat), 1), "req_per_sec": round(len(lat) / wall, 2),
             "server_ms_median": float(np.median(srv)), "n": len(lat)}
        res[str(c)] = r
        print("%4d | %9.1f %9.1f %9.1f | %10.2f | %.0f ms" % (c, r["p50_ms"], r["p95_ms"], r["max_ms"], r["req_per_sec"], r["server_ms_median"]))
    if a.out:
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
