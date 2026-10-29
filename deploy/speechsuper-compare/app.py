# -*- coding: utf-8 -*-
"""SpeechSuper ↔ 자체 엔진 나란히 비교 — **누가 들린 대로 적는가**를 보는 데모.

같은 녹음 한 개를 두 API 에 동시에 보내고, 자리마다 이렇게 늘어놓는다.
    표준발음        문장대로 읽었을 때 나야 할 소리 (우리 KoG2P)
    우리 전사       모델이 들린 대로 적은 소리 (actual_jamo · 못 들은 자리는 ∅)
    SpeechSuper    그쪽이 돌려준 음소 표기 (alpha)

모델을 올리지 않는다 — 자체 API(:8932)와 SpeechSuper 를 호출만 한다.
키는 파일에서 읽는다(`source ~/.speechsuper_keys`). 응답 저장 시 인증 항목은 지운다.

    source ~/.speechsuper_keys
    $PY deploy/speechsuper-compare/app.py --port 8934

⚠️ 사내 확인용. SpeechSuper 는 체험 키(동시 5건)라 부하 테스트에 쓰지 말 것.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, hashlib, json, os, time, urllib.error, urllib.request, uuid

SS_URL = "https://api.speechsuper.com/sent.eval.kr"
SENTENCES = ["안녕하세요", "값이 너무 비싸서 못 샀어요", "학교에서 친구를 만났어요",
             "꽃이 피었습니다", "독립문 앞에서 사진을 찍었어요", "신라면을 끓여 먹었어요",
             "밥을 먹고 커피를 마셨습니다", "무릎을 다쳐서 병원에 갔어요"]


def post(url, params, audio, token=None, timeout=45):
    """multipart 로 text(JSON) + audio 를 보낸다. 두 API 가 같은 형식을 받는다."""
    b = uuid.uuid4().hex
    body = ("--" + b + "\r\nContent-Disposition: form-data; name=\"text\"\r\n\r\n"
            + json.dumps(params) + "\r\n--" + b
            + "\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"rec.wav\""
            + "\r\nContent-Type: audio/wav\r\n\r\n").encode()
    body += audio + ("\r\n--" + b + "--\r\n").encode()
    h = {"Content-Type": "multipart/form-data; boundary=" + b, "Request-Index": "0"}
    if token:
        h["Authorization"] = "Bearer " + token
    t0 = time.monotonic()
    try:
        try:
            res = urllib.request.urlopen(urllib.request.Request(url, data=body, headers=h), timeout=timeout)
        except urllib.error.HTTPError as e:
            res = e
        with res:
            status, raw = res.status, res.read()
        data = json.loads(raw)
        for k in ("params", "applicationId"):       # 인증 항목은 남기지 않는다
            data.pop(k, None)
    except (OSError, ValueError) as e:
        status, data = None, {"error": type(e).__name__}
    return {"status": status, "ms": int((time.monotonic() - t0) * 1000), "body": data}


def ss_params(text, rid):
    app, secret = os.environ["SPEECHSUPER_APP_KEY"], os.environ["SPEECHSUPER_SECRET_KEY"]
    ts, user = str(int(time.time())), "compare-demo"
    sig = lambda v: hashlib.sha1(v.encode()).hexdigest()
    return {"connect": {"cmd": "connect", "param": {
                "sdk": {"version": 16777472, "source": 9, "protocol": 2},
                "app": {"applicationId": app, "timestamp": ts, "sig": sig(app + ts + secret)}}},
            "start": {"cmd": "start", "param": {
                "app": {"applicationId": app, "timestamp": ts, "userId": user,
                        "sig": sig(app + ts + user + secret)},
                "audio": {"audioType": "wav", "channel": 1, "sampleBytes": 2, "sampleRate": 16000},
                "request": {"coreType": "sent.eval.kr", "refText": text, "tokenId": rid}}}}


def ours_rows(body):
    """자체 응답 → 글자별 [(표준, 들린 대로)] 와 점수."""
    r = (body or {}).get("result") or {}
    words = []
    for w in r.get("words", []):
        ph = [(p.get("alpha_jamo"), p.get("actual_jamo"), p.get("pronunciation")) for p in w.get("phonemes", [])]
        words.append({"word": w.get("word"), "score": w.get("score"), "readType": w.get("readType"), "ph": ph})
    return {"words": words, "scores": {k: r.get(k) for k in ("overall", "pronunciation", "fluency", "comprehensibility")}}


def ss_rows(body):
    """SpeechSuper 응답 → 글자별 [(표기, 점수)] 와 점수. 응답 판마다 필드가 조금씩 다르다."""
    r = (body or {}).get("result") or {}
    words = []
    for w in r.get("words", []):
        ph = [(p.get("alpha"), p.get("phoneme"), p.get("pronunciation")) for p in w.get("phonemes", [])]
        sc = w.get("scores") or {}
        words.append({"word": w.get("word"), "score": sc.get("pronunciation", sc.get("overall", w.get("score"))),
                      "readType": w.get("readType", w.get("charType")), "ph": ph})
    return {"words": words, "scores": {"overall": r.get("overall"), "pronunciation": r.get("pronunciation"),
                                       "fluency": r.get("fluency"), "rhythm": r.get("rhythm")}}


def make_app(static_dir, local_url, local_token):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
    from starlette.concurrency import run_in_threadpool
    app = FastAPI(title="speechsuper-compare")

    @app.get("/health")
    def health():
        return {"ok": True, "local": local_url,
                "speechsuper_key": bool(os.environ.get("SPEECHSUPER_APP_KEY"))}

    @app.get("/sentences")
    def sentences():
        return {"sentences": SENTENCES}

    @app.post("/compare")
    async def compare(request: Request):
        form = await request.form()
        up = form.get("audio")
        text = (form.get("refText") or "").strip()
        if up is None or not text:
            return JSONResponse({"error": "audio 와 refText 가 필요하다"}, status_code=400)
        audio = await up.read()
        rid = "cmp-%d" % int(time.time() * 1000)
        ours = await run_in_threadpool(
            post, local_url, {"start": {"param": {"request": {"refText": text, "tokenId": rid}}}}, audio, local_token)
        if not os.environ.get("SPEECHSUPER_APP_KEY"):
            ss = {"status": None, "ms": 0, "body": {"error": "키가 설정되지 않았다 — source ~/.speechsuper_keys"}}
        else:
            ss = await run_in_threadpool(post, SS_URL, ss_params(text, rid), audio)
        return {"refText": text,
                "ours": {"status": ours["status"], "ms": ours["ms"], **ours_rows(ours["body"]),
                         "error": ours["body"].get("error"), "code": ours["body"].get("code")},
                "speechsuper": {"status": ss["status"], "ms": ss["ms"], **ss_rows(ss["body"]),
                                "error": ss["body"].get("error")}}

    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local-url", dest="local_url", default="http://127.0.0.1:8932/sent.eval.kr")
    ap.add_argument("--local-token", dest="local_token", default=os.environ.get("MDD_API_TOKEN"))
    ap.add_argument("--static", default=str(_pl.Path(__file__).parent / "static"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8934)
    a = ap.parse_args()
    print(f"[serve] http://{a.host}:{a.port} · 자체 {a.local_url} · "
          f"SpeechSuper 키 {'있음' if os.environ.get('SPEECHSUPER_APP_KEY') else '없음'}", flush=True)
    import uvicorn
    uvicorn.run(make_app(a.static, a.local_url, a.local_token), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
