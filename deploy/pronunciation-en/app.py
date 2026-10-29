# -*- coding: utf-8 -*-
"""영어 발음 평가 API — 한국어 API(`sent.eval.kr`)와 **요청·응답 모양이 같다**. 앱은 주소와 문장만 바꾸면 된다.

    POST /sent.eval          영어 문장 채점 (별칭 /sent.eval.promax · /sent.eval.en)
                             요청: ① SpeechSuper 방식 — multipart 에 오디오 + JSON 덩어리(connect/start; 필드 이름 무관)
                                   ② 간편 방식 — audio + refText + tokenId
    GET  /health
    GET  /                   브라우저 확인용 화면(녹음 → 결과). --api-only 면 끈다

로직은 전부 라이브러리(`mdd_en`)에 있다 — 이 파일은 HTTP 로 내보내기만 한다. 규격은 `API.md`.

    source tools/env.sh
    NLTK_DATA=/data/speech-ai/tools/nltk_data \\
    PYTHONPATH=/data/speech-ai/tools/hf_py38:/data/speech-ai/tools/en_py38 HF_HOME=... HF_HUB_OFFLINE=1 \\
    $PY deploy/pronunciation-en/app.py --port 8952

모델(기본 synth-d = lv60-espeak 을 AIHub 037 + 한국인 오류 합성 + 자가 학습으로 추가 학습한 것, mdd_en.models)과
점수기($MDD_OUTPUTS/english/scorer/<모델>)는 `scripts/eval/en_phone_bakeoff.py` → `scripts/score/en_fit_phone_scorer.py`
→ `en_fit_sentence_scorer.py` 로 만든다. 학습 과정은 results/23 §9·§10.
⚠️ 사내 확인용. `--token` 을 주면 열쇠가 필요하다(헤더 Authorization: Bearer · X-API-Key · 폼 token · ?token=).
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import FFMPEG, MDD_OUTPUTS

import argparse, os, secrets, time

HERE = _pl.Path(__file__).parent
S = {}
API_TOKEN = None
SAVE_DIR = None           # --save-dir 로 켠다. 동의한 사내 녹음만 — SpeechSuper 비교·한국인 재보정에 쓴다
MAX_SEC = 30.0            # 이보다 긴 녹음은 앞 30초만 채점한다(SpeechSuper 영어 문장 상한은 90초)
MIN_VOICED_SEC = 0.3      # 영어는 한 단어 문장(Hello)도 받으므로 한국어(1.0초)보다 낮게
MIN_VOICED_RATIO = 0.10


def check_token(request, form=None):
    """열쇠 확인(한국어 API 와 같다: Bearer 생략·앞뒤 공백 허용). 바이트로 비교해 한글 등이 섞여도 500 이 아니라 401."""
    if not API_TOKEN:
        return True
    h = request.headers.get("authorization", "").strip()
    cands = [h[7:].strip() if h.lower().startswith("bearer ") else h, request.headers.get("x-api-key", ""),
             request.query_params.get("token", ""), (form or {}).get("token", "") if form else ""]
    want = API_TOKEN.encode("utf-8")
    return any(isinstance(c, str) and c and secrets.compare_digest(c.encode("utf-8"), want) for c in cands)


def make_app(api_only=False):
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, JSONResponse
    from starlette.concurrency import run_in_threadpool
    from mdd_en.audio import SR, decode_audio, parse_request_params, save_take, trim_edges_offset, voiced_info
    from mdd_en.engine import AlignError, NoWords, TextTooLong
    from mdd_en.response import build_response

    app = FastAPI(title="영어 발음 평가")

    @app.get("/health")
    def health():
        return {"ok": True, "model": S["model"], "device": S["eng"].rec.device}

    if not api_only:
        @app.get("/")
        def root():
            return FileResponse(str(HERE / "static" / "index.html"), headers={"Cache-Control": "no-store"})

    async def sent_eval(request: Request):
        form = await request.form()
        if not check_token(request, form):
            return JSONResponse({"error": "열쇠가 없거나 틀렸다", "code": "unauthorized"}, status_code=401)
        a0 = form.get("audio")                            # 한국어 API 처럼 'audio' 필드를 먼저, 없으면 첫 파일
        audio = a0 if hasattr(a0, "read") else next((v for v in form.values() if hasattr(v, "read")), None)
        if audio is None:
            return JSONResponse({"error": "오디오 파일이 없다", "code": "no_audio"}, status_code=400)
        P = parse_request_params(dict((k, v) for k, v in form.items() if isinstance(v, str)))
        t0 = time.time()
        try:
            wav = await run_in_threadpool(decode_audio, await audio.read(), FFMPEG)
        except ValueError as e:
            return JSONResponse({"error": "오디오를 해석하지 못했다: %s" % e, "code": "decode_failed"}, status_code=400)
        if len(wav) < 0.3 * SR:
            return JSONResponse({"error": "녹음이 너무 짧다 (0.3초 미만)", "code": "too_short"}, status_code=400)
        if not P["refText"].strip():
            return JSONResponse({"error": "refText 가 비어 있다", "code": "no_reftext"}, status_code=400)
        wav = wav[: int(MAX_SEC * SR)]
        tw, off = trim_edges_offset(wav)
        vi = voiced_info(tw)                              # 앞뒤 무음을 자른 뒤에 잰다 — 늦게 끈 녹음을 거절하지 않게
        if vi["voiced_sec"] < MIN_VOICED_SEC or vi["voiced_ratio"] < MIN_VOICED_RATIO:
            return JSONResponse({"error": "말소리가 거의 없다 (말한 시간 %.1f초 · 소리 비율 %d%%) — 마이크에 가까이서 다시 녹음할 것"
                                          % (vi["voiced_sec"], vi["voiced_ratio"] * 100),
                                 "code": "low_speech", "audio": vi}, status_code=400)
        try:
            raw = await run_in_threadpool(S["eng"].assess, tw, P["refText"])
        except NoWords:
            return JSONResponse({"error": "refText 에 읽을 영어 단어가 없다", "code": "no_reftext"}, status_code=400)
        except TextTooLong as e:
            return JSONResponse({"error": "refText 가 너무 길다 (%s) — 문장을 나눠 보낼 것" % e, "code": "reftext_too_long"},
                                status_code=400)
        except AlignError:
            return JSONResponse({"error": "녹음이 refText 에 비해 너무 짧다 — 문장을 끝까지 읽었는지 확인할 것",
                                 "code": "too_short", "audio": vi}, status_code=400)
        except RuntimeError as e:                         # GPU 메모리 부족 등 — 맨 500 대신 JSON 으로
            return JSONResponse({"error": "채점 중 서버 오류: %s" % str(e)[:200], "code": "server_busy"}, status_code=503)
        body = build_response(raw, P["refText"], record_id=P["tokenId"] or None, flag_rule=S["eng"].flag_rule,
                              offset_sec=off)
        body["_meta"].update(userId=P["userId"], model=S["model"], ms=int((time.time() - t0) * 1000), audio=vi)
        if SAVE_DIR:
            r = body["result"]
            save_take(SAVE_DIR, wav, P["refText"], {"userId": P["userId"], "scores": dict(
                (k, r[k]) for k in ("overall", "pronunciation", "fluency", "rhythm", "integrity"))})
        return JSONResponse(body)

    for path in ("/sent.eval", "/sent.eval.promax", "/sent.eval.en"):
        app.add_api_route(path, sent_eval, methods=["POST"])
    return app


def main():
    global API_TOKEN, SAVE_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="synth-d",
                    help="mdd_en.models 의 이름. 기본 synth-d(사람 전사에 가장 가깝다) · synth-a(대표 오류를 더 많이 잡는다) · lv60-espeak(공개 모델 그대로)")
    ap.add_argument("--scorer-dir", dest="scorer_dir", default=None,
                    help="점수기 폴더(기본: $MDD_OUTPUTS/english/scorer/<모델>)")
    ap.add_argument("--device", default=None, help="cuda · cpu (기본: GPU 가 있으면 GPU)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--token", default=os.environ.get("PRON_EN_TOKEN"), help="API 열쇠(기본: 환경변수 PRON_EN_TOKEN)")
    ap.add_argument("--api-only", dest="api_only", action="store_true", help="확인용 화면을 끈다")
    ap.add_argument("--display-scale", dest="display_scale", choices=("human", "gentle"), default="human",
                    help="한국인 기준 문장 점수(발음·운율)의 표시 눈금. human = 사람 점수 5점 만점 ×20(사람 3·4점 준 발화 평균 66 · 79) · "
                         "gentle = 40 + 0.6×human(같은 발화 평균 80 · 87, 앱 표시용). 순서·판별력은 같다")
    ap.add_argument("--save-dir", dest="save_dir", default=None,
                    help="받은 녹음을 16k WAV + items.jsonl 로 남긴다(동의한 사내 녹음만)")
    a = ap.parse_args()
    API_TOKEN = a.token or None
    SAVE_DIR = a.save_dir
    if a.host != "127.0.0.1" and not API_TOKEN:
        raise SystemExit("127.0.0.1 밖으로 열 때는 --token 을 줄 것")
    from mdd_en.engine import EnglishAssessor
    from mdd_en.models import resolve
    hf, kind, vocab = resolve(a.model)
    sd = a.scorer_dir or os.path.join(str(MDD_OUTPUTS), "english", "scorer", a.model)
    eng = EnglishAssessor(hf, kind, sd, vocab_dir=vocab, device=a.device, display_scale=a.display_scale)
    eng.warmup()
    S.update(eng=eng, model=a.model)
    print("[load] %s (%s) · 점수기 %s · 지적 문턱 %.1f · 문장 점수 %s · 표시 눈금 %s · 장치 %s · 열쇠 %s"
          % (a.model, hf, sd, eng.flag_T, ",".join(sorted(eng.sent)) or "없음", a.display_scale, eng.rec.device,
             "켬" if API_TOKEN else "끔"), flush=True)
    import uvicorn
    uvicorn.run(make_app(a.api_only), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
