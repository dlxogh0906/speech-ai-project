# -*- coding: utf-8 -*-
"""국적 모델 전부 비교 — 문장 하나를 읽으면 모든 국적 모델의 판정을 나란히 보여 준다 (2026-10-27).

모델 목록은 models.json (서비스 구성 · 화자 분할 8:1:1 의 전체학습 · LoRA 계열 · 9월 KD · 9월 2B).
torch 계열은 GPU(bf16)에서 차례로, ONNX 계열은 CPU 스레드에서 동시에 돈다(country_classifier.model_zoo).
입력은 NPU 국적 API 의 /predict 와 같다: ffmpeg 로 앞 12초 → 0.5초 미만 too_short · 말소리 1초 미만 no_speech →
무음 정규화(선택, 기본 켬) → 앞 12초. 서비스 행은 스위치와 상관없이 NPU 설정(주 모델 정규화 · 게이트 원본)대로 계산한다.

  GET  /          데모 화면                    GET /health   로딩 진행 · 모델 목록 · 누적 요청 수
  POST /predict   audio [+ norm 1|0 · sentence · session] → 모델별 top-5 · Korea 확률 · 시간 · test 수치

토큰은 `X-API-Key` 또는 `Authorization: Bearer`. 화면과 /health 는 토큰 없이 열린다.
Funnel(`--set-path /nat-all`)은 프리픽스를 벗겨 넘기지만 / 와 /nat-all 양쪽에 마운트한다.
SPEECH_CAPTURE_DIR 이 있으면 모든 요청의 업로드 원본과 응답을 남긴다(shared/request_capture.py).
실행은 deploy/nationality-all-demo/start.sh, 운영 메모는 README.md.
"""
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import FFMPEG, NAT_OUTPUTS
from shared.request_capture import capture_app, capture_context

import argparse, datetime, hmac, importlib.util, json, os, shutil, subprocess, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 16000
MAX_BYTES = 8 * 1024 * 1024
MAX_SEC = 12                       # NPU /predict 와 같다 — 앞 12초만 본다(평가도 앞 12초)
GATE_RMS = 1.5e-3                  # 말소리 프레임 기준 — NPU 국적 API 와 같은 값
MIN_VOICED = 1.0                   # NPU 의 KD_GATE_MIN_SEC
FFMPEG_PATH = shutil.which(FFMPEG) or FFMPEG
FFMPEG_ENV = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}   # conda 함정(ENVIRONMENT.md 함정 10)

API_KEY = ""
REG = {}                           # models.json
ONNX = {}                          # id → OnnxKD
ZOO = None                         # TorchZoo
TEST = {}                          # id → {"t1", "t5", "n"} (화자 분할 8:1:1 test 웹마이크 문장 발화)
STATE = {"loading": True, "loaded": 0, "total": 0, "error": "", "failed": {}, "load_sec": None}
STATS = dict(requests=0, errors=0, started=datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
POOL = ThreadPoolExecutor(max_workers=int(os.environ.get("NAT_ALL_ONNX_WORKERS", "8")))


class AudioError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def decode(raw, max_sec=MAX_SEC):
    # m4a 는 목차(moov)가 파일 끝에 있을 수 있어 파이프가 아니라 임시 파일로 넘긴다(국적 API 와 같은 이유).
    with tempfile.NamedTemporaryFile(suffix=".audio") as tf:
        tf.write(raw)
        tf.flush()
        p = subprocess.run([FFMPEG_PATH, "-hide_banner", "-loglevel", "error", "-i", tf.name,
                            "-f", "f32le", "-ac", "1", "-ar", str(SR), "-t", str(max_sec), "pipe:1"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=False, env=FFMPEG_ENV)
    if p.returncode != 0:
        err = p.stderr.decode("utf-8", "replace").strip()
        if p.returncode in (126, 127) or "error while loading shared libraries" in err or "symbol lookup error" in err:
            raise RuntimeError("ffmpeg 실행 실패 (rc=%d): %s" % (p.returncode, err[:500]))
        raise AudioError("decode_failed", "오디오 파일을 해석할 수 없다")
    return np.frombuffer(p.stdout, np.float32).copy()


def voiced_sec(wav, frame_sec=0.03):
    f = int(SR * frame_sec)
    n = wav.size // f
    if n == 0:
        return 0.0
    rms = np.sqrt(np.square(wav[: n * f].reshape(n, f), dtype=np.float64).mean(axis=1))
    return float((rms >= GATE_RMS).sum() * frame_sec)


def resolve(path):
    return path.format(NAT=str(NAT_OUTPUTS), **os.environ) if path else path


def pros_track_fn():
    """운율 궤적 모델용 — 학습 때 궤적을 만든 스크립트(prosody_tracks.track)를 그대로 쓴다. 파일 경로를 받으므로 임시 wav 로 넘긴다."""
    path = os.environ.get("NAT_PROSODY_TRACKS_PY") or str(NAT_OUTPUTS / "prosody_20261023" / "_scripts" / "prosody_tracks.py")
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location("prosody_tracks", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import soundfile as sf

    def fn(wav):
        with tempfile.NamedTemporaryFile(suffix=".wav") as tf:
            sf.write(tf.name, wav, SR, subtype="FLOAT")
            _, tr, _ = mod.track(tf.name)
        if tr is None:
            raise RuntimeError("운율 궤적 계산 실패")
        return tr.astype(np.float32)
    return fn


def load_all(reg_path):
    """models.json 순서대로 올린다. 실패한 모델은 건너뛰고 /health 에 남긴다."""
    global ZOO
    t0 = time.time()
    try:
        from country_classifier.model_zoo import OnnxKD, TorchZoo
        models = REG["models"]
        STATE["total"] = len(models)
        for m in models:                                  # test 수치 — 화면에 보일 웹마이크 문장 발화 top-1 · top-5
            ev = resolve(m.get("eval", ""))
            if ev and os.path.isfile(os.path.join(ev, "report.json")):
                s = json.load(open(os.path.join(ev, "report.json"), encoding="utf-8"))["results"].get("web_sent")
                if s:
                    TEST[m["id"]] = {"t1": s["top1"], "t5": s["top5"], "n": s["n"]}
        threads = int(os.environ.get("NAT_ALL_ONNX_THREADS", "4"))
        for m in models:
            if m["kind"] == "onnx":
                try:
                    ONNX[m["id"]] = OnnxKD(resolve(m["path"]), threads)
                except Exception as e:
                    STATE["failed"][m["id"]] = repr(e)[:300]
                STATE["loaded"] += 1
        ZOO = TorchZoo(pros_fn=pros_track_fn())
        for m in models:
            if m["kind"] not in ("torch", "2b"):
                continue
            t1 = time.time()
            try:
                if m["kind"] == "torch":
                    ZOO.add_ckpt(m["id"], resolve(m["path"]))
                else:
                    ZOO.add_2b_partial(m["id"], resolve(m["path"]), ONNX[m["labels_from"]].labels if not m["path"] else None)
            except Exception as e:
                STATE["failed"][m["id"]] = repr(e)[:300]
            STATE["loaded"] += 1
            print("[load] %s %.0fs" % (m["id"], time.time() - t1), flush=True)
        ZOO.drop_templates()
        for m in models:
            if m["kind"] == "service":
                if m["main"] not in ONNX or m["gate"] not in ONNX:
                    STATE["failed"][m["id"]] = "주 모델 또는 게이트 모델을 못 올렸다"
                STATE["loaded"] += 1
        # 첫 요청이 느리지 않게 한 번 돌린다(cuDNN 초기화)
        warm = (1e-2 * np.random.default_rng(0).standard_normal(3 * SR)).astype(np.float32)
        run_models(warm, True)
        STATE["load_sec"] = round(time.time() - t0)
        import torch
        gpu = torch.cuda.memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0
        print("[ready] 모델 %d개 (실패 %d) · 인코더 %d개 · GPU %.1f GB · %ds" % (
            STATE["total"] - len(STATE["failed"]), len(STATE["failed"]), len(ZOO.groups), gpu, STATE["load_sec"]), flush=True)
        for k, v in STATE["failed"].items():
            print("[fail] %s %s" % (k, v), flush=True)
    except Exception as e:
        STATE["error"] = repr(e)[:500]
        print("[load] 실패 %r" % e, flush=True)
    finally:
        STATE["loading"] = False


def run_models(wav, norm):
    """디코드한 파형(앞 12초) → 모든 모델 {id: (확률, ms[, 서비스 정보])}. 모델 입력은 norm 이면 무음 정규화 → 앞 12초.
    ONNX 는 스레드 풀에서, torch 는 이 스레드에서 동시에 돈다. 서비스는 늘 NPU 설정대로(주 모델 정규화 · 게이트 원본)이고,
    같은 (모델, 입력) 계산은 한 번만 한다 — 정규화를 켜면 run_zeroth 행이 서비스 주 모델과, 끄면 기존 배포 행이 게이트와 같다."""
    from country_classifier.model_zoo import softmax
    from country_classifier.wav_augment import silence_normalize
    svc = [m for m in REG["models"] if m["kind"] == "service" and m["id"] not in STATE["failed"]]
    inputs = {"raw": wav[: MAX_SEC * SR]}
    if norm or svc:
        inputs["norm"] = silence_normalize(wav, GATE_RMS)[: MAX_SEC * SR]
    tag = "norm" if norm else "raw"

    def timed(model, w):
        t = time.perf_counter()
        z = model.logits(w)
        return softmax(z), (time.perf_counter() - t) * 1000
    want = {(mid, tag) for mid in ONNX}
    for m in svc:
        want |= {(m["main"], "norm"), (m["gate"], "raw")}
    jobs = {k: POOL.submit(timed, ONNX[k[0]], inputs[k[1]]) for k in sorted(want)}
    out = {}
    if ZOO is not None and ZOO.models:
        logits, ms = ZOO.logits(inputs[tag])
        for mid, z in logits.items():
            out[mid] = (softmax(z), ms[mid])
    for mid in ONNX:
        out[mid] = jobs[(mid, tag)].result()
    for m in svc:
        out[m["id"]] = service_rule(m, jobs[(m["main"], "norm")].result(), jobs[(m["gate"], "raw")].result())
    return out


def service_rule(m, main, gate):
    """NPU 국적 API classify() 와 같은 규칙 — 게이트 모델의 Korea 확률이 τ 를 넘거나 주 모델 1위가 Korea 면
    Korea 를 더 확신하는 쪽 분포를 쓰고 Korea 를 1위로 둔다."""
    labels = ONNX[m["main"]].labels
    glab = ONNX[m["gate"]].labels
    p, ms = main
    pg = gate[0] if glab == labels else np.array([gate[0][glab.index(l)] for l in labels])
    ki = labels.index("Korea")
    gate_k = float(pg[ki])
    if gate_k > m["gate_tau"] or int(np.argmax(p)) == ki:
        if pg[ki] > p[ki]:
            p = pg
        if int(np.argmax(p)) != ki:
            p = p.copy()
            p[ki] = p.max() + 1e-6
            p = p / p.sum()
    top = int(np.argmax(p))
    extra = {"gate_korea": round(gate_k, 4),
             "low_confidence": bool(top != ki and float(p[top]) < m["lowconf"])}
    return p, max(ms, gate[1]), extra


def labels_of(mid):
    if mid in ONNX:
        return ONNX[mid].labels
    if ZOO is not None and mid in ZOO.models:
        return ZOO.models[mid]["labels"]
    m = next(x for x in REG["models"] if x["id"] == mid)
    return ONNX[m["main"]].labels


def make_app():
    from fastapi import APIRouter, FastAPI, File, Form, Request, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse
    from starlette.concurrency import run_in_threadpool

    app = FastAPI(title="nat-all-demo", docs_url=None, redoc_url=None, openapi_url=None)

    def fail(code, message, status=400):
        STATS["errors"] += 1
        return JSONResponse({"error": message, "code": code, "rid": (capture_context.get() or {}).get("rid")},
                            status_code=status)

    def authorized(request):
        if not API_KEY:
            return True
        given = request.headers.get("x-api-key", "")
        auth = request.headers.get("authorization", "")
        if not given and auth.lower().startswith("bearer "):
            given = auth[7:].strip()
        return hmac.compare_digest(given.encode("utf-8"), API_KEY.encode("utf-8"))

    r = APIRouter()

    @r.get("/", response_class=HTMLResponse)
    async def index():
        with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
            return f.read()

    @r.get("/health")
    async def health():
        return {"ok": not STATE["loading"] and not STATE["error"], **{k: STATE[k] for k in ("loading", "loaded", "total", "error", "load_sec")},
                "failed": sorted(STATE["failed"]), "groups": REG["groups"],
                "models": [{"id": m["id"], "name": m["name"], "group": m["group"], "kind": m["kind"], "note": m.get("note", ""),
                            "leak": bool(m.get("leak")), "test": TEST.get(m["id"])} for m in REG["models"]],
                "encoders_on_gpu": len(ZOO.groups) if ZOO else 0, "auth": bool(API_KEY),
                "capture": bool(os.environ.get("SPEECH_CAPTURE_DIR")), **STATS}

    # sentence · session 은 판정에 쓰지 않는다 — 캡처 기록(request.json)에 남기려고 받는다.
    @r.post("/predict")
    async def predict(request: Request, audio: UploadFile = File(...), norm: str = Form("1"),
                      sentence: str = Form(None), session: str = Form(None)):
        STATS["requests"] += 1
        t0 = time.perf_counter()
        if not authorized(request):
            return fail("unauthorized", "토큰이 없거나 맞지 않다", 401)
        if STATE["loading"] or STATE["error"]:
            return fail("loading", "모델을 올리는 중이다 (%d/%d)" % (STATE["loaded"], STATE["total"]) if STATE["loading"]
                        else "모델을 올리지 못했다", 503)
        raw = await audio.read()
        if len(raw) > MAX_BYTES:
            return fail("too_large", "파일이 너무 크다 (8 MB 초과)")
        try:
            wav = await run_in_threadpool(decode, raw)
        except AudioError as e:
            return fail(e.code, str(e))
        except RuntimeError as e:
            print("[decode] %s" % e, flush=True)
            return fail("server_decoder", "서버 오디오 디코더 오류", 500)
        if wav.size < SR // 2:
            return fail("too_short", "오디오가 너무 짧다 (0.5초 미만)")
        voiced = voiced_sec(wav)
        if voiced < MIN_VOICED:
            return fail("no_speech", "말소리가 1초 이상 들리지 않는다 (무음이거나 소리가 너무 작다)")
        use_norm = norm != "0"
        t1 = time.perf_counter()
        out = await run_in_threadpool(run_models, wav, use_norm)
        infer_ms = (time.perf_counter() - t1) * 1000
        results = []
        for m in REG["models"]:
            if m["id"] not in out:
                continue
            p, ms = out[m["id"]][:2]
            labels = labels_of(m["id"])
            top = np.argsort(-p)[:5]
            row = {"id": m["id"], "top5": [{"label": labels[i], "prob": round(float(p[i]), 4)} for i in top],
                   "korea": round(float(p[labels.index("Korea")]), 4) if "Korea" in labels else None, "ms": round(ms)}
            if len(out[m["id"]]) > 2:
                row.update(out[m["id"]][2])
            results.append(row)
        votes = Counter(x["top5"][0]["label"] for x in results)
        body = {"sec": round(wav.size / SR, 2), "voiced_sec": round(voiced, 1), "norm": use_norm, "n_models": len(results),
                "votes": [{"label": k, "n": v} for k, v in votes.most_common()], "results": results,
                "infer_ms": round(infer_ms), "total_ms": round((time.perf_counter() - t0) * 1000),
                "rid": (capture_context.get() or {}).get("rid")}
        print("[predict] n=%d norm=%d top=%s %dms" % (len(results), use_norm, votes.most_common(1)[0][0], body["total_ms"]), flush=True)
        return body

    app.include_router(r)
    app.include_router(r, prefix="/nat-all")
    return capture_app(app, "nat-all")


def main():
    global API_KEY, REG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=os.path.join(HERE, "models.json"))
    ap.add_argument("--only", default="", help="쉼표로 고른 id 만 올린다 (시험용)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8988)
    ap.add_argument("--no-auth", action="store_true", help="토큰 검사를 끈다 (로컬 시험용)")
    a = ap.parse_args()
    API_KEY = "" if a.no_auth else os.environ.get("NAT_ALL_KEY", "")
    if not API_KEY and not a.no_auth:
        raise SystemExit("NAT_ALL_KEY 가 없다 — 토큰 없이 띄우려면 --no-auth")
    p = subprocess.run([FFMPEG_PATH, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=False, env=FFMPEG_ENV)
    if p.returncode != 0:
        raise SystemExit("ffmpeg 을 실행할 수 없다 (%s)" % FFMPEG_PATH)
    REG = json.load(open(a.models, encoding="utf-8"))
    if a.only:
        keep = set(a.only.split(","))
        REG["models"] = [m for m in REG["models"] if m["id"] in keep]
    threading.Thread(target=load_all, args=(a.models,), daemon=True).start()
    print("[start] 모델 %d개 올리는 중 · capture=%s · port=%d" % (len(REG["models"]), os.environ.get("SPEECH_CAPTURE_DIR") or "off", a.port), flush=True)
    import uvicorn
    uvicorn.run(make_app(), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
