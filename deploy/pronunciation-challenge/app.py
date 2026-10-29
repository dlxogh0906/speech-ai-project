# -*- coding: utf-8 -*-
"""발음 챌린지 데모 — 단어가 뜨면 읽고, 말이 끝나면 바로 PERFECT / GOOD / MISS.

accent.beavertalk.im 발음 챌린지(30초 라운드 · 약 2초에 단어 하나)를 CTC 모델로 바꿨을 때를 체감하는 데모.
판정 로직은 라이브러리(`mdd.word_judge`)에 있고 이 파일은 페이지·단어 목록·HTTP 만 맡는다.
발음평가 API(8932)와 같은 모델을 쓰지만 **별도 프로세스**다 — 그 서버는 건드리지 않는다.

- 단어 세트: 학습 단어(모델이 학습에서 본 단어 읽기 문항) · 처음 보는 단어 · 직접 입력.
  정확도가 둘 사이에서 크게 다르다(results/20 §4) — 그 차이를 직접 들어 보라고 나눴다.
- 장치: GPU / CPU 를 둘 다 올려 두고 페이지에서 고른다(속도 체감용). NPU 서버에서는 `--devices npu,cpu --rbln <.rbln>`.
- 말끝 감지는 브라우저가 한다(에너지 VAD). 서버에는 16 kHz mono WAV 만 온다 — ffmpeg 을 거치지 않는다.
- 녹음은 저장하지 않는다.

    source tools/env.sh
    export PYTHONPATH=/data/speech-ai/tools/hf_py38${PYTHONPATH:+:$PYTHONPATH}     # transformers(HF) 경로
    OMP_NUM_THREADS=4 $PY deploy/pronunciation-challenge/app.py \\
        --hf-ckpt $SPEECH_AI_DATA/checkpoints/w2vbert_stage2/step_38000 --port 8946
    → http://127.0.0.1:8946/  (마이크는 localhost 나 HTTPS 에서만 열린다)

    # NPU 서버 — 학습 문항 목록은 매니페스트 대신 옮겨 온 JSON 으로(results/21)
    OMP_NUM_THREADS=4 python deploy/pronunciation-challenge/app.py --hf-ckpt <model 폴더> \\
        --devices npu,cpu --rbln <w2vbert_ctc_f192_448_1024.rbln> --rbln-device <장치> \\
        --prompts-json <train_prompts.json> --port <빈 포트>
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import argparse, io, json, os, time, wave

import numpy as np

SR = 16000
HERE = _pl.Path(__file__).parent
# 학습 문항(단어·문장 어디에도) 에 부분 문자열로도 안 들어 있는 흔한 단어. 서버 시작 때 다시 확인해 태그를 붙인다.
NEW_WORDS = ["바나나", "고양이", "강아지", "컴퓨터", "비행기", "도서관", "냉장고", "자전거", "우산", "거울",
             "책상", "모자", "신발", "바다", "하늘", "구름", "기차", "우체국", "공원", "떡볶이",
             "불고기", "딸기", "수박", "포도", "토끼", "호랑이", "코끼리", "선생님", "학생", "커피",
             "아침", "저녁", "생일", "축구", "노래", "학교", "가방", "연필", "지갑", "칫솔",
             "목도리", "장갑", "냄비", "숟가락", "햇빛", "꽃병", "낙엽", "겨울", "닭갈비", "삼겹살"]
J = None
P = {"rw": [], "all": []}


def tag(word):
    w = word.replace(" ", "")
    if word in P["rw"]:
        return "seen"
    if any(w in s.replace(" ", "") for s in P["all"]):
        return "partial"                           # 다른 문항의 일부로만 들어 있음
    return "new"


def read_wav(raw):
    """16 kHz mono 16bit WAV 만 받는다(브라우저가 만들어 보낸다)."""
    try:
        w = wave.open(io.BytesIO(raw), "rb")
    except (wave.Error, EOFError) as e:
        raise ValueError("WAV 가 아니다: %s" % e)
    with w:
        if w.getframerate() != SR or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("16 kHz mono 16bit WAV 만 받는다")
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


def make_app():
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import FileResponse, JSONResponse

    app = FastAPI(title="발음 챌린지 데모")

    @app.get("/")
    def root():
        return FileResponse(str(HERE / "static" / "index.html"), headers={"Cache-Control": "no-store"})

    @app.get("/config")
    def config():
        new = [w for w in NEW_WORDS if tag(w) == "new"]
        return {"sets": {"seen": P["rw"], "new": new}, "devices": sorted(J.decs), "default_device": J.default}

    @app.get("/tags")
    def tags(w: str = ""):
        words = [x.strip() for x in w.split(",") if x.strip()][:200]
        return [{"word": x, "tag": tag(x), "ok": J.check_word(x)} for x in words]

    @app.get("/health")
    def health():
        return {"ok": True, "devices": sorted(J.decs)}

    # 동기 핸들러 — FastAPI 가 스레드풀에서 돌린다. 장치별 잠금은 WordJudge 가 건다(GPU·CPU 요청은 서로 안 막는다).
    @app.post("/judge")
    def judge(audio: UploadFile = File(...), word: str = Form(...), device: str = Form("")):
        t0 = time.perf_counter()
        try:
            out = J.judge(read_wav(audio.file.read()), word, device or None)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        out["tag"] = tag(word)
        out["server_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
        return out

    return app


def main():
    global J
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-ckpt", dest="hf_ckpt", required=True, help="w2v-BERT 체크포인트 폴더(서비스 모델 step_38000)")
    ap.add_argument("--devices", default="cuda,cpu", help="올릴 장치(쉼표): cuda · cpu · npu. GPU 가 없으면 cuda 는 건너뛴다")
    ap.add_argument("--rbln", default=None, help="npu 용 컴파일 모델(.rbln). 버킷은 파일 이름에서 읽는다")
    ap.add_argument("--rbln-device", dest="rbln_device", type=int, default=0, help="NPU 장치 번호")
    ap.add_argument("--prompts-json", dest="prompts_json", default=None,
                    help="학습 문항 목록 JSON({rw, all}). 주면 --manifest 대신 이걸 읽는다(매니페스트 없는 서버)")
    ap.add_argument("--manifest", default=os.path.join(str(MDD_OUTPUTS), "manifest_holdout_v1"),
                    help="학습 문항 목록을 읽을 매니페스트(메타의 prompt·task)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8946)
    a = ap.parse_args()
    import torch
    print("[threads] torch %d (OMP_NUM_THREADS=%s)" % (torch.get_num_threads(), os.environ.get("OMP_NUM_THREADS")), flush=True)
    from mdd.word_judge import WordJudge
    J = WordJudge(a.hf_ckpt, devices=tuple(x.strip() for x in a.devices.split(",") if x.strip()),
                  rbln=a.rbln, rbln_device=a.rbln_device)
    J.warmup()
    from mdd.word_judge import load_prompts_json, training_prompts
    P["rw"], P["all"] = load_prompts_json(a.prompts_json) if a.prompts_json else training_prompts(a.manifest)
    print("[load] 장치 %s · 학습 단어 %d · 학습 문항 전체 %d" % (sorted(J.decs), len(P["rw"]), len(P["all"])), flush=True)
    print("[serve] http://%s:%d/" % (a.host, a.port), flush=True)
    import uvicorn
    uvicorn.run(make_app(), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
