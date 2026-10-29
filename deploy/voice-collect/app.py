# -*- coding: utf-8 -*-
"""녹음 수집 페이지 — 팀원·지인이 핸드폰이나 노트북으로 문장을 읽어 올리면 화자 정보와 함께 서버에 저장한다.

목적: 국적 분류·발음 진단 모델의 **실제 기기(브라우저·폰 마이크) 검증 세트**. 학습 데이터가 스튜디오·앱 녹음뿐이라
노트북 마이크의 한국인 음성에서 run5 가 미국으로 뒤집힌 문제(2026-09-23)를 재현·수정하려면 이 채널의 데이터가 필요하다.

  GET  <prefix>/            부스 이벤트 화면 (= <prefix>/booth, 2026-10-01부터)
  GET  <prefix>/legacy      예전 수집 화면 (이름 · 모국어 · 국적 · 기기 입력 → 문장별 녹음 → 자동 업로드)
  GET  <prefix>/sentences   문장 목록 (sentences.json)
  POST <prefix>/upload      audio + 메타 → <out>/<speaker_id>/<item_id>.webm 원본 + .wav(16 kHz mono) + meta.jsonl 한 줄
  GET  <prefix>/status      수집 현황 (화자 수 · 발화 수)

부스 이벤트(2026-10-01 추가) — 외국인이 문장 10개를 읽으면 간식을 주는 행사용. 문장은 booth_sentences.json(5세트 × 10).
  GET  <prefix>/booth              참가 화면 (동의 → 국적·모국어·수준 → 10문장 녹음 → 참여 코드)
  GET  <prefix>/booth/staff?pin=   스태프 현황 (QR · 국가별 참여자 · 최근 완료). PIN 은 BOOTH_DIR/.staff_pin
  POST <prefix>/booth/api/start    동의·프로필 → 참여 코드 + 세트(가장 적게 쓰인 세트)
  POST <prefix>/booth/api/upload   code · item(1~10) · audio → BOOTH_DIR/<code>/<세트>_<문장>_<시도>.{webm|m4a} + .wav(16 kHz) + meta.jsonl
  GET  <prefix>/booth/api/check    code 의 완료 문장 수
  GET  <prefix>/booth/api/stats    현황 집계 (pin 필요)
저장 위치는 BOOTH_DIR(기본 /data/speech-ai/out/booth_collect). 참가자 목록은 BOOTH_DIR/participants.jsonl.

Funnel 경로(`tailscale funnel --set-path /collect ...`)는 실측(2026-10-01) 프리픽스를 벗겨서 넘긴다 — 그래도 둘 다 되도록 COLLECT_PREFIX(기본 /collect)와 / 양쪽에 마운트한다.
루트(/)로도 같은 앱을 서빙해 로컬 포워딩에서도 열린다. 저장 위치는 COLLECT_DIR(기본 /data/speech-ai/out/novelty/collect).

Run:  $PY deploy/voice-collect/app.py --port 8923      (tmux 안에서)
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, base64, collections, json, os, random, re, secrets, subprocess, tempfile, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("COLLECT_DIR", "/data/speech-ai/out/novelty/collect")
PREFIX = os.environ.get("COLLECT_PREFIX", "/collect")
SR = 16000
LOCK = threading.Lock()


def ffmpeg():
    """FFMPEG_BIN → PATH 의 ffmpeg → imageio. imageio 4.2.2 는 iPhone Safari MP4 를 첫 조각만 읽는다(2026-09-24 실측) — 시스템 ffmpeg(6.x)을 우선."""
    import shutil
    p = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
    if p:
        return p
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def decode_to_wav(raw, dst):
    """브라우저 녹음(webm/opus · m4a 등) → 16 kHz mono 16-bit wav. 반환 초. 실패하면 None."""
    with tempfile.NamedTemporaryFile(suffix=".audio") as tf:
        tf.write(raw); tf.flush()
        p = subprocess.run([ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", tf.name, "-ac", "1", "-ar", str(SR),
                            "-sample_fmt", "s16", dst], capture_output=True)
    if p.returncode != 0 or not os.path.exists(dst):
        return None
    return round((os.path.getsize(dst) - 44) / (SR * 2), 2)


def speaker_id(name):
    s = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", name.strip())[:32]
    return s or "anon"


# ── 부스 이벤트 ──
BOOTH_DIR = os.environ.get("BOOTH_DIR", "/data/speech-ai/out/booth_collect")
BOOTH_LOCK = threading.Lock()
BOOTH_PARTS = {}                       # code → 참가자 기록 (participants.jsonl 에서 복원)
CODE_CHARS = "ACDEFGHJKMNPQRTUVWXY3479"  # 0/O · 1/I/L · 2/Z · 5/S · 8/B 처럼 헷갈리는 글자 제외
COUNTRY_ALIAS = {
    "south korea": "Korea", "republic of korea": "Korea", "korea, republic of": "Korea", "대한민국": "Korea", "한국": "Korea",
    "usa": "United States", "us": "United States", "u.s.": "United States", "u.s.a.": "United States", "america": "United States",
    "united states of america": "United States", "미국": "United States", "uk": "United Kingdom", "england": "United Kingdom",
    "great britain": "United Kingdom", "britain": "United Kingdom", "scotland": "United Kingdom", "wales": "United Kingdom",
    "영국": "United Kingdom", "czechia": "Czech Republic", "체코": "Czech Republic", "viet nam": "Vietnam", "베트남": "Vietnam",
    "türkiye": "Turkey", "turkiye": "Turkey", "튀르키예": "Turkey", "터키": "Turkey", "russian federation": "Russia", "러시아": "Russia",
    "hong kong sar": "Hong Kong", "홍콩": "Hong Kong", "중국": "China", "prc": "China", "일본": "Japan", "필리핀": "Philippines",
    "인도네시아": "Indonesia", "태국": "Thailand", "말레이시아": "Malaysia", "인도": "India", "몽골": "Mongolia", "우즈베키스탄": "Uzbekistan",
    "카자흐스탄": "Kazakhstan", "네팔": "Nepal", "미얀마": "Myanmar", "burma": "Myanmar", "파키스탄": "Pakistan", "방글라데시": "Bangladesh",
    "캄보디아": "Cambodia", "키르기스스탄": "Kyrgyzstan", "kyrgyz republic": "Kyrgyzstan", "이란": "Iran", "iran, islamic republic of": "Iran",
    "프랑스": "France", "독일": "Germany", "스페인": "Spain", "이탈리아": "Italy", "캐나다": "Canada", "호주": "Australia", "대만": "Taiwan",
    "헝가리": "Hungary", "폴란드": "Poland", "아제르바이잔": "Azerbaijan", "타지키스탄": "Tajikistan", "투르크메니스탄": "Turkmenistan",
}


def norm_country(name):
    """자유 입력 국적 → 국적 모델 라벨 표기(label_en). 모르는 이름은 단어 첫 글자만 대문자로 둔다."""
    k = name.strip().lower()
    if k in COUNTRY_ALIAS:
        return COUNTRY_ALIAS[k]
    return " ".join(w[:1].upper() + w[1:] for w in name.strip().split())


def booth_sets():
    d = json.load(open(os.path.join(HERE, "booth_sentences.json"), encoding="utf-8"))
    return {st["id"]: {it["id"]: it for it in st["items"]} for st in d["sets"]}


def booth_load():
    os.makedirs(BOOTH_DIR, exist_ok=True)
    p = os.path.join(BOOTH_DIR, "participants.jsonl")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
                BOOTH_PARTS[r["code"]] = r
            except Exception:
                pass


def booth_pin():
    f = os.path.join(BOOTH_DIR, ".staff_pin")
    if not os.path.exists(f):
        os.makedirs(BOOTH_DIR, exist_ok=True)
        with open(f, "w") as fh:
            fh.write(f"{secrets.randbelow(10 ** 6):06d}")
        os.chmod(f, 0o600)
    return open(f).read().strip()


def booth_page(name):
    """템플릿 조각(제목·스타일·본문) + 문장·비버 이미지 주입 → 문서 하나."""
    frag = open(os.path.join(HERE, name), encoding="utf-8").read()
    sets = open(os.path.join(HERE, "booth_sentences.json"), encoding="utf-8").read()
    avatar = base64.b64encode(open(os.path.join(HERE, "booth_beaver.webp"), "rb").read()).decode()
    frag = frag.replace("/*__SETS_JSON__*/null", sets).replace("__AVATAR__", "data:image/webp;base64," + avatar)
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">' + frag + "</html>")


def booth_done_items(code):
    f = os.path.join(BOOTH_DIR, code, "meta.jsonl")
    if not os.path.exists(f):
        return set()
    out = set()
    for line in open(f, encoding="utf-8"):
        try:
            out.add(int(json.loads(line)["item"]))
        except Exception:
            pass
    return out


def audio_ext(raw, filename=""):
    """브라우저 녹음 원본 확장자 — Chrome·Android 는 webm, iPhone Safari 는 mp4(m4a)."""
    if raw[:4] == b"\x1aE\xdf\xa3" or b"webm" in raw[:64]:
        return ".webm"
    if raw[4:8] == b"ftyp":
        return ".m4a"
    if raw[:4] == b"OggS":
        return ".ogg"
    if raw[:4] == b"RIFF":
        return ".src.wav"                # 변환본 .wav 와 이름이 겹치지 않게
    return os.path.splitext(filename)[1] or ".bin"


def make_app():
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import HTMLResponse
    sub = FastAPI(title="녹음 수집")

    @sub.get("/", response_class=HTMLResponse)
    def index():
        # 2026-10-01: 공개 주소 첫 화면(/collect)은 부스 이벤트 페이지. 예전 수집 화면은 /legacy.
        # (예전 화면은 상대 경로로 sentences·upload 를 불러서, 끝에 / 없는 /collect 로 열면 문장이 안 나왔다)
        return booth_page("booth.html")

    @sub.get("/legacy", response_class=HTMLResponse)
    def legacy():
        return open(os.path.join(HERE, "index.html"), encoding="utf-8").read()

    @sub.get("/sentences")
    def sentences():
        return json.load(open(os.path.join(HERE, "sentences.json"), encoding="utf-8"))

    @sub.get("/status")
    def status():
        spk = [d for d in os.listdir(OUT) if os.path.isdir(os.path.join(OUT, d))] if os.path.isdir(OUT) else []
        n = sum(len([f for f in os.listdir(os.path.join(OUT, d)) if f.endswith(".wav")]) for d in spk)
        return {"speakers": len(spk), "utterances": n, "out": OUT}

    @sub.post("/upload")
    def upload(audio: UploadFile = File(...), name: str = Form(...), item_id: int = Form(...), text: str = Form(""),
               native_language: str = Form(""), nationality: str = Form(""), device: str = Form(""), gender: str = Form(""),
               age_group: str = Form(""), korean_level: str = Form(""), item_type: str = Form("read")):
        sid = speaker_id(name)
        d = os.path.join(OUT, sid); os.makedirs(d, exist_ok=True)
        raw = audio.file.read()
        if len(raw) < 1000:
            raise HTTPException(400, "녹음이 비어 있어요. 다시 녹음해 주세요.")
        ext = ".webm" if b"webm" in raw[:64] or audio.filename.endswith(".webm") else (".m4a" if audio.filename.endswith(".m4a") else ".bin")
        with LOCK:
            k = 1
            while os.path.exists(os.path.join(d, f"{item_id:02d}_{k}.wav")):
                k += 1
            base = os.path.join(d, f"{item_id:02d}_{k}")
            open(base + ext, "wb").write(raw)
            sec = decode_to_wav(raw, base + ".wav")
            if sec is None:
                os.remove(base + ext)
                raise HTTPException(400, "오디오를 읽지 못했어요. 브라우저를 바꿔 다시 시도해 주세요.")
            rec = {"speaker": sid, "name": name.strip(), "item_id": item_id, "item_type": item_type, "text": text, "sec": sec,
                   "native_language": native_language, "nationality": nationality, "device": device, "gender": gender,
                   "age_group": age_group, "korean_level": korean_level, "raw": os.path.basename(base + ext),
                   "wav": os.path.basename(base + ".wav"), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "ua": ""}
            with open(os.path.join(d, "meta.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {"ok": True, "speaker": sid, "item_id": item_id, "sec": sec, "take": k}

    # ── 부스 이벤트 ──
    from fastapi import Request
    SETS = booth_sets()

    @sub.get("/booth", response_class=HTMLResponse)
    def booth():
        return booth_page("booth.html")

    @sub.get("/booth/staff", response_class=HTMLResponse)
    def booth_staff(pin: str = ""):
        if pin != booth_pin():
            raise HTTPException(403, "스태프 PIN 이 필요합니다 (?pin=...)")
        return booth_page("booth_staff.html")

    @sub.get("/booth/api/ping")
    def booth_ping():
        return {"ok": True}

    @sub.post("/booth/api/start")
    async def booth_start(req: Request):
        b = await req.json()
        c, p = b.get("consent") or {}, b.get("profile") or {}
        if not (c.get("agree") is True and c.get("age14") is True):
            raise HTTPException(400, "동의가 필요해요.")
        nat, l1 = str(p.get("nationality", "")).strip()[:60], str(p.get("native_language", "")).strip()[:60]
        if not nat or not l1:
            raise HTTPException(400, "국적과 모국어를 입력해 주세요.")
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with BOOTH_LOCK:
            used = collections.Counter(r["set_id"] for r in BOOTH_PARTS.values())
            set_id = min(SETS, key=lambda k: (used.get(k, 0), random.random()))
            code = None
            while code is None or code in BOOTH_PARTS:
                code = "BT-" + "".join(secrets.choice(CODE_CHARS) for _ in range(4))
            rec = {"code": code, "set_id": set_id, "nationality": nat, "label_en": norm_country(nat), "native_language": l1,
                   "korean_level": str(p.get("korean_level", ""))[:20], "gender": str(p.get("gender", ""))[:20],
                   "age_group": str(p.get("age_group", ""))[:20], "lang": str(b.get("lang", ""))[:5],
                   "device": str(b.get("device", ""))[:200],
                   "consent": {"version": str(c.get("version", ""))[:20], "agree": True, "age14": True, "time": now}, "time": now}
            os.makedirs(BOOTH_DIR, exist_ok=True)
            with open(os.path.join(BOOTH_DIR, "participants.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            BOOTH_PARTS[code] = rec
        return {"code": code, "set_id": set_id}

    @sub.post("/booth/api/upload")
    def booth_upload(audio: UploadFile = File(...), code: str = Form(...), item: int = Form(...),
                     duration_ms: int = Form(0), peak: float = Form(0.0), mime: str = Form(""),
                     guide: str = Form("")):
        r = BOOTH_PARTS.get(code)
        if r is None:
            raise HTTPException(404, "참여 코드를 찾지 못했어요. 처음부터 다시 시작해 주세요.")
        if not 1 <= item <= 10:
            raise HTTPException(400, "문장 번호가 잘못됐어요.")
        raw = audio.file.read()
        if len(raw) < 1000:
            raise HTTPException(400, "녹음이 비어 있어요. 다시 녹음해 주세요.")
        if len(raw) > 10 * 1024 * 1024:
            raise HTTPException(413, "녹음이 너무 커요.")
        ext = audio_ext(raw, audio.filename or "")
        d = os.path.join(BOOTH_DIR, code)
        with BOOTH_LOCK:
            os.makedirs(d, exist_ok=True)
            k = 1
            while os.path.exists(os.path.join(d, f"{r['set_id']}_{item:02d}_{k}.wav")):
                k += 1
            base = os.path.join(d, f"{r['set_id']}_{item:02d}_{k}")
            open(base + ext, "wb").write(raw)
            sec = decode_to_wav(raw, base + ".wav")
            if sec is None:
                os.remove(base + ext)
                raise HTTPException(400, "오디오를 읽지 못했어요. 다시 녹음해 주세요.")
            it = SETS[r["set_id"]][item]
            row = {"code": code, "set_id": r["set_id"], "item": item, "take": k, "text": it["text"], "slot": it.get("slot", ""),
                   "sec": sec, "nationality": r["nationality"], "label_en": r["label_en"], "native_language": r["native_language"],
                   "korean_level": r["korean_level"], "gender": r["gender"], "age_group": r["age_group"], "device": r["device"],
                   "duration_ms": duration_ms, "peak": peak, "mime": mime[:60], "guide": guide[:10], "raw": os.path.basename(base + ext),
                   "wav": os.path.basename(base + ".wav"), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            with open(os.path.join(d, "meta.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {"ok": True, "item": item, "take": k, "sec": sec}

    @sub.get("/booth/api/check")
    def booth_check(code: str):
        if code not in BOOTH_PARTS:
            raise HTTPException(404, "참여 코드를 찾지 못했어요.")
        items = booth_done_items(code)
        return {"code": code, "items": sorted(items), "n": len(items)}

    @sub.get("/booth/api/stats")
    def booth_stats(pin: str = ""):
        if pin != booth_pin():
            raise HTTPException(403, "스태프 PIN 이 필요합니다.")
        parts = list(BOOTH_PARTS.values())
        done = {r["code"]: len(booth_done_items(r["code"])) for r in parts}
        by_c = collections.defaultdict(lambda: [0, 0])
        for r in parts:
            by_c[r["label_en"]][0] += 1
            by_c[r["label_en"]][1] += int(done[r["code"]] >= 10)
        recent = sorted(parts, key=lambda r: r["time"], reverse=True)[:20]
        return {"participants": len(parts), "completed": sum(n >= 10 for n in done.values()), "recordings": sum(done.values()),
                "by_country": sorted(([k, v[0], v[1]] for k, v in by_c.items()), key=lambda x: (-x[1], x[0])),
                "by_set": dict(sorted(collections.Counter(r["set_id"] for r in parts).items())),
                "recent": [{"code": r["code"], "nationality": r["label_en"], "n": done[r["code"]], "time": r["time"][5:16]} for r in recent]}

    root = FastAPI()
    root.mount(PREFIX, sub)                 # /collect 로 직접 들어오는 요청용 (Funnel 은 프리픽스를 벗겨서 / 로 온다)
    root.mount("/", sub)                    # 로컬 포워딩용
    return root


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8923)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    booth_load()
    print(f"[booth] 저장 {BOOTH_DIR} · 참가자 {len(BOOTH_PARTS)}명", flush=True)
    print(f"[collect] 저장 {OUT} · prefix {PREFIX} · ffmpeg {ffmpeg()}", flush=True)
    import uvicorn
    uvicorn.run(make_app(), host="0.0.0.0", port=a.port)
