# -*- coding: utf-8 -*-
"""사람 TAG 파일럿 P0 전사 화면 — 들린 그대로 한글로 적고, 작업 시간을 자동으로 남긴다.

`build_tag_pilot_p0.py` 가 고른 발화(items.json)를 전사자 A · B 가 따로 전사한다. 발화마다 한 사람은 빈칸에서,
다른 사람은 표준발음을 미리 채운 칸에서 시작한다(items.json 의 prefill — 미리 채움 끌림 측정).
전사자에게는 **원문 · 소리만** 보여준다(모국어 · 모델 결과는 숨긴다 — 끌림 방지).

저장: `<out>/labels_<전사자>.jsonl` 에 제출할 때마다 한 줄(나중 제출이 이긴다). 일치율은 `tag_pilot_agreement.py`.
**독립 전사**: 전사자마다 열쇠를 따로 준다(`--keys`). 열쇠가 전사자를 정하므로 A 는 B 의 답을 볼 수 없다.
AIHub 오디오를 내보내므로 **사내 전사자만** — 열쇠 없이 띄우면(시험용) 127.0.0.1 에서만 연다.

    source tools/env.sh
    $PY deploy/tag-pilot/app.py --items $MDD_OUTPUTS/tag_pilot/p0_v1/items.json --port <빈 포트> --keys <열쇠 파일>
    → 전사자 A: http://…/?t=<A 열쇠>   전사자 B: http://…/?t=<B 열쇠>
    열쇠 파일(JSON): {"A": "...", "B": "..."} — 없으면 새로 만든다(권한 600)
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, json, os, secrets, threading, time

HERE = _pl.Path(__file__).parent
ITEMS, OUT, KEYS = [], None, {}
LOCK = threading.Lock()
WHO = ("A", "B")


def last_labels(who):
    p = os.path.join(OUT, "labels_%s.jsonl" % who)
    d = {}
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            d[r["id"]] = r
    return d


def make_app():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse

    app = FastAPI(title="TAG 파일럿 P0")

    def who_of(request, claimed=None):
        """열쇠가 있으면 열쇠가 전사자를 정한다(다른 사람 답을 못 본다). 열쇠 없이 띄웠으면(시험용) 고른 이름."""
        if KEYS:
            t = request.query_params.get("t") or request.headers.get("x-token")
            for w, k in KEYS.items():
                if t and secrets.compare_digest(t, k):
                    return w
            raise HTTPException(403, "열쇠가 필요합니다")
        if claimed not in WHO:
            raise HTTPException(400, "전사자는 A 또는 B")
        return claimed

    @app.get("/whoami")
    def whoami(request: Request):
        return {"who": who_of(request, "A") if KEYS else None}

    @app.get("/")
    def root():
        return FileResponse(str(HERE / "static" / "index.html"), headers={"Cache-Control": "no-store"})

    @app.get("/items")
    def items(request: Request, who: str = ""):
        who = who_of(request, who)
        done = last_labels(who)
        return [{"id": it["id"], "text": it["text"], "sec": it["sec"],
                 "prefill": it["canon_hangul"] if it["prefill"][who] == "canon" else "",
                 "mode": it["prefill"][who], "done": it["id"] in done,
                 "answer": done[it["id"]]["text"] if it["id"] in done else None} for it in ITEMS]

    @app.get("/audio/{i}")
    def audio(request: Request, i: int):
        who_of(request, "A")
        if not 0 <= i < len(ITEMS):
            raise HTTPException(404)
        return FileResponse(ITEMS[i]["wav"], media_type="audio/wav")

    @app.post("/label")
    async def label(request: Request):
        d = await request.json()
        d["who"] = who_of(request, d.get("who"))
        if not 0 <= int(d.get("id", -1)) < len(ITEMS):
            return JSONResponse({"error": "잘못된 요청"}, status_code=400)
        it = ITEMS[int(d["id"])]
        rec = {"id": it["id"], "stem": it["stem"], "who": d["who"], "mode": it["prefill"][d["who"]],
               "text": str(d.get("text", "")), "flag": str(d.get("flag", "")), "note": str(d.get("note", ""))[:500],
               "time_sec": float(d.get("time_sec", 0)), "first_play_sec": d.get("first_play_sec"),
               "plays": int(d.get("plays", 0)), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with LOCK, open(os.path.join(OUT, "labels_%s.jsonl" % d["who"]), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {"ok": True}

    return app


def main():
    global ITEMS, OUT, KEYS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True, help="build_tag_pilot_p0.py 가 만든 items.json")
    ap.add_argument("--out", default=None, help="전사 결과 폴더(기본: items.json 이 있는 폴더)")
    ap.add_argument("--keys", default=None, help="전사자별 열쇠 JSON({A, B}). 없으면 만든다. 안 주면 열쇠 없이(127.0.0.1 시험용)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    ITEMS = json.load(open(a.items, encoding="utf-8"))
    OUT = a.out or os.path.dirname(os.path.abspath(a.items))
    os.makedirs(OUT, exist_ok=True)
    if a.keys:
        if not os.path.exists(a.keys):
            json.dump({w: secrets.token_urlsafe(12) for w in WHO}, open(a.keys, "w"))
            os.chmod(a.keys, 0o600)
        KEYS = json.load(open(a.keys))
    elif a.host != "127.0.0.1":
        raise SystemExit("열쇠 없이는 127.0.0.1 에서만 띄운다(--keys)")
    print("[load] 발화 %d개 · 저장 %s · 전사자 열쇠 %s" % (len(ITEMS), OUT, "켬" if KEYS else "끔(시험용)"), flush=True)
    import uvicorn
    uvicorn.run(make_app(), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
