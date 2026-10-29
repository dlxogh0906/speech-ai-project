"""Local, authenticated blind listening for frozen sibilant evidence packets.

Only opaque trial IDs, audio and a reviewer's own responses leave the server.
The packet's transcription, prompt, model output, source paths, and other
reviewers' answers are never included in the listening API.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import threading
import time

HERE = Path(__file__).resolve().parent
CHOICES = {"s", "ss", "other", "uncertain", "inaudible"}


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]


def validate_label(data, permitted):
    if not isinstance(data, dict) or data.get("trial") not in permitted:
        raise ValueError("unknown trial")
    if data.get("choice") not in CHOICES:
        raise ValueError("a listening choice is required")
    if type(data.get("confidence")) is not int or data["confidence"] not in (1, 2, 3):
        raise ValueError("confidence must be 1, 2 or 3")
    if data.get("native_korean") is not True:
        raise ValueError("native Korean listener declaration is required")
    counts = {}
    for mode in ("local", "context"):
        v = data.get("plays", {}).get(mode, 0)
        if type(v) is not int or not 0 <= v <= 10000:
            raise ValueError("invalid playback count")
        counts[mode] = v
    if not sum(counts.values()):
        raise ValueError("play audio before submitting")
    elapsed = data.get("elapsed_seconds", 0)
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or not 0 <= elapsed <= 86400:
        raise ValueError("invalid elapsed time")
    return {"trial": data["trial"], "choice": data["choice"], "confidence": data["confidence"],
            "native_korean": True, "plays": counts, "elapsed_seconds": elapsed,
            "note": str(data.get("note", ""))[:1000]}


def make_app(packet, keys, responses):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse

    packet, responses = Path(packet), Path(responses)
    items = {r["id"]: r for r in read_jsonl(packet / "items.private.jsonl")}
    trials = json.loads((packet / "trials.private.json").read_text())
    if set(keys) != set(trials) or len(set(keys.values())) != len(keys):
        raise ValueError("each reviewer needs a separate key")
    lookup = {who: {t["trial"]: t["item"] for t in ts} for who, ts in trials.items()}
    responses.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    manifest_sha = hashlib.sha256((packet / "items.private.jsonl").read_bytes()).hexdigest()
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def who_of(request):
        token = request.headers.get("x-review-key") or request.query_params.get("t", "")
        for who, key in keys.items():
            if secrets.compare_digest(token, key):
                return who
        raise HTTPException(401, "reviewer link required")

    def answers(who):
        return {r["trial"]: r for r in read_jsonl(responses / ("labels_" + who + ".jsonl"))}

    @app.middleware("http")
    async def private_response(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/health")
    def health():
        return {"ok": True, "task": "sibilant-blind-listening"}

    @app.get("/")
    def index(request: Request):
        who_of(request)
        return HTMLResponse((HERE / "index.html").read_text(encoding="utf-8"))

    @app.get("/trials")
    def get_trials(request: Request):
        who = who_of(request)
        done = answers(who)
        return {"reviewer": who, "trials": [{"trial": t["trial"],
                "answer": {k: done[t["trial"]][k] for k in ("choice", "confidence", "note")}
                if t["trial"] in done else None} for t in trials[who]]}

    @app.get("/audio/{trial}/{mode}")
    def get_audio(request: Request, trial: str, mode: str):
        who = who_of(request)
        if trial not in lookup[who] or mode not in ("local", "context"):
            raise HTTPException(404)
        item = lookup[who][trial]
        return FileResponse(packet / "audio" / (item + "_" + mode + ".wav"), media_type="audio/wav")

    @app.post("/label")
    async def label(request: Request):
        who = who_of(request)
        try:
            data = validate_label(await request.json(), lookup[who])
        except (ValueError, TypeError, AttributeError) as e:
            raise HTTPException(400, str(e))
        rec = {**data, "reviewer": who, "item": lookup[who][data["trial"]],
               "actor_type": "human_self_declared", "mode": "script_and_model_blinded",
               "packet_sha256": manifest_sha, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        with lock, (responses / ("labels_" + who + ".jsonl")).open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return {"ok": True}

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--responses", type=Path)
    ap.add_argument("--keys", type=Path, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    if not a.keys.exists():
        a.keys.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(os.open(str(a.keys), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
            json.dump({w: secrets.token_urlsafe(24) for w in ("A", "B", "C")}, f)
    keys = json.loads(a.keys.read_text())
    app = make_app(a.packet, keys, a.responses or a.packet.parent / "human_responses")
    print("Listening service ready at %s:%s; keys remain in the specified local file." % (a.host, a.port), flush=True)
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, access_log=False, log_level="warning")


if __name__ == "__main__":
    main()
