# -*- coding: utf-8 -*-
"""실제 발음 CTC 확인용 로컬 데모 — 샘플 청취 페이지 + 마이크 입력 추론.

샘플 페이지(`make_listen_page.py` 산출물)를 그대로 서비스하고, 거기에 마이크 녹음을 붙인다.
모델을 여러 개 올려 같은 녹음을 나란히 비교한다(`--ckpt` fairseq · `--hf-ckpt` HuggingFace w2v-BERT).
    POST /transcribe   audio(파일) + sentence(선택) + country(선택) → 모델별 실제 발음 · 자리별 판정 · 국적 피드백
    GET  /countries    국적 목록 (교정 문장이 있는 34개국 먼저)
    GET  /models       올라간 모델 이름 목록 (페이지가 줄을 이 순서로 만든다)
디코딩은 greedy CTC(LM·사전·문장 힌트 없음) — 평가와 같은 경로다.

    $PY deploy/ctc-listen-demo/app.py --ckpt $SPEECH_AI_DATA/checkpoints/FINAL_ctc/checkpoint_final.pt \\
        --static $MDD_OUTPUTS/experiments/final_ctc/listen --port 8930

⚠️ 로컬 확인용이다. 인증이 없고 AIHub 오디오가 들어 있으니 외부에 열지 말 것.
   마이크는 보안 컨텍스트에서만 열린다 — http://127.0.0.1:<port> (SSH 터널/포트 포워딩) 로 접속할 것.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import FFMPEG, G2P_TOOLS, MDD_ANALYSIS, ensure_kog2p
from shared.fairseq_guard import ensure_fairseq

import argparse, io, json, os, re, shutil, subprocess, tempfile, time

import numpy as np

SR = 16000
S = {}


def load_g2p():
    ensure_kog2p()
    argv, _sys.argv = _sys.argv, _sys.argv[:1]     # g2p.py 는 import 시점에 argv 를 파싱한다
    try:
        from g2p import readRules, graph2prono, ver_info
    finally:
        _sys.argv = argv
    from mdd.jamo_to_kog2p import phones_to_jamo, jamo_to_phones, INV
    S["g2p"] = graph2prono
    S["rules"] = readRules(ver_info[0], str(G2P_TOOLS / "KoG2P" / "rulebook.txt"))
    S["p2j"], S["j2p"], S["inv"] = phones_to_jamo, jamo_to_phones, INV


def canonical(sentence):
    """문장 → 표준발음 자모열. KoG2P 는 공백이 정확히 2칸이면 음절을 깨뜨려서 1칸으로 맞춘다."""
    text = re.sub(r"\s+", " ", re.sub(r"[^0-9A-Za-z가-힣\s]", " ", sentence or "")).strip()
    if not text:
        return []
    ri, ro = S["rules"]
    out = []
    for j in S["p2j"]([p for p in S["g2p"](text, ri, ro).split() if p in S["inv"]]):
        out += {"ㅙ": ["ㅗ", "ㅐ"], "ㅞ": ["ㅜ", "ㅔ"]}.get(j, [j])   # TAG/LAB 표기 관례
    return out


def _device(torch):
    return "cuda" if (torch.cuda.is_available() and os.environ.get("MDD_FORCE_CPU") != "1") else "cpu"


def load_model(ckpt, dict_dir, label):
    """fairseq wav2vec_ctc (XLS-R 계열)."""
    ensure_fairseq()
    import torch
    from fairseq import checkpoint_utils
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt], arg_overrides={"data": dict_dir})
    m = models[0].eval()
    dev = _device(torch)
    S["torch"] = torch; S["device"] = dev
    S.setdefault("models", []).append(
        {"name": label, "kind": "fairseq", "model": (m.cuda().half() if dev == "cuda" else m.float()),
         "d": task.target_dictionary, "normalize": cfg.task.normalize,
         "blank": task.target_dictionary.index(task.blank_symbol) if hasattr(task, "blank_symbol") else task.target_dictionary.bos()})
    print(f"[model] {label} · fairseq {ckpt} · vocab {len(task.target_dictionary)} · {dev}", flush=True)


def load_hf_model(mdir, label):
    """HuggingFace w2v-BERT 2.0 (train_w2vbert_ctc.py 산출물) — 필터뱅크 입력."""
    import torch
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertForCTC
    vocab = json.load(open(os.path.join(mdir, "jamo_vocab.json"), encoding="utf-8"))
    dev = _device(torch)
    m = Wav2Vec2BertForCTC.from_pretrained(mdir).eval().to(dev)
    S["torch"] = torch; S["device"] = dev
    S.setdefault("models", []).append(
        {"name": label, "kind": "hf", "model": (m.half() if dev == "cuda" else m.float()),
         "fe": SeamlessM4TFeatureExtractor.from_pretrained(mdir),
         "inv": {i: t for t, i in vocab["vocab"].items()}, "blank": vocab["blank"]})
    print(f"[model] {label} · hf {mdir} · vocab {len(vocab['vocab'])} · {dev}", flush=True)


def decode_audio(raw: bytes) -> np.ndarray:
    """브라우저 녹음(webm/opus 등) → 16k mono float32. ffmpeg 이 읽는 형식이면 뭐든 된다."""
    exe = shutil.which(FFMPEG) or FFMPEG
    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        f.write(raw); f.flush()
        p = subprocess.run([exe, "-nostdin", "-i", f.name, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0 or not p.stdout:
        raise ValueError(p.stderr.decode("utf-8", "replace").strip()[-300:] or "빈 오디오")
    return np.frombuffer(p.stdout, np.float32)


def transcribe(E, wav: np.ndarray):
    """모델 하나로 greedy CTC 디코드 → 자모열."""
    torch = S["torch"]
    import torch.nn.functional as F
    q = next(E["model"].parameters())
    with torch.inference_mode():
        if E["kind"] == "fairseq":
            t = torch.from_numpy(wav.copy())
            if E["normalize"]:
                t = F.layer_norm(t, t.shape)
            net = E["model"](source=t[None, :].to(q.device, q.dtype),
                             padding_mask=torch.zeros(1, len(wav), dtype=torch.bool, device=q.device))
            lp = E["model"].get_normalized_probs(net, log_probs=True).transpose(0, 1).float()[0]
            toks = lp.argmax(-1).unique_consecutive().tolist()
            out = [E["d"][i] for i in toks if i != E["blank"]]
        else:
            enc = E["fe"]([wav], sampling_rate=SR, return_tensors="pt")
            lg = E["model"](input_features=enc["input_features"].to(q.device, q.dtype),
                            attention_mask=enc["attention_mask"].to(q.device)).logits.float()[0]
            toks = lg.argmax(-1).unique_consecutive().tolist()
            out = [E["inv"][i] for i in toks if i != E["blank"] and i in E["inv"]]
    return [t for t in out if len(t) == 1 and 0x3131 <= ord(t[0]) <= 0x3163]


def align(canon, hyp):
    """표준발음 자리마다 모델이 무엇을 냈는지. (자리, 표준, 모델, 판정, 모델 쪽 위치 j)"""
    import edlib
    c, h = "".join(canon), "".join(hyp)
    if not c:
        return []
    if not h:
        return [{"i": i, "canon": x, "pred": None, "verdict": "삭제"} for i, x in enumerate(canon)]
    a = edlib.getNiceAlignment(edlib.align(c, h, task="path"), c, h)
    out, i, j = [], 0, 0
    for x, y in zip(a["query_aligned"], a["target_aligned"]):
        if x == "-":
            if out:
                out[-1]["inserted"] = (out[-1].get("inserted") or "") + y
                out[-1].setdefault("ins_j", []).append(j)
            j += 1
            continue
        out.append({"i": i, "canon": x, "pred": None if y == "-" else y, "j": None if y == "-" else j,
                    "verdict": "일치" if x == y else ("삭제" if y == "-" else "다름")})
        i += 1
        j += y != "-"
    return out


def make_app(static_dir):
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
    app = FastAPI(title="ctc-listen-demo")

    @app.get("/health")
    def health():
        return {"ok": True, "device": S["device"], "models": [e["name"] for e in S["models"]]}

    @app.get("/models")
    def models():
        return {"models": [{"name": e["name"], "kind": e["kind"]} for e in S["models"]]}

    @app.get("/countries")
    def countries():
        return {"countries": S["l1fb"].countries() if S.get("l1fb") else []}

    @app.post("/transcribe")
    async def api(audio: UploadFile = File(...), sentence: str = Form(""), country: str = Form("")):
        t0 = time.time()
        raw = await audio.read()
        try:
            wav = decode_audio(raw)
        except ValueError as e:
            return JSONResponse({"error": f"오디오를 읽지 못했다: {e}"}, status_code=400)
        if len(wav) < SR * 0.3:
            return JSONResponse({"error": "녹음이 너무 짧다 (0.3초 미만)"}, status_code=400)
        rms = float(np.sqrt(np.mean(wav[: SR * 30] ** 2)))
        if rms < 1.5e-3:
            return JSONResponse({"error": "소리가 거의 없다 — 마이크 입력을 확인할 것"}, status_code=400)
        canon = canonical(sentence)
        cph = S["j2p"](canon) if canon else []
        out = []
        for E in S["models"]:
            t1 = time.time()
            jamo = transcribe(E, wav[: SR * 20])
            pph = S["j2p"](jamo)
            sites = align(canon, jamo) if canon else []
            for s in sites:   # 음소기호 병기용 — j2p 는 자모와 1:1 이라 위치로 대응한다
                s["canon_ph"] = cph[s["i"]] if len(cph) == len(canon) else None
                s["pred_ph"] = pph[s["j"]] if (s["j"] is not None and len(pph) == len(jamo)) else None
                if s.get("ins_j") and len(pph) == len(jamo):
                    s["inserted_ph"] = [pph[k] for k in s["ins_j"]]
            out.append({"name": E["name"], "ms": int((time.time() - t1) * 1000),
                        "pred_jamo": jamo, "pred_phone": pph, "sites": sites,
                        "feedback": S["l1fb"].annotate(canon, sites, country or None)
                                    if (S.get("l1fb") and canon) else None})
        first = out[0]
        return {"sec": round(len(wav) / SR, 2), "rms": round(rms, 4), "ms": int((time.time() - t0) * 1000),
                "canon_jamo": canon, "canon_phone": cph, "models": out,
                # 옛 페이지 호환 — 첫 모델 결과를 그대로 둔다
                "pred_jamo": first["pred_jamo"], "pred_phone": first["pred_phone"],
                "sites": first["sites"], "feedback": first["feedback"]}

    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", default=[],
                    help="fairseq 체크포인트. '이름=경로' 로 줄 이름을 정한다. 여러 번 줄 수 있다")
    ap.add_argument("--hf-ckpt", dest="hf_ckpt", action="append", default=[],
                    help="HuggingFace w2v-BERT 폴더. '이름=경로' 형식 가능")
    ap.add_argument("--static", required=True, help="index.html 과 wav/ 가 있는 폴더")
    ap.add_argument("--dict-dir", dest="dict_dir", default=os.environ.get("MDD_DATA_DIR"),
                    help="dict.ltr.txt 폴더 (기본 $MDD_DATA_DIR)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8930)
    a = ap.parse_args()
    if not (a.ckpt or a.hf_ckpt):
        ap.error("--ckpt 나 --hf-ckpt 중 하나는 필요하다")
    load_g2p()
    split = lambda v, d: (v.split("=", 1) if "=" in v else (d, v))
    for v in a.hf_ckpt:
        name, path = split(v, "w2v-BERT"); load_hf_model(path, name)
    for v in a.ckpt:
        name, path = split(v, "FINAL v2"); load_model(path, a.dict_dir, name)
    wp = os.path.join(str(MDD_ANALYSIS), "l1_prior_product", "l1_prior_weights.json")
    ep = os.path.join(str(MDD_ANALYSIS), "l1_prior_product", "nationality_feedback_enriched.json")
    if os.path.exists(wp):
        from mdd.l1_feedback import L1Feedback
        S["l1fb"] = L1Feedback(wp, ep)
        print(f"[l1] 가중치 {len(S['l1fb'].priors)}개국 · 교정 문장 {len(S['l1fb'].text)}개국", flush=True)
    else:
        print(f"[l1] {wp} 없음 — 국적 피드백 없이 동작", flush=True)
    import uvicorn
    uvicorn.run(make_app(a.static), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
