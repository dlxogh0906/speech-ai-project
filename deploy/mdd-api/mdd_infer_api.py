# -*- coding: utf-8 -*-
"""발음진단 통합 추론 API — MDD(어디를 틀렸나) + 점수 3종(얼마나 잘하나)을 한 번에.

핵심: 점수용 GOP 특징은 MDD 인코더 forward 의 **부산물**이다. 따라서 모델을 한 번만
돌리고 두 결과를 모두 낸다(공식 팩 방식이라면 모델 4번 + 모국어 판별이 필요하다).

  POST /diagnose        오디오 + 문장  → 음소별 진단 (+ 세션 누적 시 점수)
  GET  /session/{id}    세션 점수 조회
  DELETE /session/{id}  세션 초기화
  GET  /health

⚠️ 점수는 **세션 단위로만 유효**하다. 발화 1개 PCC는 0.43, 20~30발화면 0.66~0.76으로 포화한다.
   그래서 발화 수가 MIN_UTTS 미만이면 점수를 내지 않고 `pending` 으로 돌려준다.
   (사람 채점자도 발화가 아니라 학습자를 보고 점수를 매겼다 — 라벨 분산의 74~94%가 화자로 설명됨)

Run:  python mdd_infer_api.py --port 8903
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import os, sys, json, time, argparse, threading, collections
import numpy as np
import torch
import torch.nn.functional as F
import editdistance
import joblib

from mdd import gop_feats_v2 as GF                     # forward_lprobs / viterbi_align / align_feats
from mdd import pipeline as MD                         # canonical / _nw / ctype / syllable_ids / decode_audio

from mdd.mdd_eval import Decoder
from mdd.jamo_to_kog2p import jamo_to_phones

SR = 16000
MIN_UTTS = 5                                  # 이 밑에서는 점수를 내지 않는다
GOOD_UTTS = 20                                # 여기부터 신뢰 구간 (30에서 포화)
# 서빙하는 모델을 여기서 못 박는다. `shared.paths.MDD_CKPT` 기본값(= tag_all_lar)을 쓰지
# 않는 이유: 그 기본값은 평가 스크립트들도 공유해서, 바꾸면 results/ 에 기록된 수치가
# 재현되지 않는다. 배포가 무엇을 쓰는지는 배포 파일이 직접 밝히는 편이 안전하다.
#   tag_all_native = tag_all_lar + 원어민 자유대화. 원어민 PER 31.45%→8.58%,
#   외국인 TAG test weighted F1 0.9547→0.9558 (2026-08-04 교체)
CKPT_PATH = os.environ.get("MDD_SERVE_CKPT",
                           f"{DATA_ROOT}/mdd_ckpt_tag_all_native/checkpoint_best.pt")
# 회귀기도 같은 체크포인트의 특징으로 학습된 것이어야 한다. 짝이 어긋나면 점수가 조용히 틀린다.
MODELS_PATH = os.environ.get("MDD_SERVE_SCORER",
                             f"{MDD_OUTPUTS}/score_models_v2_native.joblib")
# /health 가 보고하는 TAG test weighted F1. 근거는 eval_tag71{,_native}.json.
# ⚠️ 전부 "정상"으로 찍어도 0.9384 가 나온다 — 이 값만 떼어 인용하지 말 것.
MDD_F1 = {"mdd_ckpt_tag_all_lar": 0.9547, "mdd_ckpt_tag_all_native": 0.9558}
# 인코더만 ONNX 로 갈아 끼운다. 비우면 지금까지와 같은 PyTorch 경로다(기본값 유지).
#   CPU 는 int8(339 MiB · RSS 1.1 GB · 1.6~1.8배), GPU 는 fp16(603 MiB · 2.3배).
#   정확도는 같다 — TAG test 1만 발화 오류 F1 차이 int8 +0.0002 · fp16 +0.0005,
#   LAR·원어민 표본에서도 PER 차이 없음 (results/11_경량화.md).
ONNX_PATH = os.environ.get("MDD_SERVE_ONNX")
ONNX_DEVICE = os.environ.get("MDD_SERVE_DEVICE", "cpu")
ONNX_THREADS = int(os.environ.get("MDD_SERVE_THREADS", "4"))
S = {"dec": None, "bundle": None, "lock": threading.Lock(), "gate": None, "backend": None,
     "sessions": collections.defaultdict(list)}


def _load_onnx():
    """인코더 forward 만 ONNX 세션으로 바꾼다 — 진단·정렬·점수 코드는 그대로 쓴다.

    `analyze()` 는 `GF.forward_lprobs` 와 `dec` 의 속성 몇 개(d/blank/normalize)만 보므로
    그 둘만 갈아 끼우면 된다. 회귀기는 바꾸지 않는다(점수 상관 0.999 이상, 11_경량화.md).
    """
    import onnxruntime as ort
    from fairseq.data import Dictionary
    so = ort.SessionOptions()
    so.intra_op_num_threads = ONNX_THREADS
    so.inter_op_num_threads = 1
    prov = (["CUDAExecutionProvider", "CPUExecutionProvider"] if ONNX_DEVICE == "cuda"
            else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(ONNX_PATH, so, providers=prov)
    if ONNX_DEVICE == "cuda" and sess.get_providers()[0] != "CUDAExecutionProvider":
        # onnxruntime-gpu 는 CUDA 11 라이브러리를 찾아야 한다 — LD_LIBRARY_PATH 확인
        raise SystemExit(f"CUDA provider 가 안 잡혔다: {sess.get_providers()}")
    in_dt = np.float16 if "float16" in sess.get_inputs()[0].type else np.float32
    dict_path = os.path.join(os.environ.get("MDD_DATA_DIR", os.path.dirname(CKPT_PATH)), "dict.ltr.txt")

    class OrtDec:                             # analyze() 가 읽는 속성만 (model 은 안 쓴다)
        d = Dictionary.load(dict_path)
        blank = 0                             # task.blank_symbol = <s>
        normalize = True                      # 학습 설정과 같다
        device = ONNX_DEVICE

    def ort_forward(dec, wavs):               # GF.forward_lprobs 와 같은 (lp, Ls) 를 돌려준다
        w = wavs[0]                           # 서빙은 발화 1건씩이다
        x = (w - w.mean()) / np.sqrt(w.var() + 1e-5)     # normalize=True 의 layer_norm 과 같다
        lp = sess.run(None, {"source": x[None, :].astype(in_dt)})[0].astype(np.float32)
        return torch.from_numpy(lp), [lp.shape[1]]

    GF.forward_lprobs = ort_forward
    S["dec"] = OrtDec()
    S["backend"] = {"kind": "onnx", "path": ONNX_PATH, "device": ONNX_DEVICE,
                    "threads": ONNX_THREADS, "mib": round(os.path.getsize(ONNX_PATH) / 2 ** 20)}


def load():
    t0 = time.time()
    import g2p
    MD.S["ri"], MD.S["ro"] = g2p.readRules(g2p.ver_info[0], MD.RULEBOOK)
    if ONNX_PATH:
        _load_onnx()
        # ONNX 세션은 스레드 안전하다 — 직렬화할 이유가 없다. 워커 프로세스 없이도 동시 처리된다.
        S["gate"] = __import__("contextlib").nullcontext()
    else:
        import mdd.mdd_eval as _me
        _me.CKPT = CKPT_PATH                  # Decoder() 가 이 전역을 읽는다
        S["dec"] = Decoder()
        S["backend"] = {"kind": "torch", "path": CKPT_PATH, "device": S["dec"].device}
        S["gate"] = S["lock"]                 # 단일 GPU 모델 — 직렬화
    S["bundle"] = joblib.load(MODELS_PATH)
    print(f"[load] {S['backend']} + {os.path.basename(MODELS_PATH)} · "
          f"특징 {len(S['bundle']['feature_names'])}개 ({time.time()-t0:.0f}s)", flush=True)


@torch.inference_mode()
def analyze(wave, canon_toks):
    """MDD 진단 + 점수 특징을 **같은 forward 한 번**으로 뽑는다."""
    dec = S["dec"]
    lp, Ls = GF.forward_lprobs(dec, [wave])
    l = lp[0, :Ls[0]].numpy()
    dur = len(wave) / SR
    frame_sec = dur / max(1, l.shape[0])

    y = [dec.d.index(t) for t in canon_toks]
    if dec.d.unk() in y:
        raise ValueError("표준발음에 사전 밖 기호가 있습니다")

    g = l.argmax(-1)
    import itertools
    pred_idx = [int(v) for v, _ in itertools.groupby(g) if v != dec.blank]
    pred = [dec.d[v] for v in pred_idx]
    pred = [p for p in pred if p in MD.JAMO]
    per = editdistance.eval(pred_idx, y) / max(1, len(y))

    segs = GF.viterbi_align(l, y, blank=dec.blank)
    if segs is None:
        gops = [-10.0] * len(y)
    else:
        gops = []
        for si, fs in enumerate(segs):
            if not fs:
                gops.append(-10.0); continue
            fa = np.asarray(fs)
            gops.append(float((l[fa, y[si]] - l[fa].max(-1)).mean()))
    ga = np.asarray(gops)

    blanks = (g == dec.blank)
    spans = [(fs[0] * frame_sec, (fs[-1] + 1) * frame_sec) for fs in (segs or []) if fs]
    if len(spans) >= 2:
        lead_sil = spans[0][0]; trail_sil = max(0.0, dur - spans[-1][1])
        speech_span = max(0.1, spans[-1][1] - spans[0][0])
        gaps = [b0 - e0 for (_, e0), (b0, _) in zip(spans, spans[1:])]
        pauses = [gp for gp in gaps if gp >= GF.PAUSE_SEC]
    else:
        lead_sil = trail_sil = 0.0; speech_span = max(0.1, dur); pauses = []
    pause_tot = float(sum(pauses)); speak = max(0.1, speech_span - pause_tot)

    tl = torch.from_numpy(l).unsqueeze(1)
    nll = float(F.ctc_loss(tl, torch.tensor([y]), torch.tensor([l.shape[0]]),
                           torch.tensor([len(y)]), blank=dec.blank,
                           reduction="sum", zero_infinity=True)) / max(1, len(y))

    feat = {
        "gop": float(ga.mean()), "gop_p10": float(np.percentile(ga, 10)),
        "gop_p25": float(np.percentile(ga, 25)), "gop_med": float(np.median(ga)),
        "gop_min": float(ga.min()), "gop_std": float(ga.std()),
        "gop_bad": float((ga < -2.0).mean()),
        "neg_nll": -nll, "neg_per": -per, "dur": dur, "n_canon": len(y),
        "rate": len(y) / max(0.1, dur), "speech_rate": len(y) / speech_span,
        "artic_rate": len(y) / speak, "blank_frac": float(blanks.mean()),
        "lead_sil": lead_sil, "trail_sil": trail_sil, "speech_span": speech_span,
        "n_pause": len(pauses), "pause_tot": pause_tot,
        "pause_mean": float(np.mean(pauses)) if pauses else 0.0,
        "max_pause": float(max(pauses)) if pauses else 0.0,
        "pause_rate": len(pauses) / speech_span, "pause_frac": pause_tot / speech_span,
    }
    feat.update(GF.align_feats(canon_toks, pred))
    spans_out = [{"start": round(s0, 3), "end": round(e0, 3)} for s0, e0 in spans]
    return pred, feat, gops, spans_out


import re
_PAREN = re.compile(r"[（(][^）)]*[）)]")


def read_text(sentence):
    """낭독 대상만 남긴다. AIHub 프롬프트는 괄호 안에 '지시문'을 넣는데
    (예: "창문이 열렸네요. (닫아 주세요.)") 화자는 그 부분을 읽지 않는다.
    그대로 표준발음에 넣으면 멀쩡한 발화가 탈락 9개로 잡힌다 — 실측 확인."""
    t = _PAREN.sub(" ", sentence)
    return " ".join(t.split())


def diagnose_ops(canon, pred, suppress=True):
    """정렬 결과를 UI 가 그대로 그릴 수 있는 형태로.

    suppress=True 면 삭제형(모델이 그 자리 음소를 내지 못함)을 **오류 지적에서 빼고**
    `not_recognized` 로 따로 돌려준다. 그건 발음 오류가 아니라 인식 실패다.
    실측(2026-08-03, test 178,049발화): 레슨(30발화)당 억울한 지적 4.82 → 2.97회,
    억울한 지적이 0회인 레슨 16.5% → 23.3%. 대가는 정확 포착 3.97 → 3.31개로,
    교환비가 2.8배 유리하다. 근거 `outputs/suppress_tag_all_lar.json`.

    근본 원인(287문장 암기)은 이걸로 고쳐지지 않는다 — 출력 해석을 바로잡는 것뿐이다.
    자세한 건 `src/mdd/suppress.py` 와 `results/08_문장암기_일반화.md`.
    """
    from mdd.suppress import classify_ops, SuppressPolicy

    ops = MD._nw(canon, pred)
    n = {"sub": 0, "del": 0, "ins": 0}
    for op in ops:
        if op["kind"] != "match":
            n[op["kind"]] += 1

    pol = SuppressPolicy(hide_deletions=suppress, hide_insertions=suppress)
    shown_ops, hidden_ops = classify_ops(ops, pol)

    def render(op):
        tpos = op["tpos"]
        is_coda = False
        if op["kind"] != "ins" and tpos < len(canon) and canon[tpos] not in MD.VOWELS:
            is_coda = (canon[tpos] == "ㅇ") or not (tpos + 1 < len(canon)
                                                    and canon[tpos + 1] in MD.VOWELS)
        return {"pos": tpos, "kind": op["kind"],
                "type": MD.ctype(op.get("target") or "", op.get("actual") or "",
                                 op["kind"], is_coda),
                "target": op.get("target") or "∅", "actual": op.get("actual") or "∅"}

    errors = [render(o) for o in shown_ops]
    not_recognized = [{**render(o), "reason": o.get("reason")} for o in hidden_ops]
    hit = len({e["pos"] for e in errors if e["kind"] != "ins"})
    return errors, n, round(1.0 - hit / max(1, len(canon)), 3), not_recognized


def session_scores(sid):
    """세션에 쌓인 발화들로 점수 3종. 발화가 적으면 점수를 내지 않는다."""
    rows = S["sessions"].get(sid) or []
    b = S["bundle"]; names = b["feature_names"]
    out = {"n_utts": len(rows), "min_utts": MIN_UTTS, "reliable_at": GOOD_UTTS}
    if len(rows) < MIN_UTTS:
        out["status"] = "pending"
        out["message"] = f"{MIN_UTTS - len(rows)}문장 더 필요합니다 (권장 {GOOD_UTTS}문장)"
        out["scores"] = None
        return out
    X = np.array([[r["feat"][k] for k in names] for r in rows])
    scores = {}
    for dim, meta in b["metrics"].items():
        v = float(np.mean(b["models"][dim].predict(X)))
        scores[meta["name"]] = {"score": round(min(5.0, max(1.0, v)), 2),
                                "key": dim, "pcc": meta["pcc_spk"]}
    out["status"] = "ok" if len(rows) >= GOOD_UTTS else "provisional"
    out["confidence"] = "높음" if len(rows) >= GOOD_UTTS else "낮음 (문장 수 부족)"
    out["scores"] = scores
    # 점수의 근거 — 팩과 달리 왜 그 점수인지 말할 수 있다
    agg = lambda k: float(np.mean([r["feat"][k] for r in rows]))
    out["evidence"] = {
        "평균_쉼_횟수": round(agg("n_pause"), 2), "평균_최장쉼_초": round(agg("max_pause"), 2),
        "조음속도_음소per초": round(agg("artic_rate"), 2),
        "음소오류율": round(-agg("neg_per"), 3),
        "초성오류율": round(agg("r_onset"), 3), "종성오류율": round(agg("r_coda"), 3),
        "모음오류율": round(agg("r_vowel"), 3), "오류_집중도": round(agg("err_clump"), 3),
    }
    return out


def make_app():
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException, APIRouter
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="발음진단 통합 API (MDD + 점수 3종)")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])
    r = APIRouter()

    from fastapi.responses import FileResponse

    HERE = os.path.dirname(os.path.abspath(__file__))
    # 실제 AIHub 발화 샘플. compare 데모와 같은 파일을 쓴다.
    # wav 경로에 "$SPEECH_AI_DATA" 가 문자열로 들어 있어 확장이 필요하다.
    SAMPLES = []
    _sj = os.path.join(HERE, "..", "model-compare", "compare_samples.json")
    if os.path.exists(_sj):
        for _s in json.load(open(_sj, encoding="utf-8")):
            _s["wav"] = os.path.expandvars(_s.get("wav", ""))
            if os.path.exists(_s["wav"]):
                SAMPLES.append(_s)

    @r.get("/", response_class=HTMLResponse)
    def index():
        """데모 UI. API 만 쓰려면 이 경로를 무시하면 된다."""
        return open(os.path.join(HERE, "mdd_full.html"), encoding="utf-8").read()

    @r.get("/samples")
    def samples():
        return [{"i": i, "prompt": s["prompt"], "task": s.get("task", ""),
                 "note": s.get("note", ""), "dur": s.get("dur"),
                 "n_err": s.get("n_err", 0)} for i, s in enumerate(SAMPLES)]

    @r.get("/sample_audio/{i}")
    def sample_audio(i: int):
        if not 0 <= i < len(SAMPLES):
            raise HTTPException(404, "no such sample")
        return FileResponse(SAMPLES[i]["wav"], media_type="audio/wav")

    @r.post("/diagnose_sample")
    def diagnose_sample(i: int = Form(...), session_id: str = Form(None)):
        """샘플 발화를 세션에 넣는다. 점수는 MIN_UTTS(5)부터 나오므로
        UI 가 이걸 여러 번 호출해 한 번에 세션을 채울 수 있게 한다."""
        if not S["dec"]:
            raise HTTPException(503, "loading")
        if not 0 <= i < len(SAMPLES):
            raise HTTPException(404, "no such sample")
        s = SAMPLES[i]
        import wave as _w
        with _w.open(s["wav"], "rb") as f:
            arr = np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float32) / 32768.0
        return _run(arr, s["prompt"], session_id, sample=s)

    @r.get("/health")
    def health():
        b = S["bundle"]
        return {"status": "ok" if S["dec"] else "loading",
                # 어느 백엔드로 떠 있는지 — ONNX 로 바꿔 띄웠는데 모르고 수치를 인용하면 곤란하다
                "backend": S["backend"],
                # 하드코딩하면 체크포인트를 바꿔도 안 따라온다 (실제로 그랬다)
                "ckpt": "/".join(CKPT_PATH.split("/")[-2:]),
                "mdd_f1_weighted": MDD_F1.get(CKPT_PATH.split("/")[-2]),
                "score_pcc": {m["name"]: m["pcc_spk"] for m in (b["metrics"].values() if b else [])},
                "n_features": len(b["feature_names"]) if b else 0,
                "note": "점수는 세션 단위. 발화 1개로는 신뢰 불가(PCC 0.43)"}

    @r.post("/diagnose")
    # 동기 함수다 — FastAPI 가 스레드풀에서 돌린다. async 로 두면 추론하는 동안
    # 이벤트 루프가 멈춰서 다른 요청을 받지도 못한다(업로드·/health 까지 대기).
    def diagnose(audio: UploadFile = File(...), sentence: str = Form(...),
                 session_id: str = Form(None)):
        if not S["dec"]:
            raise HTTPException(503, "loading")
        sentence = (sentence or "").strip()
        if not sentence or len(sentence) > 200:
            raise HTTPException(400, "sentence required (<=200 chars)")
        try:
            w = MD.decode_audio(audio.file.read())
        except ValueError:
            raise HTTPException(400, "could not decode audio")
        return _run(w, sentence, session_id)

    def _run(w, sentence, session_id, sample=None):
        """업로드·녹음·내장샘플이 공유하는 진단 본체."""
        sentence = (sentence or "").strip()
        dur = len(w) / SR
        if dur < 0.3 or float(np.sqrt((w ** 2).mean())) < 1.5e-3:
            return {"no_speech": True, "duration_sec": round(dur, 2)}

        read = read_text(sentence)
        if not read:
            raise HTTPException(400, "낭독할 내용이 없습니다 (괄호 지시문만 있음)")
        canon, _ = MD.canonical(read)
        if not canon:
            raise HTTPException(400, "표준발음 생성 실패 (숫자·영문 확인)")

        t0 = time.time()
        with S["gate"]:                       # torch=직렬화(단일 GPU) · onnx=동시 실행
            pred, feat, gops, spans = analyze(w[: 30 * SR], canon)
        errors, n, match_rate, not_recognized = diagnose_ops(canon, pred)

        res = {
            "no_speech": False, "sentence": sentence, "read_text": read,
            "duration_sec": round(dur, 2),
            "took_sec": round(time.time() - t0, 3),
            "canonical": {"jamo": canon, "phones": jamo_to_phones(canon)},
            "recognized": {"jamo": pred, "phones": jamo_to_phones(pred) if pred else []},
            # errors 는 사용자에게 지적으로 보여줄 것만 담는다. 삭제·삽입은
            # not_recognized 로 분리한다 — "틀렸다"가 아니라 "못 알아들었다"이다.
            # n_sub/n_del/n_ins 는 억제 전 원본 개수라 기존 소비자와 호환된다.
            "mdd": {"errors": errors, "n_sub": n["sub"], "n_del": n["del"],
                    "n_ins": n["ins"], "match_rate": match_rate,
                    "not_recognized": not_recognized,
                    "suppression": "deletions+insertions hidden from user-facing errors"},
            "per_phone": [{"i": i, "phone": canon[i], "gop": round(gops[i], 2),
                           **({"span": spans[i]} if i < len(spans) else {})}
                          for i in range(len(canon))],
        }
        if session_id:
            S["sessions"][session_id].append({"feat": feat, "sentence": sentence})
            res["session"] = session_scores(session_id)
        else:
            res["session"] = {"status": "no_session",
                              "message": "session_id 를 보내면 문장이 누적되어 점수가 산출됩니다"}
        if sample:
            res["sample"] = {k: sample.get(k) for k in ("task", "note", "n_err")}
        return res

    @r.get("/session/{sid}")
    def get_session(sid: str):
        if sid not in S["sessions"]:
            raise HTTPException(404, "no such session")
        out = session_scores(sid)
        out["sentences"] = [r["sentence"] for r in S["sessions"][sid]]
        return out

    @r.delete("/session/{sid}")
    def del_session(sid: str):
        S["sessions"].pop(sid, None)
        return {"deleted": sid}

    app.include_router(r)
    app.include_router(r, prefix="/api")      # funnel --set-path 는 프리픽스를 벗기지 않는다
    return app


def make_server_app():
    """uvicorn --workers 용 팩토리 — 워커 프로세스마다 모델을 따로 올린다."""
    load()
    return make_app()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8903)
    # 한 프로세스로는 동시 요청이 GIL 에 막힌다. 인코더(ONNX/torch)는 GIL 을 놓지만
    # 강제정렬·특징·점수는 파이썬/numpy 라 직렬화된다 — 실측 단일 프로세스 2.7 req/s 대 8프로세스 5.25 req/s.
    # CPU ONNX 서빙 권장값: --workers (코어수/2) · MDD_SERVE_THREADS=2
    ap.add_argument("--workers", type=int, default=1, help="워커 프로세스 수 (CPU ONNX 서빙은 2 이상 권장)")
    a = ap.parse_args()
    import uvicorn
    if a.workers > 1:
        # ⚠️ 세션(`S["sessions"]`)은 프로세스 메모리에 있다. 워커가 여럿이면 같은 session_id 라도
        #    매번 다른 워커에 붙어 점수가 갈라진다. 같은 세션을 같은 워커로 보내거나(스티키),
        #    공용 저장소로 옮기기 전까지 --workers 는 **단발 진단 전용**이다.
        print(f"[warn] workers={a.workers} — /session 점수는 워커별로 갈라진다(단발 /diagnose 전용)", flush=True)
        _here = os.path.dirname(os.path.abspath(__file__))
        _sys.path.insert(0, _here)
        os.environ["PYTHONPATH"] = _here + os.pathsep + os.environ.get("PYTHONPATH", "")
        uvicorn.run("mdd_infer_api:make_server_app", factory=True,
                    host="0.0.0.0", port=a.port, workers=a.workers)
    else:
        load()
        uvicorn.run(make_app(), host="0.0.0.0", port=a.port)
