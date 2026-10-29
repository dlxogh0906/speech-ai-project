# -*- coding: utf-8 -*-
"""통합 발음 평가 데모 — 한 페이지에서 진단 · 국적 피드백 · 음소별 점수 · 세션 점수.

    POST /speak          audio + sentence + session + country → 자리별 진단 · L1 피드백 · 세션 요약
    POST /sent.eval.kr   SpeechSuper `sent.eval.kr` 호환. 요청 형식 두 가지를 다 받는다 —
                         ① 그쪽 방식: multipart 에 오디오 + **JSON 덩어리**(connect/start; 필드 이름 무관)
                         ② 간편 방식: audio + refText + tokenId
                         말소리가 모자라면 400 low_speech — 최소 말한 시간은 refText 음절 수에 맞춘다(2026-10-18,
                         음절당 0.08초 · 0.2~1.0초). 본문에 speech_s · min_speech_s · voiced_ratio · min_voiced_ratio 를 같이 준다.
    POST /word.judge     발음 챌린지용 단어 판정(2026-10-03 웹 요청). audio + targets(후보 단어 1~4개) →
                         후보별 표준발음·들린 자모·불일치 목록. 문장 점수·유창성·저장 없음. 유성 0.12초만 넘으면 받는다.
                         맞음·틀림 판정은 주지 않는다 — 허용 불일치 수는 웹이 정한다.
    GET  /session    현재 세션 요약(새로고침 후 이어보기)
    POST /reset      세션 비우기
    GET  /sentences  연습 문장 목록
    GET  /countries  국적 목록

로직은 전부 라이브러리에 있다 — `mdd.gop_feats_v2`(특징·강제정렬) · `mdd.session`(누적 점수)
· `mdd.l1_feedback`(국적별 교정). 이 파일은 묶어서 HTTP 로 내보내기만 한다.

    PYTHONPATH=/data/speech-ai/tools/hf_py38 HF_HOME=... $PY deploy/pronunciation-demo/app.py \\
        --hf-ckpt $SPEECH_AI_DATA/checkpoints/w2vbert_stage2/step_38000 \\
        --scorer $MDD_OUTPUTS/score_v2_w2vbert/score_models_v2_w2vbert.joblib \\
        --norms $MDD_OUTPUTS/score_phoneme/norms.json --port 8932

    NPU(Rebellions)로 띄울 때는 `--rbln <.rbln>` 을 더한다 — 모델만 NPU 로 가고 나머지는 같다.
    점수 모델은 sklearn 버전이 달라도 되게 `.npz`(mdd.hgbr_np) 를 준다. 실행 예: NPU 서버 `/root/pron_serve/start_pron.sh`,
    근거·수치: `tasks/pronunciation-assessment/results/17_NPU_이식_검토.md`.

⚠️ 사내 확인용이다. `--token` 을 주면 API 호출에 열쇠가 필요하다(헤더 `Authorization: Bearer …` ·
   `X-API-Key` · 폼 필드 `token` · 쿼리 `?token=` 아무거나). `--host 0.0.0.0` 으로 사내망에 열 수 있다.
   AIHub 이용정책(2026-09-27 확인): **원본 데이터(오디오·라벨)는 외부에 내보내면 안 되고**, 이 데이터로 학습한 모델의 결과를
   주는 서비스는 한국지능정보사회진흥원·AI 허브 출처를 밝히면 공개할 수 있다. 인터넷에 열 때는 토큰과 `--ask-token`(페이지에서 토큰 입력) 또는 `--api-only`(페이지 끔)를 켤 것
   (토큰을 켜면 데모 페이지가 토큰을 HTML 에 넣어 보낸다). 세션은 메모리에만 있다.
   마이크는 보안 컨텍스트에서만 열린다 — http://127.0.0.1:<port> 나 사내 HTTPS.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import FFMPEG, G2P_TOOLS, MDD_ANALYSIS, MDD_OUTPUTS, ensure_kog2p
from shared.request_capture import capture_app, capture_context

import argparse, collections, io, json, os, re, shutil, subprocess, time, wave

import numpy as np

SR = 16000
S = {}
SESS = collections.defaultdict(list)          # session_id → [발화 기록]

# 녹음 실패 거르기(실측 2026-09-23): 말소리 비율 20%·1.6초였던 녹음은 자모 2개만 인식됐다.
# 정상 녹음은 36~61% 였다. 이 아래면 채점하지 않고 다시 녹음하도록 돌려보낸다.
MIN_VOICED_RATIO = 0.30
MIN_VOICED_SEC = 1.0
# 최소 말한 시간은 기준 문장 길이에 맞춘다(2026-10-18). 1.0초 고정이면 짧은 문장(네 · 고마워요 · 감사합니다)은 정상 발화도
# 늘 걸렸다(앱 복습 14문장 중 12 거절 · 원어민 대화 1~3음절 86~96% 거절). 음절당 0.08초 · 하한 0.2초 · 상한 1.0초 —
# 긴 문장(13음절 이상)은 예전과 같다. 실측: 원어민(Kspon) 1~12음절 잘못 거절 1~5% · 새로 받게 되는 학습자 단어 1,190개의
# 화면 지적은 헛지적 자리 1.7% · 정밀도 59%(지금 받는 문장 1.5% · 40%). 소리 비율 조건은 그대로.
MIN_VOICED_SEC_PER_SYL = 0.08
MIN_VOICED_SEC_FLOOR = 0.2
# 입력 거르기 셋(2026-10-19 서비스 점검) — 실행 인자로 켠다(안 주면 예전과 같다). 셋 다 400 을 새로 내므로 앱이
# 코드(no_match · too_long · no_korean)를 처리한 뒤에 켠다. --no-korean-reject : refText 에 한글이 없으면 no_korean.
#   --nomatch-reject R : 기준 문장의 소리 중 들은 소리와 다른 비율이 R 을 넘으면 no_match. 잡음·음악·백색 잡음만 녹음해도
#                        45~62점이 나왔다. 그런 녹음은 0.94~1.00 · 학습자 실제 녹음은 99% 가 0.60 이하(0.9 초과 0.2%).
#   --too-long-reject  : 말소리(앞뒤 무음을 뺀 길이)가 MAX_SPEECH_SEC 를 넘으면 too_long. 예전엔 앞 20초만 채점하고
#                        나머지를 전부 '빠짐'으로 세 42초 녹음이 62점이 됐다.
MAX_SPEECH_SEC = 20.0
# 단어 판정(/word.judge)용 — 단어 발화의 유성 구간 중앙값 0.63초, 한 글자 단어 0.39초(results/20). 문장 기준(1.0초)이면
# 단어의 95% 가 거절된다. 0.2초도 한 글자 단어 7.4% 를 거절해서 0.12초(0.2%)로 둔다. 비율 조건은 없다(`mdd.word_judge` 와 같다).
WORD_MIN_VOICED_SEC = 0.12
WORD_MAX_SEC = 6.0           # 요청은 0.12~4초. 여유를 두고 그 뒤는 자른다(192 버킷 3.84초 → 그 이상이면 448 버킷으로 느려진다)
WORD_MAX_TARGETS = 4

API_TOKEN = None         # --token 으로 켠다. 없으면 인증 없이 받는다(기존 동작)
# --ask-token: 페이지에 진짜 토큰 대신 넣는 스크립트. 토큰을 한 번 입력받아 그 브라우저의 localStorage 에만 둔다
ASK_TOKEN_JS = ("<script>(function(){var K='pron_api_token',t='';"
                "try{if(location.search.indexOf('reset_token')>=0)localStorage.removeItem(K);t=localStorage.getItem(K)||'';}catch(e){}"
                "if(!t){t=(prompt('발음 평가 API 토큰을 입력하세요. 이 브라우저에만 저장됩니다.')||'').trim();"
                "try{if(t)localStorage.setItem(K,t);}catch(e){}}window.API_TOKEN=t;})();</script>")
SAVE_DIR = None          # --save-dir 로 켠다. 켜면 녹음을 16k WAV + 메타로 남겨 기준표 재작성에 쓴다

SENTENCES = ["안녕하세요", "오늘 날씨가 참 좋네요", "값이 너무 비싸서 못 샀어요",
             "학교에서 친구를 만났어요", "밥을 먹고 커피를 마셨습니다",
             "신라면을 끓여 먹었어요", "독립문 앞에서 사진을 찍었어요",
             "저는 한국어를 공부하고 있어요", "꽃이 피었습니다", "무릎을 다쳐서 병원에 갔어요"]


def load_all(a):
    ensure_kog2p()
    argv, _sys.argv = _sys.argv, _sys.argv[:1]            # g2p.py 는 import 시점에 argv 를 파싱한다
    try:
        from g2p import readRules, graph2prono, ver_info
    finally:
        _sys.argv = argv
    from mdd.jamo_to_kog2p import phones_to_jamo, jamo_to_phones, INV, VOWEL
    from mdd.gop_feats_v2 import HFDecoder, forward_lprobs, utt_features
    from mdd.session import SessionScorer, phone_context
    from mdd.l1_feedback import L1Feedback
    from mdd.speechsuper import SpeechSuperFormatter
    from mdd.vad import trim as trim_silence
    from mdd.pipeline import ctype

    if getattr(a, "rbln", None):                   # NPU: 컴파일된 .rbln 으로 추론(가중치는 안 읽는다)
        from mdd.npu_w2vbert import RBLNDecoder
        dec = RBLNDecoder(a.hf_ckpt, a.rbln, device=a.rbln_device)
    else:
        dec = HFDecoder(a.hf_ckpt)
    S.update(g2p=graph2prono, rules=readRules(ver_info[0], str(G2P_TOOLS / "KoG2P" / "rulebook.txt")),
             p2j=phones_to_jamo, j2p=jamo_to_phones, inv=INV, vowels=set(VOWEL),
             dec=dec, fwd=forward_lprobs, feats=utt_features, ctx=phone_context,
             scorer=SessionScorer(a.scorer, a.norms))
    S["trim"] = trim_silence
    S["nomatch"] = getattr(a, "nomatch_reject", 0.0) or 0.0
    S["too_long"] = getattr(a, "too_long_reject", False)
    S["no_korean"] = getattr(a, "no_korean_reject", False)
    if S["nomatch"] or S["too_long"] or S["no_korean"]:
        print("[load] 입력 거르기 · 말소리 못 알아들음 %s · 너무 긴 녹음 %s · 한글 없는 문장 %s"
              % ("%.2f 초과" % S["nomatch"] if S["nomatch"] else "끔", "%.0f초 초과" % MAX_SPEECH_SEC if S["too_long"] else "끔",
                 "거절" if S["no_korean"] else "끔"), flush=True)
    import mdd.speechsuper as _ss                      # 화면 지적 규칙(2026-10-19) — 인자를 안 주면 예전과 같다
    _ss.DEL_RUN_MIN = max(1, getattr(a, "del_run_min", 1))
    _ss.DEL_SCORE_MAX = getattr(a, "del_score_max", 0) or None
    _ss.LIQUID_NASAL = dict(_ss.LIQUID_NASAL_PAIRS) if getattr(a, "liquid_rule", False) else {}
    _ss.SS_RULE = bool(getattr(a, "ss_rule", False))
    S["sz"] = None
    if getattr(a, "sz_detector", None):
        from mdd.sz_detector import Detector
        S["sz"] = Detector(a.sz_detector)
        print("[load] ㅅㅆ·ㅈㅉㅊ 판별기 켬 (%s)" % a.sz_detector, flush=True)
    _ss.SZ_RULE = S["sz"] is not None
    if _ss.SS_RULE:
        print("[load] ㅅ↔ㅆ 예외 켬 (ㅅ→ㅆ 확률 %.1f · ㅆ→ㅅ 첫소리·모음 뒤만)" % _ss.SS_LOOSE_PROB, flush=True)
    if _ss.DEL_RUN_MIN > 1 or _ss.DEL_SCORE_MAX or _ss.LIQUID_NASAL:
        print("[load] 화면 지적 규칙 · 빠짐은 %d개 이상 이어질 때만 · 빠짐 강제정렬 점수 %s · ㄴ↔ㄹ 예외 %s"
              % (_ss.DEL_RUN_MIN, "%d 미만만" % _ss.DEL_SCORE_MAX if _ss.DEL_SCORE_MAX else "안 봄",
                 "켬" if _ss.LIQUID_NASAL else "끔"), flush=True)
    if getattr(a, "canon_fix", False):                 # 표준발음 보정(2026-10-16) — 끄면 예전과 똑같다
        from mdd.canonical_fix import CanonicalBuilder
        S["cb"] = CanonicalBuilder(canonical)
        S["cf_margin"] = a.canon_fix_margin
        print("[load] 표준발음 보정 켬 (mdd.canonical_fix · 변이 margin %.1f)" % a.canon_fix_margin, flush=True)
    S["ss"] = SpeechSuperFormatter(a.norms, jamo_to_phones=jamo_to_phones, flu_rule_path=a.flu_rule,
                                   utt_comp=a.utt_comp,
                                   phone_context=phone_context, ctype=ctype, calib_path=a.calib,
                                   utt_pct_path=a.utt_pct)
    wp = os.path.join(str(MDD_ANALYSIS), "l1_prior_product", "l1_prior_weights.json")
    ep = os.path.join(str(MDD_ANALYSIS), "l1_prior_product", "nationality_feedback_enriched.json")
    S["l1fb"] = L1Feedback(wp, ep) if os.path.exists(wp) else None
    print(f"[load] 점수모델 {'있음' if S['scorer'].models else '없음'} · "
          f"음소기준 {len(S['scorer'].norms)}묶음 · 보정표 {'있음' if S['ss'].cal else '없음'} · "
          f"문장눈금 {'백분위' if S['ss'].pct else '5점비례'} · "
          f"유창성 {'속도·쉼 규칙' if S['ss'].flu else '사람점수 예측'} · "
          f"L1 {'있음' if S['l1fb'] else '없음'}", flush=True)


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


def decode_audio(raw):
    """녹음 바이트 → 16k mono float32.

    16k mono WAV 는 그대로 읽는다(ffmpeg 프로세스 기동 ~100ms 가 모델 추론 43ms 보다 비싸다).
    그 밖(브라우저 webm/opus 등)은 ffmpeg 을 쓰되 **표준입력**으로 넘겨 임시파일을 없앤다.
    """
    if raw[:4] == b"RIFF":
        try:
            with wave.open(io.BytesIO(raw), "rb") as w:
                if w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2:
                    x = np.frombuffer(w.readframes(w.getnframes()), np.int16)
                    return x.astype(np.float32) / 32768.0
        except Exception:
            pass                                    # 형식이 다르면 아래 ffmpeg 경로로
    exe = shutil.which(FFMPEG) or FFMPEG
    p = subprocess.run([exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1",
                        "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1"],
                       input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0 or not p.stdout:
        raise ValueError(p.stderr.decode("utf-8", "replace").strip()[-300:] or "빈 오디오")
    return np.frombuffer(p.stdout, np.float32)


def save_take(wav, sentence, extra):
    """녹음 원본과 결과를 남긴다(기준표 재작성용). SAVE_DIR 가 없으면 아무것도 안 한다."""
    if not SAVE_DIR:
        return None
    os.makedirs(SAVE_DIR, exist_ok=True)
    stem = time.strftime("%Y%m%d-%H%M%S") + "-%03d" % (int(time.time() * 1000) % 1000)
    with wave.open(os.path.join(SAVE_DIR, stem + ".wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(wav, -1, 1) * 32767).astype(np.int16).tobytes())
    json.dump({"stem": stem, "sentence": sentence, "sec": round(len(wav) / SR, 2),
               "rms": round(float(np.sqrt(np.mean(wav ** 2))), 4), **extra},
              open(os.path.join(SAVE_DIR, stem + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
    return stem


def align(canon, hyp):
    """표준발음 자리마다 모델이 무엇을 냈는지."""
    import edlib
    c, h = "".join(canon), "".join(hyp)
    if not c:
        return []
    if not h:
        return [{"i": i, "canon": x, "pred": None, "j": None, "verdict": "삭제"} for i, x in enumerate(canon)]
    a = edlib.getNiceAlignment(edlib.align(c, h, task="path"), c, h)
    out, i, j = [], 0, 0
    for x, y in zip(a["query_aligned"], a["target_aligned"]):
        if x == "-":
            j += 1
            continue
        out.append({"i": i, "canon": x, "pred": None if y == "-" else y, "j": None if y == "-" else j,
                    "verdict": "일치" if x == y else ("삭제" if y == "-" else "다름")})
        i += 1; j += y != "-"
    return out


def min_voiced_sec(text):
    """기준 문장 → 최소 말한 시간(초). 한글 음절이 없으면 예전 기준(1.0초)."""
    n = len(re.findall(r"[가-힣]", text or ""))
    return MIN_VOICED_SEC if n == 0 else min(MIN_VOICED_SEC, max(MIN_VOICED_SEC_FLOOR, MIN_VOICED_SEC_PER_SYL * n))


def check_audio(wav, text=None):
    """채점할 만한 녹음인가. 아니면 (코드, 메시지, 정보) 를 돌려준다. text = 기준 문장(최소 말한 시간을 정한다)."""
    from mdd.vad import voiced_mask
    m, _ = voiced_mask(wav, SR)
    ratio = float(m.mean()) if m.size else 0.0
    sec = float(m.sum()) * 0.03
    min_sec = min_voiced_sec(text)
    info = {"voiced_ratio": round(ratio, 3), "voiced_sec": round(sec, 2),
            "min_voiced_ratio": MIN_VOICED_RATIO, "min_voiced_sec": round(min_sec, 2)}
    if sec < min_sec or ratio < MIN_VOICED_RATIO:
        return ("low_speech",
                "말소리가 거의 없다 (말한 시간 %.1f초 · 소리 비율 %d%%) — 마이크에 가까이서 또박또박 다시 녹음할 것"
                % (sec, round(ratio * 100)), info)
    return (None, None, info)


def mismatch_ratio(sites):
    """기준 문장 소리 중 들은 소리와 다른(다름 · 삭제) 비율. 소리가 4개 미만이면 None(판단 안 함)."""
    if len(sites) < 4:
        return None
    return sum(1 for x in sites if x.get("verdict") != "일치") / len(sites)


def reject_body(code, msg, vinfo):
    """말소리 검사 400 본문. 예전 칸(error · code · audio)은 그대로 두고 앱 요청 칸(2026-10-18)을 더한다."""
    return {"error": msg, "code": code, "audio": vinfo, "speech_s": vinfo["voiced_sec"], "min_speech_s": vinfo["min_voiced_sec"],
            "voiced_ratio": vinfo["voiced_ratio"], "min_voiced_ratio": vinfo["min_voiced_ratio"]}


def word_mismatches(canon, pred):
    """표준발음 자모열 ↔ 들린 자모열의 불일치 목록(편집 경로 전부: 다름 · 삭제 · 삽입).

    `align()` 은 표준발음 자리만 보고 삽입(모델이 더 낸 자모)을 건너뛴다. 단어 판정은 삽입도 세야
    `n_mismatch` 가 편집거리와 같아진다(`mdd.word_judge` 의 등급 기준과 같은 수).
    i: 표준발음 자리(삽입이면 None) · expected: 기대 자모(삽입이면 None) · heard: 들린 자모(삭제면 None)
    """
    import edlib
    c, h = "".join(canon), "".join(pred)
    if not c:
        return [], 0
    if not h:
        return [{"i": i, "expected": x, "heard": None, "kind": "삭제"} for i, x in enumerate(canon)], len(canon)
    r = edlib.align(c, h, task="path")
    a = edlib.getNiceAlignment(r, c, h)
    out, i = [], 0
    for x, y in zip(a["query_aligned"], a["target_aligned"]):
        if x == "-":
            out.append({"i": None, "expected": None, "heard": y, "kind": "삽입"})
            continue
        if y == "-":
            out.append({"i": i, "expected": x, "heard": None, "kind": "삭제"})
        elif x != y:
            out.append({"i": i, "expected": x, "heard": y, "kind": "다름"})
        i += 1
    return out, int(r["editDistance"])


def word_judge(wav, targets):
    """단어 판정 — 모델 1회 + 후보별 정렬만. 문장 점수·유창성·L1 은 돌리지 않는다(`mdd.word_judge.WordJudge.judge` 와 같은 규칙).

    돌려주는 것: (후보별 결과 목록, 공통 pred_jamo, 단계별 ms). 표준발음을 못 만드는 후보는 그 항목에 error 만 넣는다.
    """
    import itertools
    dec = S["dec"]
    t0 = time.perf_counter()
    canons = []
    for t in targets:
        canon = S["cb"].fixed(t) if S.get("cb") else canonical(t)
        if canon and dec.d.unk() in [dec.d.index(x) for x in canon]:
            canon = None
        canons.append(canon)
    wav, _ = S["trim"](wav, SR)
    t1 = time.perf_counter()
    lp, Ls = S["fwd"](dec, [wav])
    t2 = time.perf_counter()
    g = lp[0, :Ls[0]].numpy().argmax(-1)
    pred = [dec.d[int(v)] for v, _ in itertools.groupby(g) if v != dec.blank]
    pred = [p for p in pred if len(p) == 1 and 0x3131 <= ord(p) <= 0x3163]
    rows = []
    for t, canon in zip(targets, canons):
        if not canon:
            rows.append({"target": t, "error": "표준발음으로 바꿀 수 없는 단어", "code": "unsupported_target"})
            continue
        mm, n = word_mismatches(canon, pred)
        rows.append({"target": t, "canon_jamo": canon, "pred_jamo": pred, "mismatches": mm, "n_mismatch": n})
    t3 = time.perf_counter()
    return rows, pred, {"prep": round((t1 - t0) * 1e3, 1), "forward": round((t2 - t1) * 1e3, 1),
                        "post": round((t3 - t2) * 1e3, 1), "total": round((t3 - t0) * 1e3, 1)}


def parse_targets(form):
    """후보 단어 목록. `targets` 를 여러 번 보내도, 한 필드에 JSON 배열·쉼표·줄바꿈으로 묶어 보내도, `target` 하나만 보내도 받는다.
    공백으로는 나누지 않는다 — 잰말놀이처럼 띄어쓰기가 있는 후보("백박사 박법학박사")를 한 덩어리로 받기 위해서."""
    vals = []
    getlist = getattr(form, "getlist", None)
    for k in ("targets", "targets[]", "target"):
        vals += [v for v in (getlist(k) if getlist else [form.get(k)]) if isinstance(v, str)]
    out = []
    for v in vals:
        v = v.strip()
        if v.startswith("["):
            try:
                out += [str(x) for x in json.loads(v)]
                continue
            except Exception:
                pass
        out += re.split(r"[,\n]+", v)
    seen, uniq = set(), []
    for t in out:
        t = t.strip()
        if t and t not in seen:
            seen.add(t); uniq.append(t)
    return uniq


def analyze(wav, sentence, country):
    """발화 하나 → 진단 + 특징 + 음소별 margin. 세션 누적용 기록도 함께 만든다.

    앞뒤·중간의 빈 구간을 먼저 잘라낸다 — 무음 위에도 음소가 배치돼 점수가 무너진다(`mdd.vad`).
    """
    dec = S["dec"]
    wav, vad = S["trim"](wav, SR)
    canon = canonical(sentence)
    lp, Ls = S["fwd"](dec, [wav])
    l = lp[0, :Ls[0]].numpy()
    if S.get("cb") and canon:                          # 표준발음 보정 — KoG2P 오류 고침 + 단어 경계 허용 변이를 녹음에 맞춰 고름
        from mdd.canonical_fix import choose
        base, alts = S["cb"].variants(sentence)
        if base:
            canon, _used = choose(l, base, alts, dec.d.index, blank=dec.blank, margin=S["cf_margin"])
    y = [dec.d.index(t) for t in canon]
    if canon and dec.d.unk() in y:
        canon, y = [], []
    feat, extra = (S["feats"](dec, l, y, canon, len(wav) / SR) if canon else ({}, {"pred": [], "gops_m": []}))
    if not canon:                                  # 문장 없이 녹음한 경우 — 전사만
        g = l.argmax(-1)
        import itertools
        toks = [int(v) for v, _ in itertools.groupby(g) if v != dec.blank]
        extra = {"pred": [dec.d[t] for t in toks if 0x3131 <= ord(dec.d[t][0]) <= 0x3163], "gops_m": []}
    jamo = [t for t in extra["pred"] if len(t) == 1 and 0x3131 <= ord(t[0]) <= 0x3163]
    sites = align(canon, jamo)
    from mdd.gop_feats_v2 import attach_pred_probs
    attach_pred_probs(dec, l, sites, jamo)             # 들은 자모 확률 — 화면 지적 예외(speechsuper.STRONG_MIN_PROB, 2026-10-18)
    if S.get("sz") and canon and extra.get("segs"):    # ㅅㅆ·ㅈㅉㅊ 판별기(--sz-detector, 2026-10-22) — wav 는 로그확률을 낸 다듬은 녹음
        from mdd.sz_detector import annotate
        annotate(sites, wav, l, extra["segs"], canon, dec.d.index, extra["frame_sec"], S["sz"], blank=dec.blank)
    cph, pph = (S["j2p"](canon) if canon else []), S["j2p"](jamo)
    for s in sites:
        s["canon_ph"] = cph[s["i"]] if len(cph) == len(canon) else None
        s["pred_ph"] = pph[s["j"]] if (s["j"] is not None and len(pph) == len(jamo)) else None
    fb = S["l1fb"].annotate(canon, sites, country or None) if (S["l1fb"] and canon) else None
    gm = extra.get("gops_m") or []
    phone = [(S["ctx"](canon, i, S["vowels"]), gm[i]) for i in range(len(gm))]
    scores = S["scorer"].utt_scores(feat) if feat else None
    context = capture_context.get()
    if context is not None:
        context["pronunciation"] = {"canon": canon, "sites": sites, "margins": gm,
                                     "scores": scores, "vad": vad, "feats": feat,
                                     "pred_jamo": jamo}
    return ({"canon_jamo": canon, "canon_phone": cph, "pred_jamo": jamo, "pred_phone": pph,
             "sites": sites, "feedback": fb, "scores": scores, "vad": vad},
            {"scores": scores, "phone": phone},
            {"canon": canon, "sites": sites, "margins": gm, "scores": scores, "vad": vad,
             "feats": feat})


def parse_request_params(form):
    """SpeechSuper 요청에서 refText·tokenId·userId 를 꺼낸다.

    그쪽은 multipart 안에 connect/start JSON 을 통째로 넣는다. 필드 이름이 구현마다 달라서
    (text·params·data…) **값이 JSON 으로 파싱되고 start/request 를 품고 있으면** 그걸로 본다.
    """
    out = {"refText": "", "tokenId": "", "userId": "guest", "raw": None}
    for key, val in form.items():
        if not isinstance(val, str) or "{" not in val:
            continue
        try:
            j = json.loads(val)
        except Exception:
            continue
        if not isinstance(j, dict):
            continue
        req = (((j.get("start") or {}).get("param") or {}).get("request")
               or (j.get("param") or {}).get("request") or j.get("request"))
        app_ = (((j.get("start") or {}).get("param") or {}).get("app") or {})
        if req or app_:
            out["raw"] = key
            out["refText"] = (req or {}).get("refText", "") or out["refText"]
            out["tokenId"] = (req or {}).get("tokenId", "") or out["tokenId"]
            out["userId"] = app_.get("userId") or out["userId"]
            break
    for k, f in (("refText", "refText"), ("tokenId", "tokenId"), ("userId", "userId")):
        v = form.get(f)
        if isinstance(v, str) and v and not out[k]:
            out[k] = v
    return out


def check_token(request, form=None):
    """열쇠 확인. 토큰을 안 켰으면 항상 통과."""
    if not API_TOKEN:
        return True
    got = (request.headers.get("authorization", "").replace("Bearer ", "").strip()
           or request.headers.get("x-api-key", "")
           or request.query_params.get("token", "")
           or ((form.get("token") or "") if form else ""))
    return got == API_TOKEN


def make_app(static_dir, api_only=False, ask_token=False):
    from fastapi import FastAPI, File, Form, Request, UploadFile
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
    app = FastAPI(title="pronunciation-demo")

    @app.get("/health")
    def health():
        return {"ok": True, "device": S["dec"].device, "sessions": len(SESS)}

    @app.get("/sentences")
    def sentences():
        return {"sentences": SENTENCES}

    @app.get("/countries")
    def countries():
        return {"countries": S["l1fb"].countries() if S["l1fb"] else []}

    @app.get("/session")
    def session(request: Request, id: str = "default"):
        if not check_token(request):               # 세션 기록도 토큰으로 막는다(인터넷 공개 대비)
            return JSONResponse({"error": "인증 실패", "code": "unauthorized"}, status_code=401)
        return {"summary": S["scorer"].session(SESS[id]), "history": [h.get("brief") for h in SESS[id]]}

    @app.post("/reset")
    def reset(request: Request, id: str = Form("default"), token: str = Form("")):
        if not (check_token(request) or token == API_TOKEN):   # 데모 페이지는 폼 필드 token 으로 보낸다
            return JSONResponse({"error": "인증 실패", "code": "unauthorized"}, status_code=401)
        SESS.pop(id, None)
        return {"ok": True}

    @app.post("/speak")
    async def speak(request: Request, audio: UploadFile = File(...), sentence: str = Form(""),
                    country: str = Form(""), id: str = Form("default"), token: str = Form("")):
        if not (check_token(request) or token == API_TOKEN):
            return JSONResponse({"error": "인증 실패", "code": "unauthorized"}, status_code=401)
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
        code, msg, vinfo = check_audio(wav[: SR * 20], sentence)
        if code:
            return JSONResponse(reject_body(code, msg, vinfo), status_code=400)
        out, rec, raw_out = analyze(wav[: SR * 20], sentence, country)
        stem = save_take(wav[: SR * 20], sentence, {"source": "speak", "country": country,
                                                    "scores": out["scores"],
                                                    "n_err": sum(1 for x in out["sites"] if x["verdict"] != "일치"),
                                                    "n_site": len(out["sites"])})
        n_err = sum(1 for s in out["sites"] if s["verdict"] != "일치")
        rec["brief"] = {"sentence": sentence, "n_site": len(out["sites"]), "n_err": n_err,
                        "scores": out["scores"]}
        SESS[id].append(rec)
        out.update(sec=round(len(wav) / SR, 2), rms=round(rms, 4), ms=int((time.time() - t0) * 1000),
                   summary=S["scorer"].session(SESS[id]))
        return out

    @app.post("/sent.eval.kr")
    async def sent_eval_kr(request: Request):
        """SpeechSuper 호환. refText 는 채점 기준 문장, tokenId 는 그대로 recordId 로 돌려준다."""
        form = await request.form()
        if not check_token(request, form):
            return JSONResponse({"error": "인증 실패 — 토큰이 없거나 틀리다", "code": "unauthorized"}, status_code=401)
        up = form.get("audio")
        if up is None:                       # 파일 필드 이름이 다를 수 있다 — 첫 업로드 파일을 쓴다
            up = next((v for v in form.values() if hasattr(v, "read")), None)
        if up is None:
            return JSONResponse({"error": "오디오 파일이 없다", "code": "no_audio"}, status_code=400)
        P = parse_request_params(form)
        refText, tokenId, userId = P["refText"], P["tokenId"], P["userId"]
        raw = await up.read()
        try:
            wav = decode_audio(raw)
        except ValueError as e:
            return JSONResponse({"error": f"오디오를 읽지 못했다: {e}", "code": "decode_failed"}, status_code=400)
        if len(wav) < SR * 0.3:
            return JSONResponse({"error": "녹음이 너무 짧다 (0.3초 미만)", "code": "too_short"}, status_code=400)
        if not refText.strip():
            return JSONResponse({"error": "refText 가 비어 있다", "code": "no_reftext"}, status_code=400)
        if S.get("no_korean") and not re.search(r"[가-힣]", refText):   # 예전엔 200 에 점수 칸이 비어 나갔다(2026-10-19)
            return JSONResponse({"error": "refText 에 한글이 없다 — 한국어 문장을 보낼 것", "code": "no_korean"}, status_code=400)
        if S.get("too_long") and len(wav) > SR * MAX_SPEECH_SEC:
            w2, _ = S["trim"](wav, SR)
            if len(w2) > SR * MAX_SPEECH_SEC:
                return JSONResponse({"error": "녹음이 너무 길다 (말한 부분 %.0f초 · %.0f초까지) — 문장을 나눠 녹음할 것"
                                              % (len(w2) / SR, MAX_SPEECH_SEC),
                                     "code": "too_long", "speech_s": round(len(w2) / SR, 1), "max_speech_s": MAX_SPEECH_SEC},
                                    status_code=400)
        code, msg, vinfo = check_audio(wav[: SR * 20], refText)
        if code:
            return JSONResponse(reject_body(code, msg, vinfo), status_code=400)
        t0 = time.time()
        _, _, raw_out = analyze(wav[: SR * 20], refText, "")
        mm = mismatch_ratio(raw_out["sites"])
        if S.get("nomatch") and mm is not None and mm > S["nomatch"]:
            return JSONResponse({"error": "말소리를 알아듣지 못했다 — 기준 문장을 또박또박 다시 읽을 것", "code": "no_match",
                                 "mismatch": round(mm, 2), "max_mismatch": S["nomatch"]}, status_code=400)
        save_take(wav[: SR * 20], refText, {"source": "sent.eval.kr", "scores": raw_out["scores"]})
        body = S["ss"].build(refText, raw_out["canon"], raw_out["sites"], raw_out["margins"],
                             utt_scores=raw_out["scores"], record_id=tokenId or None, vowels=S["vowels"],
                             feats=raw_out.get("feats"))
        body["_meta"]["ms"] = int((time.time() - t0) * 1000)
        body["_meta"]["audio"] = raw_out.get("vad")
        body["_meta"]["userId"] = userId
        body["_meta"]["request_style"] = ("speechsuper:%s" % P["raw"]) if P["raw"] else "simple"
        return body

    @app.post("/word.judge")
    async def word_judge_route(request: Request):
        """발음 챌린지 단어 판정. multipart: audio(16k mono 16bit WAV 권장) + targets(후보 단어 1~4개).

        응답: {"targets": [{target, canon_jamo, pred_jamo, mismatches: [{i, expected, heard, kind}], n_mismatch}], "pred_jamo",
               "voiced_sec", "sec", "truncated"(6초 넘어 앞 6초만 봤으면 true), "ms": {prep, forward, post, total, request}}
        400 too_short · low_speech · no_audio · decode_failed · no_targets · too_many_targets  /  401 unauthorized.
        녹음은 저장하지 않는다. 등급(PERFECT/MISS)은 주지 않는다 — 허용 불일치 수는 웹이 정한다.
        """
        tr = time.perf_counter()
        form = await request.form()
        if not check_token(request, form):
            return JSONResponse({"error": "인증 실패 — 토큰이 없거나 틀리다", "code": "unauthorized"}, status_code=401)
        up = form.get("audio")
        if up is None or not hasattr(up, "read"):
            up = next((v for v in form.values() if hasattr(v, "read")), None)
        if up is None:
            return JSONResponse({"error": "오디오 파일이 없다", "code": "no_audio"}, status_code=400)
        targets = parse_targets(form)
        if not targets:
            return JSONResponse({"error": "targets 가 비어 있다", "code": "no_targets"}, status_code=400)
        if len(targets) > WORD_MAX_TARGETS:
            return JSONResponse({"error": f"후보 단어는 {WORD_MAX_TARGETS}개까지", "code": "too_many_targets",
                                 "n": len(targets)}, status_code=400)
        raw = await up.read()
        try:
            wav = decode_audio(raw)
        except ValueError as e:
            return JSONResponse({"error": f"오디오를 읽지 못했다: {e}", "code": "decode_failed"}, status_code=400)
        sec = len(wav) / SR
        if sec < WORD_MIN_VOICED_SEC:
            return JSONResponse({"error": f"녹음이 너무 짧다 ({WORD_MIN_VOICED_SEC}초 미만)", "code": "too_short",
                                 "sec": round(sec, 3)}, status_code=400)
        truncated = sec > WORD_MAX_SEC               # 단어 조각이 아니다 — 앞 6초만 보고, 응답에 표시한다
        wav = wav[: int(SR * WORD_MAX_SEC)]
        from mdd.vad import voiced_mask, FRAME_SEC
        m, _ = voiced_mask(wav, SR)
        voiced = float(m.sum()) * FRAME_SEC
        if voiced < WORD_MIN_VOICED_SEC:
            return JSONResponse({"error": "말소리가 거의 없다 (말한 시간 %.2f초) — 다시 말할 것" % voiced, "code": "low_speech",
                                 "voiced_sec": round(voiced, 2), "sec": round(sec, 2)}, status_code=400)
        rows, pred, ms = word_judge(wav, targets)
        ms["request"] = round((time.perf_counter() - tr) * 1e3, 1)
        return {"targets": rows, "pred_jamo": pred, "voiced_sec": round(voiced, 2), "sec": round(sec, 2),
                "truncated": truncated, "device": S["dec"].device, "ms": ms}

    if api_only:                                   # 공개 배포: 토큰이 주입되는 데모 페이지를 내리고 API 만 연다
        return capture_app(app, "pronunciation")

    @app.get("/{page}.html")
    def page(page: str):
        """토큰을 켠 경우 페이지에 주입한다 — 사내망에서 데모를 그대로 쓰기 위해서."""
        from fastapi.responses import HTMLResponse
        f = os.path.join(static_dir, page + ".html")
        if not os.path.isfile(f):
            return JSONResponse({"error": "not found"}, status_code=404)
        html = open(f, encoding="utf-8").read()
        if API_TOKEN and ask_token:                # 공개 배포: 토큰을 넣지 않고 사용자에게 묻는다(이 브라우저에만 저장)
            html = html.replace("<script>", ASK_TOKEN_JS + "\n<script>", 1)
        elif API_TOKEN:
            html = html.replace("<script>", "<script>window.API_TOKEN=%s;</script>\n<script>"
                                % json.dumps(API_TOKEN), 1)
        return HTMLResponse(html)

    @app.get("/")
    def root():
        return page("index")

    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return capture_app(app, "pronunciation")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-ckpt", dest="hf_ckpt", required=True, help="w2v-BERT 체크포인트 폴더")
    ap.add_argument("--rbln", default=None,
                    help="NPU 컴파일 모델(.rbln, 예: w2vbert_ctc_f192_448_1024.rbln). 주면 GPU 대신 NPU 로 추론한다. "
                         "--hf-ckpt 에서는 jamo_vocab.json·preprocessor_config.json 만 읽는다")
    ap.add_argument("--rbln-device", dest="rbln_device", type=int, default=0, help="NPU 장치 번호")
    ap.add_argument("--scorer", default=None, help="점수 3종 joblib")
    ap.add_argument("--norms", default=os.path.join(str(MDD_OUTPUTS), "score_phoneme", "norms.json"))
    ap.add_argument("--calib", default=os.path.join(str(MDD_OUTPUTS), "score_phoneme", "calibration.json"),
                    help="음소 점수 보정표(없으면 백분위로 물러난다)")
    ap.add_argument("--utt-comp", dest="utt_comp", action="store_true",
                    help="발화 1개 이해도(상관 0.433)를 result 에 넣는다. 기본은 세션에서만 쓴다")
    ap.add_argument("--flu-rule", dest="flu_rule",
                    default=os.path.join(str(MDD_OUTPUTS), "score_phoneme", "fluency_rule.json"),
                    help="유창성을 속도·쉼만으로 매기는 기준표(없으면 사람 점수 예측을 쓴다)")
    ap.add_argument("--utt-pct", dest="utt_pct",
                    default=os.path.join(str(MDD_OUTPUTS), "score_phoneme", "utt_percentile.json"),
                    help="문장 점수 눈금표(없으면 5점 만점 비례로 물러난다)")
    ap.add_argument("--save-dir", dest="save_dir", default=None,
                    help="녹음을 저장할 폴더(기준표 재작성용). 주지 않으면 저장하지 않는다")
    ap.add_argument("--static", default=str(_pl.Path(__file__).parent / "static"))
    ap.add_argument("--token", default=os.environ.get("MDD_API_TOKEN") or None,
                    help="API 열쇠(기본: 환경변수 MDD_API_TOKEN). 주면 /speak·/sent.eval.kr 에 인증이 걸린다")
    ap.add_argument("--ask-token", dest="ask_token", action="store_true",
                    help="데모 페이지에 토큰을 넣지 않고 브라우저에서 입력받는다(인터넷에 페이지를 열 때). ?reset_token 으로 다시 입력")
    ap.add_argument("--api-only", dest="api_only", action="store_true",
                    help="데모 페이지를 내리고 API 만 연다. 토큰을 켜면 페이지가 토큰을 HTML 에 넣어 보내므로 인터넷에 열 때는 이걸 켠다")
    ap.add_argument("--canon-fix", dest="canon_fix", action="store_true",
                    help="표준발음 보정(mdd.canonical_fix) 켜기 — KoG2P 가 틀리는 표준발음(먹을 수[머글쑤] · 중요[중요] …)을 고치고 "
                         "단어 경계처럼 두 발음이 다 맞는 곳은 녹음에 맞춰 고른다. 빼면 예전과 같다")
    ap.add_argument("--del-run-min", dest="del_run_min", type=int, default=1,
                    help="빠진 소리(음운탈락)를 몇 개 이상 이어질 때만 지적할지(기본 1 = 다). 3 = 글자·낱말을 통째로 빼먹을 때만")
    ap.add_argument("--del-score-max", dest="del_score_max", type=int, default=0,
                    help="빠진 소리는 강제정렬 점수가 이 값 미만일 때만 지적(0 = 끔, 권장 50 · --del-run-min 은 1 로) — "
                         "speechsuper.DEL_SCORE_MAX 주석")
    ap.add_argument("--nomatch-reject", dest="nomatch_reject", type=float, default=0.0,
                    help="기준 문장 소리 중 들은 소리와 다른 비율이 이 값을 넘으면 400 no_match(0 = 끔, 권장 0.9)")
    ap.add_argument("--too-long-reject", dest="too_long_reject", action="store_true",
                    help="말소리가 MAX_SPEECH_SEC(20초)를 넘으면 400 too_long(안 주면 앞 20초만 채점)")
    ap.add_argument("--no-korean-reject", dest="no_korean_reject", action="store_true",
                    help="refText 에 한글이 없으면 400 no_korean(안 주면 예전처럼 점수 칸이 빈 200)")
    ap.add_argument("--ss-rule", dest="ss_rule", action="store_true",
                    help="ㅅ↔ㅆ 예외: ㅅ→ㅆ 는 확률 0.5 이상 · ㅆ→ㅅ 는 첫소리·모음 뒤에서만 0.6 이상이면 지적(speechsuper.SS_RULE)")
    ap.add_argument("--sz-detector", dest="sz_detector", default=None,
                    help="ㅅㅆ·ㅈㅉㅊ 첫소리 판별기 json(mdd.sz_detector) — 받아쓰기가 못 듣는 세기 오류를 소리 단서로 지적(안 주면 끔)")
    ap.add_argument("--liquid-rule", dest="liquid_rule", action="store_true",
                    help="첫소리·받침 ㄴ↔ㄹ 은 점수가 높아도 들은 소리 확률 0.6 이상이면 지적(speechsuper.LIQUID_NASAL_PAIRS)")
    ap.add_argument("--canon-fix-margin", dest="canon_fix_margin", type=float, default=1.0,
                    help="허용 변이를 고를 최소 CTC 로그우도 차(기본 1.0)")
    ap.add_argument("--host", default="127.0.0.1", help="사내망에 열려면 0.0.0.0")
    ap.add_argument("--port", type=int, default=8932)
    a = ap.parse_args()
    global SAVE_DIR, API_TOKEN
    SAVE_DIR, API_TOKEN = a.save_dir, a.token
    load_all(a)
    if SAVE_DIR:
        print(f"[save] 녹음 저장 → {SAVE_DIR}", flush=True)
    print(f"[serve] http://{a.host}:{a.port} · 인증 {'켬' if API_TOKEN else '끔'}", flush=True)
    import uvicorn
    uvicorn.run(make_app(a.static, api_only=a.api_only, ask_token=a.ask_token), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
