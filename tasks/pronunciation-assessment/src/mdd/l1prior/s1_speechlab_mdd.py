# -*- coding: utf-8 -*-
"""
s1_speechlab_mdd.py
~~~~~~~~~~~~~~~~~~~
Speech_lab 처리 파이프라인 (작업지시서 3-A). 실현형 축.

대상: **파일명에 'LAR' 이 포함된 레코드만** G2P 처리 (읽기 과제 = 고정 프롬프트).

레코드 1건 처리 순서:
  0) orthographic 정제 (MDD 이전):
     - 간투사(아/어/음/그/저 …) 중 **prompt 에 대응되지 않는 '삽입된' 것만** 삭제
       (예: 'ㄱ 사람은'의 '그'=지시관형사는 prompt와 매칭되므로 보존)
     - 학습자의 더듬음/되풀이(반복)는 하나로 합쳐 정답열과 길이를 맞춤
     - 삭제/병합한 위치는 disfluency 플래그로 기록(삭제만 하고 정보는 남긴다)
  1) prompt --KoG2P(규칙 전체)--> 정답 음소열 target_phones
  2) 정제된 orthographic --KoG2P--> 실제 음소열 actual_phones
  3) 두 음소열 Needleman-Wunsch 정렬 -> (pos,target,actual) 대치/삭제/삽입
  4) taxonomy(s0)로 error_type 자동부여
  5) 어절 붕괴 필터 -> quality_flag
  -> 통합 오류 스키마(작업지시서 2절) 레코드

정제 로직은 어절(word) 단위 정렬을 먼저 하여 '삽입' 여부를 판정한다.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import os
import sys
import json
import argparse

from mdd.l1prior.s0_kog2p_wrapper import text_to_phones
from mdd.l1prior.s0_taxonomy import classify_op

COLLAPSE_RATIO = 0.5   # 어절 오류비율 임계(작업지시서 3-A-5)

# 간투사(hesitation filler) 사전 — 표면형. '삽입된' 경우에만 제거된다.
FILLERS = {
    "아", "어", "음", "그", "저", "으", "에", "응", "엄", "뭐",
    "아아", "어어", "음음", "으음", "에에", "그그", "저저", "어…", "음…",
}


# ── 범용 Needleman-Wunsch (토큰 리스트) ──────────────────────
def _nw(target, actual, match=0, mismatch=1, gap=1):
    """토큰열 정렬. ops = [{kind,target,actual,tpos,apos}] 반환.
    kind ∈ match|sub|del|ins.  tpos=target 인덱스, apos=actual 인덱스."""
    n, m = len(target), len(actual)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i * gap
    for j in range(1, m + 1):
        D[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = match if target[i - 1] == actual[j - 1] else mismatch
            D[i][j] = min(D[i - 1][j - 1] + c, D[i - 1][j] + gap, D[i][j - 1] + gap)
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            c = match if target[i - 1] == actual[j - 1] else mismatch
            if D[i][j] == D[i - 1][j - 1] + c:
                kind = "match" if target[i - 1] == actual[j - 1] else "sub"
                ops.append({"kind": kind, "target": target[i - 1],
                            "actual": actual[j - 1], "tpos": i - 1, "apos": j - 1})
                i -= 1; j -= 1; continue
        if i > 0 and D[i][j] == D[i - 1][j] + gap:
            ops.append({"kind": "del", "target": target[i - 1],
                        "actual": None, "tpos": i - 1, "apos": None})
            i -= 1; continue
        ops.append({"kind": "ins", "target": None, "actual": actual[j - 1],
                    "tpos": i, "apos": j - 1})
        j -= 1
    ops.reverse()
    return ops


# ── orthographic 정제 (간투사 제거 + 반복 병합) ───────────────
def _is_spacing_artifact(w, apos, ow, pw_set):
    """간투사 후보 w 가 사실은 띄어쓰기 변형인지 판정.
    인접 orthographic 어절과 이어붙이면 prompt 어절을 이루거나 그 접두면 True.
    예: prompt '저한테' 를 '저 한테'로 발화 -> '저'는 간투사 아님."""
    nxt = ow[apos + 1] if apos + 1 < len(ow) else None
    prv = ow[apos - 1] if apos - 1 >= 0 else None
    for merged in ((w + nxt) if nxt else None, (prv + w) if prv else None):
        if not merged:
            continue
        if merged in pw_set or any(p.startswith(merged) for p in pw_set):
            return True
    return False


def clean_orthographic(prompt, ortho):
    """prompt(고정문) 대비 삽입된 간투사/반복만 제거.
    return: (cleaned_words, disfluencies)
      disfluencies = [{type:'filler'|'repetition', word, ortho_pos, note}]"""
    pw = prompt.split()
    pw_set = set(pw)
    ow = ortho.split()
    if not ow:
        return [], []
    ops = _nw(pw, ow)

    remove_apos = set()            # 제거할 orthographic 어절 인덱스
    disfluencies = []

    # actual(=ortho) 위치별로 op 정리
    for k, op in enumerate(ops):
        if op["kind"] != "ins":
            continue
        w = op["actual"]
        apos = op["apos"]
        # 인접 '유지' 어절(정렬상 match/sub 된 ortho 어절) 찾기 — 반복 판정용
        prev_kept = None
        for kk in range(k - 1, -1, -1):
            if ops[kk]["kind"] in ("match", "sub"):
                prev_kept = ops[kk]["actual"]; break
        next_kept = None
        for kk in range(k + 1, len(ops)):
            if ops[kk]["kind"] in ("match", "sub"):
                next_kept = ops[kk]["actual"]; break

        # 1) 간투사: 삽입된 어절이 filler 사전에 있음
        #    단, 인접어절과 합쳐 prompt 어절을 이루면(띄어쓰기 변형) 보존
        if w in FILLERS and not _is_spacing_artifact(w, apos, ow, pw_set):
            remove_apos.add(apos)
            disfluencies.append({"type": "filler", "word": w,
                                 "ortho_pos": apos, "note": "삽입 간투사 제거"})
            continue
        # 2) 반복/더듬음: 인접 유지어절과 동일하거나 그 접두(예: '샤 샤워')
        rep_target = None
        for cand in (next_kept, prev_kept):
            if cand and (w == cand or (len(w) < len(cand) and cand.startswith(w))
                         or (len(cand) < len(w) and w.startswith(cand))):
                rep_target = cand; break
        # 삽입어절끼리 동일 반복(예: '샤 샤') — 앞 삽입어절과 비교
        if rep_target is None and prev_kept is None:
            for kk in range(k - 1, -1, -1):
                if ops[kk]["kind"] == "ins" and ops[kk]["actual"] == w \
                        and ops[kk]["apos"] not in remove_apos:
                    rep_target = w; break
        if rep_target is not None:
            remove_apos.add(apos)
            disfluencies.append({"type": "repetition", "word": w,
                                 "ortho_pos": apos,
                                 "note": f"반복/더듬음 병합(→'{rep_target}')"})

    cleaned = [w for idx, w in enumerate(ow) if idx not in remove_apos]
    return cleaned, disfluencies


# ── 통합 스키마 레코드 생성 ──────────────────────────────────
def build_record(rec, fname=None):
    sm = rec.get("SpeakerMetadata", {})
    rm = rec.get("RecordingMetadata", {})
    ev = rec.get("EvaluationMetadata", {})
    prompt = rm.get("prompt", "") or ""
    ortho_raw = rm.get("orthographic", "") or ""

    cleaned_words, disfluencies = clean_orthographic(prompt, ortho_raw)
    ortho_clean = " ".join(cleaned_words)

    target = list(text_to_phones(prompt))
    actual = list(text_to_phones(ortho_clean))
    ops = _nw(target, actual)

    # target_next 문맥(구개음화 판정)
    for k, op in enumerate(ops):
        nxt = None
        for kk in range(k + 1, len(ops)):
            if ops[kk]["target"] is not None:
                nxt = ops[kk]["target"]; break
        op["target_next"] = nxt

    errors = []
    n_err = 0
    for op in ops:
        if op["kind"] == "match":
            continue
        n_err += 1
        et, conf, why = classify_op(op)
        errors.append({
            "pos": op["tpos"], "target_phone": op["target"],
            "actual_phone": op["actual"], "op": op["kind"],
            "error_type": et, "t_start": None, "t_end": None,
            "source": "speech_lab", "confidence": conf, "rationale": why,
        })

    err_ratio = (n_err / len(target)) if target else 0.0

    # 품질 플래그(옵션 A): 재시작/붕괴 레코드를 집계에서 배제하기 위한 신호
    #  - collapsed: 전체 오류비율이 임계 이상(문장·어휘 이탈)
    #  - restart_suspect: 같은 target 위치에 연속 삽입이 몰림(재시작 조각 특징)
    #    예: '그 음식 그 식당은…'의 '음식' 잔여 -> pos0 삽입 뭉치
    max_ins_run, cur_run, cur_pos = 0, 0, None
    for op in ops:
        if op["kind"] == "ins" and op["tpos"] == cur_pos:
            cur_run += 1
        elif op["kind"] == "ins":
            cur_pos, cur_run = op["tpos"], 1
        else:
            cur_pos, cur_run = None, 0
        max_ins_run = max(max_ins_run, cur_run)

    if err_ratio >= COLLAPSE_RATIO:
        quality_flag = "collapsed"
    elif max_ins_run >= 3:
        quality_flag = "restart_suspect"
    else:
        quality_flag = "ok"
    usable_for_aggregation = (quality_flag == "ok")

    return {
        "utterance_id": rec.get("sound_id"),
        "file": fname,
        "task": (fname.split("-")[5][:3] if fname else None),
        "user_id": rec.get("UserID"),
        "L1": sm.get("nationality"), "language": sm.get("language"),
        "proficiency": sm.get("proficiency"),
        "recording_level": sm.get("recording_level"),
        "prompt": prompt,
        "orthographic_raw": ortho_raw,
        "orthographic_clean": ortho_clean,
        "disfluencies": disfluencies,
        "target_phones": target, "actual_phones": actual,
        "n_target": len(target), "n_actual": len(actual),
        "n_errors": n_err, "err_ratio": round(err_ratio, 3),
        "max_ins_run": max_ins_run,
        "quality_flag": quality_flag,
        "usable_for_aggregation": usable_for_aggregation,
        "errors": errors,
        "eval": {"Delivery": ev.get("DeliveryEval"),
                 "LanguageUse": ev.get("LanguageUseEval"),
                 "Content": ev.get("ContentEval"),
                 "PronunProf": None, "Fluency": None},
        "source": "speech_lab",
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--n", type=int, default=3, help="출력 샘플 수")
    ap.add_argument("--dir", default=os.path.join(
        "040.교육용 영어 모국어 사용자의 한국어 음성 데이터",
        "01-1.정식개방데이터", "Training", "02.라벨링데이터", "TL_Speech_lab"))
    ap.add_argument("--scan", type=int, default=8000)
    args = ap.parse_args()

    if args.demo:
        import glob
        picks = []
        for i, f in enumerate(sorted(glob.glob(os.path.join(args.dir, "*LAR*.json")))):
            if i >= args.scan:
                break
            try:
                r = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            if not r.get("RecordingMetadata", {}).get("orthographic"):
                continue
            out = build_record(r, os.path.basename(f))
            # 정제가 실제로 일어난(간투사/반복 제거) 샘플 우선
            if out["disfluencies"]:
                picks.append(out)
            if len(picks) >= args.n:
                break
        for out in picks:
            print(json.dumps(out, ensure_ascii=False, indent=2))
            print("=" * 70)
        if not picks:
            print("정제 발생 샘플을 못 찾음 — --scan 증가 필요.")
