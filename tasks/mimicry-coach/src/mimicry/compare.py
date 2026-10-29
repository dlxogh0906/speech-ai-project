# -*- coding: utf-8 -*-
"""원본 ↔ 녹음 음절별 비교 → 어긋남 후보 · 심각도 · 고칠 순서.

허용치·가중치는 계획서 6.5·6.6 초안 그대로다. 보정(Phase 2) 전까지는 "출발점"일 뿐이다.
    심각도 = (차이 ÷ 허용치) × 종류 가중치 × 위치 가중치(대사 끝 음절 1.2)
차이가 허용치를 넘은 것(ratio > 1)만 후보. 한 음절엔 심각도가 가장 큰 것 하나만, 대사당 최대 `top` 개.
이어진 음절이 같은 지적(같은 종류·같은 방향)이면 하나로 묶는다 — "'안 나요'가 높아요" 한 번이지 세 번이 아니다.

정규화(계획서 6.3): 음높이는 각자 중앙값 기준 반음, 길이는 음절별 길이 비의 중앙값(전체 빠르기)으로 나눈 값,
세기는 각자 음절 세기 중앙값 기준 dB. 전체 빠르기 차이는 음절이 아니라 대사 단위 지적(tempo)으로 따로 본다.
"""
import math

import numpy as np

TOL = {"pitch_level": 2.0,              # 반음
       "pitch_dir": 3.0,                # 음절 안 변화(끝−앞) 차, 반음 — 방향이 다를 때
       "duration": math.log(1.4),       # 길이 비(빠르기 보정 후) 1/1.4 ~ 1.4
       "pause": 0.25,                   # 초
       "energy": 6.0,                   # dB
       "tempo": math.log(1.3)}          # 대사 전체 빠르기 비
WEIGHT = {"pitch_dir": 1.3, "duration": 1.2, "pitch_level": 1.0, "pause": 1.0, "energy": 0.8, "tempo": 0.9, "voicing": 1.5}
LAST_WEIGHT = 1.2
DIR_TH = 1.5           # 음절 안 변화가 ±이 값을 넘으면 올림/내림, 아니면 평평
SAME_DIR_FACTOR = 2.0  # 방향은 같고 크기만 다를 땐 허용치를 이만큼 넓혀 본다(덜/더 올림)
PAUSE_SIG = 0.2        # 둘 중 긴 쪽 쉼이 이보다 짧으면(파열음 막힘 정도) 쉼 비교를 안 한다
PAUSE_SURE = 0.3       # 녹음 쪽 쉼이 이 이상이고 원본보다 길면 경계 교차 확인이 안 맞은 음절이라도 쉼만은 잰다("더 쉬었다"만) —
                       # 잡음 속에선 쉼 앞뒤가 "확신 없음"으로 빠져 쉼을 못 짚었는데, 긴 쉼은 경계가 조금 흔들려도 분명하다.
                       # "덜 쉬었다"는 안 잰다: 잡음이 쉼을 채우면 녹음 쉼이 짧게 잡힌다(「하와이」 '하' 앞 1.69초 쉼을 웅성거림·
                       # 음악 속에서 0.2~0.37초로 재서 "짧게 쉬었어요"가 나왔다)
MIN_DUR = 0.05         # 원본에서 이보다 짧게 잡힌 음절은 길이 비교를 안 한다
AGREE_MIN = 0.6        # 경계 일치율(analyze.crosscheck)이 이보다 낮으면 녹음 자체를 다시 받는다
MAIN_RATIO = {"pitch_level": 1.5, "duration": 1.25, "energy": 1.5}
                       # 허용치의 이 배를 넘는 것만 "고칠 곳"(보여 줄 지적) — 그 아래는 "다듬을 곳"(minor). 고칠 곳이 없으면 통과.
                       # 사람마다 원래 많이 다른 세 종류만 올린다(평범한 두 낭독에서 음절의 16~24% 가 허용치를 넘는다 —
                       # 시킨 대로 고쳐도 다시 녹음할 때 흔들림이 새 지적을 만들어 교정 루프가 안 끝났다, results/08 §5).
                       # 쉼·끝 억양·빠르기·목소리는 그대로 1배 — 드물고(0.4%·5.8%) 귀에 잘 띈다. 길이는 1.25배(빠르기 맞춘 뒤 1.52배)
                       # — 1.5배로 하면 1.8배로 늘린 음절의 절반이 다듬을 곳으로 내려갔다(재면 1.6~1.7배)
FINAL_CONTOUR = 3.0    # 원본 끝 음절이 이만큼(반음) 넘게 오르내리는데 녹음이 반대·평평하면 그 음절은 높이보다 끝 억양을 짚는다
                       # (「밥은 먹고 다니냐」 '냐'의 의문 올림이 "'다니냐'가 낮아요"에 묻혔다)
WHISPER_REF = 0.35     # 유성 비율(음절 유성 프레임 비율을 길이로 가중) — 원본이 이 이상인데 녹음이 WHISPER_TAKE 아래면 속삭인 것.
WHISPER_TAKE = 0.25    # 원작 0.40~0.92(숨 섞인 「밥은 먹고」·「어이가」가 0.4) · 보통 녹음 0.70~0.95 · 속삭임 0.00~0.19
VOICED_REF = (0.15, 0.5)   # 거꾸로 원본이 0.15 아래(속삭인 대사)인데 녹음이 0.5 이상이면 "원본은 속삭이듯"
WHISPER_HEARD = 0.2    # 잡음·음악만 들어와도 유성 비율은 0 이다 — 대사 토큰을 이만큼은 받아 적었을 때(line_heard)만 속삭임으로 본다
                       # (잡음·음악만 0.00 · 웅성거림만 0~0.25 · 속삭임 0.38~1.00). 못 받아 적었으면 "말소리가 잘 안 들려요"
PITCH_IMPLAUSIBLE = 10.0   # 음높이·끝억양 차가 이보다 크면 옥타브 추적 오류로 보고 안 짚는다(prosody.clean_octaves 뒤 안전망)
LINE_JER_MAX = 0.6     # 녹음 전체 받아 적기가 대사와 이보다 다르면(analyze 의 line_jer) 다른 대사·잡음 — 음절을 재지 않는다.
                       # 실제 말소리에서 다른 대사는 전부 0.6 초과, 맞는 대사는 원작(사투리·외침) 최대 0.46 · 폰 조건 2% 만 초과
LINE_JER_MARGIN = 0.3  # 원본 자체의 받아 적기 JER + 이만큼까지는 봐준다 — 사투리 대사(「뭣이 중허냐고」 원본 0.46)는 맞게 해도 높다
LINE_EXTRA_MAX = 0.4   # 대사 밖에 다른 말(토큰 2개 이상)이 붙었으면 대사 부분 JER 이 max(이 값, 원본 JER + 0.15) 아래여야 한다 —
                       # 짧은 대사(「살아있네」)는 긴 다른 대사 속 "살제" 같은 조각과도 0.5 쯤 맞는다
LINE_PROB_MIN = 0.45   # 대사 확인 JER 이 문턱 0.2 안쪽까지 올라왔는데 강제 정렬 확신(음절 평균)도 이보다 낮으면 다른 대사 —
                       # 자모가 비슷한 다른 대사(「하와이」↔「너나 잘하세요」 0.58)가 0.40·0.44, 맞는 대사는 최저 0.44(사투리라 문턱이 높다)
LINE_HEARD_MIN = 0.4   # 받아 적은 토큰이 대사 토큰의 이 비율도 안 되면 "말소리가 잘 안 들려요"(아니면 "대사가 다르게 들려요")
ABSENT_DEL = 0.66      # 받아 적기에서 이 비율 넘게 못 받은 음절이(locate_line 의 absent)
ABSENT_PROB = 0.3      # 강제 정렬 확신도 이보다 낮으면 "안 들리는" 음절
MISSING_WORD = 0.5     # 어절 음절의 이 비율 이상이 안 들리면 그 어절이 빠졌다(한 어절 대사는 첫·끝 음절) — 빠뜨린 녹음 78% ·
                       # 깨끗한 정상 녹음 70개 중 0개 · phone 조건 정상 10% 가 걸렸다(그래서 문구는 "빠졌어요"가 아니라 "안 들려요")
WORD_KEEP_PROB = 0.95  # 단 어절 안에 강제 정렬 확신이 이 이상인 음절이 있으면 어절은 있다 — 「종로구에 있는」의 '있'을 빨리
                       # 읽으면 '있'만 안 들린다('는' 1.00). 빠진 어절 104개 중 이런 음절이 있는 건 3개


def direction(d):
    if d is None:
        return None
    return "rise" if d >= DIR_TH else ("fall" if d <= -DIR_TH else "flat")


def _ok(s):
    return s.get("bound_ok", True)


def tempo_ratio(R, T):
    rat = [t["dur"] / r["dur"] for r, t in zip(R, T) if r["dur"] >= MIN_DUR and t["dur"] > 0.02 and _ok(t)]
    if len(rat) >= 3:
        return float(np.median(rat))
    return float(sum(t["dur"] for t in T) / max(1e-6, sum(r["dur"] for r in R)))


def _span_key(iss):
    if iss["type"] == "pause":
        return None                                   # 쉼은 음절 사이 한 자리라 묶지 않는다
    if iss["type"] == "pitch_dir":
        return ("pitch_dir", iss["kind"], iss["ref_dir"], iss["take_dir"])
    return (iss["type"], iss["dir"])


def merge_spans(issues):
    """음절 순서로 보며 바로 이웃한 같은 지적을 묶는다 → i(첫 음절)·i_end(끝 음절)·members. 수치는 가장 큰 음절 것."""
    out = []
    for iss in sorted(issues, key=lambda x: x["i"]):
        k = _span_key(iss)
        last = out[-1] if out else None
        if last is not None and k is not None and _span_key(last) == k and last["i_end"] + 1 == iss["i"]:
            last["members"].append(dict(iss))
            last["i_end"] = iss["i"]
            if iss["ratio"] > last["ratio"]:
                keep = dict(i=last["i"], i_end=last["i_end"], members=last["members"])
                last.clear()
                last.update(iss, **keep)
            continue
        out.append(dict(iss, i_end=iss["i"], members=[dict(iss)]))
    return out


def voiced_frac(S):
    """음절들의 유성 비율을 길이로 가중한 값(분석에 voiced 가 없으면 1 — 검증용 복원 행)."""
    w = [max(s.get("dur", 0.0), 1e-3) for s in S]
    return float(sum(s.get("voiced", 1.0) * x for s, x in zip(S, w)) / sum(w))


def missing_syllables(T, absent):
    """녹음 음절들 + 음절별 못 받아 적은 비율 → (빠진 음절 번호들, 대사 중 빠진 몫 — 어절 비율, 한 어절 대사면 음절 비율).
    안 들리는 음절 = 받아 적기에서 ABSENT_DEL 넘게 못 받고 강제 정렬 확신도 ABSENT_PROB 미만. 어절의 MISSING_WORD 이상이
    안 들리고 확신 높은(WORD_KEEP_PROB) 음절이 하나도 없으면 그 어절 전체가 빠진 것. 한 어절 대사는 첫·끝에서 이어지는 안 들리는 음절."""
    gone = [a >= ABSENT_DEL and (t.get("prob") is not None and t["prob"] < ABSENT_PROB) for a, t in zip(absent, T)]
    words = sorted({t["word"] for t in T})
    if len(words) == 1:
        out = []
        for order in (range(len(T)), range(len(T) - 1, -1, -1)):
            for i in order:
                if not gone[i]:
                    break
                out.append(i)
        out = sorted(set(out))
        return out, len(out) / float(len(T))
    out, nw = [], 0
    for w in words:
        idx = [i for i, t in enumerate(T) if t["word"] == w]
        if np.mean([gone[i] for i in idx]) >= MISSING_WORD and max(T[i].get("prob") or 0.0 for i in idx) < WORD_KEEP_PROB:
            out += idx
            nw += 1
    return out, nw / float(len(words))


def compare(ref, take, top=3, tol=None):
    """tol: TOL 일부를 바꿔 보는 dict(보정·민감도 실험용). 안 주면 초안 값.

    녹음 쪽 음절에 `bound_ok=False`(경계를 두 방법이 다르게 잡음)가 있으면 그 음절은 아예 재지 않는다(uncertain).
    녹음의 받아 적기가 대사와 다르거나(line_jer > max(LINE_JER_MAX, 원본 line_jer + LINE_JER_MARGIN)) 어절이 빠졌으면
    (line_absent → missing_syllables) 음절을 하나도 재지 않는다. 그것 또는 경계 일치율 < AGREE_MIN 이면 quality.ok=False ·
    quality.reason(silent·line·missing·bound) — 음절 지적 대신 다시 녹음을 안내한다(feedback.retake)."""
    tl = dict(TOL, **(tol or {}))
    R, T = ref["syllables"], take["syllables"]
    if len(R) != len(T):
        raise ValueError("음절 수가 다릅니다(%d≠%d) — 같은 대사여야 합니다" % (len(R), len(T)))
    n = len(R)
    lj = take.get("line_jer")
    lim = max(LINE_JER_MAX, (ref.get("line_jer") or 0.0) + LINE_JER_MARGIN)
    if take.get("line_extra"):
        lim = min(lim, max(LINE_EXTRA_MAX, (ref.get("line_jer") or 0.0) + LINE_JER_MARGIN / 2.0))
    vr, vt = voiced_frac(R), voiced_frac(T)
    heard = take.get("line_heard")
    whisper = vr >= WHISPER_REF and vt < WHISPER_TAKE and (heard is None or heard >= WHISPER_HEARD)
    reason, missing = None, []
    probs = [t["prob"] for t in T if t.get("prob") is not None]
    weak = bool(probs) and float(np.mean(probs)) < LINE_PROB_MIN
    line_bad = lambda: "whisper" if whisper else ("silent" if (heard or 0.0) < LINE_HEARD_MIN else "line")
    if lj is not None and lj > lim:
        reason = line_bad()
    elif not whisper:                                                    # 속삭였으면 빠짐·확신은 안 믿고 "속삭이듯"으로 짚는다
        if take.get("line_absent") is not None:                         # 빠진 어절이 먼저 — "'…'가 안 들려요"가 더 쓸모 있다
            missing, gone = missing_syllables(T, take["line_absent"])
            if missing:
                reason = "line" if gone > 0.5 else "missing"             # 절반 넘게 빠졌으면 다른 대사로 본다
        if reason is None and lj is not None and lj > lim - 0.2 and weak:
            reason = line_bad()                                          # 문턱 가까이 + 정렬 확신도 낮음
    blocked = reason is not None
    tempo = tempo_ratio(R, T)
    er = [r["energy"] for r in R if r["energy"] is not None]
    et = [t["energy"] for t in T if t["energy"] is not None]
    e_r, e_t = (float(np.median(er)) if er else 0.0), (float(np.median(et)) if et else 0.0)
    rows = []
    for i in range(n):
        r, t = R[i], T[i]
        c = {}
        if blocked or not (_ok(r) and _ok(t)):
            if not blocked and i > 0 and t["pause_before"] >= PAUSE_SURE and t["pause_before"] > r["pause_before"]:
                d = t["pause_before"] - r["pause_before"]
                ratio = abs(d) / tl["pause"]
                c["pause"] = dict(ref=r["pause_before"], take=t["pause_before"], diff=d, ratio=ratio,
                                  dir="extra" if d > 0 else "missing", severity=ratio * WEIGHT["pause"], flag=ratio > 1.0)
            rows.append(dict(i=i, text=r["text"], checks=c, worst=0.0, score=None, uncertain=True))
            continue
        if r["level"] is not None and t["level"] is not None and abs(t["level"] - r["level"]) < PITCH_IMPLAUSIBLE:
            d = t["level"] - r["level"]
            c["pitch_level"] = dict(ref=r["level"], take=t["level"], diff=d, ratio=abs(d) / tl["pitch_level"],
                                    dir="high" if d > 0 else "low")
        if r["delta"] is not None and t["delta"] is not None and abs(t["delta"] - r["delta"]) < PITCH_IMPLAUSIBLE:
            dr, dt = direction(r["delta"]), direction(t["delta"])
            d = t["delta"] - r["delta"]
            if dr != dt:
                kind, ratio = "opposite", abs(d) / tl["pitch_dir"]
            elif dr != "flat":
                kind, ratio = ("more" if abs(t["delta"]) > abs(r["delta"]) else "less"), \
                    abs(d) / (tl["pitch_dir"] * SAME_DIR_FACTOR)
            else:
                kind, ratio = "same", 0.0
            c["pitch_dir"] = dict(ref=r["delta"], take=t["delta"], diff=d, ratio=ratio,
                                  ref_dir=dr, take_dir=dt, kind=kind)
        if r["dur"] >= MIN_DUR and t["dur"] > 0.02:
            q = (t["dur"] / r["dur"]) / tempo
            c["duration"] = dict(ref=r["dur"], take=t["dur"], diff=q, ratio=abs(math.log(q)) / tl["duration"],
                                 dir="long" if q > 1 else "short")
        if i > 0:
            pr, pt = r["pause_before"], t["pause_before"]
            d = pt - pr
            c["pause"] = dict(ref=pr, take=pt, diff=d, ratio=abs(d) / tl["pause"] if max(pr, pt) >= PAUSE_SIG else 0.0,
                              dir="extra" if d > 0 else "missing")
        if r["energy"] is not None and t["energy"] is not None:
            d = (t["energy"] - e_t) - (r["energy"] - e_r)
            c["energy"] = dict(ref=r["energy"] - e_r, take=t["energy"] - e_t, diff=d, ratio=abs(d) / tl["energy"],
                               dir="strong" if d > 0 else "weak")
        pos = LAST_WEIGHT if i == n - 1 else 1.0
        for k, v in c.items():
            v["severity"] = v["ratio"] * WEIGHT[k] * pos
            v["flag"] = v["ratio"] > 1.0
        worst = max([v["ratio"] for v in c.values()] + [0.0])
        rows.append(dict(i=i, text=r["text"], checks=c, worst=worst, score=int(round(100.0 / (1.0 + worst ** 2))),
                         uncertain=False))

    issues = []
    for row in rows:
        fl = [(v["severity"], k) for k, v in row["checks"].items() if v["flag"]]
        if fl:
            sev, k = max(fl)
            pd = row["checks"].get("pitch_dir")
            if row["i"] == n - 1 and pd and pd["flag"] and pd["kind"] == "opposite" and abs(pd["ref"]) > FINAL_CONTOUR:
                k = "pitch_dir"                                            # 대사 끝 억양(의문 올림·끝 내림)이 먼저
            iss = dict(row["checks"][k])
            iss.update(i=row["i"], type=k)
            issues.append(iss)
    issues = merge_spans(issues)
    lt = abs(math.log(tempo)) / tl["tempo"]
    line_issues = []
    if lt > 1.0 and not blocked:
        line_issues.append(dict(i=None, type="tempo", diff=tempo, ratio=lt, severity=lt * WEIGHT["tempo"],
                                dir="slow" if tempo > 1 else "fast", flag=True))
    if not blocked and (whisper or (vr < VOICED_REF[0] and vt >= VOICED_REF[1])):
        line_issues.append(dict(i=None, type="voicing", ref=vr, take=vt, diff=vt - vr, ratio=3.0, severity=3.0 * WEIGHT["voicing"],
                                dir="whisper" if vt < vr else "voiced", flag=True))
    if whisper:                                     # 속삭였으면 "목소리를 내서"가 먼저 — 음높이는 남은 유성 조각에서 잰 것이라 안 짚는다
        issues = [x for x in issues if x["type"] not in ("pitch_level", "pitch_dir")]   # (「서장」 속삭임에 "'제'가 높아요")
    ranked = sorted(issues + line_issues, key=lambda x: (not (whisper and x["type"] == "voicing"), -x["severity"]))
    agree = take.get("bound_agree")
    if reason is None and agree is not None and agree < AGREE_MIN:
        reason = "bound"
    quality = dict(bound_agree=agree, line_jer=lj, line_jer_max=round(lim, 3), uncertain=sum(1 for row in rows if row["uncertain"]),
                   ok=reason is None, reason=reason, missing=missing if reason == "missing" else [])
    scored = [(max(R[row["i"]]["dur"], 1e-3), row["score"]) for row in rows if row["score"] is not None]
    match = (sum(w * sc for w, sc in scored) / sum(w for w, _ in scored)) if scored else None
    is_main = lambda x: x["ratio"] > MAIN_RATIO.get(x["type"], 1.0)
    main = [x for x in ranked if is_main(x)]
    quality["grade"] = "retake" if not quality["ok"] else ("work" if main else "pass")
    return dict(tempo=tempo, rows=rows, issues=ranked, selected=main[:top] if quality["ok"] else [],
                minor=[x for x in ranked if not is_main(x)][:2] if quality["ok"] else [],
                quality=quality, match=None if match is None else int(round(match)))
