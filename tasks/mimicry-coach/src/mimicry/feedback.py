# -*- coding: utf-8 -*-
"""어긋남 → 교정 문장 두 개(무엇이 어긋났나 / 어떻게 고칠까) + 보조 숫자. 계획서 6.7 표를 옮긴 것.

원칙: 전문용어 금지, 숫자는 `detail` 에만. 같은 지적이 연달아 나오면 `variant` 를 바꿔 다른 표현을 쓴다.
"""
from .text import has_batchim


def _j(s, with_b, without_b):
    """조사 고르기 — 마지막 글자에 받침이 있으면 with_b."""
    return with_b if has_batchim(s[-1]) else without_b


def _pick(opts, variant):
    return opts[variant % len(opts)]


def span_text(syllables, i, j, word_of=None):
    """음절 i..j 를 원문 띄어쓰기대로 잇는다(word_of 가 있으면 어절이 바뀌는 자리에 공백)."""
    out = syllables[i]
    for k in range(i + 1, j + 1):
        out += (" " if word_of is not None and word_of[k] != word_of[k - 1] else "") + syllables[k]
    return out


def name_runs(idx, syllables, word_of=None):
    """음절 번호들 → 이어진 구간마다 "'…'" 이름(원문 띄어쓰기대로)."""
    idx = sorted(idx)
    runs, cur = [], [idx[0]]
    for i in idx[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    return ["'%s'" % span_text(syllables, a[0], a[-1], word_of) for a in runs]


def unsure_note(idx, syllables, word_of=None):
    """통과("원본과 거의 같아요")인데 경계 확신이 없어 못 잰 음절이 있을 때 붙이는 알림 — 그 부분은 같다고 한 게 아니다."""
    names = name_runs(idx, syllables, word_of)
    last = names[-1][1:-1]
    return "%s%s 경계가 흐려 확실히 못 쟀어요" % ("·".join(names), _j(last, "은", "는"))


def retake(quality, words, syllables=None, word_of=None):
    """다시 녹음 안내 — compare 의 quality.reason(silent·line·whisper·missing·bound) → dict(what, how).
    words: 대사 어절들 · syllables·word_of: 음절 글자와 어절 번호(빠진 곳 이름을 붙일 때)."""
    line = " ".join(words)
    q = "'%s'%s" % (line, _j(line, "을", "를"))
    r = quality.get("reason")
    if r == "silent":
        return dict(what="말소리가 잘 안 들려요", how="마이크 가까이에서 %s 소리 내어 말해 주세요" % q)
    if r == "whisper":
        return dict(what="속삭이듯 말해서 대사를 잘 못 알아들었어요", how="목소리를 내서 %s 말해 주세요" % q)
    if r == "line":
        return dict(what="대사가 다르게 들려요",
                    how="%s 빠짐없이 그대로 말해 주세요. 주변이 시끄러우면 조용한 곳에서 다시 녹음해 주세요" % q)
    if r == "missing" and quality.get("missing") and syllables:
        names = name_runs(quality["missing"], syllables, word_of)
        last = names[-1][1:-1]
        return dict(what="%s%s 안 들려요" % ("·".join(names), _j(last, "이", "가")),
                    how="빠뜨렸거나 너무 작게 말했어요 — %s 처음부터 끝까지 또렷하게 다시 녹음해 주세요" % q)
    return dict(what="녹음을 원본과 확실히 맞추지 못했어요", how="조용한 곳에서, 같은 대사를 끝까지 다시 녹음해 주세요")


def describe(issue, syllables, variant=0, word_of=None):
    """issue(compare 결과 한 항목) → dict(what, how, detail, where). 묶인 지적(i..i_end)은 그 구간을 이름으로 쓴다."""
    k = issue["type"]
    if k == "voicing":
        whisper = issue["dir"] == "whisper"
        return dict(where="대사 전체", what="속삭이듯 말했어요" if whisper else "원본은 속삭이듯 말해요",
                    how=_pick(["원본처럼 목소리를 내서 말해 보세요", "성대를 울려 소리 내어 말해 보세요"] if whisper else
                              ["목소리를 빼고 숨소리로 속삭여 보세요", "성대를 울리지 말고 속삭이듯 말해 보세요"], variant),
                    detail="소리 낸 음절: 원본 %.0f%% → 나 %.0f%%" % (100 * issue["ref"], 100 * issue["take"]))
    if k == "tempo":
        fast = issue["dir"] == "fast"
        return dict(where="대사 전체",
                    what="전체적으로 원본보다 %s" % ("빨라요" if fast else "느려요"),
                    how=_pick(["원본 속도에 맞춰 조금 %s 말해보세요" % ("천천히" if fast else "더 빠르게"),
                               "원본을 한 번 더 듣고 %s 따라 해보세요" % ("여유 있게" if fast else "속도를 붙여서")], variant),
                    detail="전체 길이 원본의 %.2f배" % issue["diff"])
    i = issue["i"]
    s = span_text(syllables, i, issue.get("i_end", i), word_of)
    q = "'%s'" % s
    prev = syllables[i - 1] if i > 0 else ""
    if k == "pitch_level":
        high = issue["dir"] == "high"
        what = "%s%s 원본보다 %s" % (q, _j(s, "이", "가"), "높아요" if high else "낮아요")
        how = _pick(["힘을 빼고 조금 낮게 내보세요", "원본처럼 한 톤 낮춰서 말해보세요"] if high else
                    ["조금 더 높게 내보세요", "원본처럼 한 톤 올려서 말해보세요"], variant)
        detail = "원본보다 %.1f반음 %s" % (abs(issue["diff"]), "높음" if high else "낮음")
    elif k == "pitch_dir":
        rd, td, kind = issue["ref_dir"], issue["take_dir"], issue["kind"]
        if kind == "opposite":
            if rd == "rise":
                what = "%s에서 끝이 %s" % (q, "내려갔어요" if td == "fall" else "안 올라갔어요")
                how = _pick(["끝을 위로 끌어올려 보세요", "끝을 물어보듯 올려보세요"], variant)
            elif rd == "fall":
                what = "%s에서 끝이 %s" % (q, "올라갔어요" if td == "rise" else "안 내려갔어요")
                how = _pick(["끝을 툭 떨어뜨려 보세요", "끝을 단정 짓듯 내려보세요"], variant)
            else:
                what = "%s에서 끝이 %s" % (q, "올라갔어요" if td == "rise" else "내려갔어요")
                how = _pick(["끝을 %s 말고 평평하게 이어가 보세요" % ("올리지" if td == "rise" else "떨어뜨리지"),
                             "높낮이 없이 고르게 말해보세요"], variant)
        elif kind == "less":
            up = rd == "rise"
            what = "%s에서 끝을 원본만큼 %s" % (q, "안 올렸어요" if up else "안 내렸어요")
            how = _pick(["더 크게 %s 보세요" % ("끌어올려" if up else "떨어뜨려"),
                         "원본처럼 과감하게 %s 보세요" % ("올려" if up else "내려")], variant)
        else:
            up = rd == "rise"
            what = "%s에서 끝을 너무 많이 %s" % (q, "올렸어요" if up else "내렸어요")
            how = _pick(["조금만 %s 보세요" % ("올려" if up else "내려"), "원본보다 살짝 덜 움직여 보세요"], variant)
        detail = "음절 안 높낮이 변화: 원본 %+.1f → 나 %+.1f반음" % (issue["ref"], issue["take"])
    elif k == "duration":
        if issue["dir"] == "short":
            what = "%s%s 너무 짧게 말했어요" % (q, _j(s, "을", "를"))
            how = _pick(["%s%s 원본처럼 길게 끌어보세요" % (q, _j(s, "을", "를")), "%s에서 조금 더 머물러 보세요" % q], variant)
        else:
            what = "%s%s 너무 길게 끌었어요" % (q, _j(s, "을", "를"))
            how = _pick(["%s%s 짧게 끊어서 말해보세요" % (q, _j(s, "은", "는")), "%s%s 가볍게 지나가 보세요" % (q, _j(s, "은", "는"))],
                        variant)
        detail = "빠르기를 맞춘 뒤 원본의 %.1f배 길이" % issue["diff"]
    elif k == "pause":
        r, t = issue["ref"], issue["take"]
        if issue["dir"] == "missing":
            if t >= 0.2:
                what, how = "%s 앞에서 원본보다 짧게 쉬었어요" % q, "%s 앞에서 조금 더 길게 쉬어보세요" % q
            else:
                what = "%s 앞에서 안 쉬었어요" % q
                how = _pick(["%s 앞에서 한 박자 쉬어보세요" % q, "%s%s 말한 뒤 잠깐 멈춰보세요" % ("'%s'" % prev, _j(prev, "을", "를"))],
                            variant)
        else:
            if r >= 0.2:
                what = "%s 앞에서 원본보다 오래 쉬었어요" % q
            else:
                what = "%s 앞에서 쉬었어요" % q
            how = _pick(["'%s' 쉬지 말고 바로 이어서 말해보세요" % span_text(syllables, i - 1, i, word_of),
                         "%s 앞 쉼을 줄여보세요" % q], variant)
        detail = "쉼: 원본 %.2f초 → 나 %.2f초" % (r, t)
    elif k == "energy":
        weak = issue["dir"] == "weak"
        if weak:
            what = "%s에 힘이 약해요" % q
            how = _pick(["%s에 힘을 실어보세요" % q, "%s%s 또렷하게 세게 말해보세요" % (q, _j(s, "을", "를"))], variant)
        else:
            what = "%s에 힘이 너무 들어갔어요" % q
            how = _pick(["%s%s 힘을 빼고 가볍게 말해보세요" % (q, _j(s, "은", "는")), "%s%s 조금 작게 말해보세요" % (q, _j(s, "은", "는"))],
                        variant)
        detail = "원본보다 %.0fdB %s" % (abs(issue["diff"]), "약함" if weak else "셈")
    else:
        raise KeyError(k)
    return dict(where=q, what=what, how=how, detail=detail)
