# -*- coding: utf-8 -*-
"""모델·GPU·오디오 없이 도는 스모크 — 텍스트 처리 · 옥타브 정리 · 비교·묶기 · 교정 문장 · DTW 투영(쉼 빼기 포함) ·
대사 찾기 · 보고서 · 대사 확인(다른 대사·무음) · 군말 자르기·빠진 어절 · 끝 억양 우선·통과 기준·속삭임 · 음절 음높이 통계(튐 빼기) · 녹음 조건 변형(SNR·찌그러짐·길이 배수 점).

    $PY tasks/mimicry-coach/tests/smoke_test.py          # 수 초. 실패하면 AssertionError
KoG2P 가 없으면(G2P_TOOLS 미설정) 그 항목만 건너뛴다.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import math

import numpy as np


def hz(st):
    return [None if not np.isfinite(v) else round(100 * 2 ** (v / 12.0)) for v in st]


def test_text():
    from mimicry.text import align_tokens, parse_line, syllabify
    w, s, wo = parse_line("니가 가라, 하와이!")
    assert w == ["니가", "가라", "하와이"] and s == list("니가가라하와이") and wo == [0, 0, 1, 1, 2, 2, 2]
    assert syllabify(list("ㄱㅏㅊㅣ")) == [["ㄱ", "ㅏ"], ["ㅊ", "ㅣ"]]
    assert syllabify(list("ㅂㅕㅇㅝㄴ")) == [["ㅂ", "ㅕ", "ㅇ"], ["ㅝ", "ㄴ"]]            # 모음 앞 ㅇ 은 앞 음절 종성
    toks, src = align_tokens(*parse_line("왜 같이")[:2])
    if src == "kog2p":
        assert toks == [["ㅗ", "ㅐ"], ["ㄱ", "ㅏ"], ["ㅊ", "ㅣ"]], toks                  # ㅙ 는 두 자모, 같이→가치
    else:
        print("  (KoG2P 없음 — 직접 분해로 확인: %s)" % src)
        assert toks[0] == ["ㅗ", "ㅐ"]
    try:
        parse_line("3번 대사")
        raise AssertionError("숫자를 받으면 안 된다")
    except ValueError:
        pass


def test_octaves():
    from mimicry.prosody import clean_octaves
    st = lambda f: 12 * np.log2(np.array(f, dtype=float) / 100.0)
    out = hz(clean_octaves(st([100, 99, 96, 92, 89, 87, 85, 85, 86, 86, 178, 181, 172, 182, 195, 186, 191, 194, 188, 186]
                              + [np.nan] * 3 + [120] * 20)))
    assert max(v for v in out[10:20]) < 100, out                                        # 이어지다 한 옥타브 뛴 끝 → 접음
    assert hz(clean_octaves(st([120] * 15 + [240] * 6 + [125] * 15)))[15:21] == [120] * 6  # 가운데 두 배 → 접음
    keep = hz(clean_octaves(st([110] * 20 + [np.nan] * 5 + [190] * 12)))
    assert keep[-1] == 190, keep                                                        # 틈 건너 9.5 반음 도약은 그대로
    dbl = hz(clean_octaves(st([100] * 12 + [np.nan] * 2 + [181] * 5 + [np.nan] * 2 + [98] * 12)))
    assert dbl[14] < 100, dbl                                                           # 틈 건너 10.3 반음(약한 기본음 → 2배음)은 접음
    gl = hz(clean_octaves(st([106, 105, 105, 103, 104, 113, 113, 114, 115, 118, 110, 108, np.nan, 212, 194, 188, 175, 171]
                             + [np.nan] * 6 + [92, 92, 92, 89, 89, 89, 76, 76, 77])))
    assert max(gl[13:18]) < 110, gl                                                     # 내려가는 중에 두 배로 잡힌 조각도 접음
    g = np.linspace(100, 190, 40)
    assert np.allclose(clean_octaves(st(g)), st(g))                                     # 이어지는 상승은 그대로
    sj = hz(clean_octaves(st([115] * 40 + [105] * 6 + [201, 200, 198, 193, 200, 208] + [np.nan] * 3
                             + [92, 92, 90, 90, 90] + [np.nan] * 2 + [155] * 8)))
    assert max(sj[46:52]) < 110 and sj[55:60] == [92, 92, 90, 90, 90], sj              # 2배 이웃을 먼저 고친다 → 90 Hz 는 그대로
    iso = hz(clean_octaves(st([100] * 30 + [99, 94, 92, 91, 90] + [np.nan] * 3
                              + [216, 219, 226, 223, 218, 221, 224, 220, 202, 202])))
    assert iso[38:48] == [None] * 10, iso                                              # 틈 뒤 혼자 한 옥타브 위 → 못 잼
    wide = hz(clean_octaves(st([260] * 40 + [np.nan] * 2 + [330] * 20 + [168] * 7 + [np.nan] + [338] * 14 + [190] * 30)))
    assert wide[42:62] == [330] * 20 and wide[62:69] == [336] * 7, wide                  # 이어지는 332 는 두고, 사이에 낀 반값만 접음
    low = hz(clean_octaves(st([120] * 30 + [125, 130, 135, 140, 145] + [np.nan] * 3 + [60, 59, 58, 58, 57, 57])))
    assert low[-1] == 57, low                                                          # 혼자 한 옥타브 아래(긁는 소리)는 그대로


def syl(i, text, word, start, dur, level=0.0, delta=0.0, energy=70.0, pause=0.0, ok=None):
    d = dict(i=i, text=text, word=word, start=start, end=start + dur, dur=dur, pause_before=pause, level=level,
             delta=delta, energy=energy, n_voiced=20, voiced=0.9, prob=0.9)
    if ok is not None:
        d["bound_ok"] = ok
    return d


def test_compare_feedback():
    from mimicry.compare import compare
    from mimicry.feedback import describe
    T = list("니가가라하와이")
    word = [0, 0, 1, 1, 2, 2, 2]
    R = [syl(i, T[i], word[i], 0.2 * i, 0.2, level=0.0, delta=(4.0 if i == 3 else 0.0)) for i in range(7)]
    S = [dict(r) for r in R]
    S[3] = dict(S[3], delta=-2.0)                                     # '라' 끝: 원본 오름(+4) → 내림(−2)
    S[5] = dict(S[5], level=3.0)
    S[6] = dict(S[6], level=3.2)                                      # '와이' 둘 다 높음 → 한 지적으로 묶여야
    S[1] = dict(S[1], dur=0.32, end=S[1]["start"] + 0.32)              # '가' 1.6배
    C = compare(dict(syllables=R), dict(syllables=S, bound_agree=1.0))
    got = {(x["type"], x["i"], x.get("i_end")) for x in C["issues"]}
    assert ("pitch_dir", 3, 3) in got and ("pitch_level", 5, 6) in got and ("duration", 1, 1) in got, got
    say = {x["type"]: describe(x, T, word_of=word) for x in C["issues"]}
    assert say["pitch_dir"]["what"] == "'라'에서 끝이 내려갔어요", say["pitch_dir"]
    assert say["pitch_level"]["what"] == "'와이'가 원본보다 높아요", say["pitch_level"]
    assert say["duration"]["what"] == "'가'를 너무 길게 끌었어요", say["duration"]
    # 받침 조사
    x = dict(type="pitch_level", i=0, i_end=0, dir="low", diff=-3.0)
    assert describe(x, ["산"])["what"] == "'산'이 원본보다 낮아요"
    # 경계 불일치 음절은 안 잰다 · 일치율 낮으면 재녹음
    S2 = [dict(s) for s in S]
    S2[3]["bound_ok"] = False
    C2 = compare(dict(syllables=R), dict(syllables=S2, bound_agree=0.9))
    assert C2["rows"][3]["uncertain"] and not any(x["i"] == 3 for x in C2["issues"] if x.get("i") is not None)
    C3 = compare(dict(syllables=R), dict(syllables=S, bound_agree=0.4))
    assert not C3["quality"]["ok"] and C3["selected"] == []
    # 옥타브 오류 크기의 차이는 안 짚는다
    S4 = [dict(s) for s in R]
    S4[2] = dict(S4[2], level=12.0)
    assert not any(x.get("i") == 2 for x in compare(dict(syllables=R), dict(syllables=S4))["issues"])
    # 같은 사람·같은 억양이면 지적 0
    assert compare(dict(syllables=R), dict(syllables=[dict(r) for r in R]))["selected"] == []
    # 전체 빠르기만 다르면 음절 길이는 안 짚는다
    S5 = [dict(r, start=r["start"] * 1.2, dur=r["dur"] * 1.2, end=r["end"] * 1.2) for r in R]
    C5 = compare(dict(syllables=R), dict(syllables=S5))
    assert not any(x["type"] == "duration" for x in C5["issues"]) and abs(C5["tempo"] - 1.2) < 1e-6


def test_dtw():
    from mimicry.dtw import project_feats
    rng = np.random.default_rng(0)
    proto = rng.standard_normal((6, 8))
    ref = np.repeat(proto, [10, 10, 10, 10, 10, 10], axis=0)            # 음절 6개, 각 10 프레임
    take = np.repeat(proto, [10, 20, 10, 5, 10, 10], axis=0)            # 둘째 음절 2배 · 넷째 절반
    out = project_feats(ref + 0.05 * rng.standard_normal(ref.shape), take, [10 * k for k in range(7)], 1.0, 0.0)
    want = [0, 10, 30, 40, 45, 55, 65]
    assert all(abs(o - w) <= 1 for o, w in zip(out, want)), out


def test_locate():
    from mimicry.locate import find, jer
    rng = np.random.default_rng(1)
    tgt = [int(v) for v in rng.integers(4, 40, 12)]
    emb = list(tgt)
    emb[5] = 99
    del emb[8]
    emb.insert(3, 77)                                                   # 바꿈·빠짐·끼어듦 하나씩
    H = [int(v) for v in rng.integers(4, 40, 30)] + emb + [int(v) for v in rng.integers(4, 40, 20)]
    s0, s1, e, (i0, i1) = find([(k, 0.1 * t) for t, k in enumerate(H)], tgt)
    assert (i0, i1) == (30, 30 + len(emb) - 1) and abs(e - 0.25) < 1e-9, (i0, i1, e)
    # 끝 모음을 다른 모음으로 받아 적었으면 포함하되, 쉬고 나온 다음 말까지 끌고 가지 않는다
    assert find([(1, 0.0), (2, 0.1), (3, 0.2), (4, 0.22), (5, 0.64)], [1, 2, 3, 5])[1] == 0.22
    assert abs(jer([1, 2, 3, 4], [1, 2, 4]) - 0.25) < 1e-9


def test_project_keep():
    from mimicry.dtw import project_keep
    rng = np.random.default_rng(2)
    proto = rng.standard_normal((4, 8))
    sil = np.full((1, 8), 0.01)
    ref = np.concatenate([np.repeat(proto[:2], 10, axis=0), np.repeat(sil, 30, axis=0), np.repeat(proto[2:], 10, axis=0)])
    take = np.repeat(proto, 10, axis=0)                                 # 원본엔 3초(30프레임) 쉼, 녹음엔 없음
    keep_ref = [(0.0, 20.0), (50.0, 70.0)]
    out = project_keep(ref + 0.01 * rng.standard_normal(ref.shape), take, [10.0, 20.0, 50.0, 60.0], 1.0, -0.5,
                       keep_ref, [(0.0, 40.0)])
    want = [10, 20, 20, 30]                                             # 쉼 양끝은 녹음의 한 점(20)으로 모인다
    assert all(abs(o[0] - w) <= 1 and abs(o[1] - w) <= 1 for o, w in zip(out, want)), out


def test_report():
    from mimicry.compare import compare
    from mimicry.feedback import describe
    from mimicry.report import html, terminal
    T = list("어이가없네")
    R = [syl(i, T[i], 0 if i < 3 else 1, 0.1 + 0.2 * i, 0.2) for i in range(5)]
    S = [dict(r) for r in R]
    S[4] = dict(S[4], level=4.0)
    A = dict(words=["어이가", "없네"], syllables=R, duration=1.3, token_source="kog2p",
             tracks=dict(t_pitch=[0.1 + 0.01 * k for k in range(100)], st=[0.0] * 100, t_int=[], db=[]))
    B = dict(A, syllables=S, bound_agree=1.0)
    C = compare(A, B)
    for x in C["issues"]:
        x["say"] = describe(x, T)
    wav = (0.1 * np.sin(2 * math.pi * 150 * np.arange(20800) / 16000.0)).astype(np.float32)
    h = html(A, B, C, wav, wav, "스모크")
    assert h.startswith("<!doctype html>") and "'네'가 원본보다 높아요" in h and "data:audio/wav;base64," in h
    assert "'네'가 원본보다 높아요" in terminal(A, B, C)


def test_line_gate():
    from mimicry.compare import compare
    from mimicry.feedback import retake
    from mimicry.locate import line_check
    T = list("살아있네")
    R = [syl(i, T[i], 0, 0.1 + 0.2 * i, 0.2) for i in range(4)]
    S = [dict(r, level=5.0) for r in R]                                           # 음높이가 다 다름 — 재면 지적이 나온다
    assert compare(dict(syllables=R), dict(syllables=S, line_jer=0.1, line_heard=1.0))["selected"]
    C = compare(dict(syllables=R), dict(syllables=S, line_jer=0.9, line_heard=1.0))
    assert not C["quality"]["ok"] and C["quality"]["reason"] == "line" and not C["selected"] and C["match"] is None
    assert all(row["uncertain"] for row in C["rows"])
    hard = compare(dict(syllables=R, line_jer=0.46), dict(syllables=S, line_jer=0.7, line_heard=1.0))   # 사투리 대사: 원본도 0.46
    assert hard["quality"]["ok"] and hard["quality"]["line_jer_max"] == 0.76
    assert retake(C["quality"], ["살아있네"])["what"] == "대사가 다르게 들려요"
    q = compare(dict(syllables=R), dict(syllables=S, line_jer=1.0, line_heard=0.1))["quality"]
    assert q["reason"] == "silent" and "'살아있네'를" in retake(q, ["살아있네"])["how"]
    lp = np.log(np.full((6, 4), 0.01))                                            # 프레임 6개, 토큰 0(blank)·1·2·3
    for f, k in enumerate([1, 1, 0, 2, 0, 3]):
        lp[f, k] = 0.0
    assert line_check(lp, [1, 2, 3], 0) == (0.0, 1.0) and line_check(lp, [1, 2], 0)[0] == 0.5


def test_trim_missing():
    from mimicry.compare import compare
    from mimicry.feedback import retake
    from mimicry.locate import align_ops, locate_line, trim_bounds
    assert align_ops([1, 2, 3, 4], [1, 9, 4]) == ["m", "s", "d", "m"] or align_ops([1, 2, 3, 4], [1, 9, 4]) == ["m", "d", "s", "m"]
    # 프레임 20개: 군말 토큰 7(프레임 1) · 쉼 · 대사 1 2 3(프레임 10·12·14), 대사 토큰 4·5(둘째 어절)는 안 나옴
    lp = np.log(np.full((20, 8), 1e-3))
    seq = [0] * 20
    for f, k in ((1, 7), (10, 1), (12, 2), (14, 3)):
        seq[f] = k
    for f, k in enumerate(seq):
        lp[f, k] = 0.0
    loc = locate_line(lp, [1, 2, 3, 4, 5], [0, 0, 1, 2, 2], 3, 0)
    assert loc["before"] and not loc["after"] and loc["absent"] == [0.0, 0.0, 1.0], loc
    sr = 16000
    wav = np.zeros(int(0.45 * sr), dtype=np.float32)
    burst = lambda a, b: np.sin(2 * math.pi * 200 * np.arange(int((b - a) * sr)) / sr).astype(np.float32) * 0.3
    wav[int(0.0 * sr):int(0.06 * sr)] = burst(0.0, 0.06)                # 군말
    wav[int(0.17 * sr):int(0.33 * sr)] = burst(0.17, 0.33)              # 대사
    a, b = trim_bounds(wav, dict(span=(0.2075, 0.2875), before=[0.0275], after=[]), sr)
    assert 0.06 < a < 0.11 and b == len(wav) / sr, (a, b)               # 군말과 대사 사이 조용한 곳에서, 뒤는 안 자름
    T = list("니가가라하와이")
    word = [0, 0, 1, 1, 2, 2, 2]
    R = [dict(syl(i, T[i], word[i], 0.1 + 0.2 * i, 0.2), prob=0.9) for i in range(7)]
    S = [dict(r, prob=(0.05 if i >= 4 else 0.9)) for i, r in enumerate(R)]
    C = compare(dict(syllables=R), dict(syllables=S, line_jer=0.3, line_heard=0.6, line_absent=[0, 0, 0, 0, 1, 1, 1]))
    assert C["quality"]["reason"] == "missing" and C["quality"]["missing"] == [4, 5, 6] and not C["selected"]
    say = retake(C["quality"], ["니가", "가라", "하와이"], T, word)
    assert say["what"] == "'하와이'가 안 들려요", say
    from mimicry.feedback import unsure_note
    assert unsure_note([5, 4], list("뭣이중허냐고"), [0, 0, 1, 1, 1, 1]) == "'냐고'는 경계가 흐려 확실히 못 쟀어요"
    assert unsure_note([0, 3], T, word) == "'니'·'라'는 경계가 흐려 확실히 못 쟀어요"
    C = compare(dict(syllables=R), dict(syllables=S, line_jer=0.3, line_absent=[0, 0, 0, 0, 1, 0, 0]))   # 한 음절만은 안 막는다
    assert C["quality"]["reason"] is None
    one = [dict(syl(i, c, 0, 0.1 + 0.2 * i, 0.2), prob=(0.05 if i == 3 else 0.9)) for i, c in enumerate("살아있네")]
    q = compare(dict(syllables=one), dict(syllables=one, line_jer=0.2, line_absent=[0, 0, 0, 1]))["quality"]
    assert q["reason"] == "missing" and q["missing"] == [3] and retake(q, ["살아있네"], list("살아있네"), [0] * 4)["what"] == "'네'가 안 들려요"


def test_rank_tiers():
    from mimicry.compare import compare
    T = list("다니냐")
    R = [syl(i, T[i], 0, 0.1 + 0.2 * i, 0.2, level=0.0, delta=(4.5 if i == 2 else 0.0)) for i in range(3)]
    S = [dict(r) for r in R]
    S[2] = dict(S[2], level=-6.0, delta=-2.0)                            # 끝 음절: 낮고(6반음) 의문 올림 대신 내림
    C = compare(dict(syllables=R), dict(syllables=S))
    assert C["selected"][0]["type"] == "pitch_dir" and C["selected"][0]["i"] == 2, C["selected"]
    S = [dict(r, level=r["level"] + (2.6 if i == 1 else 0.0)) for i, r in enumerate(R)]   # 허용치 1.3배 — 다듬을 곳
    C = compare(dict(syllables=R), dict(syllables=S))
    assert C["quality"]["grade"] == "pass" and not C["selected"] and C["minor"][0]["type"] == "pitch_level"
    Rv = [dict(r, voiced=0.9) for r in R]
    Sv = [dict(r, voiced=0.05, level=None, delta=None) for r in R]
    C = compare(dict(syllables=Rv), dict(syllables=Sv))
    assert C["selected"][0]["type"] == "voicing" and C["selected"][0]["dir"] == "whisper"
    Sv[1] = dict(Sv[1], level=6.0, delta=0.0, voiced=0.3, dur=0.6)       # 남은 유성 조각이 높게 잡히고 한 음절은 길게 끎
    C = compare(dict(syllables=Rv), dict(syllables=Sv))
    assert C["selected"][0]["type"] == "voicing" and not any(x["type"].startswith("pitch") for x in C["issues"]), C["issues"]
    assert any(x["type"] == "duration" for x in C["selected"]), C["selected"]
    from mimicry.feedback import describe, retake
    assert "목소리를 내서" in describe(C["selected"][0], T)["how"] or "성대를 울려" in describe(C["selected"][0], T)["how"]
    q = compare(dict(syllables=Rv), dict(syllables=Sv, line_jer=0.9, line_heard=0.8))["quality"]   # 속삭여서 받아 적기도 나쁨
    assert q["reason"] == "whisper" and "목소리를 내서" in retake(q, ["다니냐"])["how"]
    q = compare(dict(syllables=Rv), dict(syllables=Sv, line_jer=1.0, line_heard=0.0))["quality"]   # 잡음만 — 유성 0 이어도 속삭임 아님
    assert q["reason"] == "silent", q


def test_pause_uncertain():
    from mimicry.compare import compare
    T = list("가라하와이")
    R = [syl(i, T[i], 0 if i < 2 else 1, 0.1 + 0.3 * i, 0.2, pause=(1.69 if i == 2 else 0.0)) for i in range(5)]
    S = [dict(r) for r in R]
    S[2] = dict(S[2], pause_before=0.3, bound_ok=False)                  # 잡음이 쉼을 채워 짧게 잡힘 + 경계 확신 없음
    C = compare(dict(syllables=R), dict(syllables=S))
    assert "pause" not in C["rows"][2]["checks"], C["rows"][2]           # "덜 쉬었다"는 확신 없는 음절에서 안 짚는다
    R2 = [dict(r, pause_before=0.0) for r in R]
    S2 = [dict(r) for r in R2]
    S2[2] = dict(S2[2], pause_before=0.5, bound_ok=False)                # 원본엔 없는 0.5초 쉼 — 경계가 흔들려도 짚는다
    C = compare(dict(syllables=R2), dict(syllables=S2))
    assert C["selected"] and C["selected"][0]["type"] == "pause" and C["selected"][0]["dir"] == "extra", C["selected"]


def test_pitch_stats():
    from mimicry.analyze import baseline_st, pitch_stats
    t = np.arange(100) * 0.01
    st = np.where((t > 0.2) & (t < 0.8), 12.0 * (t - 0.2), np.nan)          # 0.2~0.8초 사이 0 → 7.2 반음 오름
    level, delta, nv, nf = pitch_stats(t, st, 0.2, 0.8)
    assert nv == nf > 50 and abs(level - 3.6) < 0.2 and 4.0 < delta < 5.5, (level, delta, nv, nf)   # 끝 1/3 − 앞 1/3 ≈ 2/3 경사
    assert pitch_stats(t, st, 0.0, 0.05)[:2] == (None, None)
    jumpy = np.where(t < 0.25, -9.0, -5.0 - 2.0 * (t - 0.25))                    # 발성 시작 3프레임이 4반음 아래로 튐
    level, delta, _, _ = pitch_stats(t, jumpy, 0.21, 0.37)
    assert abs(level + 5.1) < 0.1 and abs(delta) < 0.3, (level, delta)               # 튄 3프레임을 빼고 잰다
    half = np.where(t < 0.3, 0.0, 6.0)                                               # 반반으로 갈림 → 어느 쪽인지 모름
    assert pitch_stats(t, half, 0.2, 0.4)[:2] == (None, None)
    assert abs(baseline_st(t, st, (0.0, 1.0)) - 3.6) < 0.2 and baseline_st(t, st, (0.0, 0.1)) == 0.0


def test_channels():
    from mimicry import synth
    rng = np.random.default_rng(0)
    x = (0.3 * np.sin(2 * math.pi * 200 * np.arange(16000) / 16000.0)).astype(np.float32)
    n = rng.standard_normal(16000).astype(np.float32) * 5.0
    y = synth.mix_at(x, n, 10.0)
    assert abs(10 * np.log10(np.mean(x ** 2) / np.mean((y - x) ** 2)) - 10.0) < 0.2   # 활성 구간 세기 기준 SNR
    c = synth.clip(x, 12.0)
    assert np.max(np.abs(c)) < 0.3 * 0.26 and len(synth.loop_to(x[:1000], 4000, rng)) == 4000
    pts = synth._dur_points([(0.1, 0.2, 1.5), (0.2, 0.3, 0.8), (0.5, 0.6, 1.2)])
    ts = [p[0] for p in pts]
    assert ts == sorted(ts) and len(set(ts)) == len(ts) and pts[0] == (0.1 - 0.005, 1.0) and pts[-1][1] == 1.0
    assert (0.2, 0.8) in pts and (0.3 + 0.005, 1.0) in pts and (0.5 - 0.005, 1.0) in pts     # 이웃은 바로 잇고 떨어진 건 1 로


def main():
    for f in (test_text, test_octaves, test_compare_feedback, test_dtw, test_locate, test_project_keep, test_report,
              test_line_gate, test_trim_missing, test_rank_tiers, test_pause_uncertain, test_pitch_stats, test_channels):
        f()
        print("  ✓ %s" % f.__name__)
    print("통과")


if __name__ == "__main__":
    main()
