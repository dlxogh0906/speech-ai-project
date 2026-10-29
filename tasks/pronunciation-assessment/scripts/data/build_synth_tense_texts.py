# -*- coding: utf-8 -*-
"""세기 쌍(평음·경음·격음) 오류 합성용 문장 목록 — "한국 → 한쿡" 처럼 글자 하나의 초성을 같은 계열로 바꿔 쓰고,
TTS 가 그 글자를 읽게 해 **정답(실제로 낸 소리)이 붙은 오류 음성**을 만든다.

문장 = KsponSpeech 전사(일상 대화, 학습 프롬프트 287문장 밖). 잡음 표시·이중 전사를 정리하고 2~8어절 · 6~30음절 · 한글만 남긴다.
시험용 단어(한국·가방·공부 … `--exclude-words`)가 든 문장은 뺀다 — 일부러 틀린 발음 시험셋을 깨끗하게 두기 위해.
오류 = 글자 초성 하나를 바꾼 뒤 KoG2P 표준발음끼리 비교해 **정확히 한 자리**가 같은 계열 다른 자모로 바뀐 경우만 쓴다
       (음운규칙 때문에 쓰인 글자와 소리가 다른 자리는 표준발음 기준 쌍으로 기록된다. 예: 입구→입쿠 는 ㄲ→ㅋ).
쌍 고르기 = 목표 비율 ∝ 실제 오류 수 × 서비스 모델이 표준으로 덮은 비율(`--pair-stats`, eval_pair_rescue 결과) — "자주 틀리고
       자주 놓치는" 쌍. 문장마다 가능한 쌍 중 **목표 대비 가장 모자란 쌍**을 고른다(평음 ㄱ·ㅅ·ㅈ 가 글에 훨씬 흔해서
       그냥 뽑으면 그쪽으로 쏠린다).
--correct-frac 만큼은 바꾸지 않은 문장(맞게 읽은 합성 음성)으로 둔다 — 합성 목소리 = 오류 라는 지름길을 막는다.
목소리 = Zeroth 한국어 train 화자(CC BY 4.0)의 녹음 하나씩(test 화자는 쓰지 않는다).

출력 한 줄 = {sid, voice, text_ref(원문), text_spoken(TTS 가 읽을 글), canon(원문 표준 자모), actual(읽을 글 표준 자모), pair}

    $PY .../build_synth_tense_texts.py --n 12000
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, G2P_TOOLS, MDD_OUTPUTS, ensure_kog2p

import argparse, collections, json, os, random, re, time

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
TEST_WORDS = "한국,가방,공부,다리,바다,자동차,사과,고기,두부,바지,자두,커피,토끼,포도,친구"


def clean_kspon(t):
    t = re.sub(r"\(([^)]*)\)/\(([^)]*)\)", r"\2", t)          # (숫자)/(읽은 대로) → 읽은 대로
    t = re.sub(r"\b[bnoul]/", " ", t)                          # 잡음 표시
    t = t.replace("*", " ").replace("+", " ").replace("/", " ")
    return re.sub(r"\s+", " ", t).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trn", default=os.path.join(str(DATA_ROOT), "aihub", "10.한국어음성", "KsponSpeech_scripts", "train.trn"))
    ap.add_argument("--zeroth", default=os.path.join(str(DATA_ROOT), "zeroth", "wav16k", "train_data_01", "003"))
    ap.add_argument("--pair-stats", dest="pair_stats",
                    default=os.path.join(str(MDD_OUTPUTS), "analysis", "pair_rescue_2026-10-10", "report.json"))
    ap.add_argument("--exclude-words", dest="exclude_words", default=TEST_WORDS)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--correct-frac", dest="correct_frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(str(MDD_OUTPUTS), "synth_tense_v1", "texts.jsonl"))
    a = ap.parse_args()

    ensure_kog2p()
    argv, _sys.argv = _sys.argv, _sys.argv[:1]
    try:
        from g2p import readRules, graph2prono, ver_info
    finally:
        _sys.argv = argv
    from mdd.jamo_to_kog2p import phones_to_jamo, INV
    from mdd.pair_rescue import FAMILY_OF
    import edlib
    ri, ro = readRules(ver_info[0], str(G2P_TOOLS / "KoG2P" / "rulebook.txt"))

    def canonical(s):
        out = []
        for j in phones_to_jamo([p for p in graph2prono(s, ri, ro).split() if p in INV]):
            out += {"ㅙ": ["ㅗ", "ㅐ"], "ㅞ": ["ㅜ", "ㅔ"]}.get(j, [j])
        return out

    def one_change(c0, c1):
        if len(c0) == len(c1):
            d = [j for j in range(len(c0)) if c0[j] != c1[j]]
            return d[0] if len(d) == 1 else None
        return None

    # 쌍 무게: 실제 오류 수 × 표준으로 덮은 비율(서비스 경로)
    st = json.load(open(a.pair_stats, encoding="utf-8"))["budgets"]["0.01"]["모든 쌍"]["pairs"]
    W = dict((p, v["err"] * v["revert"][0]) for p, v in st.items())
    P = dict((p, w / sum(W.values())) for p, w in W.items())
    print("[쌍 무게] " + " · ".join("%s→%s %.0f" % (p[0], p[1], w) for p, w in sorted(W.items(), key=lambda x: -x[1])), flush=True)

    excl = [w for w in a.exclude_words.split(",") if w]
    seen, sents = set(), []
    for line in open(a.trn, encoding="utf-8"):
        t = clean_kspon(line.split("::", 1)[-1])
        if not re.fullmatch(r"[가-힣 ]+[.?!]?", t) or t in seen:
            continue
        nsyl, nw = len(re.findall(r"[가-힣]", t)), len(t.split())
        if not (6 <= nsyl <= 30 and 2 <= nw <= 8) or any(w in t for w in excl):
            continue
        seen.add(t); sents.append(t)
    rng = random.Random(a.seed)
    rng.shuffle(sents)
    voices = []
    for spk in sorted(os.listdir(a.zeroth)):
        wavs = sorted(f for f in os.listdir(os.path.join(a.zeroth, spk)) if f.endswith(".wav"))
        if wavs:
            voices.append(os.path.join(a.zeroth, spk, wavs[len(wavs) // 2]))
    print("[문장] 후보 %d · 목소리 %d" % (len(sents), len(voices)), flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out, cnt, t0 = [], collections.Counter(), time.time()
    for s in sents:
        if len(out) >= a.n:
            break
        c0 = canonical(s)
        if not c0:
            continue
        rec = {"sid": "st%06d" % len(out), "voice": voices[len(out) % len(voices)], "text_ref": s, "canon": c0}
        if rng.random() < a.correct_frac:
            rec.update(text_spoken=s, actual=c0, pair=None)
        else:
            idx = [k for k, ch in enumerate(s) if "가" <= ch <= "힣" and CHO[(ord(ch) - 0xAC00) // 588] in FAMILY_OF]
            rng.shuffle(idx)
            opts = []
            for k in idx[:8]:
                code = ord(s[k]) - 0xAC00
                cho, rest = code // 588, code % 588
                for alt in FAMILY_OF[CHO[cho]]:
                    if alt == CHO[cho]:
                        continue
                    s1 = s[:k] + chr(0xAC00 + CHO.index(alt) * 588 + rest) + s[k + 1:]
                    c1 = canonical(s1)
                    j = one_change(c0, c1)
                    if j is not None and FAMILY_OF.get(c0[j]) == FAMILY_OF.get(c1[j]) and (c0[j] + c1[j]) in W:
                        opts.append((s1, c1, c0[j] + c1[j]))
            if not opts:
                continue
            tot = max(1, sum(v for k_, v in cnt.items() if k_ != "맞게"))
            deficit = lambda o: P[o[2]] - cnt[o[2]] / tot + 1e-6 * rng.random()
            s1, c1, pr = max(opts, key=deficit)
            rec.update(text_spoken=s1, actual=c1, pair=pr)
        out.append(rec); cnt[rec["pair"] or "맞게"] += 1
        if len(out) % 1000 == 0:
            print("  %d · %.0fs" % (len(out), time.time() - t0), flush=True)
    with open(a.out, "w", encoding="utf-8") as fo:
        for r in out:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[완료] %s · %d문장 · %s" % (a.out, len(out), " · ".join("%s %d" % (k if k == "맞게" else k[0] + "→" + k[1], v)
                                                              for k, v in cnt.most_common())), flush=True)


if __name__ == "__main__":
    main()
