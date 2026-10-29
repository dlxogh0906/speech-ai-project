# -*- coding: utf-8 -*-
"""세기 쌍 합성 2차 — **같은 목소리로 맞게/틀리게 읽은 짝(최소대립)** 문장 목록. `build_synth_tense_texts.py`(1차)를
전문가 리뷰(2026-10-11)대로 고쳤다.

1차와 다른 점
  - 짝: 틀린 문장마다 같은 목소리·같은 엔진으로 맞는 문장도 만든다(sid …c / …w). '합성 목소리 = 오류' 지름길을 막고,
    한 자리만 다른 두 녹음을 같이 보여 준다. 짝 검증(`build_synth_tense_manifest.py --pairs`)에서 틀린 쪽이 실제로
    다르게 들리지 않으면 짝째 버린다.
  - 외운 문장 반례: 학습 문항(train_prompts.json — 155만 발화가 이 문항뿐이라 모델이 외워 읽는다) 자체에 오류를 넣는다.
    "한국어"를 "한쿡어"로 읽어도 외운 대로 "한국어"라고 적는 문제를 직접 겨냥한다.
  - 자주 외운 낱말(--boost-words: 한국어·공부·두부·사과·친구 …)이 든 Kspon 문장은 그 낱말 안에서 바꾼다.
  - 시험 오염 방지: 사람 녹음 시험셋(evalsets/deliberate_human_v1.csv)의 '일반'·'최소대립' 낱말이 든 문장과 시험 문장
    자체는 쓰지 않는다('외운 단어' 묶음은 일부러 학습에 넣고 효과를 본다 — 일반화는 '일반' 묶음으로 잰다).
  - 자리: 첫소리만 — 발화 첫 자리 · 모음 사이 · 공명음 뒤(`mdd.pair_rescue.position`). 받침과 장애음 뒤(경음화 자리)는 뺀다.
  - 표준발음: `mdd.canonical_fix.CanonicalBuilder.fixed`(KoG2P 경음화 누락 등 보정) — 1차 라벨의 37개는 실제로는
    표준발음이었다.
  - 엔진: Chatterbox Multilingual(MIT) · Qwen3-TTS-12Hz-1.7B-Base(Apache-2.0) 를 짝 단위로 번갈아(--qwen-frac).
    엔진 하나에서만 얻은 효과가 사람에게 안 옮겨 간 1차 결과 때문.
쌍 고르기는 1차와 같다(실제 오류 수 × 표준으로 덮은 비율, 모자란 쌍 먼저).
목소리 = Zeroth 한국어 train 화자(CC BY 4.0)의 녹음 하나와 그 전사(Qwen ICL 참조용). test 화자는 쓰지 않는다.

출력: <out>/texts_chatterbox.jsonl · <out>/texts_qwen.jsonl — 한 줄 =
  {sid, pid, version(c|w), voice, voice_text, engine, src(kspon|prompt), text_ref, text_spoken, canon, actual, pair, target, position}

    $PY .../build_synth_tense_pairs.py --n-kspon 6000 --per-prompt 6
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, G2P_TOOLS, MDD_OUTPUTS, ensure_kog2p

import argparse, collections, csv, json, os, random, re, time

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
BOOST = "한국어,한국말,공부,두부,사과,친구"
POSITIONS = ("initial", "intervocalic", "post_sonorant")


def clean_kspon(t):
    t = re.sub(r"\(([^)]*)\)/\(([^)]*)\)", r"\2", t)          # (숫자)/(읽은 대로) → 읽은 대로
    t = re.sub(r"\b[bnoul]/", " ", t)                          # 잡음 표시
    t = t.replace("*", " ").replace("+", " ").replace("/", " ")
    return re.sub(r"\s+", " ", t).strip()


def eval_exclusions(path):
    """사람 녹음 시험셋 → (빼야 할 낱말, 시험 문장 글자들). '외운 단어' 묶음의 낱말은 빼지 않는다."""
    words, sents = set(), set()
    norm = lambda s: re.sub(r"[^가-힣]", "", s)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        sents.add(norm(r["ref_text"]))
        if r["wrong_text"]:
            sents.add(norm(r["wrong_text"]))
        if r["group"] in ("일반", "최소대립") and r["wrong_text"]:
            a, b = r["ref_text"].split(), r["wrong_text"].split()
            for x, y in zip(a, b):
                if x != y:                                      # 바뀐 어절의 줄기(조사 앞 두 글자까지)
                    words.add(re.sub(r"[^가-힣]", "", x)[:2]); words.add(re.sub(r"[^가-힣]", "", y)[:2])
    return words, sents


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trn", default=os.path.join(str(DATA_ROOT), "aihub", "10.한국어음성", "KsponSpeech_scripts", "train.trn"))
    ap.add_argument("--prompts", default=os.path.join(str(MDD_OUTPUTS), "analysis", "npu_bench_pack_2026-10-03", "train_prompts.json"))
    ap.add_argument("--zeroth", default=os.path.join(str(DATA_ROOT), "zeroth", "wav16k", "train_data_01", "003"))
    ap.add_argument("--pair-stats", dest="pair_stats",
                    default=os.path.join(str(MDD_OUTPUTS), "analysis", "pair_rescue_2026-10-10", "report.json"))
    ap.add_argument("--evalset", default=str(_r / "tasks" / "pronunciation-assessment" / "evalsets" / "deliberate_human_v1.csv"))
    ap.add_argument("--boost-words", dest="boost_words", default=BOOST)
    ap.add_argument("--n-kspon", dest="n_kspon", type=int, default=6000, help="Kspon 문장 짝 수")
    ap.add_argument("--per-prompt", dest="per_prompt", type=int, default=6, help="학습 문항 하나당 오류 짝 수")
    ap.add_argument("--qwen-frac", dest="qwen_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(str(MDD_OUTPUTS), "synth_tense_v2"))
    a = ap.parse_args()

    ensure_kog2p()
    argv, _sys.argv = _sys.argv, _sys.argv[:1]
    try:
        from g2p import readRules, graph2prono, ver_info
    finally:
        _sys.argv = argv
    from mdd.jamo_to_kog2p import phones_to_jamo, INV
    from mdd.pair_rescue import FAMILY_OF, position
    from mdd.canonical_fix import CanonicalBuilder
    ri, ro = readRules(ver_info[0], str(G2P_TOOLS / "KoG2P" / "rulebook.txt"))

    def kog2p(s):                                    # = 발음 API canonical()
        text = re.sub(r"\s+", " ", re.sub(r"[^0-9A-Za-z가-힣\s]", " ", s or "")).strip()
        out = []
        for j in phones_to_jamo([p for p in graph2prono(text, ri, ro).split() if p in INV]):
            out += {"ㅙ": ["ㅗ", "ㅐ"], "ㅞ": ["ㅜ", "ㅔ"]}.get(j, [j])
        return out
    B = CanonicalBuilder(kog2p)

    st = json.load(open(a.pair_stats, encoding="utf-8"))["budgets"]["0.01"]["모든 쌍"]["pairs"]
    W = dict((p, v["err"] * v["revert"][0]) for p, v in st.items())
    P = dict((p, w / sum(W.values())) for p, w in W.items())
    excl_words, excl_sents = eval_exclusions(a.evalset)
    boost = [w for w in a.boost_words.split(",") if w]
    print("[시험 오염 방지] 빼는 낱말 %d: %s" % (len(excl_words), " ".join(sorted(excl_words))), flush=True)

    def swaps(s, rng, span=None):
        """s 의 글자 하나 초성을 같은 계열로 바꾼 후보 [(s1, c0, c1, j, pair, pos)] — 표준발음 한 자리만 바뀌고 첫소리 자리인 것."""
        c0 = B.fixed(s)
        if not c0:
            return None, []
        avoid = set()
        for w in excl_words:                         # 학습 문항 안의 시험 낱말(지금 · 자리 …)은 건드리지 않는다
            for m in re.finditer(re.escape(w), s):
                avoid.update(range(m.start(), m.end()))
        idx = [k for k, ch in enumerate(s) if "가" <= ch <= "힣" and CHO[(ord(ch) - 0xAC00) // 588] in FAMILY_OF
               and (span is None or span[0] <= k < span[1]) and k not in avoid]
        rng.shuffle(idx)
        opts = []
        for k in idx[:10]:
            code = ord(s[k]) - 0xAC00
            cho, rest = code // 588, code % 588
            for alt in FAMILY_OF[CHO[cho]]:
                if alt == CHO[cho]:
                    continue
                s1 = s[:k] + chr(0xAC00 + CHO.index(alt) * 588 + rest) + s[k + 1:]
                c1 = B.fixed(s1)
                if len(c1) != len(c0):
                    continue
                d = [j for j in range(len(c0)) if c0[j] != c1[j]]
                if len(d) != 1:
                    continue
                j = d[0]
                pos = position(c0, j)
                if FAMILY_OF.get(c0[j]) != FAMILY_OF.get(c1[j]) or (c0[j] + c1[j]) not in W or pos not in POSITIONS:
                    continue
                opts.append((s1, c0, c1, j, c0[j] + c1[j], pos))
        return c0, opts

    rng = random.Random(a.seed)
    # 목소리: 화자마다 가운데 녹음 하나 + 전사
    voices = []
    for spk in sorted(os.listdir(a.zeroth)):
        d = os.path.join(a.zeroth, spk)
        wavs = sorted(f for f in os.listdir(d) if f.endswith(".wav"))
        trans = [f for f in os.listdir(d) if f.endswith(".trans.txt")]
        if not wavs or not trans:
            continue
        tr = dict(l.strip().split(" ", 1) for l in open(os.path.join(d, trans[0]), encoding="utf-8") if " " in l.strip())
        w = wavs[len(wavs) // 2]
        if w[:-4] in tr:
            voices.append((os.path.join(d, w), tr[w[:-4]]))
    print("[목소리] %d" % len(voices), flush=True)

    cnt = collections.Counter()
    pairs = []

    def pick(opts):
        tot = max(1, sum(cnt.values()))
        return max(opts, key=lambda o: P[o[4]] - cnt[o[4]] / tot + 1e-6 * rng.random())

    # 1) 학습 문항 반례
    norm = lambda s: re.sub(r"[^가-힣]", "", s)
    prompts = [p for p in json.load(open(a.prompts, encoding="utf-8"))["all"] if norm(p) not in excl_sents]
    for p in prompts:
        c0, opts = swaps(p, rng)
        rng.shuffle(opts)
        used = set()
        for _ in range(a.per_prompt):
            opts = [o for o in opts if o[3] not in used]
            if not opts:
                break
            o = pick(opts)
            used.add(o[3]); cnt[o[4]] += 1
            pairs.append(("prompt", p, o))
    n_prompt = len(pairs)
    # 2) Kspon 문장(외운 낱말이 있으면 그 낱말 안에서)
    seen, sents = set(), []
    for line in open(a.trn, encoding="utf-8"):
        t = clean_kspon(line.split("::", 1)[-1])
        if not re.fullmatch(r"[가-힣 ]+[.?!]?", t) or t in seen:
            continue
        nsyl, nw = len(re.findall(r"[가-힣]", t)), len(t.split())
        if not (6 <= nsyl <= 30 and 2 <= nw <= 8) or any(w in t for w in excl_words) or norm(t) in excl_sents:
            continue
        seen.add(t); sents.append(t)
    rng.shuffle(sents)
    with_boost = [s for s in sents if any(w in s for w in boost)]
    rest = [s for s in sents if not any(w in s for w in boost)]
    print("[Kspon] 후보 %d (외운 낱말 포함 %d)" % (len(sents), len(with_boost)), flush=True)
    t0 = time.time()
    n_boost_target = min(len(with_boost), a.n_kspon // 4)          # 짝의 1/4 까지 외운 낱말 문장
    for s in with_boost:
        if len(pairs) - n_prompt >= n_boost_target:
            break
        w = next(w for w in boost if w in s)
        k = s.index(w)
        _c0, opts = swaps(s, rng, span=(k, k + len(w)))
        if opts:
            o = pick(opts); cnt[o[4]] += 1
            pairs.append(("kspon_boost", s, o))
    for s in rest:
        if len(pairs) - n_prompt >= a.n_kspon:
            break
        _c0, opts = swaps(s, rng)
        if opts:
            o = pick(opts); cnt[o[4]] += 1
            pairs.append(("kspon", s, o))
            if len(pairs) % 1000 == 0:
                print("  짝 %d · %.0fs" % (len(pairs), time.time() - t0), flush=True)

    os.makedirs(a.out, exist_ok=True)
    files = {"chatterbox": open(os.path.join(a.out, "texts_chatterbox.jsonl"), "w", encoding="utf-8"),
             "qwen": open(os.path.join(a.out, "texts_qwen.jsonl"), "w", encoding="utf-8")}
    order = list(range(len(pairs)))
    rng.shuffle(order)
    n_eng = collections.Counter()
    for n, i in enumerate(order):
        src, s, (s1, c0, c1, j, pr, pos) = pairs[i]
        eng = "qwen" if n < a.qwen_frac * len(pairs) else "chatterbox"
        voice, vtext = voices[i % len(voices)]
        pid = "sp%06d" % i
        base = {"pid": pid, "voice": voice, "voice_text": vtext, "engine": eng, "src": src, "text_ref": s, "canon": c0,
                "pair": pr, "target": j, "position": pos}
        for ver, txt, act in (("c", s, c0), ("w", s1, c1)):
            r = dict(base, sid=pid + ver, version=ver, text_spoken=txt, actual=act)
            files[eng].write(json.dumps(r, ensure_ascii=False) + "\n")
        n_eng[eng] += 1
    for f in files.values():
        f.close()
    stats = {"pairs": len(pairs), "by_src": dict(collections.Counter(p[0] for p in pairs)), "by_engine": dict(n_eng),
             "by_pair": dict(cnt.most_common()), "by_position": dict(collections.Counter(p[2][5] for p in pairs)),
             "voices": len(voices), "excluded_words": sorted(excl_words)}
    json.dump(stats, open(os.path.join(a.out, "texts_stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
