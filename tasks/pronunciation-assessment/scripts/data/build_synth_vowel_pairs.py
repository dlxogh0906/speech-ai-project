# -*- coding: utf-8 -*-
"""모음 · ㅅ/ㅆ 짝 합성 문장 목록 — `build_synth_tense_pairs.py`(세기 쌍 2차)와 같은 꼴(texts_*.jsonl)로 만든다.

왜(2026-10-19 서비스 점검): 학습자 모음 오류의 화면 지적이 17% 이고 ㅗ→ㅓ · ㅛ→ㅕ 는 모델이 5~6% 만 듣는다.
ㅆ→ㅅ 는 4%. 세기 쌍 2차 합성에는 모음 짝이 없었고 ㅅ/ㅆ 짝은 검증에서 34~46% 가 버려졌다.

  - 모음: 학습자 실제 혼동(라벨 대비 TAG, 6,720발화) 상위 쌍만 — ㅗ↔ㅓ · ㅗ↔ㅜ · ㅜ↔ㅡ · ㅡ↔ㅣ · ㅛ↔ㅕ · ㅓ↔ㅕ.
    쌍마다 몫 = 실제 횟수 비율(바닥 3%). 글자 하나의 가운뎃소리를 바꾸고, 표준발음이 그 자리 하나만 달라지는 것만 쓴다.
  - ㅅ/ㅆ: 첫소리(발화 첫 자리 · 모음 사이 · 공명음 뒤)만 — 1·2차와 같은 자리 규칙. 버려지는 몫을 생각해 넉넉히 만든다.
  - 문장: 학습 문항(외운 문장 반례) + Kspon 문장. 사람 녹음 시험셋 낱말은 빼는 규칙(1·2차와 같다)을 그대로 쓴다.
  - 목소리 · 엔진 · 짝(…c/…w) 꼴은 2차와 같다 → `synth_tense_tts_qwen.py` · `build_synth_tense_pairs_manifest.py` 를 그대로 쓴다.

    $PY .../build_synth_vowel_pairs.py --n-kspon 4500 --per-prompt 3 --sss-frac 0.35 --out $MDD_OUTPUTS/synth_vowel_v3
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, collections, importlib.util, json, os, random, re, time

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
# 학습자 실제 모음 혼동 횟수(2026-10-19, 정답→실제) — 몫을 정하는 데만 쓴다
VOWEL_PAIRS = {"ㅗㅓ": 49, "ㅗㅜ": 29, "ㅓㅗ": 28, "ㅡㅣ": 27, "ㅜㅡ": 25, "ㅛㅕ": 22, "ㅓㅕ": 16, "ㅜㅗ": 12, "ㅡㅜ": 9,
               "ㅕㅛ": 7, "ㅕㅓ": 7, "ㅣㅡ": 6}
SSS_PAIRS = ("ㅅㅆ", "ㅆㅅ")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_pl.Path(__file__).with_name(name + ".py")))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trn", default=os.path.join(str(DATA_ROOT), "aihub", "10.한국어음성", "KsponSpeech_scripts", "train.trn"))
    ap.add_argument("--prompts", default=os.path.join(str(MDD_OUTPUTS), "analysis", "npu_bench_pack_2026-10-03", "train_prompts.json"))
    ap.add_argument("--zeroth", default=os.path.join(str(DATA_ROOT), "zeroth", "wav16k", "train_data_01", "003"))
    ap.add_argument("--evalset", default=str(_r / "tasks" / "pronunciation-assessment" / "evalsets" / "deliberate_human_v1.csv"))
    ap.add_argument("--n-kspon", dest="n_kspon", type=int, default=4500)
    ap.add_argument("--per-prompt", dest="per_prompt", type=int, default=3)
    ap.add_argument("--sss-frac", dest="sss_frac", type=float, default=0.35, help="ㅅ/ㅆ 짝 몫(나머지는 모음)")
    ap.add_argument("--qwen-frac", dest="qwen_frac", type=float, default=1.0, help="Qwen 몫(나머지 Chatterbox). Chatterbox 는 4배 느리다")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(str(MDD_OUTPUTS), "synth_vowel_v3"))
    a = ap.parse_args()

    v2 = _load("build_synth_tense_pairs")              # eval_exclusions · clean_kspon 를 같이 쓴다
    from shared.paths import G2P_TOOLS, ensure_kog2p
    from mdd.canonical_fix import CanonicalBuilder
    from mdd.pair_rescue import position
    ensure_kog2p()
    argv, _sys.argv = _sys.argv, _sys.argv[:1]
    try:
        from g2p import readRules, graph2prono, ver_info
    finally:
        _sys.argv = argv
    from mdd.jamo_to_kog2p import phones_to_jamo, INV
    ri, ro = readRules(ver_info[0], str(G2P_TOOLS / "KoG2P" / "rulebook.txt"))

    def kog2p(text):
        text = re.sub(r"\s+", " ", re.sub(r"[^0-9A-Za-z가-힣\s]", " ", text or "")).strip()
        out = []
        for j in phones_to_jamo([p for p in graph2prono(text, ri, ro).split() if p in INV]):
            out += {"ㅙ": ["ㅗ", "ㅐ"], "ㅞ": ["ㅜ", "ㅔ"]}.get(j, [j])
        return out
    B = CanonicalBuilder(kog2p)
    excl_words, excl_sents = v2.eval_exclusions(a.evalset)
    tot_v = sum(VOWEL_PAIRS.values())
    P = dict((k, (1 - a.sss_frac) * max(0.03, v / tot_v)) for k, v in VOWEL_PAIRS.items())
    norm_p = sum(P.values())
    P = dict((k, (1 - a.sss_frac) * v / norm_p) for k, v in P.items())
    P.update((k, a.sss_frac / 2) for k in SSS_PAIRS)
    rng = random.Random(a.seed)

    def swaps(s):
        c0 = B.fixed(s)
        if not c0:
            return []
        avoid = set()
        for w in excl_words:
            for m in re.finditer(re.escape(w), s):
                avoid.update(range(m.start(), m.end()))
        idx = [k for k, ch in enumerate(s) if "가" <= ch <= "힣" and k not in avoid]
        rng.shuffle(idx)
        opts = []
        for k in idx[:12]:
            code = ord(s[k]) - 0xAC00
            cho, jung, jong = code // 588, (code % 588) // 28, code % 28
            alts = []
            for pr in VOWEL_PAIRS:                                  # 가운뎃소리 바꾸기
                if JUNG[jung] == pr[0]:
                    alts.append((chr(0xAC00 + cho * 588 + JUNG.index(pr[1]) * 28 + jong), pr, "vowel"))
            for pr in SSS_PAIRS:                                    # ㅅ/ㅆ 첫소리 바꾸기
                if CHO[cho] == pr[0]:
                    alts.append((chr(0xAC00 + CHO.index(pr[1]) * 588 + jung * 28 + jong), pr, "onset"))
            for ch, pr, kind in alts:
                s1 = s[:k] + ch + s[k + 1:]
                c1 = B.fixed(s1)
                if len(c1) != len(c0):
                    continue
                d = [j for j in range(len(c0)) if c0[j] != c1[j]]
                if len(d) != 1 or c0[d[0]] + c1[d[0]] != pr:
                    continue
                j = d[0]
                pos = "vowel" if kind == "vowel" else position(c0, j)
                if kind == "onset" and pos not in v2.POSITIONS:
                    continue
                opts.append((s1, c0, c1, j, pr, pos))
        return opts

    cnt = collections.Counter()

    def pick(opts):
        tot = max(1, sum(cnt.values()))
        return max(opts, key=lambda o: P[o[4]] - cnt[o[4]] / tot + 1e-6 * rng.random())

    voices = []
    for spk in sorted(os.listdir(a.zeroth)):
        d = os.path.join(a.zeroth, spk)
        wavs = sorted(f for f in os.listdir(d) if f.endswith(".wav"))
        trans = [f for f in os.listdir(d) if f.endswith(".trans.txt")]
        if not wavs or not trans:
            continue
        tr = dict(l.strip().split(" ", 1) for l in open(os.path.join(d, trans[0]), encoding="utf-8") if " " in l.strip())
        w = wavs[len(wavs) // 3]                                    # 2차(가운데)와 다른 참조 녹음
        if w[:-4] in tr:
            voices.append((os.path.join(d, w), tr[w[:-4]]))
    print("[목소리] %d" % len(voices), flush=True)

    norm = lambda s: re.sub(r"[^가-힣]", "", s)
    pairs = []
    prompts = [p for p in json.load(open(a.prompts, encoding="utf-8"))["all"] if norm(p) not in excl_sents]
    for p in prompts:
        opts = swaps(p)
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
    seen, sents = set(), []
    for line in open(a.trn, encoding="utf-8"):
        t = v2.clean_kspon(line.split("::", 1)[-1])
        if not re.fullmatch(r"[가-힣 ]+[.?!]?", t) or t in seen:
            continue
        nsyl, nw = len(re.findall(r"[가-힣]", t)), len(t.split())
        if not (6 <= nsyl <= 30 and 2 <= nw <= 8) or any(w in t for w in excl_words) or norm(t) in excl_sents:
            continue
        seen.add(t); sents.append(t)
    rng.shuffle(sents)
    t0 = time.time()
    for s in sents:
        if len(pairs) - n_prompt >= a.n_kspon:
            break
        opts = swaps(s)
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
        pid = "vp%06d" % i
        base = {"pid": pid, "voice": voice, "voice_text": vtext, "engine": eng, "src": src, "text_ref": s, "canon": c0,
                "pair": pr, "target": j, "position": pos}
        for ver, txt, act in (("c", s, c0), ("w", s1, c1)):
            files[eng].write(json.dumps(dict(base, sid=pid + ver, version=ver, text_spoken=txt, actual=act), ensure_ascii=False) + "\n")
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
