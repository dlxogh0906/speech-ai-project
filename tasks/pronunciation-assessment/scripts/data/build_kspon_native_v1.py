# -*- coding: utf-8 -*-
"""KsponSpeech(AIHub 한국어 음성) → 원어민 보조 매니페스트 (46자모 ltr, 외국인 TAG 와 같은 dict).

역할: 서비스 대상이 아니라 **문장 다양성** 보조 데이터다 (외국인 TAG = PRIMARY GOLD).

전사 정리 (KsponSpeech 전사 규칙)
  - 이중 전사 (철자)/(발음) → **발음** 쪽을 쓴다      예) (70%)/(칠십 퍼센트) → 칠십 퍼센트
  - b/ n/ l/ o/ (숨·잡음·웃음·다른 화자) 표시는 지운다
  - u/ (알아들을 수 없음) 또는 * (불명확) 이 있으면 첫 clean 실험에서 제외
  - "아/" "그/" 같은 간투사는 말한 소리이므로 남기고 "/" 만 지운다
  - "+" (반복) 표시만 지우고 반복한 말은 남긴다
  - 문장부호 제거. 발음 쪽을 골랐는데도 한글 이외 글자(영문·숫자)가 남으면 제외
  - 괄호·슬래시 짝이 깨진 전사는 제외(corrupt)

자모 변환 — 두 방식을 --audit 으로 먼저 비교한 뒤 전체에 적용한다
  A direct   음절을 그대로 자모로 쪼갠다(초성 ㅇ 은 TAG 관례대로 생략). 음운 규칙 미적용 = 철자 수준
  B kog2p    기존 원어민 전처리·pipeline.canonical 과 같다: 텍스트 → KoG2P(음운 규칙) → phones_to_jamo
  TAG 는 "들린 표면 발음"(예: 좋겠네요 → ㅈㅗㅋㅔㄴㄴㅔㅛ)이라 음운 규칙이 적용된 B 가 관례에 맞는다.
  KoG2P 는 입력 글자에 규칙만 적용하므로 발음 전사("몬")를 철자("뭔")로 되돌리지 않는다 — audit 에서 확인한다.

오디오: .pcm = 헤더 없는 16kHz · 16bit LE · mono (확인함). 리샘플 없이 wav 헤더만 붙여 --wav-out 에 쓴다.

    $PY .../build_kspon_native_v1.py --audit 500 --out-audit tasks/pronunciation-assessment/results/kspon_jamo_audit.md
    $PY .../build_kspon_native_v1.py --out $MDD_OUTPUTS/manifest_kspon_native_v1 --workers 12
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, G2P_TOOLS, ensure_kog2p

import argparse, collections, hashlib, json, os, random, re, time, wave
from multiprocessing import Pool

SR = 16000
L_ = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
V_ = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
T_ = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ",
      "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
DUAL = re.compile(r"\(([^()]*)\)/\(([^()]*)\)")

_G = {}


def g2p_init():
    ensure_kog2p()
    argv, _sys.argv = _sys.argv, _sys.argv[:1]          # g2p.py 가 import 시 argv 를 파싱한다
    try:
        from g2p import readRules, graph2prono, ver_info
    finally:
        _sys.argv = argv
    from mdd.jamo_to_kog2p import phones_to_jamo, INV
    _G.update(ri_ro=readRules(ver_info[0], str(G2P_TOOLS / "KoG2P" / "rulebook.txt")), g2p=graph2prono,
              p2j=phones_to_jamo, inv=INV)


def norm_text(t):
    return re.sub(r"[^0-9A-Za-z가-힣]", "", t or "")


def clean_transcript(raw):
    """→ (정리된 한글 문장, 사유). 사유가 None 이 아니면 제외."""
    t = raw.strip()
    if "*" in t:
        return None, "unclear_star"
    if re.search(r"(^|\s)u/", t):
        return None, "unintelligible_u"
    t = DUAL.sub(lambda m: m.group(2), t)
    if "(" in t or ")" in t:
        return None, "corrupt_paren"
    t = re.sub(r"(^|\s)[bnlo]/(?=\s|$)", " ", t)           # 비음성 표시
    t = re.sub(r"([가-힣A-Za-z0-9]+)/", r"\1", t)            # 간투사 "아/" → "아"
    if "/" in t:
        return None, "corrupt_slash"
    t = t.replace("+", " ")
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None, "empty_after_clean"
    if re.search(r"[^가-힣\s]", t):
        return None, "non_hangul_left"
    return t, None


def to_jamo_direct(text):
    out = []
    for ch in text:
        o = ord(ch) - 0xAC00
        if 0 <= o < 11172:
            l, v, tt = o // 588, (o % 588) // 28, o % 28
            if L_[l] != "ㅇ":
                out.append(L_[l])
            out.append(V_[v])
            if tt:
                out.append(T_[tt])
    return out


# 외국인 TAG/LAB 관례: ㅙ·ㅞ 를 한 글자로 적지 않고 두 자모로 적는다
# (manifest_tag_rebuilt train: LAB ㅙ 0 · "ㅗ ㅐ" 35,260 / ㅞ 0 · "ㅜ ㅔ" 12,411, 2026-09-15 실측). KoG2P 는 ㅙ·ㅞ 를 낸다.
TAG_VOWEL_SPLIT = {"ㅙ": ["ㅗ", "ㅐ"], "ㅞ": ["ㅜ", "ㅔ"]}


def to_jamo_kog2p(text):
    if not _G:
        g2p_init()
    ri, ro = _G["ri_ro"]
    # KoG2P 는 단어 사이 공백이 정확히 2칸이면 다음 음절을 깨뜨린다("c 0v v" → 음절 소실, 실측). 1칸으로 맞춘다.
    text = re.sub(r"\s+", " ", text).strip()
    phs = [p for p in _G["g2p"](text, ri, ro).split() if p in _G["inv"]]
    out = []
    for j in _G["p2j"](phs):
        out += TAG_VOWEL_SPLIT.get(j, [j])
    return out


def read_trn(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        if " :: " not in line:
            continue
        p, t = line.rstrip("\n").split(" :: ", 1)
        rows.append((p.strip(), t))
    return rows


# ───────────────────────────── audit ─────────────────────────────
def audit(a):
    rows = read_trn(os.path.join(a.kspon_root, "KsponSpeech_scripts", "train.trn"))
    rng = random.Random(a.seed)
    dual = [r for r in rows if DUAL.search(r[1])]
    pick = rng.sample(dual, min(a.audit // 2, len(dual))) + rng.sample(rows, a.audit - min(a.audit // 2, len(dual)))
    g2p_init()
    dict_syms = {l.split()[0] for l in open(a.dict, encoding="utf-8") if l.strip()}
    res, reasons = [], collections.Counter()
    for p, raw in pick:
        t, why = clean_transcript(raw)
        if why:
            reasons[why] += 1
            continue
        pairs = DUAL.findall(raw)
        # 철자 쪽이 한글인 쌍만 철자 쪽으로 바꿔 본다 — 숫자·영문 철자는 그대로 KoG2P 에 넣을 수 없어서 비교 불가
        spell = DUAL.sub(lambda m: m.group(1) if re.fullmatch(r"[가-힣\s]+", m.group(1).strip() or "x") else m.group(2), raw)
        spell_t, _ = clean_transcript(spell)
        A, B = to_jamo_direct(t), to_jamo_kog2p(t)
        Bs = to_jamo_kog2p(spell_t) if pairs and spell_t and spell_t != t else None
        res.append({"path": p, "raw": raw, "clean": t, "A": A, "B": B, "B_spelling": Bs, "n_dual": len(pairs),
                    "dual_nonhangul_spelling": sum(not re.fullmatch(r"[가-힣\s]+", s.strip() or "x") for s, _ in pairs),
                    "oovA": sorted({x for x in A if x not in dict_syms}), "oovB": sorted({x for x in B if x not in dict_syms})})
    same = sum(r["A"] == r["B"] for r in res)
    dual_rows = [r for r in res if r["B_spelling"] is not None]
    follows_phonetic = sum(r["B"] != r["B_spelling"] for r in dual_rows)
    diff_ops = collections.Counter()
    import edlib
    for r in res:
        if r["A"] == r["B"]:
            continue
        x, y = "".join(r["A"]), "".join(r["B"])
        al = edlib.getNiceAlignment(edlib.align(x, y, task="path"), x, y)
        for qa, tb in zip(al["query_aligned"], al["target_aligned"]):
            if qa != tb:
                diff_ops[f"{qa}→{tb}"] += 1
    L = ["# KsponSpeech 발음 전사 → 46자모 변환 audit", "",
         f"생성 {time.strftime('%Y-%m-%d %H:%M')} · 표본 {len(pick)} (이중 전사 포함 {min(a.audit // 2, len(dual))}) · seed {a.seed}", "",
         "## 요약", "",
         f"- 정리 후 사용 {len(res)} · 제외 {dict(reasons)}",
         f"- A(음절 직접 분해) == B(KoG2P→자모) : {same}/{len(res)}",
         f"- 이중 전사 발화 {len(dual_rows)}개 중 B 결과가 **발음 쪽**을 따른 것(철자 쪽 B 와 다름): {follows_phonetic}",
         f"- OOV(46자모 밖) — A: {sum(len(r['oovA']) > 0 for r in res)}발화 · B: {sum(len(r['oovB']) > 0 for r in res)}발화",
         f"- A↔B 차이 상위(A→B): {diff_ops.most_common(25)}", "",
         "## 판단 기준", "",
         "- TAG 는 음운 규칙이 적용된 표면 발음을 적는다(예: 좋겠네요 → ㅈㅗㅋㅔㄴㄴㅔㅛ). A↔B 차이가 음운 규칙(경음화·비음화·연음·"
         "받침 대표음 등)으로만 설명되면 B 가 TAG 관례와 맞다.",
         "- 이중 전사에서 B 가 발음 쪽을 따라야 한다(철자로 되돌아가면 실패).", "",
         "## 이중 전사 사례 (최대 25)", "", "| 원문 | 정리(발음 쪽) | B 발음 쪽 | B 철자 쪽 |", "|---|---|---|---|"]
    for r in dual_rows[:25]:
        L.append(f"| {r['raw']} | {r['clean']} | {''.join(r['B'])} | {''.join(r['B_spelling'])} |")
    L += ["", "## A↔B 가 다른 일반 사례 (최대 40)", "", "| 정리 문장 | A 직접 | B KoG2P |", "|---|---|---|"]
    for r in [r for r in res if r["A"] != r["B"]][:40]:
        L.append(f"| {r['clean']} | {''.join(r['A'])} | {''.join(r['B'])} |")
    os.makedirs(os.path.dirname(os.path.abspath(a.out_audit)), exist_ok=True)
    open(a.out_audit, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:12]), flush=True)


# ───────────────────────────── build ─────────────────────────────
def work(job):
    rel, raw, split, root, wav_out, method = job
    t, why = clean_transcript(raw)
    rec = {"rel": rel, "split": split, "raw": raw}
    if why:
        return {**rec, "fail": why}
    src = os.path.join(root, rel)
    try:
        n = os.path.getsize(src)
    except OSError:
        return {**rec, "fail": "audio_missing"}
    # eval_clean·eval_other 의 .pcm 은 전부 끝에 1바이트가 더 붙어 있다(train 은 0건, 2026-09-15 실측).
    # 앞부분은 train 과 같은 16bit LE 샘플이라 마지막 1바이트만 버린다.
    trimmed = n % 2
    frames = n // 2
    rec["pcm_trailing_byte_trimmed"] = bool(trimmed)
    sec = frames / SR
    try:
        ltr = to_jamo_kog2p(t) if method == "kog2p" else to_jamo_direct(t)
    except Exception as e:
        return {**rec, "fail": f"g2p_error:{type(e).__name__}"}
    if not ltr:
        return {**rec, "fail": "empty_label"}
    dst = os.path.join(wav_out, rel[:-4] + ".wav")
    if 0.5 <= sec <= 15.0 and not os.path.exists(dst):
        pcm = open(src, "rb").read()[: frames * 2]
        if not pcm.strip(b"\x00"):
            return {**rec, "fail": "pcm_all_zero"}
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = f"{dst}.{os.getpid()}.tmp"
        with wave.open(tmp, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm)
        os.replace(tmp, dst)
    return {**rec, "text": t, "norm_text": norm_text(t), "ltr": ltr, "frames": frames, "sec": round(sec, 3), "wav": dst}


def build(a):
    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit(f"출력 폴더가 비어 있지 않다(덮어쓰기 금지): {a.out}")
    dict_syms = [l.split()[0] for l in open(a.dict, encoding="utf-8") if l.strip()]
    assert len(dict_syms) == 46
    dset = set(dict_syms)
    sdir = os.path.join(a.kspon_root, "KsponSpeech_scripts")
    splits = {"train": "train.trn", "dev": "dev.trn", "eval_clean": "eval_clean.trn", "eval_other": "eval_other.trn"}
    if a.splits:
        splits = {k: v for k, v in splits.items() if k in a.splits.split(",")}
    jobs = []
    for sp, fn in splits.items():
        for rel, raw in read_trn(os.path.join(sdir, fn)):
            jobs.append((rel, raw, sp, a.kspon_root, a.wav_out, a.method))
    print(f"[kspon] 전사 {len(jobs):,}줄 · 변환 {a.method}", flush=True)
    t0 = time.time()
    recs = []
    with Pool(a.workers, initializer=g2p_init if a.method == "kog2p" else None) as pool:
        for i, r in enumerate(pool.imap(work, jobs, chunksize=200), 1):
            recs.append(r)
            if i % 100000 == 0:
                print(f"  {i:,}/{len(jobs):,} {time.time() - t0:.0f}s", flush=True)
    excl = collections.defaultdict(collections.Counter)
    out = collections.defaultdict(list)
    oov_list = []
    for r in recs:
        if "fail" in r:
            excl[r["split"]][r["fail"]] += 1; continue
        if not (0.5 <= r["sec"] <= 15.0):
            excl[r["split"]]["duration"] += 1; continue
        bad = sorted({x for x in r["ltr"] if x not in dset})
        if bad:
            excl[r["split"]]["oov"] += 1; oov_list.append({"rel": r["rel"], "text": r["text"], "oov": bad}); continue
        out[r["split"]].append(r)
    os.makedirs(a.out, exist_ok=True)
    wav_root = os.path.abspath(a.wav_out)
    stats = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(a), "excluded": {k: dict(v) for k, v in excl.items()},
             "oov_excluded": len(oov_list), "splits": {}}
    for sp, rs in out.items():
        with open(os.path.join(a.out, f"{sp}.tsv"), "w", encoding="utf-8") as ft, \
             open(os.path.join(a.out, f"{sp}.ltr"), "w", encoding="utf-8") as fl, \
             open(os.path.join(a.out, f"{sp}.meta.jsonl"), "w", encoding="utf-8") as fm:
            ft.write(wav_root + "\n")
            for r in rs:
                ft.write(f"{r['rel'][:-4]}.wav\t{r['frames']}\n")
                fl.write(" ".join(r["ltr"]) + "\n")
                fm.write(json.dumps({"rel": r["rel"], "raw": r["raw"], "text": r["text"], "norm_text": r["norm_text"],
                                     "sec": r["sec"]}, ensure_ascii=False) + "\n")
        toks = collections.Counter(x for r in rs for x in r["ltr"])
        secs = sorted(r["sec"] for r in rs)
        stats["splits"][sp] = {"utterances": len(rs), "hours": round(sum(secs) / 3600, 2),
                               "unique_texts": len({r["norm_text"] for r in rs}),
                               "duration_median": secs[len(secs) // 2] if secs else None,
                               "label_tokens_total": sum(toks.values()), "label_tokens_unique": len(toks),
                               "oov_tokens": {k: v for k, v in toks.items() if k not in dset}}
        with open(os.path.join(a.out, f"{sp}.texts.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(sorted({r["norm_text"] for r in rs})) + "\n")
    with open(os.path.join(a.out, "dict.ltr.txt"), "w", encoding="utf-8") as f:
        f.write(open(a.dict, encoding="utf-8").read())
    with open(os.path.join(a.out, "oov_excluded.jsonl"), "w", encoding="utf-8") as f:
        for o in oov_list:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    h = hashlib.sha256()
    for sp in sorted(out):
        for ext in ("tsv", "ltr"):
            h.update(open(os.path.join(a.out, f"{sp}.{ext}"), "rb").read())
    stats["manifest_sha256"] = h.hexdigest()
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), ensure_ascii=False, indent=1)
    for sp in stats["splits"]:
        assert not stats["splits"][sp]["oov_tokens"], sp
    print(json.dumps({k: v for k, v in stats.items() if k != "args"}, ensure_ascii=False, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kspon-root", dest="kspon_root", default=str(DATA_ROOT / "aihub" / "10.한국어음성"))
    ap.add_argument("--dict", default=os.environ.get("MDD_DATA_DIR", "") + "/dict.ltr.txt")
    ap.add_argument("--audit", type=int, default=0, help="N>0 이면 표본 N개로 A/B 변환 비교만 하고 끝낸다")
    ap.add_argument("--out-audit", dest="out_audit", default="kspon_jamo_audit.md")
    ap.add_argument("--out", default=None)
    ap.add_argument("--wav-out", dest="wav_out", default=str(DATA_ROOT / "kspon_wav16k"))
    ap.add_argument("--method", choices=["kog2p", "direct"], default="kog2p")
    ap.add_argument("--splits", default=None, help="일부 split 만 (쉼표). 예: eval_clean,eval_other")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.audit:
        audit(a)
    else:
        if not a.out:
            raise SystemExit("--out 필요")
        build(a)


if __name__ == "__main__":
    main()
