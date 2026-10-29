# -*- coding: utf-8 -*-
"""Foreign Clean V2 매니페스트 — 외국인 발음평가 TAG(실제 발음)만으로 CTC 학습·평가 split 을 만든다.

목적: 문장 암기(같은 문항 수천 번 반복 → 표준발음으로 되돌림)를 줄이는 E1 실험의 입력.

CTC target = TextGrid `phones` 티어(사람이 들은 실제 발음)의 자모. 아래는 target 으로 쓰지 않는다:
    prompt · LAB 표준발음(phonemic.phones) · KoG2P(prompt) · error_tags · LAR(표준발음 라벨)
LAB 표준발음은 `.canon` 으로 따로 저장한다 — 되돌림/보존 지표 계산용이지 학습용이 아니다.

규칙 (기존 build_tag_manifest_combined / rebuild_tag_manifest 와 같게 둔 것)
  - phones 티어 라벨 중 "" . - spn <unk> 는 버리고, 나머지에서 호환 자모(U+3131~3163)만 남긴다.
    TAG "_"(그 음소를 발음하지 않음)는 자모가 아니므로 여기서 빠진다. CTC blank 와는 무관하다.
  - 화자 split: md5(화자ID)[:8] % 10 → 0=test · 1=valid · 나머지=train  (기존 test 와 같은 화자)
  - 오디오: $SPEECH_AI_DATA/mdd_gold_wav16k (resample_gold_16k.py 산출, _index.tsv 에 원본 샘플레이트·길이)

V2 에서 새로 한 것
  - 소스: 040~043 Training+Validation 전부 (041 Validation 포함)
  - 길이 0.5~15초 · 원본 8kHz 제외 · 빈 라벨 제외 · dict.ltr.txt 밖 기호(OOV)가 있는 발화 제외(목록 저장)
  - 문항 홀드아웃: 정규화 prompt 기준으로 RW·RS 문항 일부를 학습에서 통째로 뺀다 (--holdout-rw/--holdout-rs, --seed)
  - train 문항당 상한 --cap-per-prompt (화자 라운드로빈, --seed 로 결정적)
  - split 누수 assert: train∩test_* 화자 = 0 · train∩test_unseen 문항 = 0 (위반 시 아무것도 쓰지 않고 멈춤)

출력 split
  train          train 화자 × 홀드아웃 아닌 문항 × cap
  valid          valid 화자 × 홀드아웃 아닌 문항 (--valid-size 개 표본 — 학습 중 검증 비용 때문)
  valid_unseen   valid 화자 × 홀드아웃 문항 (--valid-size 개 표본)
  test_seen      test 화자 × 홀드아웃 아닌 문항 (기존 test 계열 · 참고용)
  test_unseen    test 화자 × 홀드아웃 문항 (PRIMARY — 새 화자 + 학습에서 안 본 문장)
  각 split: .tsv · .ltr(TAG 실제 발음) · .canon(LAB 표준) · .meta.jsonl(프롬프트·L1·점수·error_tags·TAG/LAB 음소 타임스탬프)

파싱 결과는 --cache 에 저장해 두고 재사용한다 (cap·seed ablation 이 수십 초로 끝난다).

    $PY tasks/pronunciation-assessment/scripts/data/build_foreign_clean_v2.py \\
        --out $MDD_OUTPUTS/manifest_foreign_clean_v2 --cap-per-prompt 1000 --seed 42
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, collections, glob, gzip, hashlib, json, os, random, re, statistics, time
from multiprocessing import Pool

SKIP_LBL = {"", ".", "-", "spn", "<unk>"}
PARTS = {"TL": ("Training", "TS"), "VL": ("Validation", "VS")}
SR = 16000


def is_jamo(c):
    return 0x3131 <= ord(c) <= 0x3163


def norm_prompt(p):
    """공백·문장부호를 없앤 비교용 문항 텍스트."""
    return re.sub(r"[^0-9A-Za-z가-힣]", "", p or "")


def split_of(spk):
    h = int(hashlib.md5(spk.encode()).hexdigest()[:8], 16) % 10
    return "test" if h == 0 else "valid" if h == 1 else "train"


def read_textgrid(path):
    b = open(path, "rb").read()
    if b[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return b.decode("utf-16")
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("utf-16")


def tiers(txt):
    out = {}
    for m in re.finditer(r'name = "([^"]+)"(.*?)(?=name = "|\Z)', txt, re.S):
        ivs = re.findall(r'xmin = ([-\d.eE]+)\s+xmax = ([-\d.eE]+)\s+text = "([^"]*)"', m.group(2))
        out[m.group(1)] = [(round(float(a), 3), round(float(b), 3), t) for a, b, t in ivs]
    return out


def parse_one(job):
    """TAG + LAB 1쌍 → 레코드. 실패는 {"stem", "fail"} 로 돌려준다."""
    stem, ds, part, tag_path, lab_path = job
    rec = {"stem": stem, "ds": ds, "part": part}
    try:
        tv = tiers(read_textgrid(tag_path))
    except Exception as e:
        return {**rec, "fail": f"tag_read:{type(e).__name__}"}
    if "phones" not in tv:
        return {**rec, "fail": "no_phones_tier"}
    ltr, dropped = [], collections.Counter()
    tag_ph = []
    for a, b, t in tv["phones"]:
        s = t.strip()
        tag_ph.append([a, b, s])
        if s in SKIP_LBL:
            if s:
                dropped[s] += 1
            continue
        for ch in s:
            if is_jamo(ch):
                ltr.append(ch)
            elif not ch.isspace():
                dropped[ch] += 1
    try:
        d = json.load(open(lab_path, encoding="utf-8"))
    except Exception as e:
        return {**rec, "fail": f"lab_read:{type(e).__name__}"}
    rm = d.get("RecordingMetadata", {}) or {}
    ph = rm.get("phonemic", {}) or {}
    canon, lab_ph = [], []
    for x in ph.get("phones", []) or []:
        s = str(x.get("phone", "")).strip()
        lab_ph.append([x.get("start"), x.get("end"), s])
        canon += [ch for ch in s if is_jamo(ch)]
    sm, ev = d.get("SpeakerMetadata", {}) or {}, d.get("EvaluationMetadata", {}) or {}
    toks = stem.split("-")
    task_item = toks[5]
    rec.update({
        "spk": toks[0], "task": re.match(r"[A-Z]+", task_item).group(0), "item": task_item,
        "prompt": rm.get("prompt"), "norm_prompt": norm_prompt(rm.get("prompt")),
        "language": sm.get("language"), "nationality": sm.get("nationality"), "lang_code": toks[3],
        "proficiency": sm.get("proficiency"),
        "scores": {k: ev.get(k) for k in ("PronunProfEval", "FluencyEval", "ComprehendEval")},
        "error_tags": [[e.get("error_tag"), e.get("start"), e.get("end")] for e in ph.get("error_tags", []) or []],
        "ltr": ltr, "canon": canon, "dropped": dict(dropped),
        "tag_phones": tag_ph, "lab_phones": lab_ph,
        "tag_words": [[a, b, t] for a, b, t in tv.get("words", []) if t.strip()],
    })
    return rec


def collect_jobs(root):
    jobs, missing = [], collections.Counter()
    for base in sorted(glob.glob(os.path.join(root, "0[4][0-3].*", "01-1.정식개방데이터"))):
        ds = os.path.basename(os.path.dirname(base)).split(".")[0]
        for part, (top, _) in PARTS.items():
            lbl = os.path.join(base, top, "02.라벨링데이터")
            tdir, ldir = os.path.join(lbl, f"{part}_pronunciation_tag"), os.path.join(lbl, f"{part}_pronunciation_lab")
            labs = {e.name[:-5] for e in os.scandir(ldir) if e.name.endswith(".json")}
            for e in os.scandir(tdir):
                if not e.name.endswith(".TextGrid"):
                    continue
                st = e.name[:-9]
                if st not in labs:
                    missing[f"{ds}:{part}:no_lab"] += 1
                    continue
                jobs.append((st, ds, part, e.path, os.path.join(ldir, st + ".json")))
    return sorted(jobs), missing


def load_or_parse(a):
    if a.cache and os.path.exists(a.cache):
        print(f"[cache] {a.cache}", flush=True)
        with gzip.open(a.cache, "rt", encoding="utf-8") as f:
            return [json.loads(l) for l in f], {}
    t0 = time.time()
    jobs, missing = collect_jobs(a.aihub_root)
    print(f"[parse] TAG+LAB {len(jobs):,}쌍 · 짝 없음 {dict(missing)}", flush=True)
    recs = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(parse_one, jobs, chunksize=200), 1):
            recs.append(r)
            if i % 200000 == 0:
                print(f"  {i:,}/{len(jobs):,}  {time.time() - t0:.0f}s", flush=True)
    recs.sort(key=lambda r: r["stem"])
    if a.cache:
        os.makedirs(os.path.dirname(a.cache), exist_ok=True)
        tmp = a.cache + ".tmp"
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, a.cache)
    print(f"[parse] 완료 {time.time() - t0:.0f}s", flush=True)
    return recs, dict(missing)


def cap_round_robin(recs, cap, rng):
    """문항별 상한. 화자 순서를 seed 로 섞고 화자마다 한 발화씩 돌아가며 뽑는다."""
    by_prompt = collections.defaultdict(list)
    for r in recs:
        by_prompt[r["norm_prompt"]].append(r)
    keep = []
    for p in sorted(by_prompt):
        rs = by_prompt[p]
        if cap <= 0 or len(rs) <= cap:
            keep += rs
            continue
        per_spk = collections.defaultdict(list)
        for r in sorted(rs, key=lambda r: r["stem"]):
            per_spk[r["spk"]].append(r)
        spks = sorted(per_spk)
        rng.shuffle(spks)
        picked, k = [], 0
        while len(picked) < cap:
            took = False
            for s in spks:
                if k < len(per_spk[s]):
                    picked.append(per_spk[s][k]); took = True
                    if len(picked) >= cap:
                        break
            if not took:
                break
            k += 1
        keep += picked
    return sorted(keep, key=lambda r: r["stem"])


def dist(vals):
    if not vals:
        return {}
    s = sorted(vals)
    q = lambda p: s[min(len(s) - 1, int(p * (len(s) - 1)))]
    return {"n": len(s), "mean": round(sum(s) / len(s), 3), "min": s[0], "p5": q(.05), "p25": q(.25),
            "median": q(.5), "p75": q(.75), "p95": q(.95), "max": s[-1]}


def split_stats(rs, dict_syms):
    per_prompt = collections.Counter(r["norm_prompt"] for r in rs)
    toks = collections.Counter(t for r in rs for t in r["ltr"])
    secs = [r["sec"] for r in rs]
    return {
        "utterances": len(rs), "hours": round(sum(secs) / 3600, 2),
        "speakers": len({r["spk"] for r in rs}), "prompts": len(per_prompt),
        "unique_ltr": len({" ".join(r["ltr"]) for r in rs}),
        "utt_per_prompt": dist(list(per_prompt.values())),
        "duration_sec": dist([round(x, 2) for x in secs]),
        "sample_rate": dict(collections.Counter(r["src_sr"] for r in rs).most_common()),
        "task": dict(collections.Counter(r["task"] for r in rs).most_common()),
        "dataset": dict(collections.Counter(f"{r['ds']}:{r['part']}" for r in rs).most_common()),
        "language": dict(collections.Counter(r["language"] for r in rs).most_common()),
        "label_tokens_total": sum(toks.values()), "label_tokens_unique": len(toks),
        "label_token_dist": dict(toks.most_common()),
        "oov_tokens": {t: c for t, c in toks.items() if t not in dict_syms},
        "utt_with_mispron_site": sum(r["ltr"] != r["canon"] for r in rs),
    }


def write_split(out, name, rs, wav_root):
    with open(os.path.join(out, f"{name}.tsv"), "w", encoding="utf-8") as ft, \
         open(os.path.join(out, f"{name}.ltr"), "w", encoding="utf-8") as fl, \
         open(os.path.join(out, f"{name}.canon"), "w", encoding="utf-8") as fc, \
         open(os.path.join(out, f"{name}.meta.jsonl"), "w", encoding="utf-8") as fm:
        ft.write(wav_root + "\n")
        for r in rs:
            ft.write(f"{r['stem'][:2]}/{r['stem']}.wav\t{r['frames']}\n")
            fl.write(" ".join(r["ltr"]) + "\n")
            fc.write(" ".join(r["canon"]) + "\n")
            meta = {k: r[k] for k in ("stem", "spk", "ds", "part", "task", "item", "prompt", "norm_prompt",
                                      "language", "nationality", "proficiency", "src_sr", "sec", "scores",
                                      "error_tags", "tag_words", "tag_phones", "lab_phones", "dropped")}
            fm.write(json.dumps(meta, ensure_ascii=False) + "\n")


def sha256_files(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        with open(p, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aihub-root", dest="aihub_root", default=str(DATA_ROOT / "aihub"))
    ap.add_argument("--wav16k", default=str(DATA_ROOT / "mdd_gold_wav16k"))
    ap.add_argument("--dict", default=os.environ.get("MDD_DATA_DIR", "") + "/dict.ltr.txt",
                    help="CTC 사전 (46자모). 기본 $MDD_DATA_DIR/dict.ltr.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=str(MDD_OUTPUTS / "cache" / "foreign_tag_lab_parsed.jsonl.gz"))
    ap.add_argument("--cap-per-prompt", dest="cap", type=int, default=1000, help="train 문항당 최대 발화 수 (0=무제한)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout-rw", dest="holdout_rw", type=int, default=20, help="학습에서 뺄 RW(단어) 문항 수")
    ap.add_argument("--holdout-rs", dest="holdout_rs", type=int, default=10, help="학습에서 뺄 RS(문장) 문항 수")
    ap.add_argument("--min-test-utt", dest="min_test_utt", type=int, default=200,
                    help="홀드아웃 후보 문항이 test 화자에게서 가져야 할 최소 발화 수")
    ap.add_argument("--min-sec", dest="min_sec", type=float, default=0.5)
    ap.add_argument("--max-sec", dest="max_sec", type=float, default=15.0)
    ap.add_argument("--exclude-sr", dest="exclude_sr", default="8000", help="제외할 원본 샘플레이트(쉼표)")
    ap.add_argument("--valid-size", dest="valid_size", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()

    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit(f"출력 폴더가 비어 있지 않다(덮어쓰기 금지): {a.out}")
    dict_syms = [l.split()[0] for l in open(a.dict, encoding="utf-8") if l.strip()]
    assert len(dict_syms) == 46, f"dict.ltr.txt 가 46개가 아니다: {len(dict_syms)}"
    dict_set = set(dict_syms)
    excl_sr = {int(x) for x in a.exclude_sr.split(",") if x}

    recs, missing = load_or_parse(a)
    index = {}
    with open(os.path.join(a.wav16k, "_index.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            st, sr, ch, fr = line.rstrip("\n").split("\t")[:4]
            index[st] = (int(sr) if sr else None, int(fr))

    # ── 1. 품질 필터 ──
    excl = collections.Counter()
    oov_files, clean = [], []
    for r in recs:
        if "fail" in r:
            excl[r["fail"]] += 1; continue
        if r["stem"] not in index:
            excl["no_wav16k"] += 1; continue
        r["src_sr"], r["frames"] = index[r["stem"]]
        r["sec"] = round(r["frames"] / SR, 3)
        if not r["ltr"]:
            excl["empty_ltr"] += 1; continue
        if not r["canon"]:
            excl["empty_canon"] += 1; continue
        if not r["norm_prompt"]:
            excl["empty_prompt"] += 1; continue
        if r["sec"] < a.min_sec or r["sec"] > a.max_sec:
            excl["duration"] += 1; continue
        if r["src_sr"] in excl_sr:
            excl[f"src_sr_{r['src_sr']}"] += 1; continue
        bad = sorted({t for t in r["ltr"] if t not in dict_set})
        if bad:
            excl["oov_ltr"] += 1
            oov_files.append({"stem": r["stem"], "oov": bad, "ltr": " ".join(r["ltr"]),
                              "tag_phones": [p for p in r["tag_phones"] if any(c in bad for c in p[2])]})
            continue
        r["split"] = split_of(r["spk"])
        clean.append(r)
    print(f"[filter] 전체 {len(recs):,} → clean {len(clean):,} · 제외 {dict(excl)}", flush=True)

    # ── 2. 문항 홀드아웃 ──
    rng = random.Random(a.seed)
    prompt_task = {}
    test_cnt = collections.Counter()
    for r in clean:
        prompt_task.setdefault(r["norm_prompt"], collections.Counter())[r["task"]] += 1
        if r["split"] == "test":
            test_cnt[r["norm_prompt"]] += 1
    ptask = {p: c.most_common(1)[0][0] for p, c in prompt_task.items()}
    held = set()
    for task, n in (("RW", a.holdout_rw), ("RS", a.holdout_rs)):
        cands = sorted(p for p, t in ptask.items() if t == task and test_cnt[p] >= a.min_test_utt)
        if len(cands) < n:
            raise SystemExit(f"{task} 홀드아웃 후보 {len(cands)}개 < 요청 {n}개")
        held |= set(rng.sample(cands, n))

    groups = collections.defaultdict(list)
    for r in clean:
        seen = r["norm_prompt"] not in held
        groups[(r["split"], seen)].append(r)
    train_all = groups[("train", True)]
    train = cap_round_robin(train_all, a.cap, random.Random(a.seed))
    vrng = random.Random(a.seed + 1)
    sample = lambda rs, n: sorted(vrng.sample(rs, n) if 0 < n < len(rs) else rs, key=lambda r: r["stem"])
    splits = {
        "train": train,
        "valid": sample(groups[("valid", True)], a.valid_size),
        "valid_unseen": sample(groups[("valid", False)], a.valid_size),
        "test_seen": groups[("test", True)],
        "test_unseen": groups[("test", False)],
    }

    # ── 3. 누수 assert (위반이면 아무것도 쓰지 않는다) ──
    spk = {k: {r["spk"] for r in v} for k, v in splits.items()}
    txt = {k: {r["norm_prompt"] for r in v} for k, v in splits.items()}
    leak = {
        "speaker train∩test_unseen": len(spk["train"] & spk["test_unseen"]),
        "speaker train∩test_seen": len(spk["train"] & spk["test_seen"]),
        "speaker train∩valid": len(spk["train"] & spk["valid"]),
        "speaker train∩valid_unseen": len(spk["train"] & spk["valid_unseen"]),
        "prompt train∩test_unseen": len(txt["train"] & txt["test_unseen"]),
        "prompt train∩valid_unseen": len(txt["train"] & txt["valid_unseen"]),
    }
    print(f"[leak] {leak}", flush=True)
    assert all(v == 0 for v in leak.values()), f"누수 발생: {leak}"
    for k, v in splits.items():
        oov = {t for r in v for t in r["ltr"] if t not in dict_set}
        assert not oov, f"{k} 에 OOV 가 남았다: {oov}"
        assert v, f"{k} 가 비었다"

    # 참고: 홀드아웃 문항이 학습 문항의 부분 문자열로 들어 있는 경우(정확 일치 누수는 아님)
    substr = sorted(h for h in held if any(h in t for t in txt["train"]))

    # ── 4. 쓰기 ──
    os.makedirs(a.out, exist_ok=True)
    wav_root = os.path.abspath(a.wav16k)
    for k, v in splits.items():
        write_split(a.out, k, v, wav_root)
    with open(os.path.join(a.out, "dict.ltr.txt"), "w", encoding="utf-8") as f:
        f.write(open(a.dict, encoding="utf-8").read())
    with open(os.path.join(a.out, "oov_excluded.jsonl"), "w", encoding="utf-8") as f:
        for o in oov_files:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    stats = {
        "name": os.path.basename(a.out.rstrip("/")),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(a),
        "source_pairs": len(recs), "pairs_missing_lab": missing,
        "excluded": dict(excl), "oov_excluded_utterances": len(oov_files),
        "clean_total": len(clean),
        "train_before_cap": split_stats(train_all, dict_set)["utterances"],
        "heldout_prompts": {t: sorted(p for p in held if ptask[p] == t) for t in ("RW", "RS")},
        "heldout_prompt_substring_of_train_prompt": substr,
        "leakage": leak,
        "splits": {k: split_stats(v, dict_set) for k, v in splits.items()},
        "dropped_tag_labels_train": dict(sum((collections.Counter(r["dropped"]) for r in train), collections.Counter())),
        "manifest_sha256": sha256_files([os.path.join(a.out, f"{k}.{e}") for k in splits for e in ("tsv", "ltr")]),
    }
    with open(os.path.join(a.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)

    L = [f"# {stats['name']}", "", f"생성 {stats['created']} · seed {a.seed} · cap {a.cap} · "
         f"길이 {a.min_sec}~{a.max_sec}s · 제외 샘플레이트 {sorted(excl_sr)}", "",
         f"- 원본 TAG+LAB 쌍 {len(recs):,} → clean {len(clean):,}",
         f"- 제외 사유: {dict(excl)}",
         f"- OOV 로 제외한 발화: {len(oov_files)} (목록 `oov_excluded.jsonl`)",
         f"- 누수 검사: {leak}",
         f"- manifest sha256 (tsv+ltr): `{stats['manifest_sha256']}`", "",
         "| split | 발화 | 시간 | 화자 | 문항 | 고유 ltr | 문항당 평균/중앙/최대 | 오발음 포함 발화 | OOV |",
         "|---|--:|--:|--:|--:|--:|---|--:|--:|"]
    for k, s in stats["splits"].items():
        u = s["utt_per_prompt"]
        L.append(f"| {k} | {s['utterances']:,} | {s['hours']} | {s['speakers']:,} | {s['prompts']} | "
                 f"{s['unique_ltr']:,} | {u['mean']} / {u['median']} / {u['max']} | "
                 f"{s['utt_with_mispron_site']:,} | {sum(s['oov_tokens'].values())} |")
    L += ["", f"train 상한 적용 전 {stats['train_before_cap']:,} 발화", "",
          "## 홀드아웃 문항 (test_unseen / valid_unseen)", "",
          f"- RW {len(stats['heldout_prompts']['RW'])}: {', '.join(stats['heldout_prompts']['RW'])}",
          f"- RS {len(stats['heldout_prompts']['RS'])}: {', '.join(stats['heldout_prompts']['RS'])}",
          f"- 참고(정확 일치 누수 아님) — 학습 문항 안에 부분 문자열로 들어 있는 홀드아웃 문항: {substr or '없음'}", "",
          "## split 별 세부 (샘플레이트 · 과제 · 데이터셋 · 모국어 상위 10 · 길이)", ""]
    for k, s in stats["splits"].items():
        L += [f"### {k}", f"- 샘플레이트 {s['sample_rate']}", f"- 과제 {s['task']}", f"- 데이터셋 {s['dataset']}",
              f"- 모국어 상위 {dict(list(s['language'].items())[:10])}", f"- 길이(초) {s['duration_sec']}",
              f"- 라벨 토큰 {s['label_tokens_total']:,} · 고유 {s['label_tokens_unique']}", ""]
    with open(os.path.join(a.out, "stats.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:14]), flush=True)
    print(f"[done] {a.out}", flush=True)


if __name__ == "__main__":
    main()
