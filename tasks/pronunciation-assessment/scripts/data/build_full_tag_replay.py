# -*- coding: utf-8 -*-
"""전체 TAG(전 문항) + 오발음 비율 조정 + 원어민 replay — 서비스용 Stage 2 매니페스트.

서비스는 **주어진 문장을 읽는** 구조라, 학습에 모든 문항을 쓰고 평가는 "같은 문항 · 안 본 화자"로 한다.

CTC target 은 외국인·원어민 모두 실제 발음이다:
  외국인 = TAG phones 자모 (LAB·KoG2P 는 target 으로 쓰지 않는다. `.canon` 은 평가용으로만 저장)
  원어민 = KsponSpeech 발음 전사 → KoG2P → 자모 (manifest_kspon_native_v1 그대로)

train 구성
  외국인  train 화자(md5 분할) × 전 문항 · Foreign Clean V2 와 같은 품질 필터(0.5~15초 · 8kHz 제외 · OOV 제외)
          TAG==LAB 발화 : TAG!=LAB 발화 = --clean-ratio : (1-clean-ratio)  (기본 60:40, 발화 수 기준)
          → 정상 발화는 전부 쓰고, 오발음 발화를 복제(정수배 + 무작위 나머지)해서 비율을 맞춘다
  원어민  전체 train 의 --native-ratio (기본 10%, 발화 수 기준) 를 KsponSpeech 에서 문장 중복 없이 뽑는다
valid   valid 화자 × 전 문항, 자연 분포에서 --valid-size 개 (선택 PRIMARY: 같은 문항 + 안 본 화자)
test    test 화자 × 전 문항 전부 (기존 test_seen + test_unseen 과 같은 화자·발화)

    $PY .../build_full_tag_replay.py --dict $MDD_DATA_DIR/dict.ltr.txt --out $MDD_OUTPUTS/manifest_full_tag_replay10
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, collections, hashlib, importlib.util, json, os, random, time

# 파싱·필터·쓰기 규칙은 Foreign Clean V2 와 한 곳에서 관리한다
_spec = importlib.util.spec_from_file_location("fcv2", str(_pl.Path(__file__).with_name("build_foreign_clean_v2.py")))
fcv2 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(fcv2)
SR = 16000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dict", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=str(MDD_OUTPUTS / "cache" / "foreign_tag_lab_parsed.jsonl.gz"))
    ap.add_argument("--aihub-root", dest="aihub_root", default=str(DATA_ROOT / "aihub"))
    ap.add_argument("--wav16k", default=str(DATA_ROOT / "mdd_gold_wav16k"))
    ap.add_argument("--native", default=str(MDD_OUTPUTS / "manifest_kspon_native_v1"))
    ap.add_argument("--clean-ratio", dest="clean_ratio", type=float, default=0.6, help="외국인 train 중 TAG==LAB 발화 비율")
    ap.add_argument("--native-ratio", dest="native_ratio", type=float, default=0.1, help="전체 train 중 원어민 발화 비율")
    ap.add_argument("--valid-size", dest="valid_size", type=int, default=10000)
    ap.add_argument("--min-sec", dest="min_sec", type=float, default=0.5)
    ap.add_argument("--max-sec", dest="max_sec", type=float, default=15.0)
    ap.add_argument("--exclude-sr", dest="exclude_sr", default="8000")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--holdout-items", dest="holdout_items", default=None,
                    help="select_holdout_prompts.py 가 만든 JSON. 주면 그 문항을 학습에서 빼고 "
                         "valid_seen/valid_unseen/test_seen/test_unseen 을 만든다(없으면 예전과 똑같이 동작)")
    a = ap.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit(f"출력 폴더가 비어 있지 않다(덮어쓰기 금지): {a.out}")
    dict_syms = [l.split()[0] for l in open(a.dict, encoding="utf-8") if l.strip()]
    assert len(dict_syms) == 46
    dset = set(dict_syms)
    excl_sr = {int(x) for x in a.exclude_sr.split(",") if x}

    recs, _ = fcv2.load_or_parse(a)
    index = {}
    with open(os.path.join(a.wav16k, "_index.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            st, sr, ch, fr = line.rstrip("\n").split("\t")[:4]
            index[st] = (int(sr) if sr else None, int(fr))

    excl = collections.Counter()
    by_split = collections.defaultdict(list)
    for r in recs:
        if "fail" in r:
            excl[r["fail"]] += 1; continue
        if r["stem"] not in index:
            excl["no_wav16k"] += 1; continue
        r["src_sr"], r["frames"] = index[r["stem"]]
        r["sec"] = round(r["frames"] / SR, 3)
        if not r["ltr"] or not r["canon"] or not r["norm_prompt"]:
            excl["empty_label_or_prompt"] += 1; continue
        if not (a.min_sec <= r["sec"] <= a.max_sec):
            excl["duration"] += 1; continue
        if r["src_sr"] in excl_sr:
            excl[f"src_sr_{r['src_sr']}"] += 1; continue
        if any(t not in dset for t in r["ltr"]):
            excl["oov_ltr"] += 1; continue
        r["split"] = fcv2.split_of(r["spk"])
        by_split[r["split"]].append(r)
    print(f"[filter] 제외 {dict(excl)} · split {{k: len(v) for k, v in by_split.items()}}".replace("{k: len(v) for k, v in by_split.items()}",
          str({k: len(v) for k, v in by_split.items()})), flush=True)

    # ── 문항 hold-out (선택) — 화자 분할은 그대로, 문항만 추가로 가른다 ──
    hold = None
    if a.holdout_items:
        H = json.load(open(a.holdout_items, encoding="utf-8"))
        hold = {"valid": set(H["valid_items"]), "test": set(H["test_items"])}
        assert not (hold["valid"] & hold["test"]), "valid/test hold-out 문항이 겹친다"
        allh = hold["valid"] | hold["test"]
        dropped = collections.Counter()
        for sp in ("train", "valid", "test"):
            keep = []
            for r in by_split[sp]:
                it = r.get("item")
                if sp == "train" and it in allh:
                    dropped["train×holdout"] += 1; continue            # 학습 화자의 hold-out 발화는 어디에도 안 쓴다
                if sp == "valid" and it in hold["test"]:
                    dropped["valid×test_items"] += 1; continue          # test 문항은 선택 과정에서도 안 본다
                if sp == "test" and it in hold["valid"]:
                    dropped["test×valid_items"] += 1; continue
                keep.append(r)
            by_split[sp] = keep
        print(f"[holdout] valid 문항 {len(hold['valid'])} · test 문항 {len(hold['test'])} · 버린 발화 {dict(dropped)}", flush=True)

    rng = random.Random(a.seed)
    # ── 외국인 train: 60(정상) : 40(오발음) ──
    tr = by_split["train"]
    clean = [r for r in tr if r["ltr"] == r["canon"]]
    mis = [r for r in tr if r["ltr"] != r["canon"]]
    target_mis = round(len(clean) * (1 - a.clean_ratio) / a.clean_ratio)
    reps, rem = divmod(target_mis, len(mis))
    mis_rows = mis * reps + rng.sample(mis, rem)
    foreign = clean + mis_rows
    rng.shuffle(foreign)
    # ── 원어민 replay ──
    n_native = round(len(foreign) * a.native_ratio / (1 - a.native_ratio))
    ntsv = open(os.path.join(a.native, "train.tsv"), encoding="utf-8").read().splitlines()
    nroot, nrows = ntsv[0], ntsv[1:]
    nltr = open(os.path.join(a.native, "train.ltr"), encoding="utf-8").read().splitlines()
    nmeta = [json.loads(l) for l in open(os.path.join(a.native, "train.meta.jsonl"), encoding="utf-8")]
    order = list(range(len(nrows))); rng.shuffle(order)
    seen, native_idx = set(), []
    for i in order:
        if len(native_idx) >= n_native:
            break
        if nmeta[i]["norm_text"] in seen:
            continue
        seen.add(nmeta[i]["norm_text"]); native_idx.append(i)
    if len(native_idx) < n_native:
        raise SystemExit(f"원어민 문장이 모자라다: {len(native_idx)} < {n_native}")
    for i in native_idx:
        assert all(t in dset for t in nltr[i].split()), f"원어민 OOV: {nrows[i]}"

    # ── valid / test ──
    vr = random.Random(a.seed + 1)
    if hold is None:
        valid = sorted(vr.sample(by_split["valid"], min(a.valid_size, len(by_split["valid"]))), key=lambda r: r["stem"])
        test = sorted(by_split["test"], key=lambda r: r["stem"])
    else:
        vs = [r for r in by_split["valid"] if r.get("item") not in hold["valid"]]
        valid = sorted(vr.sample(vs, min(a.valid_size, len(vs))), key=lambda r: r["stem"])   # valid_seen
        valid_unseen = sorted([r for r in by_split["valid"] if r.get("item") in hold["valid"]], key=lambda r: r["stem"])
        test = sorted([r for r in by_split["test"] if r.get("item") not in hold["test"]], key=lambda r: r["stem"])  # test_seen
        test_unseen = sorted([r for r in by_split["test"] if r.get("item") in hold["test"]], key=lambda r: r["stem"])
    spk = lambda rs: {r["spk"] for r in rs}
    leak = {"train∩valid": len(spk(tr) & spk(valid)), "train∩test": len(spk(tr) & spk(test)), "valid∩test": len(spk(valid) & spk(test))}
    assert all(v == 0 for v in leak.values()), f"화자 누수: {leak}"
    prompts_train = {r["norm_prompt"] for r in tr}
    cover = {"valid": len({r["norm_prompt"] for r in valid} - prompts_train), "test": len({r["norm_prompt"] for r in test} - prompts_train)}

    # ── 쓰기 ──
    os.makedirs(a.out, exist_ok=True)
    wav_root = os.path.abspath(a.wav16k)
    with open(os.path.join(a.out, "train.tsv"), "w", encoding="utf-8") as ft, \
         open(os.path.join(a.out, "train.ltr"), "w", encoding="utf-8") as fl, \
         open(os.path.join(a.out, "train.source"), "w", encoding="utf-8") as fs:
        ft.write("/\n")
        rows = [("foreign_clean" if r["ltr"] == r["canon"] else "foreign_mis",
                 os.path.join(wav_root, r["stem"][:2], r["stem"] + ".wav"), r["frames"], " ".join(r["ltr"])) for r in foreign]
        for i in native_idx:
            p, fr = nrows[i].split("\t")
            rows.append(("native", os.path.join(nroot, p), int(fr), nltr[i]))
        rng.shuffle(rows)
        for src, path, fr, ltr in rows:
            ft.write(f"{path.lstrip('/')}\t{fr}\n"); fl.write(ltr + "\n"); fs.write(src + "\n")
    fcv2.write_split(a.out, "valid", valid, wav_root)
    fcv2.write_split(a.out, "test", test, wav_root)
    if hold is not None:
        fcv2.write_split(a.out, "valid_unseen", valid_unseen, wav_root)
        fcv2.write_split(a.out, "test_unseen", test_unseen, wav_root)
        # 누출 보고서 — 화자·문항·부분문자열
        items = lambda rs: {r.get("item") for r in rs}
        prompts = lambda rs: {r["norm_prompt"] for r in rs}
        sets = {"train": tr, "valid_seen": valid, "valid_unseen": valid_unseen, "test_seen": test, "test_unseen": test_unseen}
        rep_ = {"holdout_file": os.path.abspath(a.holdout_items), "speaker_overlap": {}, "item_overlap": {},
                "prompt_overlap": {}, "sizes": {}}
        names = list(sets)
        for i, x in enumerate(names):
            rep_["sizes"][x] = {"utterances": len(sets[x]), "speakers": len(spk(sets[x])), "items": len(items(sets[x]))}
            for y in names[i + 1:]:
                k = f"{x}∩{y}"
                rep_["speaker_overlap"][k] = len(spk(sets[x]) & spk(sets[y]))
                rep_["item_overlap"][k] = len(items(sets[x]) & items(sets[y]))
                rep_["prompt_overlap"][k] = len(prompts(sets[x]) & prompts(sets[y]))
        tp = prompts(tr)
        rep_["unseen_prompt_inside_train_prompt"] = sorted(
            p for p in prompts(valid_unseen) | prompts(test_unseen) if any(p in q for q in tp))
        must_zero = [rep_["speaker_overlap"][f"train∩{s}"] for s in names[1:]] + \
                    [rep_["item_overlap"]["train∩valid_unseen"], rep_["item_overlap"]["train∩test_unseen"],
                     rep_["item_overlap"]["valid_unseen∩test_unseen"], rep_["speaker_overlap"]["valid_seen∩test_seen"]]
        rep_["ok"] = all(v == 0 for v in must_zero) and not rep_["unseen_prompt_inside_train_prompt"]
        json.dump(rep_, open(os.path.join(a.out, "leakage_report.json"), "w"), ensure_ascii=False, indent=1)
        print(f"[leak] ok={rep_['ok']} · 화자겹침 {rep_['speaker_overlap']} · 문항겹침 {rep_['item_overlap']}", flush=True)
        assert rep_["ok"], "hold-out 누출 — leakage_report.json 확인"
    with open(os.path.join(a.out, "dict.ltr.txt"), "w", encoding="utf-8") as f:
        f.write(open(a.dict, encoding="utf-8").read())

    src_cnt = collections.Counter(r[0] for r in rows)
    src_sec = collections.Counter()
    for src, _, fr, _ in rows:
        src_sec[src] += fr / SR
    h = hashlib.sha256()
    for fn in ("train.tsv", "train.ltr", "valid.tsv", "valid.ltr"):
        h.update(open(os.path.join(a.out, fn), "rb").read())
    stats = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(a), "excluded": dict(excl), "speaker_leak": leak,
        "valid_test_prompts_not_in_train": cover,
        "foreign_train_pool": {"utterances": len(tr), "clean": len(clean), "mis": len(mis), "prompts": len(prompts_train),
                               "speakers": len(spk(tr))},
        "mis_repeat": {"full_copies": reps, "extra_sampled": rem, "rows": len(mis_rows)},
        "train_rows": dict(src_cnt), "train_hours": {k: round(v / 3600, 1) for k, v in src_sec.items()},
        "train_ratio_utt": {k: round(v / len(rows), 4) for k, v in src_cnt.items()},
        "train_ratio_hours": {k: round(v / sum(src_sec.values()), 4) for k, v in src_sec.items()},
        "native_unique_texts": len(seen),
        "splits": dict([("valid", fcv2.split_stats(valid, dset)), ("test", fcv2.split_stats(test, dset))] +
                       ([("valid_unseen", fcv2.split_stats(valid_unseen, dset)),
                         ("test_unseen", fcv2.split_stats(test_unseen, dset))] if hold is not None else [])),
        "manifest_sha256": h.hexdigest(),
    }
    for k in list(stats["splits"]):
        stats["splits"][k].pop("label_token_dist", None)
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k not in ("args", "splits")}, ensure_ascii=False, indent=1), flush=True)
    for k, s in stats["splits"].items():
        print(f"{k}: {s['utterances']:,} 발화 · {s['hours']}h · 화자 {s['speakers']:,} · 문항 {s['prompts']} · 오발음 포함 {s['utt_with_mispron_site']:,}")


if __name__ == "__main__":
    main()
