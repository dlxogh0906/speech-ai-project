# -*- coding: utf-8 -*-
"""AIHub 학습자 녹음(학습에 안 쓴 시험 화자)을 **지금 운영과 같은 모델 · 규칙**으로 판정해 국적별 발음오류 카탈로그용 자료를 만든다.

운영 NPU(P6)와 같게: M3(`w2vbert_synth_tense_v2/step_3000`) · 표준발음 보정(choose, margin 1) · 빠짐 점수 50 미만 · ㄴ↔ㄹ 예외 ·
통째로 안 들린 글자 글자 단위 판정. 계산 경로는 `mdd.api_offline.ApiEmulator`(app.py 그대로).
국적은 AIHub 화자 정보(SpeakerMetadata.nationality — 실제 값). 같은 녹음의 사람 라벨(TAG) 오류도 함께 남긴다(비교용).

    $PY .../dataset_error_catalog.py extract --shard 0/8 --out $MDD_OUTPUTS/analysis/dataset_catalog_2026-11-02/shard
    $PY .../dataset_error_catalog.py merge --dir $MDD_OUTPUTS/analysis/dataset_catalog_2026-11-02
       → catalog_model.json(우리 음소인식기) · catalog_tag.json(사람 TAG) — tools/capture_db/build_catalog.py render 로 그린다
"""
# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, collections, csv, datetime, glob, importlib.util, json, os, re, time, wave

VOWELS = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
OUT_DIR = os.path.join(str(MDD_OUTPUTS), "analysis", "dataset_catalog_2026-11-02")


def position(seq, i):
    if seq[i] in VOWELS:
        return "중성"
    return "초성" if (i + 1 < len(seq) and seq[i + 1] in VOWELS) else "종성"


def cmd_extract(a):
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    import mdd.speechsuper as ss
    ss.DEL_RUN_MIN = 1; ss.DEL_SCORE_MAX = 50; ss.LIQUID_NASAL = dict(ss.LIQUID_NASAL_PAIRS); ss.SS_RULE = False; ss.SZ_RULE = False
    from mdd.api_offline import ApiEmulator
    from mdd import canonical_fix as cf
    from mdd.pipeline import ctype
    spec = importlib.util.spec_from_file_location("ecs", str(_pl.Path(__file__).with_name("eval_canonical_source.py")))
    ecs = importlib.util.module_from_spec(spec); spec.loader.exec_module(ecs)
    shard, n_shard = (int(x) for x in a.shard.split("/"))
    meta = json.load(open(a.speaker_meta, encoding="utf-8"))
    emu = ApiEmulator(a.ckpt, scorer=a.scorer)
    B = cf.CanonicalBuilder(emu.canonical)
    j2p = emu.ss.j2p
    M = a.manifest
    with open(os.path.join(M, a.split + ".tsv"), encoding="utf-8") as f:
        root = f.readline().rstrip("\n"); rows = [l.rstrip("\n").split("\t")[0] for l in f]
    rd = lambda e: open(os.path.join(M, "%s.%s" % (a.split, e)), encoding="utf-8").read().splitlines()
    ltr, can = [l.split() for l in rd("ltr")], [l.split() for l in rd("canon")]
    by_item, seen_path = collections.defaultdict(list), set()
    for i, p in enumerate(rows):
        m = ecs.ITEM_RE.search(p)
        if p in seen_path:                    # replay 매니페스트는 같은 녹음을 여러 번 담는다 — 한 번만
            continue
        seen_path.add(p)
        if m and can[i] and not any(len(x) != 1 for x in can[i] + ltr[i]):
            by_item[m.group(1)].append(i)
    codes = sorted(by_item)
    lab = dict((k, collections.Counter(" ".join(can[i]) for i in by_item[k]).most_common(1)[0][0].replace(" ", "")) for k in codes)
    sentences = [r["sentence"] for r in csv.DictReader(open(a.sentences, encoding="utf-8"))]
    prompts = json.load(open(a.prompts, encoding="utf-8"))["all"]
    item_text = ecs.item_texts(emu, sentences, prompts, codes, lab)
    cache = {}
    for k, t in item_text.items():
        base, alts = B.variants(t); cache[k] = (base, alts, emu.canonical(t))
    pick = sorted(i for k, v in by_item.items() if k in item_text for i in v)[shard::n_shard]
    out = "%s.%d-of-%d.jsonl" % (a.out, shard, n_shard)
    done = set()
    if os.path.exists(out):
        done = set(json.loads(l)["stem"] for l in open(out, encoding="utf-8"))
    print("[표본] 문항 %d(원문 있음) · 이 조각 %d · 이미 %d" % (len(item_text), len(pick), len(done)), flush=True)
    t0, n_new = time.time(), 0
    with open(out, "a", encoding="utf-8") as fo:
        for n, i in enumerate(pick, 1):
            stem = os.path.basename(rows[i])[:-4]
            if stem in done:
                continue
            k = ecs.ITEM_RE.search(rows[i]).group(1); text = item_text[k]
            with wave.open(os.path.join(root, rows[i]), "rb") as w:
                x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
            rejected, _ = emu.check_audio(x[: 16000 * 20], text)
            lp = emu.lprobs(x)
            base, alts, api = cache[k]
            canon = cf.choose(lp[0], base, alts, emu.dec.d.index, blank=emu.dec.blank, margin=1.0)[0] if base else api
            body, raw = emu.assess(x, text, canon=canon, lp=lp, gate=False)
            ph = ApiEmulator.phonemes(body)
            model = [[p.get("position"), p.get("alpha"), p.get("alpha_jamo"), p.get("actual"), p.get("actual_jamo"), p["errorType"]]
                     for p in ph if "errorType" in p]
            # 사람 TAG(라벨 표준음 축) 오류
            labc = can[i]; tag = ecs.proj(labc, ltr[i])
            try:
                lab_ph = j2p(labc)
            except Exception:
                lab_ph = None
            human = []
            for q, (c, t) in enumerate(zip(labc, tag)):
                if t == c:
                    continue
                pos = position(labc, q)
                a_sym = None
                if t:
                    try:
                        seq = list(labc); seq[q] = t; a_sym = j2p(seq)[q]
                    except Exception:
                        a_sym = None
                et = ctype(c, t, "sub", pos == "종성") if t else "음운탈락"
                human.append([pos, lab_ph[q] if lab_ph and q < len(lab_ph) else None, c, a_sym, t, et])
            spk = stem[:5]; sm = meta.get(spk) or {}
            fo.write(json.dumps({"stem": stem, "split": a.split, "spk": spk, "item": k, "kind": k[:2], "country": sm.get("nationality"),
                                 "language": sm.get("language"), "proficiency": sm.get("proficiency"), "rejected": bool(rejected),
                                 "n_phonemes": len(ph), "n_lab": len(labc), "model": model, "human": human,
                                 "scores": {s: body["result"].get(s) for s in ("overall", "pronunciation", "fluency")}},
                                ensure_ascii=False) + "\n")
            n_new += 1
            if n_new % 500 == 0:
                fo.flush(); el = time.time() - t0
                print("  %d/%d · %.1f발화/s · 남은 %.0f분" % (n, len(pick), n_new / el, (len(pick) - n) / (n_new / el) / 60), flush=True)
    print("[끝] %s · %.0f초" % (out, time.time() - t0), flush=True)


def cmd_merge(a):
    files = sorted(glob.glob(os.path.join(a.dir, "shard*.*-of-*.jsonl")))
    recs = [json.loads(l) for f in files for l in open(f, encoding="utf-8")]
    for r in recs:
        r.setdefault("split", "test")
    want = set(a.splits.split(",")) if a.splits else {r["split"] for r in recs}
    recs = [r for r in recs if r["split"] in want]
    scope = "시험화자" if want == {"test"} else ("전체" if want >= {"train", "test"} else "+".join(sorted(want)))
    split_name = {"train": "학습 화자", "valid": "검증 화자", "test": "시험 화자"}
    kind_name = {"RS": "문장(RS)", "RW": "낱말(RW)"}
    for which, fname, desc, short, ftag in (("model", "catalog_model_%s.json" % scope, "우리 음소인식기(운영 M3·P6 규칙) 판정", "우리 음소인식기 · " + scope, "우리모델_" + scope),
                                            ("human", "catalog_tag_%s.json" % scope, "사람 라벨(AIHub TAG) — 사람이 들은 실제 발음", "사람 TAG · " + scope, "사람TAG_" + scope)):
        utts, flags = [], []
        for r in recs:
            ui = len(utts)
            fl = r[which]
            utts.append({"spk": r["spk"], "date": None, "version": "M3·P6" if which == "model" else "사람 TAG", "source": split_name.get(r["split"], r["split"]) + " · " + kind_name.get(r["kind"], r["kind"]),
                         "ref_text": r["item"], "n_phonemes": r["n_phonemes"] if which == "model" else r["n_lab"], "n_flags": len(fl),
                         "country": r["country"] or "(국적 미상)", "nat_status": "meta", "rejected": r["rejected"]})
            for f in fl:
                flags.append([ui] + f)
        meta = {"kind": "dataset", "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                "label": desc, "short": short, "file_tag": ftag, "n_requests": len(recs), "date_min": None, "date_max": None,
                "nationality": "AIHub 화자 정보(SpeakerMetadata.nationality) — 실제 값",
                "dataset": "AIHub 040~043 발음 녹음 · " + ("학습에 안 쓴 시험 화자" if scope == "시험화자" else "학습·검증·시험 화자 전체(" + ",".join(sorted(want)) + ")"),
                "scope": scope}
        json.dump({"meta": meta, "versions": [["M3·P6", "", "운영과 같은 모델·규칙"], ["사람 TAG", "", "사람 라벨"]], "utts": utts,
                   "flags": flags, "flag_cols": ["utt", "position", "alpha", "alpha_jamo", "actual", "actual_jamo", "error_type"]},
                  open(os.path.join(a.dir, fname), "w", encoding="utf-8"), ensure_ascii=False)
        print("[합침] %s · 녹음 %d · 지적 %d → %s" % (desc, len(utts), len(flags), fname), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--manifest", default=os.path.join(str(MDD_OUTPUTS), "manifest_full_tag_replay10"))
    e.add_argument("--speaker-meta", dest="speaker_meta", default=os.path.join(OUT_DIR, "speaker_meta.json"))
    e.add_argument("--sentences", default=os.path.join(str(MDD_OUTPUTS), "analysis", "mdd_error_2026-09-27",
                                                       "w2vbert_step38000_seen_sentence", "sentence_analysis.csv"))
    e.add_argument("--prompts", default=os.path.join(str(MDD_OUTPUTS), "analysis", "npu_bench_pack_2026-10-03", "train_prompts.json"))
    e.add_argument("--ckpt", default=os.path.join(str(DATA_ROOT), "checkpoints", "w2vbert_synth_tense_v2", "step_3000"))
    e.add_argument("--scorer", default=os.path.join(str(MDD_OUTPUTS), "score_v2_w2vbert", "score_models_v2_w2vbert.joblib"))
    e.add_argument("--split", default="test", choices=("train", "valid", "test"))
    e.add_argument("--shard", default="0/1")
    e.add_argument("--threads", type=int, default=2)
    e.add_argument("--out", default=os.path.join(OUT_DIR, "shard"))
    m = sp.add_parser("merge")
    m.add_argument("--dir", default=OUT_DIR)
    m.add_argument("--splits", default="", help="합칠 split(쉼표). 비우면 있는 것 전부")
    a = ap.parse_args()
    {"extract": cmd_extract, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
