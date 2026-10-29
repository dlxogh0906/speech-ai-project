# -*- coding: utf-8 -*-
"""세기 쌍 합성 2차(짝) + 실제 학습 데이터 → 이어 학습용 매니페스트(`train_w2vbert_ctc.py --manifest … --weights`).

합성 짝(`build_synth_tense_pairs.py` → `synth_tense_tts.py` · `synth_tense_tts_qwen.py`)
  - 짝(…c/…w)이 둘 다 있고 길이가 정상인 것만(한글 음절 × 0.3초 기준 0.3배~2.5배+2초).
  - 검증 모델(--check-model, 기본 서비스 모델)로
      · 받아쓴 자모열과 라벨의 차이율 ≤ --max-cer (둘 다) — TTS 가 문장을 빼먹거나 뭉갠 녹음을 거른다.
      · **짝 대비**: 대비 = [LL(w|틀린 표준) − LL(w|맞는 표준)] − [LL(c|틀린 표준) − LL(c|맞는 표준)] ≥ --min-contrast.
        틀린 쪽이 맞는 쪽보다 '틀린 소리'에 뚜렷이 더 가깝지 않으면 TTS 가 글자를 바꿔 읽지 않은 것 — 짝째 버린다.
        (1차는 이 검증이 없어 '오류' 라벨이 붙은 맞는 발음이 섞였을 수 있다.)
      · 라벨의 단어 경계 동화는 `canonical_fix.choose` 로 녹음에 맞춘다(목표 자리와 겹치는 변이는 쓰지 않는다) —
        TTS 가 [밥가치]로 읽었는데 라벨이 [밥까치]면 목표와 무관한 자리에서 엉뚱한 것을 가르친다.
  - 짝 단위로 --dev-frac 을 synth_dev 로 떼어 학습에 넣지 않는다.
실제: manifest_full_tag_replay10 train 에서 무작위 --n-real 행. **행 가중치**: 세기 쌍 오류(같은 계열 다른 자모,
  첫소리 자리)가 있는 외국인 행 = 1 + --pair-bonus (results/19 H4 의 b=0.5 와 같은 꼴). 판정 기준 표준발음 = **문항 단위
  라벨 표준발음**(--item-lab-split 의 .canon 에서 문항 코드별 최빈값) — 행마다 표준발음이 든 캐시는 화자 정보가 섞여 있어
  쓰지 않는다. 문항 코드는 녹음 파일 이름에서만 뽑는다. 원어민 행·문항을 모르는 행 = 1.
valid: replay10 valid 그대로.

    $PY .../build_synth_tense_pairs_manifest.py --texts $MDD_OUTPUTS/synth_tense_v2/texts_qwen.jsonl,$MDD_OUTPUTS/synth_tense_v2/texts_chatterbox.jsonl
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, G2P_TOOLS, MDD_OUTPUTS, ensure_kog2p

import argparse, collections, json, os, random, re, shutil, wave

ITEM_RE = re.compile(r"-(R[WSP]\d{3})-")
ONSET_POS = ("initial", "intervocalic", "post_sonorant")


def expected(t):
    return max(1, len(re.findall(r"[가-힣]", t))) * 0.3


def hard_pair_row(lab, tag, FAMILY_OF, position):
    """라벨 표준발음 축에서 TAG 가 같은 계열 다른 자모인 첫소리 자리가 있나."""
    import edlib
    c, t = "".join(lab), "".join(tag)
    if not c or not t:
        return False
    al = edlib.getNiceAlignment(edlib.align(c, t, task="path"), c, t)
    j = 0
    for q, s in zip(al["query_aligned"], al["target_aligned"]):
        if q == "-":
            continue
        if s != "-" and s != q and FAMILY_OF.get(q) and FAMILY_OF.get(q) == FAMILY_OF.get(s) and position(lab, j) in ONSET_POS:
            return True
        j += 1
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--texts", default=",".join(os.path.join(str(MDD_OUTPUTS), "synth_tense_v2", f)
                                                 for f in ("texts_qwen.jsonl", "texts_chatterbox.jsonl")))
    ap.add_argument("--wav-dir", dest="wav_dir", default=os.path.join(str(MDD_OUTPUTS), "synth_tense_v2", "wav"))
    ap.add_argument("--real", default=os.path.join(str(MDD_OUTPUTS), "manifest_full_tag_replay10"))
    ap.add_argument("--n-real", dest="n_real", type=int, default=150000)
    ap.add_argument("--pair-bonus", dest="pair_bonus", type=float, default=0.5)
    ap.add_argument("--item-lab-split", dest="item_lab_split", default="test")
    ap.add_argument("--replicate", type=int, default=2)
    ap.add_argument("--dev-frac", dest="dev_frac", type=float, default=0.05)
    ap.add_argument("--check-model", dest="check_model",
                    default=os.path.join(str(DATA_ROOT), "checkpoints", "w2vbert_stage2", "step_38000"))
    ap.add_argument("--max-cer", dest="max_cer", type=float, default=0.35)
    ap.add_argument("--min-contrast", dest="min_contrast", type=float, default=2.0)
    ap.add_argument("--no-junction", dest="no_junction", action="store_true", help="라벨 단어 경계 변이 고르기를 끈다")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(str(MDD_OUTPUTS), "manifest_synth_tense_v2"))
    a = ap.parse_args()
    import edlib
    import numpy as np
    import torch
    import torch.nn.functional as F
    from mdd.pair_rescue import FAMILY_OF, position
    from mdd.canonical_fix import CanonicalBuilder, choose
    from mdd.gop_feats_v2 import HFDecoder, forward_lprobs
    from mdd.vad import trim

    ensure_kog2p()
    argv, _sys.argv = _sys.argv, _sys.argv[:1]
    try:
        from g2p import readRules, graph2prono, ver_info
    finally:
        _sys.argv = argv
    from mdd.jamo_to_kog2p import phones_to_jamo, INV
    ri, ro = readRules(ver_info[0], str(G2P_TOOLS / "KoG2P" / "rulebook.txt"))

    def kog2p(s):
        text = re.sub(r"\s+", " ", re.sub(r"[^0-9A-Za-z가-힣\s]", " ", s or "")).strip()
        out = []
        for j in phones_to_jamo([p for p in graph2prono(text, ri, ro).split() if p in INV]):
            out += {"ㅙ": ["ㅗ", "ㅐ"], "ㅞ": ["ㅜ", "ㅔ"]}.get(j, [j])
        return out
    B = CanonicalBuilder(kog2p)

    vocab = set(l.split()[0] for l in open(os.path.join(a.real, "dict.ltr.txt"), encoding="utf-8") if l.strip())
    drop = collections.Counter()
    by_pid = collections.defaultdict(dict)
    for f in a.texts.split(","):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            by_pid[r["pid"]][r["version"]] = r
    pairs = []
    for pid, d in sorted(by_pid.items()):
        if set(d) != {"c", "w"}:
            drop["incomplete_text"] += 1; continue
        ok = True
        for v in ("c", "w"):
            r = d[v]
            p = os.path.join(a.wav_dir, r["sid"] + ".wav")
            if not os.path.exists(p):
                drop["no_wav"] += 1; ok = False; break
            with wave.open(p, "rb") as w:
                fr = w.getnframes()
            e = expected(r["text_spoken"])
            if not (0.3 * e <= fr / 16000.0 <= 2.5 * e + 2):
                drop["length"] += 1; ok = False; break
            if any(t not in vocab for t in r["actual"] + r["canon"]):
                drop["vocab"] += 1; ok = False; break
            r["path"], r["frames"] = p, fr
        if ok:
            pairs.append(d)
    print("[짝] 후보 %d · 파일·길이 통과 %d · 버림 %s" % (len(by_pid), len(pairs), dict(drop)), flush=True)

    dec = HFDecoder(a.check_model)
    to_id = dec.d.index

    def ll(lp, seq):
        lp_t = torch.as_tensor(lp).float().unsqueeze(1)
        return -F.ctc_loss(lp_t, torch.tensor([[to_id(c) for c in seq]]), torch.tensor([lp_t.shape[0]]),
                           torch.tensor([len(seq)]), blank=dec.blank, reduction="sum").item()

    keep, contrasts, junction_used = [], [], collections.Counter()
    for n, d in enumerate(pairs):
        info = {}
        for v in ("c", "w"):
            r = d[v]
            with wave.open(r["path"], "rb") as w:
                x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
            x, _ = trim(x, 16000)
            lp, Ls = forward_lprobs(dec, [x])
            lp = lp[0, :Ls[0]].numpy()
            hyp, prev = [], None
            for t in lp.argmax(-1).tolist():
                if t != prev and t != dec.blank:
                    s = dec.d[t]
                    if len(s) == 1 and 0x3131 <= ord(s) <= 0x3163:
                        hyp.append(s)
                prev = t
            info[v] = (lp, hyp)
        c0, c1 = d["c"]["canon"], d["w"]["actual"]
        llr = lambda v: ll(info[v][0], c1) - ll(info[v][0], c0)
        con = llr("w") - llr("c")
        contrasts.append(con)
        d["contrast"] = round(con, 3)
        if not (con >= a.min_contrast):
            drop["contrast"] += 1; continue
        bad = False
        for v in ("c", "w"):
            r = d[v]
            lab = list(r["actual"])
            if not a.no_junction:
                j = r["target"]
                base, alts = B.variants(r["text_spoken"])
                if base == lab:                        # 라벨이 기본 보정 결과와 같을 때만(목표 자리 좌표가 같다)
                    alts = [x_ for x_ in alts if not (x_[0] <= j + 1 and j - 1 <= x_[1])]
                    lab2, used = choose(info[v][0], base, alts, to_id, blank=dec.blank)
                    if used:
                        junction_used.update(u[3] for u in used)
                        lab = lab2
            r["label"] = lab
            ref = "".join(lab)
            hyp = info[v][1]
            cer = edlib.align("".join(hyp), ref)["editDistance"] / max(1, len(ref)) if hyp else 1.0
            r["check_cer"] = round(cer, 4)
            if cer > a.max_cer:
                bad = True
        if bad:
            drop["check_cer"] += 1; continue
        keep.append(d)
        if n % 500 == 0:
            print("  확인 %d/%d · 통과 %d" % (n + 1, len(pairs), len(keep)), flush=True)
    rng = random.Random(a.seed)
    rng.shuffle(keep)
    n_dev = int(len(keep) * a.dev_frac)
    dev, syn = keep[:n_dev], keep[n_dev:]

    # 실제 행 + 가중치
    with open(os.path.join(a.real, "train.tsv"), encoding="utf-8") as f:
        root = f.readline().rstrip("\n")
        real_rows = [l.rstrip("\n") for l in f]
    real_ltr = open(os.path.join(a.real, "train.ltr"), encoding="utf-8").read().splitlines()
    assert root == "/" and len(real_rows) == len(real_ltr), (root, len(real_rows), len(real_ltr))
    with open(os.path.join(a.real, a.item_lab_split + ".tsv"), encoding="utf-8") as f:
        f.readline()
        lab_paths = [l.split("\t")[0] for l in f]
    lab_can = open(os.path.join(a.real, a.item_lab_split + ".canon"), encoding="utf-8").read().splitlines()
    votes = collections.defaultdict(collections.Counter)
    for p, c in zip(lab_paths, lab_can):
        m = ITEM_RE.search(p)
        if m and c.strip():
            votes[m.group(1)][c.strip()] += 1
    item_lab = dict((k, v.most_common(1)[0][0].split()) for k, v in votes.items())
    pick = sorted(random.Random(a.seed + 1).sample(range(len(real_rows)), min(a.n_real, len(real_rows))))
    wts, wkind = [], collections.Counter()
    for i in pick:
        m = ITEM_RE.search(real_rows[i])
        if not m or m.group(1) not in item_lab:
            wts.append(1.0); wkind["문항 모름(원어민 등)"] += 1; continue
        hp = hard_pair_row(item_lab[m.group(1)], real_ltr[i].split(), FAMILY_OF, position)
        wts.append(1.0 + a.pair_bonus if hp else 1.0)
        wkind["세기 쌍 오류 행" if hp else "그 밖의 외국인 행"] += 1

    os.makedirs(a.out, exist_ok=True)
    rel = lambda p: os.path.abspath(p).lstrip("/")
    with open(os.path.join(a.out, "train.tsv"), "w", encoding="utf-8") as ft, \
            open(os.path.join(a.out, "train.ltr"), "w", encoding="utf-8") as fl, \
            open(os.path.join(a.out, "train.kind"), "w", encoding="utf-8") as fk, \
            open(os.path.join(a.out, "train.weights"), "w", encoding="utf-8") as fw:
        ft.write("/\n")
        for i, w in zip(pick, wts):
            ft.write(real_rows[i] + "\n"); fl.write(real_ltr[i] + "\n"); fk.write("real\n"); fw.write("%g\n" % w)
        for _ in range(a.replicate):
            for d in syn:
                for v in ("c", "w"):
                    r = d[v]
                    ft.write("%s\t%d\n" % (rel(r["path"]), r["frames"])); fl.write(" ".join(r["label"]) + "\n")
                    fk.write("synth_%s_%s_%s\n" % (r["engine"], "ok" if v == "c" else r["pair"], r["src"])); fw.write("1\n")
    with open(os.path.join(a.out, "synth_dev.tsv"), "w", encoding="utf-8") as ft, \
            open(os.path.join(a.out, "synth_dev.ltr"), "w", encoding="utf-8") as fl, \
            open(os.path.join(a.out, "synth_dev.canon"), "w", encoding="utf-8") as fc, \
            open(os.path.join(a.out, "synth_dev.jsonl"), "w", encoding="utf-8") as fj:
        ft.write("/\n")
        for d in dev:
            for v in ("c", "w"):
                r = d[v]
                ft.write("%s\t%d\n" % (rel(r["path"]), r["frames"])); fl.write(" ".join(r["label"]) + "\n")
                fc.write(" ".join(r["canon"]) + "\n")
                fj.write(json.dumps(dict((k, r[k]) for k in ("sid", "pid", "version", "engine", "src", "pair", "target",
                                                              "position", "text_ref", "text_spoken")), ensure_ascii=False) + "\n")
    for ext in ("tsv", "ltr"):
        shutil.copy(os.path.join(a.real, "valid." + ext), os.path.join(a.out, "valid." + ext))
    shutil.copy(os.path.join(a.real, "dict.ltr.txt"), os.path.join(a.out, "dict.ltr.txt"))
    cs = sorted(contrasts)
    q = lambda p: round(cs[int(p * (len(cs) - 1))], 2) if cs else None
    stats = {"real_rows": len(pick), "real_weight_kinds": dict(wkind), "pair_bonus": a.pair_bonus,
             "synth_pairs_train": len(syn), "synth_rows_train": len(syn) * 2 * a.replicate, "synth_dev_pairs": len(dev),
             "dropped": dict(drop), "replicate": a.replicate, "min_contrast": a.min_contrast,
             "contrast_quantiles(5/25/50/75/95%)": [q(0.05), q(0.25), q(0.5), q(0.75), q(0.95)],
             "junction_variants_used": dict(junction_used),
             "synth_by_engine": dict(collections.Counter(d["c"]["engine"] for d in syn)),
             "synth_by_src": dict(collections.Counter(d["c"]["src"] for d in syn)),
             "synth_pairs_by_pair": dict(collections.Counter(d["w"]["pair"] for d in syn).most_common()),
             "synth_hours_train": round(sum(d[v]["frames"] for d in syn for v in ("c", "w")) / 16000 / 3600, 2),
             "synth_voices_train": len(set(d["c"]["voice"] for d in syn))}
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
