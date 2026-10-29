# -*- coding: utf-8 -*-
"""실제 발음 CTC 평가 — "사용자가 실제로 낸 소리를 그대로 받아쓰는가".

정답 = TAG(사람이 들은 실제 발음, `.ltr`) · 기준 = LAB 표준발음(`.canon`) · 예측 = greedy CTC.
LM·lexicon·prompt 조건부 디코딩은 쓰지 않는다 (표준발음으로 교정되는 걸 가리지 않도록).

지표
  JER (PRIMARY)   (S+D+I)/N, N = TAG 자모 수. 치환·삭제·삽입률을 따로 낸다.
  표준발음 축(canonical 위치 j) 위에서 TAG·예측을 각각 canon 과 2자 정렬해 행동을 센다
  (정답 x 가 모델과 무관한 고정 정답지 방식 — mdd_eval_fixedref.py 와 같은 축):
    오발음 자리 TAG[j] != LAB[j]:  보존 pred==TAG · 되돌림 pred==LAB · 그 외(other)
    정상 자리   TAG[j] == LAB[j]:  정상 자리 오류 pred != TAG
  MDD (SECONDARY)  TA=정상&pred==LAB · FR=정상&pred!=LAB · FA=오발음&pred==LAB · TR=오발음&pred!=LAB
                   Precision=TR/(TR+FR) · Recall=TR/(TR+FA) · FAR=FA/(FA+TR) · FRR=FR/(TA+FR)
  ⚠️ 표준발음 축에 자리가 없는 **삽입형** 오발음은 자리 지표에서 빠진다(JER 에는 들어간다).
  기존 weighted F1 (REFERENCE)  공식 3자 정렬(mdd_reproduce_official.align/fixup), 자모 공간 — 연속성 확인용.

사용
  # 디코드(GPU) + 채점
  $PY .../eval_actual_ctc.py score --manifest <dir> --split test_unseen --ckpt <pt> --out <dir>/old_test_unseen
  # 두 모델 사례 비교 (A: OLD 되돌림→NEW 보존 · B: 둘 다 되돌림 · C: NEW other · D: 정상 자리를 NEW 가 망침)
  $PY .../eval_actual_ctc.py compare --manifest <dir> --split test_unseen --old <out prefix> --new <out prefix> --out <dir>

산출: <out>.preds.jsonl(+.meta.json) · <out>.metrics.json · <out>.utt.jsonl(발화별 자리 사건)
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, collections, hashlib, json, os, random, time, wave

import numpy as np

N_JAMO, N_OUT = 46, 50


# ─────────────────────────── 입력 ───────────────────────────
def read_split(mdir, split):
    with open(os.path.join(mdir, f"{split}.tsv"), encoding="utf-8") as f:
        root = f.readline().rstrip("\n")
        rows = [l.rstrip("\n").split("\t") for l in f]
    rd = lambda ext: [l.split() for l in open(os.path.join(mdir, f"{split}.{ext}"), encoding="utf-8").read().splitlines()]
    ltr = rd("ltr")
    # 원어민 매니페스트에는 .canon 이 없다(표준발음 = 전사 발음). 그때는 ltr 을 canon 으로 써서 JER 만 의미가 있게 한다
    canon = rd("canon") if os.path.exists(os.path.join(mdir, f"{split}.canon")) else [list(x) for x in ltr]
    meta_p = os.path.join(mdir, f"{split}.meta.jsonl")
    meta = [json.loads(l) for l in open(meta_p, encoding="utf-8")] if os.path.exists(meta_p) else [{} for _ in rows]
    assert len(rows) == len(ltr) == len(canon) == len(meta), (len(rows), len(ltr), len(canon), len(meta))
    return [(os.path.join(root, p), int(fr)) for p, fr in rows], ltr, canon, meta


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


# ─────────────────────────── 디코드 ───────────────────────────
def load_model(ckpt, dict_dir):
    from shared.fairseq_guard import ensure_fairseq
    ensure_fairseq()
    import torch
    from fairseq import checkpoint_utils
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt], arg_overrides={"data": dict_dir})
    m = models[0].eval()
    d = task.target_dictionary
    syms = [d[i] for i in range(d.nspecial, len(d))]
    ref = [l.split()[0] for l in open(os.path.join(dict_dir, "dict.ltr.txt"), encoding="utf-8") if l.strip()]
    assert cfg.task.labels == "ltr", cfg.task.labels
    assert len(ref) == N_JAMO and syms == ref, "체크포인트 사전이 dict.ltr.txt(46자모)와 다르다"
    proj = m.w2v_encoder.proj.weight.shape[0]
    assert len(d) == N_OUT and proj == N_OUT, f"CTC 출력 {len(d)}/{proj} != {N_OUT}"
    use_cuda = torch.cuda.is_available() and os.environ.get("MDD_FORCE_CPU") != "1"
    m = m.cuda().half() if use_cuda else m.float()
    blank = d.index(task.blank_symbol) if hasattr(task, "blank_symbol") else d.bos()
    print(f"[model] {ckpt} · vocab {len(d)} · proj {proj} · blank {blank} · normalize {cfg.task.normalize} · "
          f"{'cuda fp16' if use_cuda else 'cpu fp32'}", flush=True)
    return m, d, blank, cfg.task.normalize


def decode(ckpt, dict_dir, wavs, preds_path, batch_samples):
    import torch
    import torch.nn.functional as F
    meta = {"ckpt": os.path.abspath(ckpt), "ckpt_size": os.path.getsize(ckpt), "n": len(wavs),
            "first": wavs[0][0], "last": wavs[-1][0], "decoder": "greedy_ctc_no_lm"}
    mp = preds_path + ".meta.json"
    if os.path.exists(preds_path) and os.path.exists(mp):
        old = json.load(open(mp))
        if all(old.get(k) == v for k, v in meta.items()):
            print(f"[preds] 캐시 재사용 {preds_path}", flush=True)
            P = {}
            for l in open(preds_path, encoding="utf-8"):
                o = json.loads(l); P[o["i"]] = o["pred"]
            return [P[i] for i in range(len(wavs))]
        raise SystemExit(f"예측 캐시가 다른 조건에서 만들어졌다(덮어쓰지 않음): {mp}")
    model, d, blank, normalize = load_model(ckpt, dict_dir)
    order = sorted(range(len(wavs)), key=lambda i: wavs[i][1])
    batches, cur, b = [], 0, []
    for i in order:
        if b and cur + wavs[i][1] > batch_samples:
            batches.append(b); b, cur = [], 0
        b.append(i); cur += wavs[i][1]
    if b:
        batches.append(b)
    P = [None] * len(wavs)
    t0, done = time.time(), 0
    q = next(model.parameters())
    with torch.inference_mode():
        for bi, idx in enumerate(batches):
            xs = []
            for i in idx:
                with wave.open(wavs[i][0], "rb") as w:
                    assert w.getframerate() == 16000 and w.getnchannels() == 1
                    xs.append(np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0)
            mx = max(len(x) for x in xs)
            src = torch.zeros(len(xs), mx); pm = torch.zeros(len(xs), mx, dtype=torch.bool)
            for k, x in enumerate(xs):
                t = torch.from_numpy(x)
                if normalize:
                    t = F.layer_norm(t, t.shape)
                src[k, :len(x)] = t; pm[k, len(x):] = True
            net = model(source=src.to(q.device, q.dtype), padding_mask=pm.to(q.device))
            lp = model.get_normalized_probs(net, log_probs=True).transpose(0, 1).float()
            opm = net.get("padding_mask")
            for k, i in enumerate(idx):
                l = lp[k] if opm is None else lp[k][: int((~opm[k]).sum())]
                toks = l.argmax(-1).unique_consecutive().tolist()
                P[i] = [d[t] for t in toks if t != blank]
            done += len(idx)
            if bi % 50 == 0:
                print(f"  decode {done:,}/{len(wavs):,} ({done / max(1e-9, time.time() - t0):.0f}/s)", flush=True)
    tmp = preds_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for i, p in enumerate(P):
            f.write(json.dumps({"i": i, "pred": p}, ensure_ascii=False) + "\n")
    os.replace(tmp, preds_path)
    json.dump({**meta, "sec": round(time.time() - t0, 1)}, open(mp, "w"), ensure_ascii=False, indent=1)
    print(f"[preds] {len(wavs):,} 발화 {time.time() - t0:.0f}s → {preds_path}", flush=True)
    return P


# ─────────────────────────── 채점 ───────────────────────────
def jer_counts(ref, hyp):
    """(S, D, I). 자모 1개 = 유니코드 1글자라 문자열로 바로 정렬한다."""
    import edlib
    r, h = "".join(ref), "".join(hyp)
    if not h:
        return 0, len(r), 0
    if not r:
        return 0, 0, len(h)
    a = edlib.getNiceAlignment(edlib.align(h, r, task="path"), h, r)   # query=hyp, target=ref
    S = D = I = 0
    for qh, tr in zip(a["query_aligned"], a["target_aligned"]):
        if qh == "-":
            D += 1
        elif tr == "-":
            I += 1
        elif qh != tr:
            S += 1
    return S, D, I


def project_on_canon(canon, seq):
    """canonical 각 위치 j 에 대응하는 seq 기호(삭제면 None). seq 의 삽입은 버린다."""
    import edlib
    c, s = "".join(canon), "".join(seq)
    if not s:
        return [None] * len(c)
    a = edlib.getNiceAlignment(edlib.align(c, s, task="path"), c, s)
    out = []
    for qc, ts in zip(a["query_aligned"], a["target_aligned"]):
        if qc == "-":
            continue
        out.append(None if ts == "-" else ts)
    assert len(out) == len(c)
    return out


def score_utt(ref, canon, hyp):
    S, D, I = jer_counts(ref, hyp)
    man, prd = project_on_canon(canon, ref), project_on_canon(canon, hyp)
    ev = {"S": S, "D": D, "I": I, "N": len(ref), "mis": [], "nerr": [], "n_normal": 0}
    for j, c in enumerate(canon):
        m, p = man[j], prd[j]
        if m != c:
            cls = "preserve" if p == m else "revert" if p == c else "other"
            ev["mis"].append([j, c, m, p, cls])
        else:
            ev["n_normal"] += 1
            if p != m:
                ev["nerr"].append([j, c, p])
    return ev


def legacy_xy(canon, ref, hyp):
    from mdd.mdd_reproduce_official import align, fixup
    n1, n2, n3 = fixup(*align("".join(canon), "".join(ref), "".join(hyp) or "-"))
    L = min(len(n1), len(n2), len(n3))
    return [int(n1[k] != n2[k]) for k in range(L)], [int(n1[k] != n3[k]) for k in range(L)]


def aggregate(evs, legacy=None):
    S = sum(e["S"] for e in evs); D = sum(e["D"] for e in evs); I = sum(e["I"] for e in evs); N = sum(e["N"] for e in evs)
    mis = [m for e in evs for m in e["mis"]]
    cls = collections.Counter(m[4] for m in mis)
    n_normal = sum(e["n_normal"] for e in evs)
    nerr = [x for e in evs for x in e["nerr"]]
    fr = sum(1 for x in nerr if x[2] != x[1])          # 정상 자리에서 pred != LAB (== TAG)
    ta = n_normal - fr
    fa = cls["revert"]; tr = cls["preserve"] + cls["other"]
    r = lambda a, b: round(a / b, 4) if b else None
    pr, rc = (tr / (tr + fr) if tr + fr else 0), (tr / (tr + fa) if tr + fa else 0)
    out = {
        "utterances": len(evs),
        "jer": {"jer": r(S + D + I, N), "sub": r(S, N), "del": r(D, N), "ins": r(I, N), "S": S, "D": D, "I": I, "N": N},
        "sites": {"mispron_sites": len(mis), "preservation": r(cls["preserve"], len(mis)),
                  "reversion": r(cls["revert"], len(mis)), "other": r(cls["other"], len(mis)),
                  "other_deletion": r(sum(1 for m in mis if m[4] == "other" and m[3] is None), len(mis)),
                  "normal_sites": n_normal, "normal_site_error": r(len(nerr), n_normal)},
        "mdd": {"precision": round(pr, 4), "recall": round(rc, 4),
                "f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0,
                "far": r(fa, fa + tr), "frr": r(fr, ta + fr), "TA": ta, "FR": fr, "FA": fa, "TR": tr,
                "CD": cls["preserve"], "DE": cls["other"]},
    }
    if legacy is not None:
        from sklearn.metrics import f1_score
        X = [v for x, _ in legacy for v in x]; Y = [v for _, y in legacy for v in y]
        tp = sum(a & b for a, b in zip(X, Y)); p_ = tp / max(1, sum(Y)); r_ = tp / max(1, sum(X))
        out["legacy"] = {"f1_weighted": round(f1_score(X, Y, average="weighted"), 4),
                         "err_f1": round(2 * p_ * r_ / max(1e-9, p_ + r_), 4), "positions": len(X)}
    return out


def cmd_score(a):
    wavs, ltr, canon, meta = read_split(a.manifest, a.split)
    preds_path = a.out + ".preds.jsonl"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    P = decode(a.ckpt, a.dict_dir or a.manifest, wavs, preds_path, a.batch_samples)
    t0 = time.time()
    evs, leg = [], []
    for i in range(len(wavs)):
        hyp = [t for t in P[i] if len(t) == 1 and 0x3131 <= ord(t) <= 0x3163]
        evs.append(score_utt(ltr[i], canon[i], hyp))
        leg.append(legacy_xy(canon[i], ltr[i], hyp))
    res = {"name": a.name, "ckpt": os.path.abspath(a.ckpt), "manifest": os.path.abspath(a.manifest), "split": a.split,
           "tsv_sha16": file_sha(os.path.join(a.manifest, f"{a.split}.tsv")),
           "ltr_sha16": file_sha(os.path.join(a.manifest, f"{a.split}.ltr")),
           "decoder": "greedy CTC, no LM/lexicon/prompt", **aggregate(evs, leg)}
    by = collections.defaultdict(list)
    for e, m in zip(evs, meta):
        by[m.get("language") or "?"].append(e)
    res["by_language"] = {k: {kk: v for kk, v in aggregate(v).items() if kk in ("utterances", "jer", "sites")}
                          for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])) if len(v) >= a.min_group}
    if a.exclude_texts:
        ex = {l.strip() for l in open(a.exclude_texts, encoding="utf-8") if l.strip()}
        keep = [e for e, m in zip(evs, meta) if m.get("norm_prompt") not in ex]
        res["strict_text_unseen"] = {"exclude_texts": os.path.abspath(a.exclude_texts), "excluded_utts": len(evs) - len(keep),
                                     **aggregate(keep)}
    json.dump(res, open(a.out + ".metrics.json", "w"), ensure_ascii=False, indent=1)
    with open(a.out + ".utt.jsonl", "w", encoding="utf-8") as f:
        for i, e in enumerate(evs):
            f.write(json.dumps({"i": i, **e}, ensure_ascii=False) + "\n")
    j, s, m = res["jer"], res["sites"], res["mdd"]
    print(f"[{a.name} · {a.split}] n={res['utterances']:,}  JER {j['jer']} (S {j['sub']} D {j['del']} I {j['ins']})  "
          f"보존 {s['preservation']} 되돌림 {s['reversion']} other {s['other']}  정상자리오류 {s['normal_site_error']}  "
          f"MDD P {m['precision']} R {m['recall']} F1 {m['f1']} FAR {m['far']} FRR {m['frr']}  "
          f"legacy wF1 {res['legacy']['f1_weighted']}  ({time.time() - t0:.0f}s)", flush=True)


def cmd_compare(a):
    wavs, ltr, canon, meta = read_split(a.manifest, a.split)
    load = lambda pre: ([json.loads(l) for l in open(pre + ".utt.jsonl", encoding="utf-8")],
                        {o["i"]: o["pred"] for o in map(json.loads, open(pre + ".preds.jsonl", encoding="utf-8"))})
    (uo, po), (un, pn) = load(a.old), load(a.new)
    assert len(uo) == len(un) == len(wavs)
    cats = collections.defaultdict(list)
    for i in range(len(wavs)):
        mo = {m[0]: m for m in uo[i]["mis"]}
        for m in un[i]["mis"]:
            o = mo.get(m[0])
            if o is None:
                continue
            if o[4] == "revert" and m[4] == "preserve":
                cats["A_old_revert_new_preserve"].append((i, m[0], m[1], m[2], o[3], m[3]))
            elif o[4] == "revert" and m[4] == "revert":
                cats["B_both_revert"].append((i, m[0], m[1], m[2], o[3], m[3]))
            if m[4] == "other":
                cats["C_new_other"].append((i, m[0], m[1], m[2], o[3], m[3]))
        old_bad = {x[0] for x in uo[i]["nerr"]}
        for x in un[i]["nerr"]:
            if x[0] not in old_bad:
                cats["D_new_breaks_normal"].append((i, x[0], x[1], x[1], x[1], x[2]))
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    summary = {}
    for k in ("A_old_revert_new_preserve", "B_both_revert", "C_new_other", "D_new_breaks_normal"):
        v = cats[k]
        summary[k] = len(v)
        pick = rng.sample(v, min(a.n, len(v)))
        with open(os.path.join(a.out, f"{k}.jsonl"), "w", encoding="utf-8") as f:
            for i, j, c, t, po_, pn_ in sorted(pick):
                f.write(json.dumps({"audio": wavs[i][0], "stem": meta[i].get("stem"), "prompt": meta[i].get("prompt"),
                                    "language": meta[i].get("language"), "site": j, "LAB": c, "TAG": t,
                                    "pred_old_at_site": po_, "pred_new_at_site": pn_,
                                    "TAG_seq": " ".join(ltr[i]), "LAB_seq": " ".join(canon[i]),
                                    "OLD_seq": " ".join(po[i]), "NEW_seq": " ".join(pn[i])}, ensure_ascii=False) + "\n")
    json.dump({"old": a.old, "new": a.new, "split": a.split, "counts": summary}, open(os.path.join(a.out, "summary.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"[compare] {summary} → {a.out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--manifest", required=True); s.add_argument("--split", required=True)
    s.add_argument("--ckpt", required=True); s.add_argument("--out", required=True, help="산출 파일 접두어")
    s.add_argument("--name", default="model")
    s.add_argument("--dict-dir", dest="dict_dir", default=None, help="dict.ltr.txt 폴더 (기본: 매니페스트)")
    s.add_argument("--batch-samples", dest="batch_samples", type=int, default=1_600_000)
    s.add_argument("--exclude-texts", dest="exclude_texts", default=None,
                   help="정규화 텍스트 목록 — 여기에 없는 발화만 따로 strict_text_unseen 으로 채점")
    s.add_argument("--min-group", dest="min_group", type=int, default=500)
    c = sub.add_parser("compare")
    c.add_argument("--manifest", required=True); c.add_argument("--split", required=True)
    c.add_argument("--old", required=True); c.add_argument("--new", required=True); c.add_argument("--out", required=True)
    c.add_argument("--n", type=int, default=100); c.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cmd_score(a) if a.cmd == "score" else cmd_compare(a)


if __name__ == "__main__":
    main()
