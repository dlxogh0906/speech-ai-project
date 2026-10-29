# -*- coding: utf-8 -*-
"""점수 3종 보존 검증 — 경량화한 인코더가 뽑은 특징으로 **기존 회귀기**가 같은 점수를 내는가.

회귀기(`score_models_v2_native.joblib`)는 체크포인트가 뽑은 특징 41종으로 학습됐다. 인코더를 바꾸면
(ONNX · int8 · 층 축소) 특징 분포가 조금이라도 움직일 수 있고, 그러면 **에러 없이 점수만 틀린다.**
그래서 같은 발화에서 두 백엔드의 특징 → 점수를 나란히 놓고, 사람 점수와의 PCC 도 같이 잰다.

특징 계산은 서빙 코드(`deploy/mdd-api/mdd_infer_api.py` 의 `analyze`)를 그대로 부른다 — 복제하지 않는다.
ort 백엔드는 그 안의 `GF.forward_lprobs` 만 ONNX 세션으로 바꿔 끼운다. 표준발음은 LAB 의 canonical
(매니페스트 `.canon`), 사람 점수는 LAB `EvaluationMetadata` — `gop_feats_v2.py` 와 같은 출처다.

같은 forward 에서 **MDD 채점용 greedy 출력(`pred_raw`)도 함께 저장**한다(`Decoder.decode` 와 같은 규칙:
특수 기호 포함, blank 제외). CPU 에서 인코더를 두 번 돌리지 않으려는 것이다 — `to_preds` 로
`mdd_eval_tag71.py` / `mdd_eval_fixedref.py` 의 `--preds` 형식으로 바꾼다.

    # 1) 백엔드별 특징 추출 (+ greedy 출력)
    $PY .../eval_score_backend.py extract --manifest <dir> --backend torch --out feats_torch.jsonl
    $PY .../eval_score_backend.py extract --manifest <dir> --backend ort --onnx mdd_dyn_matmul.onnx \\
        --workers 4 --threads 4 --out feats_int8.jsonl
    # 2) 비교 (CPU 수 초)
    $PY .../eval_score_backend.py compare --a feats_torch.jsonl --b feats_int8.jsonl --out cmp.json
    # 3) MDD 채점용 예측 캐시
    $PY .../eval_score_backend.py to_preds --manifest <dir> --feats feats_int8.jsonl --out preds_int8.jsonl

PCC 집계는 `train_score_models.py` 와 같다: 라벨 0 제외 · 화자 = stem 앞 5자리 · 화자당 3발화 이상.
⚠️ 부분집합(화자당 발화가 적음)의 화자 PCC 는 문서값(0.67/0.69/0.76, test 화자당 ~149발화)보다 낮게 나온다.
   **두 백엔드의 차이**만 판단에 쓴다.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import DATA_ROOT, REPO_ROOT

import argparse, collections, glob, importlib.util, json, os, time, wave
import numpy as np

DIMS = (("pron", "PronunProfEval"), ("comp", "ComprehendEval"), ("flu", "FluencyEval"))
_W = {}


def load_wav(path):
    with wave.open(path, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


def read_manifest(mdir, split):
    with open(os.path.join(mdir, f"{split}.tsv")) as f:
        root = f.readline().rstrip("\n")
        return [os.path.join(root, l.split("\t")[0]) for l in f.read().splitlines()]


def lab_index(root, stems):
    want, out = set(stems), {}
    for d in glob.glob(os.path.join(root, "04[0-3].*", "01-1.정식개방데이터", "*", "02.라벨링데이터",
                                    "*_pronunciation_lab")):
        for e in os.scandir(d):
            st = e.name[:-5]
            if st in want and st not in out:
                out[st] = e.path
    return out


def import_api():
    spec = importlib.util.spec_from_file_location(
        "mdd_infer_api", str(REPO_ROOT / "deploy" / "mdd-api" / "mdd_infer_api.py"))
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)                  # load() 는 부르지 않는다 — 모델은 여기서 직접 붙인다
    return api


def init_backend(backend, onnx_path, ckpt, threads, dict_path, device="cpu"):
    import torch
    torch.set_num_threads(threads)
    api = import_api()
    if backend == "torch":
        from mdd import mdd_eval
        mdd_eval.CKPT = ckpt
        api.S["dec"] = mdd_eval.Decoder()
    else:
        import onnxruntime as ort
        from fairseq.data import Dictionary
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        # 정확도 추출은 여러 작업이 CPU 를 나눠 쓰는 중에 돈다. 기본값(스레드 스피닝)은 그때 CPU 를 태우기만 한다.
        # 속도 측정(bench_backends.py)은 기본값 그대로 둔다.
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        sess = ort.InferenceSession(onnx_path, so, providers=prov)
        if device == "cuda" and sess.get_providers()[0] != "CUDAExecutionProvider":
            raise SystemExit(f"CUDA provider 가 안 잡혔다: {sess.get_providers()}")
        in_dt = np.float16 if "float16" in sess.get_inputs()[0].type else np.float32

        class OrtDec:                                   # analyze() 가 쓰는 속성만
            d = Dictionary.load(dict_path)
            blank = 0                                   # <s> — task.blank_symbol
            normalize = True

        def ort_forward(dec, wavs):
            w = wavs[0]
            x = (w - w.mean()) / np.sqrt(w.var() + 1e-5)
            lp = sess.run(None, {"source": x[None, :].astype(in_dt)})[0].astype(np.float32)
            return torch.from_numpy(lp), [lp.shape[1]]

        api.GF.forward_lprobs = ort_forward             # analyze 는 모듈 속성 GF.forward_lprobs 를 부른다
        api.S["dec"] = OrtDec()

    base_fwd = api.GF.forward_lprobs

    def stash_forward(dec, wavs):                       # greedy 원출력을 옆으로 빼 둔다
        lp, Ls = base_fwd(dec, wavs)
        _W["last_argmax"] = lp[0, :Ls[0]].argmax(-1).numpy()
        return lp, Ls

    api.GF.forward_lprobs = stash_forward
    dec = api.S["dec"]
    _W.update(api=api, symbols=[dec.d[i] for i in range(len(dec.d))], blank=dec.blank)


def extract_one(job):
    i, stem, wav_path, canon, lab_path = job
    api = _W["api"]
    _W["last_argmax"] = None
    rec = {"i": i, "stem": stem}
    try:
        pred, feat, gops, _ = api.analyze(load_wav(wav_path), canon)
        rec["pred"] = "".join(pred)
        rec["feat"] = {k: float(v) for k, v in feat.items()}
    except ValueError as e:                             # 사전 밖 기호 — forward 는 이미 돌았다
        rec["skip"] = str(e)
    g = _W["last_argmax"]
    if g is not None:
        keep = np.ones(len(g), bool)
        keep[1:] = g[1:] != g[:-1]
        rec["pred_raw"] = [_W["symbols"][int(t)] for t in g[keep] if t != _W["blank"]]
    ev = {}
    if lab_path:
        try:
            ev = json.load(open(lab_path, encoding="utf-8")).get("EvaluationMetadata", {})
        except ValueError:
            pass
    for k, src in DIMS:
        rec[k] = ev.get(src, 0) or 0
    return rec


def cmd_extract(a):
    rows = read_manifest(a.manifest, a.split)
    canon = [l.split() for l in open(os.path.join(a.manifest, f"{a.split}.canon"), encoding="utf-8")]
    stems = [os.path.basename(p)[:-4] for p in rows]
    labs = lab_index(a.aihub_root, stems)
    print(f"[data] {len(rows):,}발화 · LAB {len(labs):,}", flush=True)

    meta = {"manifest": os.path.abspath(a.manifest), "split": a.split, "n": len(rows), "backend": a.backend,
            "model": os.path.abspath(a.onnx if a.backend == "ort" else a.ckpt), "ort_device": a.device}
    meta["model_bytes"] = os.path.getsize(meta["model"])
    meta_path = a.out + ".meta.json"
    done = set()
    if os.path.exists(a.out):
        old = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        if any(old.get(k) != meta[k] for k in ("manifest", "split", "n", "backend", "model", "model_bytes")):
            raise SystemExit(f"[stop] {a.out} 가 다른 설정의 결과다 — 다른 경로를 쓸 것\n  old={old}\n  new={meta}")
        for line in open(a.out, encoding="utf-8"):
            try:
                done.add(json.loads(line)["i"])
            except ValueError:
                pass
        print(f"[resume] {len(done):,} 이미 있음", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(meta, open(meta_path, "w"), ensure_ascii=False, indent=1)
    jobs = [(i, stems[i], rows[i], canon[i], labs.get(stems[i])) for i in range(len(rows)) if i not in done]

    t0 = time.time()
    with open(a.out, "a", encoding="utf-8") as fo:
        pool = None
        if a.workers == 1:
            init_backend(a.backend, a.onnx, a.ckpt, a.threads, a.dict, a.device)
            it = map(extract_one, jobs)
        else:
            from multiprocessing import Pool
            pool = Pool(a.workers, initializer=init_backend,
                        initargs=(a.backend, a.onnx, a.ckpt, a.threads, a.dict, a.device))
            it = pool.imap_unordered(extract_one, jobs, chunksize=8)
        for k, rec in enumerate(it, 1):
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if k % 1000 == 0:
                fo.flush()
                print(f"  {k:,}/{len(jobs):,}  {k/(time.time()-t0):.1f}/s", flush=True)
        if pool is not None:
            pool.close(); pool.join()
    el = time.time() - t0
    meta.update(extract_sec=round(el, 1), workers=a.workers, threads=a.threads)
    json.dump(meta, open(meta_path, "w"), ensure_ascii=False, indent=1)
    print(f"[완료] {a.out} ({el:.0f}s)")
    return 0


def cmd_to_preds(a):
    n = len(read_manifest(a.manifest, a.split))
    got = {}
    for line in open(a.feats, encoding="utf-8"):
        r = json.loads(line)
        if "pred_raw" in r:
            got[r["i"]] = r["pred_raw"]
    missing = [i for i in range(n) if i not in got]
    if missing:
        raise SystemExit(f"[stop] pred_raw 가 없는 행 {len(missing):,}개 (예: {missing[:5]}) — extract 를 끝까지 돌릴 것")
    with open(a.out, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"i": i, "pred": got[i]}, ensure_ascii=False) + "\n")
    fmeta = json.load(open(a.feats + ".meta.json"))
    json.dump({**fmeta, "from_feats": os.path.abspath(a.feats)}, open(a.out + ".meta.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"[to_preds] {n:,}행 → {a.out}")
    return 0


def cmd_compare(a):
    import joblib
    load = lambda p: {r["i"]: r for r in map(json.loads, open(p, encoding="utf-8")) if "feat" in r}
    A, B = load(a.a), load(a.b)
    keys = sorted(set(A) & set(B))
    bundle = joblib.load(a.scorer)
    names = bundle["feature_names"]
    XA = np.array([[A[i]["feat"][k] for k in names] for i in keys])
    XB = np.array([[B[i]["feat"][k] for k in names] for i in keys])
    pcc = lambda x, y: float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None

    feats = {}
    for j, k in enumerate(names):
        d = np.abs(XA[:, j] - XB[:, j])
        feats[k] = {"pcc_a_b": pcc(XA[:, j], XB[:, j]), "median_abs_diff": float(np.median(d)),
                    "p99_abs_diff": float(np.percentile(d, 99)), "std_a": float(XA[:, j].std())}
    rep = {"a": a.a, "b": a.b, "n_common": len(keys), "scorer": a.scorer,
           "pred_equal_rate": float(np.mean([A[i]["pred"] == B[i]["pred"] for i in keys])),
           "features": feats, "scores": {}}
    for dim, _ in DIMS:
        m = bundle["models"][dim]
        pa, pb = m.predict(XA), m.predict(XB)
        y = np.array([A[i][dim] for i in keys], float)
        ok = y > 0
        spk = [A[i]["stem"][:5] for i in keys]

        def spk_pcc(p):
            g, t = collections.defaultdict(list), collections.defaultdict(list)
            for s, q, yy, o in zip(spk, p, y, ok):
                if o:
                    g[s].append(q); t[s].append(yy)
            ks = [s for s in g if len(g[s]) >= a.min_utts]
            return pcc([np.mean(g[s]) for s in ks], [np.mean(t[s]) for s in ks]), len(ks)

        sa, n_spk = spk_pcc(pa)
        sb, _ = spk_pcc(pb)
        rep["scores"][dim] = {
            "pred_pcc_a_b": pcc(pa, pb), "pred_mean_abs_diff": float(np.mean(np.abs(pa - pb))),
            "pred_p99_abs_diff": float(np.percentile(np.abs(pa - pb), 99)),
            "n_labeled": int(ok.sum()), "n_speakers": n_spk,
            "pcc_utt_a": pcc(pa[ok], y[ok]), "pcc_utt_b": pcc(pb[ok], y[ok]),
            "pcc_spk_a": sa, "pcc_spk_b": sb,
            "d_pcc_spk": (sb - sa) if sa is not None and sb is not None else None}
        s = rep["scores"][dim]
        f4 = lambda v, sign="": "n/a" if v is None else format(v, sign + ".4f")   # 사람 점수 없는 매니페스트(원어민 등)
        print(f"[{dim}] 점수 상관 a↔b {f4(s['pred_pcc_a_b'])} · 평균|Δ| {s['pred_mean_abs_diff']:.4f} · "
              f"사람 PCC 화자 a {f4(sa)} → b {f4(sb)} ({f4(s['d_pcc_spk'], '+')}, {n_spk}명) · "
              f"발화 a {f4(s['pcc_utt_a'])} → b {f4(s['pcc_utt_b'])}")
    worst = sorted(feats.items(), key=lambda kv: (kv[1]["pcc_a_b"] if kv[1]["pcc_a_b"] is not None else 1))[:5]
    print("[특징] a↔b 상관이 가장 낮은 5개:", [(k, round(v["pcc_a_b"], 4) if v["pcc_a_b"] else None) for k, v in worst])
    print(f"[인식] 인식 자모열 동일 {rep['pred_equal_rate']:.4f} ({len(keys):,}발화)")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"→ {a.out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--manifest", required=True)
    e.add_argument("--split", default="test")
    e.add_argument("--backend", required=True, choices=["torch", "ort"])
    e.add_argument("--ckpt", default=os.environ.get("MDD_SERVE_CKPT"))
    e.add_argument("--onnx", default=None)
    e.add_argument("--dict", default=os.path.join(os.environ.get("MDD_DATA_DIR", ""), "dict.ltr.txt"))
    e.add_argument("--aihub_root", default=str(DATA_ROOT / "aihub"))
    e.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="ort 백엔드 실행 장치 (torch 는 Decoder 가 정한다)")
    e.add_argument("--workers", type=int, default=1)
    e.add_argument("--threads", type=int, default=4)
    e.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True, help="기준 (원본)")
    c.add_argument("--b", required=True, help="비교 대상 (경량화)")
    c.add_argument("--scorer", default=os.environ.get("MDD_SERVE_SCORER"))
    c.add_argument("--min_utts", type=int, default=3)
    c.add_argument("--out", required=True)
    t = sub.add_parser("to_preds")
    t.add_argument("--manifest", required=True)
    t.add_argument("--split", default="test")
    t.add_argument("--feats", required=True)
    t.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "extract" and a.backend == "ort" and not a.onnx:
        ap.error("ort 백엔드는 --onnx 가 필요하다")
    return {"extract": cmd_extract, "compare": cmd_compare, "to_preds": cmd_to_preds}[a.cmd](a)


if __name__ == "__main__":
    _sys.exit(main())
