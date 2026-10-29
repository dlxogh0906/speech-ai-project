# -*- coding: utf-8 -*-
"""매니페스트 디코드 — 실행 백엔드(PyTorch / ONNX Runtime)와 무관하게 같은 형식의 예측 캐시를 만든다.

경량화 변형(ONNX fp32 · int8 · 층 축소 …)의 정확도를 원본과 **같은 잣대로** 재기 위한 것이다.
출력은 `mdd_eval_tag71.py` / `mdd_eval_fixedref.py` 의 `--preds` 형식 그대로다:
    {"i": 매니페스트 행 번호(헤더 제외 0-based), "pred": [자모 ...]}      ← i 오름차순으로 저장
greedy 규칙은 `mdd_eval.Decoder.decode` 와 같다: argmax → 연속 중복 제거 → blank 제거 → 사전 기호.

백엔드
    torch  --torch_mode batched  원래 평가 경로 그대로(길이 정렬 배치 + 패딩, GPU fp16). 헤드라인 재현용
    torch  --torch_mode single   발화 1건씩 — 서빙 API 와 같은 조건
    ort    --onnx <그래프>        발화 1건씩. CPU 는 --workers 프로세스 × --threads 스레드

⚠️ 캐시가 어느 모델에서 나왔는지 채점 스크립트는 확인하지 않는다(재사용하면 조용히 옛 예측을 잰다).
   그래서 옆에 `<preds>.meta.json` 을 남기고, 이미 있는 캐시의 meta 가 지금 인자와 다르면 멈춘다.

    $PY tasks/pronunciation-assessment/scripts/eval/decode_manifest_backend.py --manifest <dir> --split test \\
        --backend ort --onnx mdd_int8.onnx --workers 4 --threads 4 --preds preds_int8.jsonl
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

import argparse, json, os, time, wave
import numpy as np

_W = {}          # ort 워커 프로세스 전역 (세션·사전)


def load_wav(path):
    """mdd_eval.load_wav 와 같은 식 (int16 / 32768)."""
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def greedy_syms(lp, blank, symbols):
    g = lp.argmax(-1)
    keep = np.ones(len(g), bool)
    keep[1:] = g[1:] != g[:-1]
    return [symbols[int(t)] for t in g[keep] if t != blank]


def read_manifest(mdir, split):
    with open(os.path.join(mdir, f"{split}.tsv")) as f:
        root = f.readline().rstrip("\n")
        rows = []
        for line in f:
            p, fr = line.rstrip("\n").split("\t")[:2]
            rows.append((os.path.join(root, p), int(fr)))
    return rows


def ort_init(onnx_path, threads, dict_path):
    import onnxruntime as ort
    from fairseq.data import Dictionary
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    _W["sess"] = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    in_type = _W["sess"].get_inputs()[0].type
    _W["dt"] = np.float16 if "float16" in in_type else np.float32
    d = Dictionary.load(dict_path)              # AudioFinetuningTask 의 target_dictionary 와 같은 색인
    _W["symbols"] = [d[i] for i in range(len(d))]
    _W["blank"] = d.bos()                        # blank = <s> = 0 (task.blank_symbol)


def ort_one(job):
    i, path = job
    w = load_wav(path)
    x = (w - w.mean()) / np.sqrt(w.var() + 1e-5)  # F.layer_norm(t, t.shape) — eps 1e-5, affine 없음
    lp = _W["sess"].run(None, {"source": x[None, :].astype(_W["dt"])})[0][0]
    return i, greedy_syms(lp.astype(np.float32), _W["blank"], _W["symbols"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--backend", required=True, choices=["torch", "ort"])
    ap.add_argument("--ckpt", default=os.environ.get("MDD_SERVE_CKPT"))
    ap.add_argument("--torch_mode", default="batched", choices=["batched", "single"])
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--dict", default=os.path.join(os.environ.get("MDD_DATA_DIR", ""), "dict.ltr.txt"))
    ap.add_argument("--workers", type=int, default=1, help="ort CPU 프로세스 수")
    ap.add_argument("--threads", type=int, default=4, help="프로세스당 intra-op 스레드")
    ap.add_argument("--preds", required=True)
    a = ap.parse_args()

    rows = read_manifest(a.manifest, a.split)
    meta = {"manifest": os.path.abspath(a.manifest), "split": a.split, "n": len(rows), "backend": a.backend,
            "model": os.path.abspath(a.onnx if a.backend == "ort" else a.ckpt),
            "torch_mode": a.torch_mode if a.backend == "torch" else None}
    meta["model_bytes"] = os.path.getsize(meta["model"])
    meta_path = a.preds + ".meta.json"
    partial = a.preds + ".partial"

    if os.path.exists(a.preds):
        old = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        same = all(old.get(k) == meta[k] for k in ("manifest", "split", "n", "backend", "model", "model_bytes"))
        if not same:
            raise SystemExit(f"[stop] {a.preds} 가 이미 있고 meta 가 다르다 — 다른 경로를 쓸 것\n  old={old}\n  new={meta}")
        print(f"[skip] {a.preds} 이미 완료 ({len(rows):,}행)")
        return 0

    done = {}
    if os.path.exists(partial):
        old = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        if any(old.get(k) != meta[k] for k in ("manifest", "split", "n", "backend", "model")):
            raise SystemExit(f"[stop] {partial} 가 다른 설정의 잔재다 — 지우고 다시 돌릴 것")
        for line in open(partial, encoding="utf-8"):
            try:
                r = json.loads(line); done[r["i"]] = r["pred"]
            except ValueError:
                pass                                   # 중단 시 잘린 마지막 줄
        print(f"[resume] {len(done):,}/{len(rows):,} 이미 디코드됨", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.preds)), exist_ok=True)
    json.dump(meta, open(meta_path, "w"), ensure_ascii=False, indent=1)

    todo = [i for i in range(len(rows)) if i not in done]
    t0 = time.time()
    out = open(partial, "a", encoding="utf-8")

    def emit(i, pred, k):
        out.write(json.dumps({"i": i, "pred": pred}, ensure_ascii=False) + "\n")
        done[i] = pred
        if k % 2000 == 0:
            out.flush()
            el = time.time() - t0
            print(f"  {len(done):,}/{len(rows):,}  {k/el:.1f} 발화/초  ({el:.0f}s)", flush=True)

    if a.backend == "ort":
        if not a.onnx:
            ap.error("ort 백엔드는 --onnx 가 필요하다")
        jobs = [(i, rows[i][0]) for i in todo]
        if a.workers == 1:
            ort_init(a.onnx, a.threads, a.dict)
            for k, job in enumerate(jobs, 1):
                emit(*ort_one(job), k)
        else:
            from multiprocessing import Pool
            with Pool(a.workers, initializer=ort_init, initargs=(a.onnx, a.threads, a.dict)) as pool:
                for k, (i, pred) in enumerate(pool.imap_unordered(ort_one, jobs, chunksize=8), 1):
                    emit(i, pred, k)
    else:
        import torch
        torch.set_num_threads(a.threads)
        from mdd import mdd_eval
        from mdd.gop_feats_v2 import forward_lprobs
        mdd_eval.CKPT = a.ckpt                       # import 뒤에 넣는다 (mdd.pipeline 이 덮어쓴다)
        dec = mdd_eval.Decoder()
        meta["device"] = dec.device
        json.dump(meta, open(meta_path, "w"), ensure_ascii=False, indent=1)
        symbols = [dec.d[i] for i in range(len(dec.d))]
        if a.torch_mode == "batched":
            items = [(rows[i][1], rows[i][0], i) for i in todo]   # batched() 는 items[i][0] = 프레임 수
            k = 0
            for idxs in mdd_eval.batched(items):
                preds = dec.decode([load_wav(items[j][1]) for j in idxs])
                for j, p in zip(idxs, preds):
                    k += 1
                    emit(items[j][2], p, k)
        else:
            for k, i in enumerate(todo, 1):
                lp, Ls = forward_lprobs(dec, [load_wav(rows[i][0])])
                emit(i, greedy_syms(lp[0, :Ls[0]].numpy(), dec.blank, symbols), k)
    out.close()

    if len(done) != len(rows):
        raise SystemExit(f"[stop] {len(done)}/{len(rows)} 만 디코드됨 — 다시 실행하면 이어서 한다")
    with open(a.preds + ".tmp", "w", encoding="utf-8") as f:
        for i in range(len(rows)):
            f.write(json.dumps({"i": i, "pred": done[i]}, ensure_ascii=False) + "\n")
    os.replace(a.preds + ".tmp", a.preds)
    os.remove(partial)
    el = time.time() - t0
    meta.update(decode_sec=round(el, 1), utts_per_sec=round(len(todo) / max(el, 1e-9), 2),
                workers=a.workers, threads=a.threads)
    json.dump(meta, open(meta_path, "w"), ensure_ascii=False, indent=1)
    print(f"[완료] {a.preds}  {len(rows):,}행  {el:.0f}s")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
