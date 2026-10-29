# -*- coding: utf-8 -*-
"""국적 KD 인코더(ONNX) — fp32 vs int8 변형을 **같은 발화**에서 정답 대비로 비교한다.

왜 이 방식인가
    배포 모델(encoder_kd_l17.onnx + head_np.npz, top-1 0.8245)의 test 셋은 이 머신에서 재구성할 수 없다
    (505·한국어 원어민 코퍼스와 speaker_split.json 이 없다). 그래서 지금 있는 AIHub 040~043 **발음평가**
    데이터로 평가 발화를 만들어 fp32 와 변형을 나란히 잰다.
    - 절대 정확도는 배포 수치와 비교하면 안 된다: 041·043 화자 상당수가 KD 학습에 쓰였고(오염),
      040(영어권)·042(중·일)는 국적 학습에 안 쓰인 도메인이다. **변형 간 차이만 유효하다.**
    - 판단은 정답 대비 top-1/top-5 로 한다. 예측 일치율로 판단하지 말 것(DECISIONS #12).

평가 발화
    과제 RS(문장) · 프롬프트 4어절 이상 · 국적이 헤드 41라벨에 있는 화자 · 라벨당 최대 --per_label 발화(시드 고정).
    화자는 md5(화자ID)%10==0 인 쪽만 쓴다(발음 쪽 test split 과 같은 규칙 — 고정된 화자 집합을 쓰려는 것).
전처리
    서빙(deploy/nationality-kd-demo/app.py)과 같다: ffmpeg → f32le · mono · 16 kHz · 앞 12초, 진폭 정규화 없음.
    헤드: nan_to_num → clip ±50 → (x-mu)/sd → W·x+b → softmax.

변형
    fp32           encoder_kd_l17.onnx (배포본)
    dyn_matmul     int8 동적 양자화 MatMul 가중치만 (0705 구버전과 같은 방식)
    dyn_matmul_pc  위와 같되 채널별 스케일

    $PY tasks/nationality-classification/scripts/eval/eval_kd_onnx_variants.py --out /data/speech-ai/out/compress/nationality
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, FFMPEG

import argparse, collections, csv, glob, hashlib, json, os, subprocess, time
from multiprocessing import Pool
import numpy as np

SR, MAX_SEC = 16000, 12
# AIHub SpeakerMetadata.nationality → 헤드 라벨. 여기 없는 문자열은 이름이 같으면 그대로 쓴다.
ALIAS = {"Russian Federation": "Russia", "Iran, Islamic Republic Of": "Iran", "Viet Nam": "Vietnam",
         "United States of America": "United States", "USA": "United States", "UK": "United Kingdom",
         "Korea, Republic Of": "Korea", "Hongkong": "Hong Kong"}
PARTS = {"TL": ("Training", "TS"), "VL": ("Validation", "VS")}
_W = {}


def is_fixed_speaker(spk):
    return int(hashlib.md5(spk.encode()).hexdigest()[:8], 16) % 10 == 0


def build_items(root, labels, per_label, seed):
    want = set(labels)
    rows, unmapped = [], collections.Counter()
    for ds in ("040", "041", "042", "043"):
        base = glob.glob(os.path.join(root, f"{ds}.*", "01-1.정식개방데이터"))[0]
        for part, (top, snd) in PARTS.items():
            lab_dir = os.path.join(base, top, "02.라벨링데이터", f"{part}_pronunciation_lab")
            wavs = {}
            for sd in glob.glob(os.path.join(base, top, "01.원천데이터", f"{snd}_pronunciation_sound*")):
                if os.path.isdir(sd):
                    for e in os.scandir(sd):
                        if e.name.endswith(".wav"):
                            wavs[e.name[:-4]] = e.path
            for e in os.scandir(lab_dir):
                if not e.name.endswith(".json"):
                    continue
                st = e.name[:-5]
                tok = st.split("-")
                if len(tok) < 7 or not tok[5].startswith("RS") or not is_fixed_speaker(tok[0]) or st not in wavs:
                    continue
                try:
                    j = json.load(open(e.path, encoding="utf-8"))
                except ValueError:
                    continue
                prompt = str(j.get("RecordingMetadata", {}).get("prompt", ""))
                if len(prompt.split()) < 4:
                    continue
                nat = str(j.get("SpeakerMetadata", {}).get("nationality", "")).strip()
                lab = ALIAS.get(nat, nat)
                if lab not in want:
                    unmapped[nat] += 1
                    continue
                rows.append({"stem": st, "wav": wavs[st], "label": lab, "speaker": tok[0], "dataset": ds})
        print(f"  [{ds}] 누적 {len(rows):,}", flush=True)
    rng = np.random.default_rng(seed)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    picked = []
    for lab in sorted(by):
        rs = sorted(by[lab], key=lambda r: r["stem"])
        idx = rng.permutation(len(rs))[:per_label]
        picked += [rs[i] for i in sorted(idx)]
    return picked, unmapped


def decode(path):
    """app.py decode() 와 같은 ffmpeg 인자."""
    p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", path,
                        "-f", "f32le", "-ac", "1", "-ar", str(SR), "-t", str(MAX_SEC), "pipe:1"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        return None
    return np.frombuffer(p.stdout, np.float32).copy()


def init_worker(onnx_path, threads, head_path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    _W["sess"] = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    h = np.load(head_path, allow_pickle=True)
    _W["head"] = (h["W"], h["b"], h["mu"], h["sd"])


def run_one(job):
    i, off, n, mm_path, total = job
    wav = np.memmap(mm_path, dtype=np.float32, mode="r", shape=(total,))[off:off + n]
    t = time.perf_counter()
    pooled = _W["sess"].run(["pooled"], {"source": np.ascontiguousarray(wav)[None, :]})[0]
    ms = (time.perf_counter() - t) * 1000
    W, b, mu, sd = _W["head"]
    x = np.clip(np.nan_to_num(pooled), -50, 50)
    z = ((x - mu) / sd) @ W.T + b
    return i, z[0].astype(np.float32), ms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=os.environ.get("KD_MODEL_DIR", "/data/speech-ai/weights/final/nationality"))
    ap.add_argument("--aihub_root", default=str(DATA_ROOT / "aihub"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_label", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variants", default="fp32,dyn_matmul,dyn_matmul_pc")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    fp32 = os.path.join(a.weights, "encoder_kd_l17.onnx")
    head = os.path.join(a.weights, "head_np.npz")
    labels = [str(x) for x in np.load(head, allow_pickle=True)["labels"]]
    L2I = {l: i for i, l in enumerate(labels)}

    # ── 1. 평가 발화 (캐시) ──
    items_csv = os.path.join(a.out, "eval_items.csv")
    if os.path.exists(items_csv):
        items = list(csv.DictReader(open(items_csv, encoding="utf-8")))
        print(f"[items] 캐시 {len(items):,}발화", flush=True)
    else:
        items, unmapped = build_items(a.aihub_root, labels, a.per_label, a.seed)
        with open(items_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["stem", "wav", "label", "speaker", "dataset"])
            w.writeheader(); w.writerows(items)
        json.dump(dict(unmapped.most_common()), open(os.path.join(a.out, "unmapped_nationality.json"), "w"),
                  ensure_ascii=False, indent=1)
        print(f"[items] {len(items):,}발화 · 라벨 {len({r['label'] for r in items})} · "
              f"헤드에 없는 국적 상위 {unmapped.most_common(8)}", flush=True)

    # ── 2. 디코드 1회 → memmap (모든 변형이 같은 파형을 받게) ──
    mm_path = os.path.join(a.out, "wav_cache.f32")
    idx_path = os.path.join(a.out, "wav_cache_index.json")
    if os.path.exists(idx_path):
        spans = json.load(open(idx_path))
    else:
        with Pool(12) as pool:
            wavs = pool.map(decode, [r["wav"] for r in items], chunksize=16)
        spans, off = [], 0
        with open(mm_path, "wb") as f:
            for wv in wavs:
                n = 0 if wv is None or wv.size < SR // 2 else wv.size
                if n:
                    f.write(wv.tobytes())
                spans.append([off, n]); off += n
        json.dump(spans, open(idx_path, "w"))
    total = sum(n for _, n in spans)
    keep = [i for i, (_, n) in enumerate(spans) if n > 0]
    print(f"[decode] 유효 {len(keep):,}/{len(items):,} · {total/SR/3600:.1f} h", flush=True)

    # ── 3. 변형 생성 ──
    paths = {"fp32": fp32}
    need = [v for v in a.variants.split(",") if v != "fp32"]
    if need:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        for v in need:
            dst = os.path.join(a.out, f"encoder_kd_l17_{v}.onnx")
            if not os.path.exists(dst):
                t0 = time.time()
                quantize_dynamic(fp32, dst, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"],
                                 per_channel=(v == "dyn_matmul_pc"))
                print(f"[quant] {v} {os.path.getsize(dst)/2**20:.0f} MiB ({time.time()-t0:.0f}s)", flush=True)
            paths[v] = dst

    # ── 4. 추론 + 채점 ──
    y = np.array([L2I[items[i]["label"]] for i in keep])
    ds = np.array([items[i]["dataset"] for i in keep])
    report = {"n": len(keep), "n_labels": int(len(set(y.tolist()))), "per_label_cap": a.per_label,
              "by_dataset_n": dict(collections.Counter(ds.tolist())),
              "_caveat": "절대 정확도는 배포 수치(0.8245)와 비교 불가 — 오염·도메인 차이. 변형 간 차이만 유효.",
              "variants": {}}
    logits = {}
    for v, p in paths.items():
        jobs = [(k, spans[i][0], spans[i][1], mm_path, total) for k, i in enumerate(keep)]
        z = np.zeros((len(keep), len(labels)), np.float32); ms = np.zeros(len(keep))
        t0 = time.time()
        with Pool(a.workers, initializer=init_worker, initargs=(p, a.threads, head)) as pool:
            for k, zz, t in pool.imap_unordered(run_one, jobs, chunksize=8):
                z[k] = zz; ms[k] = t
        logits[v] = z
        top5 = np.argsort(-z, 1)[:, :5]
        hit1 = top5[:, 0] == y
        hit5 = (top5 == y[:, None]).any(1)
        macro5 = float(np.mean([hit5[y == c].mean() for c in np.unique(y)]))
        rec = {"file": p, "bytes": os.path.getsize(p), "top1": float(hit1.mean()), "top5": float(hit5.mean()),
               "macro_top5": macro5, "median_encode_ms_parallel": float(np.median(ms)),
               "wall_sec": round(time.time() - t0, 1),
               "by_dataset": {d: {"n": int((ds == d).sum()), "top1": float(hit1[ds == d].mean()),
                                  "top5": float(hit5[ds == d].mean())} for d in sorted(set(ds.tolist()))}}
        if v != "fp32":
            base = logits["fp32"]
            rec["top1_agree_vs_fp32"] = float((z.argmax(1) == base.argmax(1)).mean())
            rec["d_top1_vs_fp32"] = rec["top1"] - report["variants"]["fp32"]["top1"]
            rec["d_top5_vs_fp32"] = rec["top5"] - report["variants"]["fp32"]["top5"]
        report["variants"][v] = rec
        print(f"[{v}] {os.path.getsize(p)/2**20:.0f} MiB · top1 {rec['top1']:.4f} · top5 {rec['top5']:.4f} · "
              f"macro5 {macro5:.4f}" + (f" · Δtop1 {rec['d_top1_vs_fp32']:+.4f} Δtop5 {rec['d_top5_vs_fp32']:+.4f}"
                                        if v != "fp32" else ""), flush=True)
    np.savez_compressed(os.path.join(a.out, "logits.npz"), y=y, **logits)
    with open(os.path.join(a.out, "variants_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"→ {os.path.join(a.out, 'variants_report.json')}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
