# -*- coding: utf-8 -*-
"""국적 KD 인코더 — 입력 길이(앞 N초)에 따른 정확도와 CPU 계산 시간.

왜
    CPU 서빙에서 처리량을 올리는 가장 싼 방법은 입력을 줄이는 것이다(동시 20건 실측: 12초 wav 1.5건/초 →
    6초 m4a 3.3건/초). 대신 정확도가 얼마나 떨어지는지를 **같은 발화 쌍**으로 잰다.

평가 발화
    AIHub 040~043 **Speech 과제**(말하기) Validation — 국적이 헤드 41라벨에 있는 화자.
    기본은 길이 --min_sec(12초) 이상만 쓴다 → 모든 길이 조건에서 입력이 실제로 달라진다.
    --min_sec 0.5 로 주면 자연 길이 분포 그대로(서버가 N초에서 자를 때의 전체 영향).
    라벨당 최대 --per_label 발화(시드 고정).
    ⚠️ 절대 정확도는 배포 수치(0.8245)와 비교하면 안 된다 — 041·043 화자 일부가 KD 학습에 쓰였을 수 있고,
       040(영어권)·042(중·일)는 국적 학습 도메인 밖이다. **길이 조건 간 차이만 유효하다.**
전처리·헤드
    서빙(deploy/nationality-kd-demo/app.py)과 같다: ffmpeg → f32le · mono · 16 kHz · 앞 12초.
    짧은 조건은 그 파형의 앞 N초를 자른다. 헤드: nan_to_num → clip ±50 → (x-mu)/sd → W·x+b.
차이 구간
    12초 대비 발화별 적중 차이의 **화자 단위 부트스트랩 95%**(2,000회).

    KD_DEVICE 와 무관하게 --device 로 고른다. GPU 는 CUDA 11 라이브러리 경로가 필요하다:
    LD_LIBRARY_PATH=<CUDA 11 링크 폴더> $PY tasks/nationality-classification/scripts/eval/eval_kd_input_length.py \
        --out /data/speech-ai/out/nationality/input_length --device cuda
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, FFMPEG

import argparse, collections, csv, glob, json, os, subprocess, time, wave
from multiprocessing import Pool
import numpy as np

SR, MAX_SEC = 16000, 12
# AIHub SpeakerMetadata.nationality → 헤드 라벨 (eval_kd_onnx_variants.py 와 같다)
ALIAS = {"Russian Federation": "Russia", "Iran, Islamic Republic Of": "Iran", "Viet Nam": "Vietnam",
         "United States of America": "United States", "USA": "United States", "UK": "United Kingdom",
         "Korea, Republic Of": "Korea", "Hongkong": "Hong Kong"}


def find_key(x, key):
    """JSON 안에서 key 를 재귀로 찾는다(과제마다 메타데이터 위치가 조금씩 달라서)."""
    if isinstance(x, dict):
        if key in x and not isinstance(x[key], (dict, list)):
            return x[key]
        for v in x.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(x, list):
        for v in x:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def wav_sec(path):
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return 0.0


def build_items(root, datasets, labels, per_label, seed, min_sec):
    want = set(labels)
    rows, unmapped, scanned = [], collections.Counter(), collections.Counter()
    for ds in datasets:
        base = glob.glob(os.path.join(root, f"{ds}.*", "01-1.정식개방데이터"))[0]
        lab = {}
        for dp, _, fs in os.walk(os.path.join(base, "Validation", "02.라벨링데이터", "VL_Speech_lab")):
            for f in fs:
                if f.endswith(".json"):
                    lab[f[:-5]] = os.path.join(dp, f)
        for sd in glob.glob(os.path.join(base, "Validation", "01.원천데이터", "VS_Speech_sound*")):
            for dp, _, fs in os.walk(sd):
                for f in fs:
                    if not f.endswith(".wav"):
                        continue
                    st = f[:-4]
                    scanned[ds] += 1
                    if st not in lab:
                        continue
                    sec = wav_sec(os.path.join(dp, f))
                    if sec < min_sec:
                        continue
                    try:
                        j = json.load(open(lab[st], encoding="utf-8"))
                    except ValueError:
                        continue
                    nat = str(find_key(j, "nationality") or "").strip()
                    lb = ALIAS.get(nat, nat)
                    if lb not in want:
                        unmapped[nat] += 1
                        continue
                    rows.append({"stem": st, "wav": os.path.join(dp, f), "label": lb,
                                 "speaker": st.split("-")[0], "dataset": ds, "sec": round(sec, 2)})
        print(f"  [{ds}] wav {scanned[ds]:,} · 라벨 JSON {len(lab):,} · 누적 선택 {len(rows):,}", flush=True)
    rng = np.random.default_rng(seed)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    picked = []
    for lb in sorted(by):
        rs = sorted(by[lb], key=lambda r: r["stem"])
        picked += [rs[i] for i in sorted(rng.permutation(len(rs))[:per_label])]
    return picked, unmapped


def decode(path):
    """app.py decode() 와 같은 ffmpeg 인자 (앞 12초)."""
    p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", path,
                        "-f", "f32le", "-ac", "1", "-ar", str(SR), "-t", str(MAX_SEC), "pipe:1"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return np.frombuffer(p.stdout, np.float32).copy() if p.returncode == 0 else None


def speaker_boot(delta, spk, n=2000, seed=0):
    spks = np.unique(spk)
    idx = {s: np.where(spk == s)[0] for s in spks}
    rng = np.random.default_rng(seed)
    vals = np.empty(n)
    for b in range(n):
        ii = np.concatenate([idx[s] for s in rng.choice(spks, len(spks), replace=True)])
        vals[b] = delta[ii].mean()
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=os.environ.get("KD_MODEL_DIR", "/data/speech-ai/weights/final/nationality"))
    ap.add_argument("--aihub_root", default=str(DATA_ROOT / "aihub"))
    ap.add_argument("--datasets", default="040,041,042,043")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_sec", type=float, default=12.0)
    ap.add_argument("--per_label", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lengths", default="12,10,8,6,4")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cpu_bench", type=int, default=12, help="CPU 4스레드 계산 시간을 잴 발화 수 (0 이면 생략)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lengths = [int(x) for x in a.lengths.split(",")]
    assert lengths[0] == MAX_SEC, "첫 길이는 기준(12초)이어야 한다"

    enc = os.path.join(a.weights, "encoder_kd_l17.onnx")
    h = np.load(os.path.join(a.weights, "head_np.npz"), allow_pickle=True)
    labels = [str(x) for x in h["labels"]]
    L2I = {l: i for i, l in enumerate(labels)}
    W, B, MU, SD = h["W"], h["b"], h["mu"], h["sd"]

    # ── 1. 평가 발화 (캐시) ──
    items_csv = os.path.join(a.out, "eval_items.csv")
    if os.path.exists(items_csv):
        items = list(csv.DictReader(open(items_csv, encoding="utf-8")))
        print(f"[items] 캐시 {len(items):,}발화", flush=True)
    else:
        items, unmapped = build_items(a.aihub_root, a.datasets.split(","), labels, a.per_label, a.seed, a.min_sec)
        with open(items_csv, "w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["stem", "wav", "label", "speaker", "dataset", "sec"])
            wr.writeheader(); wr.writerows(items)
        json.dump(dict(unmapped.most_common()), open(os.path.join(a.out, "unmapped_nationality.json"), "w"),
                  ensure_ascii=False, indent=1)
        print(f"[items] {len(items):,}발화 · 라벨 {len({r['label'] for r in items})} · "
              f"헤드에 없는 국적 상위 {unmapped.most_common(6)}", flush=True)
    secs = np.array([float(r["sec"]) for r in items])
    print(f"[items] 길이 중앙 {np.median(secs):.1f}s · 12초 초과 {100*(secs > 12).mean():.0f}% · "
          f"6초 초과 {100*(secs > 6).mean():.0f}%", flush=True)

    # ── 2. 디코드 1회 (앞 12초) ──
    t0 = time.time()
    with Pool(12) as pool:
        wavs = pool.map(decode, [r["wav"] for r in items], chunksize=16)
    keep = [i for i, w in enumerate(wavs) if w is not None and w.size >= SR // 2]
    print(f"[decode] 유효 {len(keep):,}/{len(items):,} ({time.time()-t0:.0f}s)", flush=True)

    # ── 3. 길이별 추론 ──
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
    prov = ([("CUDAExecutionProvider", {"device_id": 0, "cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
            if a.device == "cuda" else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(enc, so, providers=prov)
    if a.device == "cuda" and "CUDAExecutionProvider" not in sess.get_providers():
        raise SystemExit("CUDA provider 를 못 올렸다 — LD_LIBRARY_PATH(CUDA 11 링크 폴더) 확인")

    def logits_of(x):
        pooled = sess.run(["pooled"], {"source": np.ascontiguousarray(x)[None, :]})[0]
        z = ((np.clip(np.nan_to_num(pooled), -50, 50) - MU) / SD) @ W.T + B
        return z[0]

    y = np.array([L2I[items[i]["label"]] for i in keep])
    spk = np.array([items[i]["speaker"] for i in keep])
    dsa = np.array([items[i]["dataset"] for i in keep])
    hits, report = {}, {"n": len(keep), "n_labels": int(len(set(y.tolist()))), "n_speakers": int(len(set(spk.tolist()))),
                        "min_sec": a.min_sec, "per_label": a.per_label, "device": a.device,
                        "by_dataset_n": dict(collections.Counter(dsa.tolist())),
                        "_caveat": "절대 정확도는 배포 수치와 비교 불가(오염·도메인 차이). 길이 조건 간 차이만 유효.",
                        "lengths": {}}
    all_logits = {}
    for L in lengths:
        t0 = time.time()
        z = np.stack([logits_of(wavs[i][: L * SR]) for i in keep])
        all_logits[f"L{L}"] = z
        top5 = np.argsort(-z, 1)[:, :5]
        h1 = (top5[:, 0] == y).astype(float); h5 = (top5 == y[:, None]).any(1).astype(float)
        hits[L] = (h1, h5)
        rec = {"top1": float(h1.mean()), "top5": float(h5.mean()),
               "macro_top5": float(np.mean([h5[y == c].mean() for c in np.unique(y)])),
               "wall_sec": round(time.time() - t0, 1),
               "by_dataset": {d: {"n": int((dsa == d).sum()), "top1": float(h1[dsa == d].mean()),
                                  "top5": float(h5[dsa == d].mean())} for d in sorted(set(dsa.tolist()))}}
        if L != lengths[0]:
            b1, b5 = hits[lengths[0]]
            rec["d_top1"] = rec["top1"] - report["lengths"][str(lengths[0])]["top1"]
            rec["d_top5"] = rec["top5"] - report["lengths"][str(lengths[0])]["top5"]
            rec["d_top1_ci95"] = speaker_boot(h1 - b1, spk)
            rec["d_top5_ci95"] = speaker_boot(h5 - b5, spk)
            rec["top1_agree_vs_12s"] = float((z.argmax(1) == all_logits[f"L{lengths[0]}"].argmax(1)).mean())
        report["lengths"][str(L)] = rec
        extra = "" if L == lengths[0] else (
            f" · Δtop1 {rec['d_top1']:+.4f} [{rec['d_top1_ci95'][0]:+.4f}, {rec['d_top1_ci95'][1]:+.4f}]"
            f" · Δtop5 {rec['d_top5']:+.4f} [{rec['d_top5_ci95'][0]:+.4f}, {rec['d_top5_ci95'][1]:+.4f}]")
        print(f"[{L:2d}s] top1 {rec['top1']:.4f} · top5 {rec['top5']:.4f} · macro5 {rec['macro_top5']:.4f}{extra} "
              f"({rec['wall_sec']}s)", flush=True)

    # ── 4. CPU 계산 시간 (4스레드 세션 1개, 인코더 forward 중앙값) ──
    if a.cpu_bench:
        so4 = ort.SessionOptions(); so4.intra_op_num_threads = 4
        cpu = ort.InferenceSession(enc, so4, providers=["CPUExecutionProvider"])
        sample = [i for i in keep if wavs[i].size >= MAX_SEC * SR][: a.cpu_bench]
        bench = {}
        for L in lengths:
            ts = []
            for k, i in enumerate(sample):
                x = np.ascontiguousarray(wavs[i][: L * SR])[None, :]
                t = time.perf_counter(); cpu.run(["pooled"], {"source": x}); dt = (time.perf_counter() - t) * 1000
                if k:                                   # 첫 회는 워밍업
                    ts.append(dt)
            bench[str(L)] = float(np.median(ts)) if ts else None
        report["cpu_encode_ms_4threads"] = {"n": len(sample), "median_ms": bench}
        print("[cpu] 4스레드 인코더 중앙값 ms: " + " · ".join(f"{L}s {bench[str(L)]:.0f}" for L in lengths), flush=True)

    np.savez_compressed(os.path.join(a.out, "logits.npz"), y=y, speaker=spk, dataset=dsa, **all_logits)
    with open(os.path.join(a.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"→ {os.path.join(a.out, 'report.json')}", flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
