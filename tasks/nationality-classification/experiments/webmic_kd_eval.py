# -*- coding: utf-8 -*-
"""웹마이크 재학습 최종 평가 — 서빙과 같은 경로(ffmpeg → ONNX → numpy 헤드)로 모델 여러 개를 같은 시험지에.

시험지: webmic_kd_prep.py 의 test.csv (화자 분리, 학습·선택에 안 쓴 화자)
    web     040~043 화자 — 문장·자유발화(sent) / 단어(word) 로 나눠서도 본다
    p505    131(505 도메인) 화자 — 라벨이 없어 ① 배포 모델과의 top1 일치율
            ② 그룹명 기준 정확도(JP→Japan · TH→Thailand · CN→China. EX 는 여러 나라라 제외)로 본다
    korean  KsponSpeech·001 원어민
차이 구간: 모델 간 발화별 적중 차이의 화자 단위 부트스트랩 95% (첫 모델이 기준)
⚠️ 041·043 화자 일부는 배포 모델 학습(옛 split)에 쓰였을 수 있다 — 두 모델 모두 그 영향을 받는다.
   040·042 는 배포 모델이 본 적 없는 화자다.

    LD_LIBRARY_PATH=<CUDA 11 링크 폴더> $PY .../webmic_kd_eval.py \
        --models deployed=$KD_MODEL_DIR new=/data/.../run1/deploy --device cuda
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import FFMPEG, NAT_OUTPUTS

import argparse, collections, csv, json, os, subprocess, time
from multiprocessing import Pool
import numpy as np

SR, MAX_SEC = 16000, 12
GROUP_LABEL = {"p505_JP": "Japan", "p505_TH": "Thailand", "p505_CN": "China"}


# 앱 업로드 포맷 흉내 — (인코더 인자, 파이프 가능한 컨테이너). m4a 는 파이프가 안 돼 같은 코덱의 ADTS 로 대신한다.
CODECS = {"wav": None,
          "aac64": (["-c:a", "aac", "-b:a", "64k"], "adts"),
          "opus32": (["-c:a", "libopus", "-b:a", "32k"], "webm"),
          "mp3_64": (["-c:a", "libmp3lame", "-b:a", "64k"], "mp3")}


def decode(path, codec="wav"):
    """deploy/nationality-kd-demo/app.py 와 같은 ffmpeg 인자 (앞 12초). codec 이면 먼저 그 포맷으로 인코딩."""
    src = path
    data = None
    if CODECS[codec]:
        args, fmt = CODECS[codec]
        # 서버는 최대 12초만 보지만 앱은 전체를 올린다 — 인코딩은 전체, 디코드에서 자른다
        enc = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", "48000",
                              *args, "-f", fmt, "pipe:1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if enc.returncode != 0:
            return None
        src, data = "pipe:0", enc.stdout
    p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", src, "-f", "f32le", "-ac", "1",
                        "-ar", str(SR), "-t", str(MAX_SEC), "pipe:1"], input=data,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return np.frombuffer(p.stdout, np.float32).copy() if p.returncode == 0 else None


def _decode_job(a):
    return decode(*a)


def speaker_boot(vals, spk, n=2000, seed=0):
    spks = np.unique(spk)
    if spks.size == 0 or len(vals) == 0:                 # 표준 그룹이 비어 있는 시험지(부스 시험지 등)
        return (float("nan"), float("nan"))
    idx = {s: np.where(spk == s)[0] for s in spks}
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for b in range(n):
        ii = np.concatenate([idx[s] for s in rng.choice(spks, len(spks), replace=True)])
        out[b] = vals[ii].mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def run_model(mdir, wavs, device):
    import onnxruntime as ort
    if device == "cuda":
        prov = [("CUDAExecutionProvider", {"device_id": 0, "cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
    else:
        prov = ["CPUExecutionProvider"]
    onnx_path = [p for p in ["encoder_kd_l17.onnx"] if os.path.isfile(os.path.join(mdir, p))][0]
    sess = ort.InferenceSession(os.path.join(mdir, onnx_path), providers=prov)
    if device == "cuda":
        assert "CUDAExecutionProvider" in sess.get_providers(), "CUDA EP 로드 실패 — LD_LIBRARY_PATH 확인"
    h = np.load(os.path.join(mdir, "head_np.npz"), allow_pickle=True)
    W, b, mu, sd = h["W"], h["b"], h["mu"], h["sd"]
    labels = [str(x) for x in h["labels"]]
    L = np.zeros((len(wavs), len(labels)), np.float32)
    t0 = time.time()
    for i, w in enumerate(wavs):
        if w is None:
            continue
        pooled = sess.run(["pooled"], {"source": w[None, :]})[0][0]
        x = np.clip(np.nan_to_num(pooled), -50, 50)
        L[i] = ((x - mu) / sd) @ W.T + b
        if i % 3000 == 0:
            print(f"    {i:,}/{len(wavs):,} ({time.time()-t0:.0f}s)", flush=True)
    return L, labels


def metrics(rows, L, labels, ok, ref_top1=None):
    c2i = {c: i for i, c in enumerate(labels)}
    dom = np.array([r["domain"] for r in rows], dtype=object)
    spk = np.array([r["speaker"] for r in rows], dtype=object)
    task = np.array(["word" if r["task"] == "RW" else "sent" for r in rows], dtype=object)
    y = np.array([c2i.get(r["label_en"], -1) for r in rows])
    top5 = np.argsort(-L, 1)[:, :5]
    h1 = (top5[:, 0] == y) & ok
    h5 = (top5 == y[:, None]).any(1) & ok
    web = ok & np.array([d.startswith("web") for d in dom])
    out = {}

    def block(m, with_ci=False):
        if not m.any():
            return None
        cls = sorted(set(y[m].tolist()))
        r = {"n": int(m.sum()), "speakers": int(len(set(spk[m]))),
             "top1": round(float(h1[m].mean()), 4), "top5": round(float(h5[m].mean()), 4),
             "macro1": round(float(np.mean([h1[m & (y == c)].mean() for c in cls])), 4),
             "macro5": round(float(np.mean([h5[m & (y == c)].mean() for c in cls])), 4)}
        if with_ci:
            r["top5_ci"] = speaker_boot(h5[m].astype(float), spk[m])
            r["top1_ci"] = speaker_boot(h1[m].astype(float), spk[m])
        return r

    out["web_all"] = block(web, True)
    out["web_sent"] = block(web & (task == "sent"), True)
    out["web_word"] = block(web & (task == "word"), True)
    for d in ["web040", "web041", "web042", "web043"]:
        out[d] = block(web & (dom == d))
        out[d + "_sent"] = block(web & (dom == d) & (task == "sent"))
    out["per_class_web"] = {}
    for c in sorted(set(y[web].tolist())):
        m = web & (y == c)
        out["per_class_web"][labels[c]] = {"n": int(m.sum()), "speakers": int(len(set(spk[m]))),
                                           "top1": round(float(h1[m].mean()), 4), "top5": round(float(h5[m].mean()), 4)}
    kor = ok & np.array([d in ("kspon", "kor001") for d in dom])
    out["korean"] = block(kor)
    p505 = {}
    for g, lb in GROUP_LABEL.items():
        m = ok & (dom == g)
        if m.any():
            gi = c2i[lb]
            p505[g] = {"n": int(m.sum()), "label": lb, "top1": round(float((top5[m, 0] == gi).mean()), 4),
                       "top5": round(float((top5[m] == gi).any(1).mean()), 4)}
    allp = ok & np.array([d.startswith("p505") for d in dom])
    if ref_top1 is not None and allp.any():
        p505["agree_top1_with_ref"] = round(float((top5[allp, 0] == ref_top1[allp]).mean()), 4)
        p505["ref_top1_in_top5"] = round(float((top5[allp] == ref_top1[allp][:, None]).any(1).mean()), 4)
        ex = ok & (dom == "p505_EX")
        p505["EX_agree_top1_with_ref"] = round(float((top5[ex, 0] == ref_top1[ex]).mean()), 4)
    lab = ok & (dom == "p505") & (y >= 0)                          # 131 을 라벨과 함께 평가할 때
    if lab.any():
        out["p505_labeled"] = block(lab, True)
        out["per_class_p505"] = {}
        for c in sorted(set(y[lab].tolist())):
            m = lab & (y == c)
            out["per_class_p505"][labels[c]] = {"n": int(m.sum()), "speakers": int(len(set(spk[m]))),
                                                "top1": round(float(h1[m].mean()), 4),
                                                "top5": round(float(h5[m].mean()), 4)}
    out["p505"] = p505
    return out, h1, h5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", default=str(NAT_OUTPUTS / "webmic_kd" / "test.csv"))
    ap.add_argument("--models", nargs="+", required=True, help="이름=모델폴더(encoder_kd_l17.onnx + head_np.npz). 첫째가 기준")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="스모크용 — 행을 고르게 솎아 이 개수만")
    ap.add_argument("--codec", default="wav", choices=sorted(CODECS), help="앱 업로드 포맷 흉내 (기본 원본 wav)")
    ap.add_argument("--only_sent", action="store_true", help="웹 문장·자유발화 행만 (코덱 비교용으로 시간 절약)")
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)

    rows = list(csv.DictReader(open(A.test, encoding="utf-8")))
    if A.only_sent:
        rows = [r for r in rows if r["domain"].startswith("web") and r["task"] != "RW"]
    if A.limit:
        rows = rows[::max(1, len(rows) // A.limit)][:A.limit]
    t0 = time.time()
    with Pool(A.workers) as pool:
        wavs = pool.map(_decode_job, [(r["audio_path"], A.codec) for r in rows], chunksize=32)
    ok = np.array([w is not None and w.size >= SR * 0.5 for w in wavs])
    print(f"[decode] {ok.sum():,}/{len(rows):,} ({time.time()-t0:.0f}s)", flush=True)

    res, hits, logits = {}, {}, {}
    ref_top1 = None
    for spec in A.models:
        name, mdir = spec.split("=", 1)
        print(f"[model] {name}: {mdir}", flush=True)
        L, labels = run_model(mdir, wavs, A.device)
        if ref_top1 is None:
            ref_top1 = L.argmax(1)
        res[name], h1, h5 = metrics(rows, L, labels, ok, ref_top1)
        hits[name] = (h1, h5)
        logits[name] = L
        r = res[name]

        def g(k, f="top5"):
            return "-" if r.get(k) is None else r[k][f]
        print(f"  web 전체 top1 {g('web_all', 'top1')} top5 {g('web_all')} {g('web_all', 'top5_ci')} "
              f"macro5 {g('web_all', 'macro5')} · 문장 top5 {g('web_sent')} · 단어 top5 {g('web_word')}", flush=True)
        print("  " + " · ".join(f"{d} {g(d)}" for d in ["web040", "web041", "web042", "web043"])
              + f" · 한국어 top1 {g('korean', 'top1')} · 131(라벨) top1 {g('p505_labeled', 'top1')} top5 "
              f"{g('p505_labeled')} macro5 {g('p505_labeled', 'macro5')}", flush=True)

    # ── 모델 간 차이 (첫 모델 기준, 화자 부트스트랩) ──
    names = list(res)
    dom = np.array([r["domain"] for r in rows], dtype=object)
    spk = np.array([r["speaker"] for r in rows], dtype=object)
    task = np.array(["word" if r["task"] == "RW" else "sent" for r in rows], dtype=object)
    web = ok & np.array([d.startswith("web") for d in dom])
    diffs = {}
    for nm in names[1:]:
        diffs[nm] = {}
        for tag, m in [("web_all", web), ("web_sent", web & (task == "sent"))]:
            for k, j in [("top1", 0), ("top5", 1)]:
                d = hits[nm][j][m].astype(float) - hits[names[0]][j][m].astype(float)
                diffs[nm][f"{tag}_{k}"] = {"delta": round(float(d.mean()), 4), "ci": speaker_boot(d, spk[m])}
    json.dump({"test": A.test, "n": len(rows), "decoded": int(ok.sum()), "models": dict(s.split("=", 1) for s in A.models),
               "device": A.device, "codec": A.codec, "only_sent": A.only_sent, "results": res, "diff_vs_first": diffs},
              open(os.path.join(A.out, "report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    np.savez(os.path.join(A.out, "logits.npz"), ok=ok, **{f"L_{k}": v for k, v in logits.items()})
    print(json.dumps(diffs, ensure_ascii=False, indent=1))
    print(f"[saved] {A.out}/report.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
