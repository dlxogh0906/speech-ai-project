# -*- coding: utf-8 -*-
"""발음 평가 API 의 **단계별 처리 시간** 측정 — 운영 서버(8932)는 건드리지 않고, 같은 코드를 이 프로세스에 올려 잰다.

`deploy/pronunciation-demo/app.py` 의 `load_all` · `analyze` · 포매터를 그대로 불러 쓰고, 단계마다 시계를 감싼다:
  decode(오디오 해석) · check(말소리 판정) · trim(묵음 자르기) · canonical(KoG2P) · forward(모델) ·
  feats(강제정렬·GOP·특징) · align(전사 정렬) · l1(국적 피드백) · score(GBM 점수) · build(응답 JSON)
발화 길이를 바꿔(앞에서 잘라) 길이 의존성을 보고, GPU·CPU 를 따로 잰다(CPU 는 MDD_FORCE_CPU=1).

⚠️ 속도 측정은 단독으로 — 다른 GPU 작업과 겹치면 값이 부풀려진다(docs/ENVIRONMENT.md 함정 9).

NPU 서버처럼 매니페스트가 없는 곳에서는 측정용 묶음을 옮겨 가서 `--clips-dir` 로 잰다(results/21):
    $PY ... --export-pack <폴더>                                   # GPU 서버: 묶음 만들기(모델 안 올림)
    python ... --clips-dir <폴더> --hf-ckpt <model> --scorer <joblib> --score-dir <norms 등이 있는 폴더> \\
        --rbln <.rbln> --rbln-device <장치> --out npu.json         # NPU 서버

    $PY .../bench_pronunciation_latency.py --hf-ckpt $SPEECH_AI_DATA/checkpoints/w2vbert_stage2/step_38000 \\
        --scorer $MDD_OUTPUTS/score_v2_w2vbert/score_models_v2_w2vbert.joblib --out <json>
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import argparse, collections, importlib.util, io, json, os, time, wave
import numpy as np


def load_app():
    p = _r / "deploy" / "pronunciation-demo" / "app.py"
    spec = importlib.util.spec_from_file_location("pron_app_bench", str(p))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def wav_bytes(x, sr=16000):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    return b.getvalue()


def read_wav16(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


def read_items(manifest, split, clips_dir=None):
    """→ (wav 경로 목록, 메타(prompt·task) 목록, 샘플 수 목록)."""
    if clips_dir:
        idx = json.load(open(os.path.join(clips_dir, "index.json"), encoding="utf-8"))
        return ([os.path.join(clips_dir, r["file"]) for r in idx], [{"prompt": r["text"], "task": r["task"]} for r in idx],
                [r["n"] for r in idx])
    lines = open(os.path.join(manifest, "%s.tsv" % split), encoding="utf-8").read().splitlines()
    root, rows = lines[0].strip(), lines[1:]
    meta = [json.loads(l) for l in open(os.path.join(manifest, "%s.meta.jsonl" % split), encoding="utf-8")]
    return [os.path.join(root, r.split("\t")[0]) for r in rows], meta, [int(r.split("\t")[1]) for r in rows]


def export_pack(manifest, split, out, n):
    """다른 서버(NPU)로 옮길 측정 묶음: 단어 발화 n개 + 8.2초 이상 발화 n개 + 학습 문항 목록 + 부하 측정용 클립 2개.

    AIHub 발화가 들어 있다 — 사내 서버끼리만 옮긴다.
    """
    import shutil
    from mdd.word_judge import save_prompts_json
    paths, meta, nsamp = read_items(manifest, split)
    rng = np.random.RandomState(0)
    word = [i for i in range(len(paths)) if meta[i].get("task") == "RW"]
    long_ = [i for i in range(len(paths)) if nsamp[i] >= 8.2 * 16000]
    picks = list(rng.choice(word, n, replace=False)) + list(rng.choice(long_, n, replace=False))
    os.makedirs(os.path.join(out, "clips"), exist_ok=True)
    idx = []
    for k, i in enumerate(picks):
        f = "clips/%03d.wav" % k
        shutil.copyfile(paths[i], os.path.join(out, f))
        read_wav16(os.path.join(out, f))                      # 16 kHz mono 16bit 인지 확인
        idx.append({"file": f, "text": meta[i].get("prompt") or "", "task": meta[i].get("task"), "n": int(nsamp[i])})
    json.dump(idx, open(os.path.join(out, "index.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    nrw, nall = save_prompts_json(manifest, os.path.join(out, "train_prompts.json"))
    lat = os.path.join(str(MDD_OUTPUTS), "analysis", "latency_2026-10-01")
    for f in ("word_2s.wav", "word_2s.txt", "sent_3s.wav", "sent_3s.txt"):
        if os.path.exists(os.path.join(lat, f)):
            shutil.copyfile(os.path.join(lat, f), os.path.join(out, f))
    print("[pack] %s · 단어 %d · 긴 발화 %d · 학습 문항 %d(단어 %d)" % (out, n, n, nall, nrw), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-ckpt", dest="hf_ckpt", default=None, help="w2v-BERT 체크포인트 폴더(NPU 면 jamo_vocab·preprocessor 만 읽는다)")
    ap.add_argument("--scorer", default=None, help="점수 3종 joblib")
    ap.add_argument("--manifest", default=os.path.join(str(MDD_OUTPUTS), "manifest_holdout_v1"))
    ap.add_argument("--split", default="valid")
    ap.add_argument("--durations", nargs="+", type=float, default=[0.6, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0])
    ap.add_argument("--clips", type=int, default=6, help="길이마다 서로 다른 발화 수")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--webm", action="store_true", help="webm/opus 해석 경로도 잰다(ffmpeg)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rbln", default=None, help="NPU 컴파일 모델(.rbln). 주면 모델 계산만 NPU 로 간다(발음 API 와 같은 방식)")
    ap.add_argument("--rbln-device", dest="rbln_device", type=int, default=0)
    ap.add_argument("--score-dir", dest="score_dir", default=os.path.join(str(MDD_OUTPUTS), "score_phoneme"),
                    help="norms.json · calibration.json · fluency_rule.json · utt_percentile.json 이 있는 폴더")
    ap.add_argument("--clips-dir", dest="clips_dir", default=None,
                    help="매니페스트 대신 --export-pack 으로 만든 묶음(index.json + clips/)에서 발화를 읽는다")
    ap.add_argument("--export-pack", dest="export_pack", default=None,
                    help="측정용 묶음을 이 폴더에 만들고 끝낸다(모델을 올리지 않는다). --hf-ckpt·--scorer 는 아무 값이나 된다")
    ap.add_argument("--pack-n", dest="pack_n", type=int, default=12, help="묶음에 넣을 단어·긴 발화 각각의 수")
    a = ap.parse_args()
    if a.export_pack:
        export_pack(a.manifest, a.split, a.export_pack, a.pack_n)
        return
    if not (a.hf_ckpt and a.scorer):
        ap.error("--hf-ckpt 와 --scorer 가 필요하다(--export-pack 만 예외)")
    app = load_app()
    sd = a.score_dir
    ns = argparse.Namespace(hf_ckpt=a.hf_ckpt, rbln=a.rbln, rbln_device=a.rbln_device, scorer=a.scorer,
                            norms=os.path.join(sd, "norms.json"), calib=os.path.join(sd, "calibration.json"),
                            utt_comp=False, flu_rule=os.path.join(sd, "fluency_rule.json"),
                            utt_pct=os.path.join(sd, "utt_percentile.json"))
    t0 = time.time(); app.load_all(ns); load_sec = time.time() - t0
    S = app.S
    import torch
    dev = S["dec"].device                              # "cuda" · "cpu" · "npu:N" 
    print("[load] %.1fs · 장치 %s · torch 스레드 %d" % (load_sec, dev, torch.get_num_threads()), flush=True)

    # 단계별 시계 — 원래 함수를 감싼다(이 프로세스 안에서만)
    T = collections.defaultdict(float)
    def timed(name, fn):
        def w(*x, **k):
            if dev == "cuda": torch.cuda.synchronize()
            s = time.perf_counter(); r = fn(*x, **k)
            if dev == "cuda": torch.cuda.synchronize()
            T[name] += time.perf_counter() - s; return r
        return w
    S["trim"] = timed("trim", S["trim"]); S["fwd"] = timed("forward", S["fwd"]); S["feats"] = timed("feats", S["feats"])
    S["scorer"].utt_scores = timed("score", S["scorer"].utt_scores)
    if S["l1fb"]: S["l1fb"].annotate = timed("l1", S["l1fb"].annotate)
    app.canonical = timed("canonical", app.canonical); app.align = timed("align", app.align)
    build = timed("build", S["ss"].build); decode = timed("decode", app.decode_audio); check = timed("check", app.check_audio)

    # 실제 발화(본 문항 valid, 또는 옮겨 온 묶음) — 길이별로 앞에서 자른다. 문장은 그 발화의 문항.
    paths, meta, nsamp = read_items(a.manifest, a.split, a.clips_dir)
    long_ix = [i for i in range(len(paths)) if nsamp[i] >= 8.2 * 16000]
    word_ix = [i for i in range(len(paths)) if meta[i].get("task") == "RW"]
    rng = np.random.RandomState(0)
    def load_wav(i):
        return read_wav16(paths[i])

    def one(x, sentence, fmt="wav"):
        raw = wav_bytes(x)
        if fmt == "webm":
            import subprocess, imageio_ffmpeg
            raw = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-f", "wav", "-i", "pipe:0",
                                  "-c:a", "libopus", "-f", "webm", "pipe:1"], input=raw, capture_output=True).stdout
        T.clear()
        s = time.perf_counter()
        wav = decode(raw)
        code, msg, vinfo = check(wav[: 16000 * 20])
        _, _, raw_out = app.analyze(wav[: 16000 * 20], sentence, "")
        build(sentence, raw_out["canon"], raw_out["sites"], raw_out["margins"], utt_scores=raw_out["scores"],
              record_id=None, vowels=S["vowels"], feats=raw_out.get("feats"))
        if dev == "cuda": torch.cuda.synchronize()
        tot = time.perf_counter() - s
        return dict(T, total=tot)

    sample = load_wav(long_ix[0])
    for _ in range(a.warmup):
        one(sample[: 32000], meta[long_ix[0]].get("prompt") or u"안녕하세요")
    if dev == "cuda": torch.cuda.reset_peak_memory_stats()

    res = {"device": dev, "load_sec": round(load_sec, 1), "torch_threads": torch.get_num_threads(), "by_duration": {}}
    stages = ["decode", "check", "trim", "canonical", "forward", "feats", "align", "l1", "score", "build", "total"]
    print("%5s | " % "길이" + " ".join("%9s" % s for s in stages) + "   (ms, 중앙값)", flush=True)
    for d in a.durations:
        pool = word_ix if d <= 2.0 else long_ix
        picks = rng.choice(pool, min(a.clips, len(pool)), replace=False)
        recs = []
        for i in picks:
            x = load_wav(i)
            if len(x) < d * 16000:                       # 짧은 단어는 반복 대신 원래 길이로
                pass
            x = x[: int(d * 16000)]
            for _ in range(a.repeat):
                recs.append(one(x, meta[i].get("prompt") or u""))
        med = dict((k, round(1000 * float(np.median([r.get(k, 0.0) for r in recs])), 1)) for k in stages)
        p95 = round(1000 * float(np.percentile([r["total"] for r in recs], 95)), 1)
        res["by_duration"][str(d)] = {"median_ms": med, "p95_total_ms": p95, "n": len(recs)}
        print("%4.1fs | " % d + " ".join("%9.1f" % med[k] for k in stages) + "   p95 합계 %.1f" % p95, flush=True)
    if a.webm:
        x = load_wav(word_ix[0])[: 32000]
        recs = [one(x, meta[word_ix[0]].get("prompt") or u"", fmt="webm") for _ in range(a.repeat * 3)]
        res["webm_2s"] = {"decode_ms": round(1000 * float(np.median([r["decode"] for r in recs])), 1),
                          "total_ms": round(1000 * float(np.median([r["total"] for r in recs])), 1)}
        print("[webm/opus 2초] decode %.1f ms · 합계 %.1f ms" % (res["webm_2s"]["decode_ms"], res["webm_2s"]["total_ms"]), flush=True)
    if dev == "cuda":
        res["gpu_peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
        # 묶음 처리 상한 — 2초 발화 N개를 한 번에
        x = load_wav(word_ix[0])[: 32000]
        from mdd.gop_feats_v2 import forward_lprobs        # 감싸지 않은 원래 함수
        res["batch_forward_2s"] = {}
        for n in (1, 4, 8, 16, 32):
            forward_lprobs(S["dec"], [x] * n)               # 이 크기 첫 호출(준비) 제외
            ts = []
            for _ in range(5):
                torch.cuda.synchronize(); s = time.perf_counter()
                forward_lprobs(S["dec"], [x] * n)
                torch.cuda.synchronize(); ts.append(time.perf_counter() - s)
            ms = 1000 * float(np.median(ts))
            res["batch_forward_2s"][str(n)] = {"ms": round(ms, 1), "per_clip_ms": round(ms / n, 1)}
            print("[묶음] 2초 × %2d개 forward %.1f ms (개당 %.1f ms)" % (n, ms, ms / n), flush=True)
        print("[gpu] 최대 메모리 %d MB" % res["gpu_peak_mem_mb"], flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
        print("[saved] %s" % a.out)


if __name__ == "__main__":
    main()
