# -*- coding: utf-8 -*-
"""국적 분류 — 모델 크기별 재평가 (정확도·용량·지연).

기존 `metrics.json` 들은 서로 다른 test 셋에서 나왔다(0702 는 043 제외 133,509 /
0704·0705 는 043 포함 135,770). 크기 비교로 인용하려면 **같은 셋** 인 0704 vs 0705
쌍을 써야 한다. 이 스크립트는 거기에 빠져 있는 것만 실측한다:

  ① int8 양자화가 예측을 바꾸는가  — 같은 오디오에 fp32 ONNX 와 int8 ONNX 를 태워
     top-1 / top-5 일치율을 잰다. 라벨이 필요 없으므로 어느 test 셋이든 유효하다.
     (지금까지 검증은 클래스별 샘플 wav 41개뿐이었다)
  ② 12초 발화 1건 처리 지연 — CPU 스레드 수를 고정하고 잰다.

    $PY tasks/nationality-classification/scripts/eval/eval_size_tradeoff.py --n 1000
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import NAT_OUTPUTS, NAT_DATA

import os, csv, json, time, argparse
import numpy as np

SR = 16000
D300 = f"{NAT_OUTPUTS}/0705_country41_sent043_300m"


def load_wav(path, max_sec=12):
    """torchaudio 없이 16 kHz 모노 float32 로 읽는다. 실패하면 None."""
    import wave
    from scipy.signal import resample_poly
    try:
        with wave.open(path, "rb") as w:
            sr, nch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(min(w.getnframes(), int(sr * max_sec) * nch))
    except Exception:
        return None
    if sw != 2:
        return None
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        x = x.reshape(-1, nch).mean(1)
    if sr != SR:
        from math import gcd
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x if x.size >= SR // 2 else None


def main():
    ap = argparse.ArgumentParser(description="국적 분류 크기별 재평가")
    ap.add_argument("--manifest", default=f"{NAT_DATA}/aihub_country41_811/test.csv")
    ap.add_argument("--n", type=int, default=1000, help="일치율 측정 표본 수")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=f"{NAT_OUTPUTS}/size_tradeoff.json")
    A = ap.parse_args()

    import onnxruntime as ort
    import torch, torch.nn as nn

    fp32, int8 = f"{D300}/encoder_l17.onnx", f"{D300}/encoder_l17_int8.onnx"

    def mk(p):
        so = ort.SessionOptions(); so.intra_op_num_threads = A.threads
        return ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
    s32, s8 = mk(fp32), mk(int8)

    d = torch.load(f"{D300}/head.pt", map_location="cpu", weights_only=False)
    head = nn.Linear(d["feat_dim"], len(d["labels"])); head.load_state_dict(d["head"]); head.eval()
    mu = torch.from_numpy(np.asarray(d["mu"], np.float32))
    sd = torch.from_numpy(np.asarray(d["sd"], np.float32))

    def top5(pooled):
        x = torch.from_numpy(pooled)
        # 서버와 동일한 전처리 — 여기가 어긋나면 일치율이 양자화 탓으로 잘못 읽힌다
        x = torch.nan_to_num(x).clamp(-50, 50).half().float()
        x = (x - mu) / sd
        return head(x).softmax(-1)[0].topk(5).indices.tolist()

    # 정답 대비 정확도도 재려면 한글 라벨을 헤드의 영문 라벨로 옮겨야 한다.
    # 두 label2id 는 같은 인덱스를 공유하므로 인덱스로 사상한다(전 행 100% 매핑 확인).
    KO = json.load(open(f"{NAT_OUTPUTS}/0702_country41_sent_best/label2id.json", encoding="utf-8"))
    EN = json.load(open(f"{NAT_OUTPUTS}/0702_country41_sent_best/label2id_en.json", encoding="utf-8"))
    i2ko = {v: k for k, v in KO.items()}; i2en = {v: k for k, v in EN.items()}
    ko2en = {i2ko[i]: i2en[i] for i in sorted(i2ko) if i in i2en}
    lab2idx = {c: i for i, c in enumerate(d["labels"])}

    rows = list(csv.DictReader(open(A.manifest, encoding="utf-8")))
    rng = np.random.default_rng(42)
    rng.shuffle(rows)

    acc = {"fp32": [0, 0], "int8": [0, 0]}      # [top1 맞은 수, top5 맞은 수]
    n = t1 = t5 = 0
    dt32 = dt8 = 0.0
    feat_err = []
    for r in rows:
        if n >= A.n:
            break
        w = load_wav(r["audio_path"])
        if w is None:
            continue
        src = w[None, :]
        a = time.perf_counter(); p32 = s32.run(["pooled"], {"source": src})[0]; b = time.perf_counter()
        p8 = s8.run(["pooled"], {"source": src})[0]; c = time.perf_counter()
        dt32 += b - a; dt8 += c - b
        feat_err.append(float(np.abs(p32 - p8).max()))
        k32, k8 = top5(p32), top5(p8)
        t1 += int(k32[0] == k8[0])
        t5 += int(set(k32) == set(k8))
        gold = lab2idx.get(ko2en.get(r["label_name"]))
        if gold is not None:
            for nm, k in (("fp32", k32), ("int8", k8)):
                acc[nm][0] += int(k[0] == gold); acc[nm][1] += int(gold in k)
        n += 1
        if n % 100 == 0:
            print(f"  {n}/{A.n}  top1일치 {t1/n:.4f}", flush=True)

    if not n:
        raise SystemExit("읽을 수 있는 오디오가 없습니다")

    res = {
        "_설명": "int8 양자화가 예측을 바꾸는지 + 크기별 지연. 정확도 자체는 각 metrics.json 참조.",
        "manifest": A.manifest, "n_utt": n, "threads": A.threads,
        "정확도": {nm: {"top1": round(v[0] / n, 4), "top5": round(v[1] / n, 4)}
                 for nm, v in acc.items()},
        "int8_vs_fp32": {
            "top1_일치율": round(t1 / n, 4),
            "top5집합_일치율": round(t5 / n, 4),
            "특징_최대절대차_중앙": round(float(np.median(feat_err)), 4),
        },
        "지연_발화1건_ms": {
            "onnx_fp32": round(dt32 / n * 1000, 1),
            "onnx_int8": round(dt8 / n * 1000, 1),
            "speedup": round(dt32 / dt8, 2),
        },
        "용량_MB": {
            "onnx_fp32": round(os.path.getsize(fp32) / 1e6),
            "onnx_int8": round(os.path.getsize(int8) / 1e6),
        },
    }
    # 40분짜리 측정이라 **먼저 출력하고 나중에 저장한다**. 저장이 실패해도
    # (경로·권한) 결과가 화면에 남는다 — 반대 순서였다가 1,000발화를 통째로 날렸다.
    q, l = res["int8_vs_fp32"], res["지연_발화1건_ms"]
    print(f"\n===== 국적 분류 크기별 재평가 · {n:,}발화 =====")
    for nm, v in res["정확도"].items():
        print(f"{nm:5} 정답 대비  top-1 {v['top1']:.4f}  top-5 {v['top5']:.4f}")
    print(f"int8 vs fp32   top-1 일치 {q['top1_일치율']*100:.2f}%  top-5 집합 일치 {q['top5집합_일치율']*100:.2f}%")
    print(f"지연(12초 상한, {A.threads}스레드)  fp32 {l['onnx_fp32']:.0f} ms  int8 {l['onnx_int8']:.0f} ms  ({l['speedup']}배)")
    print(f"용량  fp32 {res['용량_MB']['onnx_fp32']} MB  int8 {res['용량_MB']['onnx_int8']} MB")
    try:
        os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
        json.dump(res, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"→ {A.out}")
    except OSError as e:
        print(f"[저장 실패] {e}\n위 수치를 직접 옮겨 적을 것")


if __name__ == "__main__":
    main()
