# -*- coding: utf-8 -*-
"""모델 비교 데모(country_classifier.model_zoo)가 평가와 같은 답을 내는지 — 저장된 test 로짓과 대조한다 (2026-10-27).

models.json 에서 eval 폴더가 있는 torch 모델마다 그 폴더의 logits.npz 행 --n 개(고정 seed)를 뽑아,
평가와 같은 로더(load_wav_mono · 앞 12초)로 읽은 파형을 데모 경로(TorchZoo.logits, 한 발화씩)에 넣는다.
평가는 길이 묶음 배치(패딩) · fp32 가중치 + bf16 autocast 였고 데모는 한 발화 · Linear bf16 가중치라 완전히 같지는 않다.
1위 일치율과 확률 최대 차이를 모델마다 출력한다.
    $PY scripts/eval/check_zoo_parity.py --models deploy/nationality-all-demo/models.json --n 40 [--only full300m,full2b]
"""
# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import NAT_OUTPUTS

import argparse, csv, json, os, time
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="", help="결과 JSON 경로 (선택)")
    a = ap.parse_args()
    from country_classifier.audio import load_wav_mono
    from country_classifier.model_zoo import TorchZoo, softmax
    reg = json.load(open(a.models, encoding="utf-8"))
    res = lambda p: p.format(NAT=str(NAT_OUTPUTS), **os.environ) if p else p
    keep = set(a.only.split(",")) if a.only else None
    report = {}
    for m in reg["models"]:
        if m["kind"] != "torch" or not m.get("eval") or (keep and m["id"] not in keep):
            continue
        ev = res(m["eval"])
        z = np.load(os.path.join(ev, "logits.npz"), allow_pickle=True)
        with open(os.path.join(ev, "_test_loader.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ok = np.where(z["ok"])[0]
        pick = np.random.default_rng(a.seed).choice(ok, min(a.n, len(ok)), replace=False)
        zoo = TorchZoo()
        try:
            zoo.add_ckpt(m["id"], res(m["path"]))
        except RuntimeError as e:                         # 운율 궤적 모델 — 궤적 함수 없이 대조할 수 없다
            print("[skip] %s %s" % (m["id"], e), flush=True)
            continue
        agree, diff, t0 = 0, 0.0, time.time()
        for i in pick:
            w, _ = load_wav_mono(rows[i]["audio_path"], 16000)
            w = w.numpy().astype(np.float32)[: 12 * 16000]
            p = softmax(zoo.logits(w)[0][m["id"]])
            q = softmax(z["L"][i])
            agree += int(np.argmax(p) == np.argmax(q))
            diff = max(diff, float(np.abs(p - q).max()))
        report[m["id"]] = {"n": len(pick), "top1_agree": agree / len(pick), "max_prob_diff": round(diff, 4)}
        print("%-16s 1위 일치 %d/%d · 확률 최대 차이 %.4f · %.0fs" % (m["id"], agree, len(pick), diff, time.time() - t0), flush=True)
        del zoo
        import torch
        torch.cuda.empty_cache()
    if a.out:
        json.dump(report, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("PARITY_DONE", flush=True)


if __name__ == "__main__":
    main()
