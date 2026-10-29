# -*- coding: utf-8 -*-
"""영어 음소 인식 공개 모델 비교 — 같은 조건으로 돌려 "틀린 발음을 들린 대로 잡는가"를 잰다.

데이터
    l2arctic   L2-ARCTIC 한국인 4명 주석 600발화 — 사람이 적은 **실제 들린 음소**가 있다(CC BY-NC · 사내 평가 전용)
    so762      speechocean762 test 2,500발화 — 중국어 모국어 · 음소마다 사람 점수 0~2(5명 평균)
    so762train speechocean762 train 2,500발화 — 점수 보정 학습용 캐시(--data 에 따로 넣는다)

지표(표준 음소 자리마다)
    사람 오류: l2arctic = 주석이 대치·삭제 / so762 = 음소 점수 < 1.0 (5명 평균. 1 = 억양 강함, 0 = 틀림)
    모델 오류: 인식 음소열을 표준발음과 정렬해 다르거나 빠진 자리
    정밀도 = 모델이 틀렸다고 한 자리 중 사람도 틀렸다고 한 비율 · 재현율 = 사람 오류 중 모델이 잡은 비율
    잘못 지적(FRR) = 사람이 맞다고 한 자리를 모델이 틀렸다고 한 비율
    엄격 = 기호가 같아야 맞음(DX·Q 표기 차이만 봐줌) · 관대 = 미국 영어 변이(AO≈AA, 약모음 AH≈IH 등)도 봐줌
    l2arctic 은 정확 진단(CD)도 — 틀린 자리에서 들린 소리까지 맞힌 비율

산출물: $MDD_OUTPUTS/english/bakeoff/<모델>/<데이터>.pkl (발화별 인식 결과 + 음소 사후확률 — 점수 실험에 다시 쓴다)
        $MDD_OUTPUTS/english/bakeoff/summary.json

    PYTHONPATH=/data/speech-ai/tools/hf_py38:/data/speech-ai/tools/en_py38 HF_HOME=... \\
    $PY tasks/pronunciation-assessment/scripts/eval/en_phone_bakeoff.py --models all
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, json, os, pickle, re, time

import numpy as np

# 모델 목록은 mdd_en.models 한 곳에 있다(점수기 학습·평가·서버와 같은 이름).


def load_wav16(path):
    """16 kHz 로 읽고 **서버와 같이** 앞뒤 무음을 0.5초로 맞춘다(mdd_en.audio.trim_edges) — 학습·서빙 특징을 같게."""
    import soundfile as sf
    from mdd_en.audio import trim_edges
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != 16000:
        import torch, torchaudio
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
    return trim_edges(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="all", help="쉼표로. all = mdd_en.models 에 등록된 전부(폴더가 없는 것은 건너뜀)")
    ap.add_argument("--data", default="l2arctic,so762", help="l2arctic · so762 · so762train (쉼표로)")
    ap.add_argument("--l2arctic", default=os.path.join(str(DATA_ROOT), "english", "l2arctic_v5"))
    ap.add_argument("--so762", default=os.path.join(str(DATA_ROOT), "english", "speechocean762"))
    ap.add_argument("--out", default=os.path.join(str(MDD_OUTPUTS), "english", "bakeoff"))
    ap.add_argument("--limit", type=int, default=0, help="데이터마다 앞에서 N발화만(시험용)")
    ap.add_argument("--so762-err", dest="so_err", type=float, default=1.0, help="이 점수 미만을 사람 오류로")
    a = ap.parse_args()
    from mdd_en.models import MODELS, resolve
    from mdd_en.recognizer import PhoneRecognizer
    from mdd_en.l2arctic import load_speakers
    from mdd_en.metrics import prf, site_eval
    from mdd_en.speechocean import load_split

    names = list(MODELS) if a.models == "all" else a.models.split(",")
    data = {}
    if "l2arctic" in a.data:
        rows = load_speakers(a.l2arctic)
        data["l2arctic"] = rows[: a.limit] if a.limit else rows
    for key, split in (("so762", "test"), ("so762train", "train")):
        if key in a.data.split(","):
            rows = load_split(a.so762, split)
            data[key] = rows[: a.limit] if a.limit else rows
    print("[data] " + " · ".join("%s %d발화" % (k, len(v)) for k, v in data.items()), flush=True)
    sp = os.path.join(a.out, "summary.json")
    summary = json.load(open(sp)) if os.path.exists(sp) else {}

    for name in names:
        m = MODELS[name]
        try:
            md, kind, vd = resolve(name)
        except (FileNotFoundError, OSError) as e:
            print("[skip] %s — %s" % (name, e), flush=True)
            continue
        m = dict(m, kind=kind)
        rec = PhoneRecognizer(md, kind, vocab_dir=vd)
        res = summary.get(name, {})                      # 다른 데이터로 돌린 결과는 남긴다
        res.update({"license": m["license"], "hf": m.get("hf") or md})
        for ds, rows in data.items():
            cache, sites, t_ms, per_num, per_den = [], [], [], 0, 0
            for r in rows:
                wav = load_wav16(r["wav"])
                t0 = time.time()
                lp = rec.log_probs(wav)
                hyp, spans = rec.greedy(lp)
                t_ms.append((time.time() - t0) * 1000)
                post, _ = rec.phone_post(lp)
                if ds == "l2arctic":
                    ph = [p for p in r["phones"] if p["canon"] is not None]
                    canon = [p["canon"] for p in ph]
                    h_act = [p["actual"] for p in ph]
                    h_err = [p["type"] in ("s", "d") and p["actual"] != p["canon"] for p in ph]
                    stressed = [p["stressed"] for p in ph]
                    perceived = [p["actual"] for p in r["phones"] if p["actual"] is not None]
                    from mdd_en.align import align
                    from mdd_en.phones import fold
                    _, d = align(fold(perceived, m["kind"]), fold(hyp, m["kind"]))
                    per_num += d; per_den += len(perceived)
                else:
                    canon, h_err, stressed = [], [], []
                    for w in r["words"]:
                        for p, acc in zip(w["phones"], w["phones_acc"]):
                            canon.append(re.sub(r"\d", "", p))
                            h_err.append(acc < a.so_err)
                            dg = re.search(r"\d", p)
                            stressed.append(None if dg is None else dg.group(0) != "0")
                    h_act = None
                sites += site_eval(canon, h_act, h_err, stressed, hyp, m["kind"])
                cache.append({"utt": r["utt"], "hyp": hyp, "spans": spans, "canon": canon,
                              "post": post.cpu().numpy().astype(np.float16), "n_samples": len(wav)})
            os.makedirs(os.path.join(a.out, name), exist_ok=True)
            pickle.dump(cache, open(os.path.join(a.out, name, ds + ".pkl"), "wb"))
            out = {"strict": prf(sites, 1), "lenient": prf(sites, 2), "ms_median": round(float(np.median(t_ms)), 1),
                   "n_utts": len(rows)}
            if ds == "l2arctic":
                cds = [s[3] for s in sites if s[3] is not None]
                out["correct_diagnosis"] = round(float(np.mean(cds)), 4) if cds else None
                out["per_vs_perceived"] = round(per_num / max(per_den, 1), 4)
            res[ds] = out
            print("[%s] %s  엄격 P %.3f R %.3f F1 %.3f FRR %.3f | 관대 P %.3f R %.3f F1 %.3f FRR %.3f%s  %.0fms"
                  % (name, ds, out["strict"]["precision"], out["strict"]["recall"], out["strict"]["f1"],
                     out["strict"]["frr"], out["lenient"]["precision"], out["lenient"]["recall"],
                     out["lenient"]["f1"], out["lenient"]["frr"],
                     ("  PER %.3f CD %.3f" % (out["per_vs_perceived"], out["correct_diagnosis"] or 0)
                      if ds == "l2arctic" else ""), out["ms_median"]), flush=True)
        summary[name] = res
        json.dump(summary, open(sp, "w"), ensure_ascii=False, indent=1)
        del rec
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    print("[saved] " + sp)


if __name__ == "__main__":
    main()
