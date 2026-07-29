# -*- coding: utf-8 -*-
"""공식 PCC 팩(점수 3종) 재현 — 우리 화자분리 test로 채점.

팩의 실체(소스 분석 결과): "XLS-R + 회귀(MSE)"가 아니라 **CTC로 점수 숫자를 받아쓰는** 모델이다.
  · 학습 라벨 = "3 |" 처럼 점수를 문자로 (fairseq_files.py)
  · vocab 11 = fairseq 특수토큰 4 + '0'~'5' + '|'  (proj 출력 차원으로 실측 확인)
  · 추론 = greedy CTC → 첫 숫자 문자를 점수로 (correlation.py의 string[0])
  · 언어권별 모델 4개(eng/eu/sjr/asia)를 발화의 원 코퍼스에 맞춰 라우팅해야 한다

저자 보고 PCC 0.7228은 **발화 단위**이고 화자분리 여부가 불명이다.
여기서는 두 잣대를 모두 낸다:
  · 발화 단위 PCC  (팩 저자와 같은 프로토콜)
  · 화자 단위 PCC  (우리 게이트 판정 기준 — GOP와 직접 비교 가능)
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import sys, os, json, time, argparse, collections
import numpy as np
import torch
import torch.nn.functional as F
from fairseq import checkpoint_utils

ap = argparse.ArgumentParser()
ap.add_argument("--feats", default=f"{MDD_OUTPUTS}/gop_feats_v2_test.jsonl")
ap.add_argument("--packs", default=f"{DATA_ROOT}/pcc_packs")
ap.add_argument("--max_utts", type=int, default=30, help="화자당 발화 수 상한 (GOP 평가와 동일 프로토콜)")
ap.add_argument("--min_utts", type=int, default=3)
ap.add_argument("--preds", default=f"{MDD_OUTPUTS}/pcc_pack_preds.jsonl")
ap.add_argument("--out", default=f"{MDD_OUTPUTS}/eval_pcc_pack.json")
A = ap.parse_args()

WAV = f"{DATA_ROOT}/mdd_gold_wav16k"
DICT = "/tmp/claude-7778/-home-tta/10c4e7a9-7fb1-463f-b365-95e33ca89ab2/scratchpad/pcc_dict"
MAX_SAMPLES = 1_600_000

# 과제명 → (팩 디렉토리, 라벨 필드)
TASKS = [("pronunProf", "Pron_Proficiency_PCC", "pron", "발음숙달"),
         ("comprehensibility", "Pron_Comprehensibility_PCC", "comp", "이해가능"),
         ("fluency", "Pron_Fluency_PCC", "flu", "유창성")]

# 원 코퍼스 → 팩의 LANG 코드. LAB JSON이 어느 디렉토리에 있는지로 판정한다(추측 아님).
AA = f"{DATA_ROOT}/aihub/asian_korean_speech/043.교육용_아시아어(중·일어_제외)_사용자의_한국어_음성_데이터/01-1.정식개방데이터"
EE = f"{DATA_ROOT}/aihub/european_korean_speech/041.교육용_유럽어_모국어_사용자의_한국어_음성_데이터/01-1.정식개방데이터/Training/02.라벨링데이터/extracted"
CORPUS_DIRS = [("eng", f"{DATA_ROOT}/aihub/040_labels/extracted/lab"),
               ("eu", f"{EE}/TL_pronunciation_lab"),
               ("sjr", f"{DATA_ROOT}/aihub/042_labels/extracted/lab"),
               ("asia", f"{AA}/Training/02.라벨링데이터/TL_pronunciation_lab"),
               ("asia", f"{AA}/Validation/02.라벨링데이터/VL_pronunciation_lab")]


def load_wav(stem):
    import wave
    with wave.open(f"{WAV}/{stem[:2]}/{stem}.wav", "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


def build_lang_map():
    m = {}
    for lang, d in CORPUS_DIRS:
        if not os.path.isdir(d):
            print(f"  [warn] 없음: {d}", flush=True); continue
        n = 0
        for fn in os.listdir(d):
            if fn.endswith(".json"):
                m.setdefault(fn[:-5], lang); n += 1
        print(f"  {lang:5} ← {d.split('/')[-1]:28} {n:>9,}", flush=True)
    return m


@torch.inference_mode()
def decode_scores(ckpt, stems):
    """해당 언어권 체크포인트로 점수 숫자를 디코드."""
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt], arg_overrides={"data": DICT})
    m = models[0].eval().cuda().half(); d = task.target_dictionary
    norm = cfg.task.normalize
    waves = {}
    order = []
    for s in stems:
        try:
            w = load_wav(s)
        except Exception:
            continue
        if len(w) < 400: continue
        waves[s] = w[: 20 * 16000]; order.append(s)
    order.sort(key=lambda s: len(waves[s]))

    out, i, t0 = {}, 0, time.time()
    while i < len(order):
        batch, tot = [], 0
        while i < len(order):
            n = len(waves[order[i]])
            if batch and tot + n > MAX_SAMPLES: break
            batch.append(order[i]); tot += n; i += 1
        ws = [waves[s] for s in batch]; mx = max(len(w) for w in ws)
        src = torch.zeros(len(ws), mx); pm = torch.zeros(len(ws), mx, dtype=torch.bool)
        for k, w in enumerate(ws):
            t = torch.from_numpy(w)
            if norm: t = F.layer_norm(t, t.shape)
            src[k, :len(w)] = t; pm[k, len(w):] = True
        net = m(source=src.half().cuda(), padding_mask=pm.cuda())
        lp = m.get_normalized_probs(net, log_probs=True).transpose(0, 1).float().cpu()
        opm = net.get("padding_mask")
        for k, s in enumerate(batch):
            L = int((~opm[k]).sum()) if opm is not None else lp.shape[1]
            g = lp[k, :L].argmax(-1).unique_consecutive()
            toks = [d[t] for t in g.tolist() if t != 0]
            digits = [t for t in toks if t.isdigit()]
            out[s] = int(digits[0]) if digits else None      # correlation.py 와 동일: 첫 문자
        if len(out) % 8000 < len(batch):
            print(f"    {len(out):,}/{len(order):,} ({len(out)/max(1e-9,time.time()-t0):.0f}/s)", flush=True)
    del models, m
    torch.cuda.empty_cache()
    return out


def pcc(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 0 else float("nan")


rows = [json.loads(l) for l in open(A.feats, encoding="utf-8")]
print(f"[test] 발화 {len(rows):,}")
print("[언어권 매핑] LAB JSON 소재로 판정:")
langmap = build_lang_map()
miss = sum(1 for r in rows if r["stem"] not in langmap)
print(f"  매핑 실패 {miss:,}건\n")

rng = np.random.default_rng(0)
fout = open(A.preds, "w", encoding="utf-8")
res = {}
for task, pack, field, name in TASKS:
    # 화자당 표집 (GOP 평가와 동일)
    spk = collections.defaultdict(list)
    for r in rows:
        if r.get(field, 0) and r[field] > 0 and r["stem"] in langmap:
            spk[r["stem"][:5]].append(r)
    use = []
    for s, rs in spk.items():
        if len(rs) > A.max_utts:
            idx = rng.permutation(len(rs))[: A.max_utts]; rs = [rs[i] for i in idx]
        if len(rs) >= A.min_utts: use += rs
    bylang = collections.defaultdict(list)
    for r in use: bylang[langmap[r["stem"]]].append(r["stem"])
    print(f"=== {name} ({task}) · 발화 {len(use):,} · 화자 {len(set(r['stem'][:5] for r in use)):,}")
    print(f"    언어권 분포: { {k: len(v) for k, v in sorted(bylang.items())} }", flush=True)

    pred = {}
    for lang, stems in sorted(bylang.items()):
        ck = f"{A.packs}/{pack}/ckpt_pre/checkpoint_{task}_{lang}.pt"
        if not os.path.exists(ck):
            print(f"    [skip] 없음 {ck}"); continue
        print(f"    · {lang}: {len(stems):,}발화 디코드", flush=True)
        pred.update(decode_scores(ck, stems))

    ok = [r for r in use if pred.get(r["stem"]) is not None]
    for r in ok:
        fout.write(json.dumps({"stem": r["stem"], "task": task, "lang": langmap[r["stem"]],
                               "pred": pred[r["stem"]], "true": r[field]}, ensure_ascii=False) + "\n")
    u_p = [pred[r["stem"]] for r in ok]; u_t = [r[field] for r in ok]
    agg_p, agg_t = collections.defaultdict(list), collections.defaultdict(list)
    for r in ok:
        agg_p[r["stem"][:5]].append(pred[r["stem"]]); agg_t[r["stem"][:5]].append(r[field])
    ks = [s for s in agg_p if len(agg_p[s]) >= A.min_utts]
    s_p = [np.mean(agg_p[s]) for s in ks]; s_t = [np.mean(agg_t[s]) for s in ks]
    exact = float(np.mean([a == b for a, b in zip(u_p, u_t)]))
    res[task] = {"n_utt": len(ok), "n_spk": len(ks),
                 "pcc_utt": round(pcc(u_p, u_t), 4), "pcc_spk": round(pcc(s_p, s_t), 4),
                 "exact_match": round(exact, 4)}
    print(f"    → 발화단위 PCC {res[task]['pcc_utt']}  |  화자단위 PCC {res[task]['pcc_spk']}"
          f"  |  정확일치 {exact:.1%}\n", flush=True)
fout.close()

print("=" * 74)
print(f"{'차원':10}{'발화':>9}{'화자':>7}{'발화단위PCC':>13}{'화자단위PCC':>13}{'정확일치':>10}")
for task, _, _, name in TASKS:
    if task not in res: continue
    r = res[task]
    print(f"{name:10}{r['n_utt']:>9,}{r['n_spk']:>7,}{r['pcc_utt']:>13}{r['pcc_spk']:>13}{r['exact_match']:>10.1%}")
print("\n(저자 보고 0.7228 = 발화 단위 · 화자분리 불명)")
json.dump({"max_utts": A.max_utts, "tasks": res}, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"[saved] {A.out} · preds {A.preds}")
