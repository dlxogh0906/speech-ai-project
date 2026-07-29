# -*- coding: utf-8 -*-
"""공식 팩이 '학습 때 본 화자를 외운 것'인지 판별하는 교차언어 테스트.

팩은 언어권별 모델 4개(eng/eu/sjr/asia)로 나뉜다. 각 모델은 자기 코퍼스만 학습했으므로
**다른 언어권 화자는 무조건 미학습 화자**다. 따라서 대각선 밖(비대각) 성능은
'화자 암기'가 절대 도울 수 없는 순수 일반화 성능이다.

  비대각이 무너짐   → 대각선 0.94는 화자 암기일 가능성
  비대각도 잘 나옴  → 이 모델은 진짜로 발음 품질을 듣는다
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
ap.add_argument("--task", default="pronunProf")
ap.add_argument("--pack", default="Pron_Proficiency_PCC")
ap.add_argument("--field", default="pron")
ap.add_argument("--per_lang", type=int, default=2500)
ap.add_argument("--out", default=f"{MDD_OUTPUTS}/eval_pcc_crosslang.json")
A = ap.parse_args()

WAV = f"{DATA_ROOT}/mdd_gold_wav16k"
DICT = "/tmp/claude-7778/-home-tta/10c4e7a9-7fb1-463f-b365-95e33ca89ab2/scratchpad/pcc_dict"
PACKS = f"{DATA_ROOT}/pcc_packs"
LANGS = ["eng", "eu", "sjr", "asia"]
MAX_SAMPLES = 1_600_000

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


@torch.inference_mode()
def decode_scores(ckpt, waves, order):
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt], arg_overrides={"data": DICT})
    m = models[0].eval().cuda().half(); d = task.target_dictionary; norm = cfg.task.normalize
    out, i = {}, 0
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
            dig = [d[t] for t in g.tolist() if t != 0 and d[t].isdigit()]
            out[s] = int(dig[0]) if dig else None
    del models, m; torch.cuda.empty_cache()
    return out


langmap = {}
for lang, dr in CORPUS_DIRS:
    if os.path.isdir(dr):
        for fn in os.listdir(dr):
            if fn.endswith(".json"): langmap.setdefault(fn[:-5], lang)

rows = [json.loads(l) for l in open(A.feats, encoding="utf-8")]
rng = np.random.default_rng(0)
groups = collections.defaultdict(list)
for r in rows:
    if r.get(A.field, 0) and r[A.field] > 0 and r["stem"] in langmap:
        groups[langmap[r["stem"]]].append(r)
sample = {}
for lg in LANGS:
    rs = groups[lg]
    idx = rng.permutation(len(rs))[: A.per_lang]
    sample[lg] = [rs[i] for i in idx]
    print(f"[표본] {lg:5} {len(sample[lg]):,}발화 · 화자 {len(set(r['stem'][:5] for r in sample[lg])):,}")

waves, order = {}, []
for lg in LANGS:
    for r in sample[lg]:
        try:
            w = load_wav(r["stem"])
        except Exception:
            continue
        if len(w) < 400: continue
        waves[r["stem"]] = w[: 20 * 16000]; order.append(r["stem"])
order.sort(key=lambda s: len(waves[s]))
print(f"[오디오] {len(order):,}개 로드 완료\n")


def pcc(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 0 else float("nan")


mat, mat_s, mat_v = {}, {}, {}
for ml in LANGS:
    ck = f"{PACKS}/{A.pack}/ckpt_pre/checkpoint_{A.task}_{ml}.pt"
    t0 = time.time()
    pred = decode_scores(ck, waves, order)
    mat[ml], mat_s[ml], mat_v[ml] = {}, {}, {}
    for dl in LANGS:
        ok = [r for r in sample[dl] if pred.get(r["stem"]) is not None]
        # 예측이 도메인 밖에서 상수로 뭉개지면 PCC는 '암기'와 무관하게 떨어진다 → 분산도 같이 본다
        mat_v[ml][dl] = round(float(np.std([pred[r["stem"]] for r in ok])), 3)
        mat[ml][dl] = round(pcc([pred[r["stem"]] for r in ok], [r[A.field] for r in ok]), 3)
        ag, al = collections.defaultdict(list), collections.defaultdict(list)
        for r in ok:
            ag[r["stem"][:5]].append(pred[r["stem"]]); al[r["stem"][:5]].append(r[A.field])
        ks = [s for s in ag if len(ag[s]) >= 3]
        mat_s[ml][dl] = round(pcc([np.mean(ag[s]) for s in ks], [np.mean(al[s]) for s in ks]), 3)
    print(f"[{ml} 모델] {time.time()-t0:.0f}s  발화단위 " +
          "  ".join(f"{dl}={mat[ml][dl]}" for dl in LANGS), flush=True)

for title, M in (("발화 단위 PCC", mat), ("화자 단위 PCC", mat_s), ("예측값 표준편차 (상수붕괴 점검)", mat_v)):
    print(f"\n=== {title} ===")
    print("{:14}".format("모델 \\ 데이터") + "".join(f"{dl:>9}" for dl in LANGS) + f"{'비대각평균':>11}")
    for ml in LANGS:
        off = np.mean([M[ml][dl] for dl in LANGS if dl != ml])
        cells = "".join((f"{M[ml][dl]:>8}*" if dl == ml else f"{M[ml][dl]:>9}") for dl in LANGS)
        print(f"{ml:14}{cells}{off:>11.3f}")
    d = np.mean([M[l][l] for l in LANGS])
    o = np.mean([M[a][b] for a in LANGS for b in LANGS if a != b])
    print(f"  대각(학습 언어권) 평균 {d:.3f}  vs  비대각(미학습 화자) 평균 {o:.3f}   차이 {d-o:+.3f}")

json.dump({"task": A.task, "utt": mat, "spk": mat_s, "pred_std": mat_v}, open(A.out, "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"\n[saved] {A.out}")
