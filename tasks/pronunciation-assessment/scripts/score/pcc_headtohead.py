# -*- coding: utf-8 -*-
"""PCC head-to-head: 공식 점수 팩(3차원×4언어권 CTC) vs 우리 GOP+보정.

- 같은 TAG test 화자들에서 화자당 최대 40발화 표본
- 발화별로 소속 코퍼스(040/041/042/043)에 맞는 팩으로 라우팅해 점수 디코드
- 정답 = LAB JSON EvaluationMetadata (0은 미채점 → 제외)
- 보고: 차원별 (a) 발화 단위 PCC (팩 자체 방식) (b) 화자 단위 PCC (우리 방식과 비교)
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import sys, os, json, time, collections
import numpy as np
from scipy import stats
from fairseq import checkpoint_utils
import torch, torch.nn.functional as F
from mdd.mdd_eval import load_wav

M = f"{MDD_OUTPUTS}/manifest_tag_all"
OUT = f"{MDD_OUTPUTS}/eval_pcc_headtohead.json"
PRED_CACHE = f"{MDD_OUTPUTS}/pcc_pack_preds.jsonl"
MAX_PER_SPK = 40
PACKS = {
    "fluency": f"{DATA_ROOT}/pcc_packs/Pron_Fluency_PCC/ckpt_pre/checkpoint_fluency_{c}.pt",
    "comprehensibility": f"{DATA_ROOT}/pcc_packs/Pron_Comprehensibility_PCC/ckpt_pre/checkpoint_comprehensibility_{c}.pt",
    "pronunProf": f"{DATA_ROOT}/pcc_packs/Pron_Proficiency_PCC/ckpt_pre/checkpoint_pronunProf_{c}.pt",
}
GT_KEY = {"fluency": "FluencyEval", "comprehensibility": "ComprehendEval", "pronunProf": "PronunProfEval"}
LABS = {
    "eng": [f"{DATA_ROOT}/aihub/040_labels/extracted/lab"],
    "sjr": [f"{DATA_ROOT}/aihub/042_labels/extracted/lab"],
    "asia": [f"{DATA_ROOT}/aihub/asian_korean_speech/043.교육용_아시아어(중·일어_제외)_사용자의_한국어_음성_데이터/01-1.정식개방데이터/Training/02.라벨링데이터/TL_pronunciation_lab",
             f"{DATA_ROOT}/aihub/asian_korean_speech/043.교육용_아시아어(중·일어_제외)_사용자의_한국어_음성_데이터/01-1.정식개방데이터/Validation/02.라벨링데이터/VL_pronunciation_lab"],
    "eu": [f"{DATA_ROOT}/aihub/european_korean_speech/041.교육용_유럽어_모국어_사용자의_한국어_음성_데이터/01-1.정식개방데이터/Training/02.라벨링데이터/extracted/TL_pronunciation_lab"],
}

print("[1] 라벨 인덱스(코퍼스 라우팅 + 정답 점수)...", flush=True)
route = {}   # stem -> (corpus, lab_path)
for c, dirs in LABS.items():
    for d in dirs:
        for f in os.listdir(d):
            if f.endswith(".json"):
                route.setdefault(f[:-5], (c, os.path.join(d, f)))
print(f"    {len(route):,}", flush=True)

print("[2] test 화자표본...", flush=True)
rel = [l.split("\t") for l in open(f"{M}/test.tsv").read().splitlines()[1:]]
by_spk = collections.defaultdict(list)
for r in rel:
    st = os.path.basename(r[0])[:-4]
    by_spk[st[:5]].append((st, "/" + r[0], int(r[1])))
sample = []
for spk, lst in by_spk.items():
    lst.sort(key=lambda x: x[0])
    sample += lst[:MAX_PER_SPK]
sample = [s for s in sample if s[0] in route]
print(f"    화자 {len(by_spk):,} → 표본 발화 {len(sample):,}", flush=True)

print("[3] 정답 점수 로드...", flush=True)
gt = {}
for st, wav, fr in sample:
    c, lp = route[st]
    try:
        em = json.load(open(lp, encoding="utf-8")).get("EvaluationMetadata", {})
    except Exception:
        continue
    gt[st] = {k: em.get(v) for k, v in GT_KEY.items()}

# 코퍼스별 그룹
groups = collections.defaultdict(list)
for st, wav, fr in sample:
    groups[route[st][0]].append((st, wav, fr))
for c in groups: print(f"    {c}: {len(groups[c]):,}발화", flush=True)

def batched(items, budget=1_600_000):
    order = sorted(range(len(items)), key=lambda i: items[i][2])
    out, cur, mx = [], [], 0
    for i in order:
        f = items[i][2]; m = max(mx, f)
        if cur and m * (len(cur) + 1) > budget:
            out.append(cur); cur = [i]; mx = f
        else:
            cur.append(i); mx = m
    if cur: out.append(cur)
    return out

done_cache = {}
if os.path.exists(PRED_CACHE):
    for line in open(PRED_CACHE, encoding="utf-8"):
        d = json.loads(line); done_cache[(d["dim"], d["stem"])] = d["pred"]
    print(f"[cache] {len(done_cache):,}", flush=True)

fcache = open(PRED_CACHE, "a", encoding="utf-8")
preds = collections.defaultdict(dict)   # dim -> stem -> score
for (dim, st), v in done_cache.items():
    preds[dim][st] = v

for dim, pat in PACKS.items():
    for c, items in groups.items():
        todo = [it for it in items if (dim, it[0]) not in done_cache]
        if not todo:
            continue
        ck = pat.format(c=c)
        print(f"[decode] {dim} × {c} — {len(todo):,}발화 ({os.path.basename(ck)})", flush=True)
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ck])
        model = models[0].eval().cuda().half()
        d = task.target_dictionary
        blank = 0
        norm = cfg.task.normalize
        t0 = time.time(); done = 0
        with torch.inference_mode():
            for idxs in batched(todo):
                wavs = [load_wav(todo[i][1]) for i in idxs]
                lens = [len(w) for w in wavs]; mx = max(lens)
                src = torch.zeros(len(wavs), mx); pm = torch.zeros(len(wavs), mx, dtype=torch.bool)
                for j, w in enumerate(wavs):
                    t = torch.from_numpy(w)
                    if norm: t = F.layer_norm(t, t.shape)
                    src[j, :len(w)] = t; pm[j, len(w):] = True
                net = model(source=src.half().cuda(), padding_mask=pm.cuda())
                lp = model.get_normalized_probs(net, log_probs=True).transpose(0, 1).float().cpu()
                opm = net.get("padding_mask")
                for j, i in enumerate(idxs):
                    x = lp[j]
                    if opm is not None:
                        L = int((~opm[j]).sum()); x = x[:L]
                    toks = x.argmax(-1).unique_consecutive().tolist()
                    digits = [d[t] for t in toks if d[t] in "012345"]
                    score = int(digits[0]) if digits else None
                    st_ = todo[i][0]
                    preds[dim][st_] = score
                    fcache.write(json.dumps({"dim": dim, "stem": st_, "pred": score}) + "\n")
                done += len(idxs)
                if done % 4000 < 40:
                    print(f"    {done:,}/{len(todo):,} ({done/max(1e-9,time.time()-t0):.0f}/s)", flush=True)
        fcache.flush()
        del model; torch.cuda.empty_cache()

fcache.close()

print("[4] PCC 계산...", flush=True)
res = {}
for dim in PACKS:
    ref_u, hyp_u = [], []
    spk_ref = collections.defaultdict(list); spk_hyp = collections.defaultdict(list)
    n_null = 0
    for st, p in preds[dim].items():
        g = gt.get(st, {}).get(dim)
        if g in (None, 0, "0") or p in (None, 0):
            n_null += 1; continue
        g = int(g)
        ref_u.append(g); hyp_u.append(p)
        spk_ref[st[:5]].append(g); spk_hyp[st[:5]].append(p)
    utt = stats.pearsonr(ref_u, hyp_u)[0] if len(ref_u) > 2 else None
    sr = [np.mean(v) for k, v in spk_ref.items()]
    sh = [np.mean(spk_hyp[k]) for k in spk_ref]
    spk = stats.pearsonr(sr, sh)[0] if len(sr) > 2 else None
    res[dim] = {"n_utt": len(ref_u), "n_spk": len(sr), "excluded": n_null,
                "pcc_utt": round(float(utt), 4) if utt is not None else None,
                "pcc_spk": round(float(spk), 4) if spk is not None else None}
    print(f"  {dim:20s} 발화PCC {res[dim]['pcc_utt']} | 화자PCC {res[dim]['pcc_spk']} (utt {len(ref_u):,} / spk {len(sr):,})", flush=True)
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"[saved] {OUT}", flush=True)
