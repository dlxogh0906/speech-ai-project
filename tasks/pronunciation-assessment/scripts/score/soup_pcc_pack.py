# -*- coding: utf-8 -*-
"""언어권별 점수 모델 4개를 가중치 평균(model soup)으로 합쳐 평가.

동기 두 가지:
 1) 실용 — 합쳐지면 L1 라우팅 없이 모델 1개로 전 언어권을 처리(3.5G, 14G→3.5G).
 2) 진단 — 각 모델이 '자기 화자'를 외운 것이라면, 평균을 내는 순간 그 기억은 서로 상쇄돼
    대각선(자기 언어권) 성능이 비대각 수준으로 내려앉는다. 반대로 진짜 발음 품질을 듣는
    능력이라면 4개가 같은 방향을 가리키므로 살아남고, 오히려 일반화가 좋아진다.

수프는 같은 초기값에서 갈라진 모델들에만 성립한다 → w2v_path/구조 동일성을 먼저 확인한다.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import DATA_ROOT, MDD_OUTPUTS, MDD_TASK

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
ap.add_argument("--out", default=f"{MDD_OUTPUTS}/eval_pcc_soup.json")
A = ap.parse_args()

WAV = f"{DATA_ROOT}/mdd_gold_wav16k"
SCRATCH = f"{MDD_TASK}/outputs"
DICT = f"{SCRATCH}/pcc_dict"
PACKS = f"{DATA_ROOT}/pcc_packs"
LANGS = ["eng", "eu", "sjr", "asia"]
MAX_SAMPLES = 1_600_000
CK = {lg: f"{PACKS}/{A.pack}/ckpt_pre/checkpoint_{A.task}_{lg}.pt" for lg in LANGS}

AA = f"{DATA_ROOT}/aihub/asian_korean_speech/043.교육용_아시아어(중·일어_제외)_사용자의_한국어_음성_데이터/01-1.정식개방데이터"
EE = f"{DATA_ROOT}/aihub/european_korean_speech/041.교육용_유럽어_모국어_사용자의_한국어_음성_데이터/01-1.정식개방데이터/Training/02.라벨링데이터/extracted"
CORPUS_DIRS = [("eng", f"{DATA_ROOT}/aihub/040_labels/extracted/lab"),
               ("eu", f"{EE}/TL_pronunciation_lab"),
               ("sjr", f"{DATA_ROOT}/aihub/042_labels/extracted/lab"),
               ("asia", f"{AA}/Training/02.라벨링데이터/TL_pronunciation_lab"),
               ("asia", f"{AA}/Validation/02.라벨링데이터/VL_pronunciation_lab")]


def make_soup(dst):
    """4개 체크포인트의 model 텐서를 균등 평균. 정수형(카운터 등)은 첫 모델 값을 그대로 쓴다."""
    acc, base, meta = None, None, {}
    for i, lg in enumerate(LANGS):
        c = torch.load(CK[lg], map_location="cpu")
        sd = c["model"]
        w2v = c["cfg"]["model"].get("w2v_path", "?")
        meta[lg] = {"w2v_path": str(w2v).split("/")[-1], "n_param": len(sd),
                    "proj": tuple(sd["w2v_encoder.proj.weight"].shape),
                    "updates": c.get("optimizer_history", [{}])[-1].get("num_updates")}
        if acc is None:
            base = c
            acc = {k: (v.float().clone() if v.is_floating_point() else v.clone()) for k, v in sd.items()}
        else:
            assert set(acc) == set(sd), f"{lg}: 파라미터 이름 불일치"
            for k, v in sd.items():
                assert acc[k].shape == v.shape, f"{lg}:{k} 모양 불일치 {acc[k].shape} vs {v.shape}"
                if v.is_floating_point(): acc[k] += v.float()
        del c
    for k in acc:
        if acc[k].is_floating_point(): acc[k] /= len(LANGS)
    print("[구성 확인]")
    for lg in LANGS:
        m = meta[lg]
        print(f"  {lg:5} base={m['w2v_path']:16} 파라미터 {m['n_param']:,} "
              f"출력층 {m['proj']} 학습스텝 {m['updates']}")
    bases = {meta[lg]["w2v_path"] for lg in LANGS}
    print(f"  → 같은 초기값에서 갈라짐: {'예' if len(bases) == 1 else '아니오 ' + str(bases)} "
          f"(수프 성립 조건)\n")
    base["model"] = {k: (v.half() if v.is_floating_point() else v) for k, v in acc.items()}
    base.pop("last_optimizer_state", None)
    torch.save(base, dst)
    return meta


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


def pcc(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 0 else float("nan")


def score(pred, rs):
    ok = [r for r in rs if pred.get(r["stem"]) is not None]
    u = pcc([pred[r["stem"]] for r in ok], [r[A.field] for r in ok])
    ag, al = collections.defaultdict(list), collections.defaultdict(list)
    for r in ok:
        ag[r["stem"][:5]].append(pred[r["stem"]]); al[r["stem"][:5]].append(r[A.field])
    ks = [s for s in ag if len(ag[s]) >= 3]
    return round(u, 3), round(pcc([np.mean(ag[s]) for s in ks], [np.mean(al[s]) for s in ks]), 3)


SOUP = f"{SCRATCH}/soup_{A.task}.pt"
meta = make_soup(SOUP)

langmap = {}
for lang, dr in CORPUS_DIRS:
    if os.path.isdir(dr):
        for fn in os.listdir(dr):
            if fn.endswith(".json"): langmap.setdefault(fn[:-5], lang)

rows = [json.loads(l) for l in open(A.feats, encoding="utf-8")]
rng = np.random.default_rng(0)                     # 교차언어 실험과 동일 표본
groups = collections.defaultdict(list)
for r in rows:
    if r.get(A.field, 0) and r[A.field] > 0 and r["stem"] in langmap:
        groups[langmap[r["stem"]]].append(r)
sample = {lg: [groups[lg][i] for i in rng.permutation(len(groups[lg]))[: A.per_lang]] for lg in LANGS}

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
print(f"[표본] 언어권당 {A.per_lang:,} · 오디오 {len(order):,}개\n")

t0 = time.time()
soup_pred = decode_scores(SOUP, waves, order)
print(f"[수프 디코드] {time.time()-t0:.0f}s")
own = {}
for lg in LANGS:                                   # 각 언어권을 '전용 모델'로 채점 = 대각선 기준선
    stems = sorted((r["stem"] for r in sample[lg] if r["stem"] in waves), key=lambda s: len(waves[s]))
    own[lg] = decode_scores(CK[lg], waves, stems)

print(f"\n{'데이터':8}{'전용모델(대각)':>16}{'수프':>10}{'차이':>9}   [화자 단위 PCC]")
res = {}
for lg in LANGS:
    _, s_own = score(own[lg], sample[lg])
    _, s_soup = score(soup_pred, sample[lg])
    res[lg] = {"own_spk": s_own, "soup_spk": s_soup}
    print(f"{lg:8}{s_own:>16}{s_soup:>10}{s_soup - s_own:>+9.3f}")
allr = [r for lg in LANGS for r in sample[lg]]
u_all, s_all = score(soup_pred, allr)
print(f"\n[수프 전체 통합 채점] 발화단위 {u_all}  화자단위 {s_all}  (n={len(allr):,})")
res["_all"] = {"soup_utt": u_all, "soup_spk": s_all}
json.dump({"task": A.task, "meta": {k: str(v) for k, v in meta.items()}, "res": res},
          open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"[saved] {A.out}")
