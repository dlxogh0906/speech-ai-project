# -*- coding: utf-8 -*-
"""공식 MDD 체크포인트(한글 자모 출력)를 우리 gold에 채점.

- 공식 모델 = wav2vec_ctc, 출력 = 한글 자모(45), dict 내장(task_state).
- 우리 gold target_phones(KoG2P) → 자모 변환(SYMBOL_TO_JAMO) → 채점 기준.
- 로딩 함정: cfg.task.data / cfg.model.w2v_path 가 그들 경로 → arg_overrides로 우회.
- 스모크(--smoke N): N개만 로드·디코드·자모비교 후 종료.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import DATA_ROOT, MDD_OUTPUTS, MODEL_ROOT

import sys, os, json, wave, argparse, tempfile
import numpy as np
import torch
import torch.nn.functional as F
from fairseq import checkpoint_utils
from mdd.l1prior import s0_symbol_map as sm
from mdd.mdd_eval_pos import _nw, prf_at_threshold

GOLD_SPEC = f"{MDD_OUTPUTS}/gold_eval/gold_eval_spec.jsonl"
GOLD_WAV = f"{DATA_ROOT}/mdd_gold_wav16k"
XLSR = f"{MODEL_ROOT}/xlsr2_300m.pt"


def kog2p_to_jamo(phones):
    """KoG2P 음소열 → 대표 자모열 (1:1). 매핑없으면 원본 유지(드묾)."""
    return [sm.SYMBOL_TO_JAMO.get(p, p) for p in phones]


def load_official(ckpt):
    """공식 체크포인트 로드. dict를 임시폴더에 써서 task.data 우회, w2v_path 로컬 대체."""
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    d = raw["task_state"]["target_dictionary"]
    tmp = tempfile.mkdtemp()
    labels = raw["cfg"]["task"].get("labels", "phn")
    with open(os.path.join(tmp, f"dict.{labels}.txt"), "w", encoding="utf-8") as f:
        for s in d.symbols[d.nspecial:]:        # 특수토큰 4개 제외(fairseq가 자동 재추가)
            f.write(f"{s} 1\n")
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [ckpt], arg_overrides={"data": tmp, "w2v_path": XLSR})
    model = models[0].eval().cuda().half()
    dic = task.target_dictionary
    blank = 0
    print(f"[official] loaded | vocab={len(dic)} blank={blank} normalize={cfg.task.normalize}", flush=True)
    return model, dic, bool(cfg.task.normalize)


def load_wav(path):
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


@torch.inference_mode()
def decode(model, dic, wavs, normalize, blank=0):
    lens = [len(w) for w in wavs]; mx = max(lens)
    src = torch.zeros(len(wavs), mx); pm = torch.zeros(len(wavs), mx, dtype=torch.bool)
    for i, w in enumerate(wavs):
        t = torch.from_numpy(w)
        if normalize:
            t = F.layer_norm(t, t.shape)
        src[i, :len(w)] = t; pm[i, len(w):] = True
    src = src.half().cuda(); pm = pm.cuda()
    net = model(source=src, padding_mask=pm)
    lp = model.get_normalized_probs(net, log_probs=True).transpose(0, 1).float().cpu()
    opm = net.get("padding_mask")
    out = []
    for b in range(len(wavs)):
        x = lp[b]
        if opm is not None:
            x = x[: int((~opm[b]).sum())]
        toks = x.argmax(-1).unique_consecutive().tolist()
        syms = [dic[i] for i in toks if i != blank]
        syms = [s for s in syms if s != "|"]        # 어절경계 제거
        out.append(syms)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{DATA_ROOT}/mdd_official_ckpt/checkpoint_mdd_eu.pt")
    ap.add_argument("--dataset", default="041")     # gold 부분셋
    ap.add_argument("--smoke", type=int, default=0)  # >0이면 N개만 확인 후 종료
    ap.add_argument("--out", default=f"{MDD_OUTPUTS}/eval_official_041.json")
    a = ap.parse_args()

    recs = []
    for line in open(GOLD_SPEC, encoding="utf-8"):
        dd = json.loads(line)
        if dd.get("dataset") != a.dataset:
            continue
        s = dd["stem"]; wp = os.path.join(GOLD_WAV, s[:2], s + ".wav")
        if not os.path.exists(wp):
            continue
        recs.append({"path": wp, "stem": s,
                     "canon_j": kog2p_to_jamo(dd.get("target_phones") or []),
                     "err_pos": [p for p in (dd.get("err_pos") or []) if isinstance(p, int)]})
    print(f"[gold {a.dataset}] {len(recs):,} 발화", flush=True)

    model, dic, norm = load_official(a.ckpt)

    if a.smoke:
        for r in recs[:a.smoke]:
            hyp = decode(model, dic, [load_wav(r["path"])], norm)[0]
            print(f"\nstem={r['stem']}")
            print(f"  canon(자모): {' '.join(r['canon_j'])}")
            print(f"  인식(자모):  {' '.join(hyp)}")
        return

    # 전체 채점
    pos_pred, pos_gold = [], []; n_pos = n_err = 0
    B = 16
    for i in range(0, len(recs), B):
        batch = recs[i:i + B]
        hyps = decode(model, dic, [load_wav(r["path"]) for r in batch], norm)
        for r, hyp in zip(batch, hyps):
            C = r["canon_j"]
            if not C:
                continue
            gold = set(p for p in r["err_pos"] if 0 <= p < len(C))
            pred = {op["tpos"] for op in _nw(C, hyp) if op["kind"] in ("sub", "del")}
            for k in range(len(C)):
                pos_gold.append(1 if k in gold else 0)
                pos_pred.append(1 if k in pred else 0)
            n_pos += len(C); n_err += len(gold)
        if i % (B * 50) == 0:
            print(f"  {i:,}/{len(recs):,}", flush=True)
    m = prf_at_threshold(np.array(pos_pred, float), np.array(pos_gold, float), 0.5)
    res = {"detection": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
           "n_utt": len(recs), "n_positions": n_pos, "n_err_positions": n_err,
           "ckpt": os.path.basename(a.ckpt), "dataset": a.dataset}
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=2)
    d = res["detection"]
    print(f"\n[공식모델 {a.dataset}] F1={d['F1']:.4f} P={d['P']:.4f} R={d['R']:.4f} "
          f"FRR={d['FRR']:.4f} FAR={d['FAR']:.4f}  ({len(recs):,}발화)")
    print(f"[saved] {a.out}")


if __name__ == "__main__":
    main()
