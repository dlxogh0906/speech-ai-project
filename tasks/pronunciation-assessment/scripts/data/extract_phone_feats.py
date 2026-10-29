# -*- coding: utf-8 -*-
"""음소 단위 특징 추출 — "문장에 구애받지 않는 발음 진단기"(phone detector)의 입력을 만든다.

배경: 받아쓴 뒤 표준발음과 비교하는 방식은 안 본 문장에서 정밀도가 0.18 까지 떨어진다(results/13·14).
여기서는 받아쓰지 않는다. **표준발음(canonical)을 음성에 강제정렬**하고, 자리마다 음향 증거를 뽑아
"이 자리가 오발음인가"를 따로 학습한 판별기가 본다. 판별기 입력은 음소 단위 증거뿐이라 문장을 외울 수 없다.

음향 모델은 **원어민만으로 학습한 Stage 1**(KsponSpeech)을 기본으로 쓴다 — 외국인 문항을 한 번도 본 적이 없어
학습 문항과 홀드아웃 문항에서 특징 분포가 같다(암기한 모델을 쓰면 학습 쪽 특징만 깨끗해져 임계값이 안 옮겨진다).

자리(표준발음 토큰) j 마다
  lpp_spike[50]  정렬이 그 토큰에 준 프레임(스파이크)의 평균 로그사후확률
  lpp_span[50]   이웃 토큰 중심과의 중간점까지 넓힌 구간의 평균 로그사후확률
  scal[7]        표준음 로그확률 · 최강 경쟁음과의 차(GOP) · 구간 내 표준음 최댓값 · blank 평균 · 길이 2종 · 엔트로피
  emb_last / emb_mid[C]   구간 평균 은닉 벡터(마지막 층 · --mid_layer 층), fp16
라벨은 채점기(eval_actual_ctc.project_on_canon)와 **같은 정렬**로 만든다: y=1 ⇔ TAG[j] != LAB[j] (치환·삭제형).

출력 <out>/<split>/: scal.npy lpp_spike.npy lpp_span.npy emb_last.npy emb_mid.npy canon_id.npy real_id.npy y.npy
                     offsets.npy(발화별 토큰 시작) ok.npy(정렬 성공) utt.jsonl(발화 메타) meta.json

    $PY tasks/pronunciation-assessment/scripts/data/extract_phone_feats.py --manifest $MDD_OUTPUTS/manifest_foreign_clean_v2 \
        --split test_unseen --ckpt $SPEECH_AI_DATA/checkpoints/final_stage1_300m_40k/checkpoint_best.pt --out <dir>
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import REPO_ROOT
from mdd import phone_feats as PF

import argparse, importlib.util, json, os, time, wave

import numpy as np

N_SCAL = PF.N_SCAL


def load_evalmod():
    """채점기와 같은 입력 읽기·정렬을 쓰려고 스크립트를 모듈로 올린다(복제하지 않는다)."""
    p = REPO_ROOT / "tasks" / "pronunciation-assessment" / "scripts" / "eval" / "eval_actual_ctc.py"
    spec = importlib.util.spec_from_file_location("eval_actual_ctc", str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mid_layer", type=int, default=16, help="중간 층 번호(1-based). 0 이면 저장하지 않는다")
    ap.add_argument("--max_utts", type=int, default=0, help="앞에서부터 N 발화만(0=전부) — 문항 균형은 --per_item 으로")
    ap.add_argument("--per_item", type=int, default=0, help="문항(item)당 최대 발화 수(0=제한 없음), 매니페스트 순서대로")
    ap.add_argument("--batch_samples", type=int, default=2400000)
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    import torchaudio
    E = load_evalmod()
    wavs, ltr, canon, meta = E.read_split(a.manifest, a.split)

    keep = list(range(len(wavs)))
    if a.per_item:
        seen, k2 = {}, []
        for i in keep:
            it = meta[i].get("item", "?")
            seen[it] = seen.get(it, 0) + 1
            if seen[it] <= a.per_item:
                k2.append(i)
        keep = k2
    if a.max_utts:
        keep = keep[:a.max_utts]

    model, d, blank, normalize = E.load_model(a.ckpt, a.manifest)
    C = model.w2v_encoder.proj.weight.shape[1]
    V = len(d)
    n_tok = sum(len(canon[i]) for i in keep)
    od = os.path.join(a.out, a.split)
    os.makedirs(od, exist_ok=True)
    mm = lambda name, shape, dt: np.lib.format.open_memmap(os.path.join(od, name), mode="w+", dtype=dt, shape=shape)
    scal = mm("scal.npy", (n_tok, N_SCAL), np.float32)
    lps = mm("lpp_spike.npy", (n_tok, V), np.float16)
    lpn = mm("lpp_span.npy", (n_tok, V), np.float16)
    el = mm("emb_last.npy", (n_tok, C), np.float16)
    em = mm("emb_mid.npy", (n_tok, C), np.float16) if a.mid_layer else None
    cid = mm("canon_id.npy", (n_tok,), np.int16)
    rid = mm("real_id.npy", (n_tok,), np.int16)
    yy = mm("y.npy", (n_tok,), np.uint8)
    spn = mm("span.npy", (n_tok, 2), np.int32)          # 자리별 [시작, 끝) 프레임(20 ms) — 구간 재생·시각화용
    offsets = np.zeros(len(keep) + 1, np.int64)
    ok = np.zeros(len(keep), bool)
    for k, i in enumerate(keep):
        offsets[k + 1] = offsets[k] + len(canon[i])

    order = sorted(range(len(keep)), key=lambda k: wavs[keep[k]][1])
    batches, cur, b = [], 0, []
    for k in order:
        n = wavs[keep[k]][1]
        if b and cur + n > a.batch_samples:
            batches.append(b); b, cur = [], 0
        b.append(k); cur += n
    if b:
        batches.append(b)
    print(f"[data] {a.split}: {len(keep):,}발화 · 토큰 {n_tok:,} · 배치 {len(batches):,} · 은닉 {C} · mid_layer {a.mid_layer}", flush=True)

    t0, done, n_fail = time.time(), 0, 0
    with torch.inference_mode():
        for bi, idx in enumerate(batches):
            xs = []
            for k in idx:
                with wave.open(wavs[keep[k]][0], "rb") as w:
                    assert w.getframerate() == 16000 and w.getnchannels() == 1
                    xs.append(np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0)
            lp_all, h, hm, Ts = PF.forward_anchor(model, xs, normalize, a.mid_layer, torch, F)
            for j, k in enumerate(idx):
                i = keep[k]
                T = Ts[j]
                y = [d.index(s) for s in canon[i]]
                s0, L = int(offsets[k]), len(y)
                man = E.project_on_canon(canon[i], ltr[i])
                cid[s0:s0 + L] = y
                rid[s0:s0 + L] = [-1 if m is None else d.index(m) for m in man]
                yy[s0:s0 + L] = [int(m != c) for m, c in zip(man, canon[i])]
                o = PF.align_featurize(lp_all[j, :T], h[j, :T], hm[j, :T] if hm is not None else None, y, blank, torch, torchaudio)
                if o is None:
                    n_fail += 1
                    continue
                scal[s0:s0 + L] = o["scal"].cpu().numpy()
                lps[s0:s0 + L] = o["lpp_spike"].cpu().numpy().astype(np.float16)
                lpn[s0:s0 + L] = o["lpp_span"].cpu().numpy().astype(np.float16)
                el[s0:s0 + L] = o["emb_last"].cpu().numpy().astype(np.float16)
                if em is not None:
                    em[s0:s0 + L] = o["emb_mid"].cpu().numpy().astype(np.float16)
                spn[s0:s0 + L] = o["span"].cpu().numpy()
                ok[k] = True
            done += len(idx)
            if bi % 100 == 0:
                print(f"  {done:,}/{len(keep):,} ({done / max(1e-9, time.time() - t0):.0f}/s) 정렬실패 {n_fail}", flush=True)

    for arr in (scal, lps, lpn, el, cid, rid, yy, spn) + ((em,) if em is not None else ()):
        arr.flush()
    np.save(os.path.join(od, "offsets.npy"), offsets)
    np.save(os.path.join(od, "ok.npy"), ok)
    with open(os.path.join(od, "utt.jsonl"), "w", encoding="utf-8") as f:
        for i in keep:
            m = meta[i]
            f.write(json.dumps({"i": i, **{x: m.get(x) for x in ("stem", "spk", "ds", "task", "item", "language", "nationality",
                                                                  "proficiency", "scores", "sec")}}, ensure_ascii=False) + "\n")
    info = {"manifest": os.path.abspath(a.manifest), "split": a.split, "ckpt": os.path.abspath(a.ckpt), "n_utts": len(keep),
            "n_tokens": int(n_tok), "align_fail": n_fail, "err_rate": float(np.asarray(yy).mean()), "hidden": C, "vocab": V,
            "blank": int(blank), "mid_layer": a.mid_layer, "symbols": [d[i] for i in range(V)], "sec": round(time.time() - t0, 1)}
    json.dump(info, open(os.path.join(od, "meta.json"), "w"), ensure_ascii=False, indent=1)
    print(f"[완료] {od} · {len(keep):,}발화 · 토큰 {n_tok:,} · 오발음 자리 {info['err_rate']:.2%} · 정렬실패 {n_fail} · {info['sec']}s", flush=True)


if __name__ == "__main__":
    main()
