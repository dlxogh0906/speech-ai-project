# -*- coding: utf-8 -*-
"""음소 구간 임베딩 추출 — SupCon 오프라인 탐침의 입력.

질문: 인코더 표현 안에 ㅅ/ㅆ · ㄱ/ㅋ 같은 "세기 차이"를 가를 정보가 **이미 있는데 CTC 헤드가 못 쓰는 것**인가,
아니면 **표현 자체에 없는 것**인가. 앞이면 CTC+SupCon 결합 학습이 의미가 있고, 뒤면 없다.

방법: TAG(실제 발음)를 모델 출력에 CTC 강제정렬(`mdd.gop_feats_v2.viterbi_align`)해 토큰마다 프레임 구간을 얻고,
그 구간(±1 프레임)의 hidden state 평균을 임베딩으로 저장한다. 같은 구간의 CTC 로그확률 평균도 함께 저장한다
(헤드가 이미 내놓는 정보와 비교하기 위해). 대상은 세기 계열 자음 14개(ㄱㄲㅋ ㄷㄸㅌ ㅂㅃㅍ ㅈㅉㅊ ㅅㅆ).
각 토큰에 정렬된 **표준발음 자모**도 붙여, 실제 오류 자리(표준 ≠ 실제)를 따로 볼 수 있게 한다.

    $PY .../extract_phone_embeddings.py --ckpt $SPEECH_AI_DATA/checkpoints/holdout_v1_H0/step_4000 \\
        --manifest $MDD_OUTPUTS/manifest_holdout_v1 --split valid_unseen --out <npz>
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, json, os, random, time, wave
import numpy as np

TARGET = list(u"ㄱㄲㅋㄷㄸㅌㅂㅃㅍㅈㅉㅊㅅㅆ")
VOWELS = set(u"ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")


def read_wav(path, max_sec=20.0):
    with wave.open(path, "rb") as w:
        x = np.frombuffer(w.readframes(min(w.getnframes(), int(max_sec * 16000))), np.int16).astype(np.float32) / 32768.0
    return x


def tag_to_canon(canon, tag):
    """TAG 토큰마다 정렬된 표준발음 자모(삽입이면 None)."""
    import edlib
    c, t = u"".join(canon), u"".join(tag)
    if not c or not t:
        return [None] * len(tag)
    a = edlib.getNiceAlignment(edlib.align(t, c, task="path"), t, c)     # query=TAG, target=canon
    out = []
    for qt, tc in zip(a["query_aligned"], a["target_aligned"]):
        if qt == "-":
            continue
        out.append(None if tc == "-" else tc)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, default=0, help="발화 표본 수(0=전부)")
    ap.add_argument("--layers", nargs="+", type=int, default=[16, 24], help="hidden_states 번호(0=입력 투영, 24=마지막)")
    ap.add_argument("--batch-sec", dest="batch_sec", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from mdd.gop_feats_v2 import HFDecoder, viterbi_align

    lines = open(os.path.join(a.manifest, "%s.tsv" % a.split), encoding="utf-8").read().splitlines()
    root, rows = lines[0].strip(), lines[1:]
    ltr = open(os.path.join(a.manifest, "%s.ltr" % a.split), encoding="utf-8").read().splitlines()
    canp = os.path.join(a.manifest, "%s.canon" % a.split)
    can = open(canp, encoding="utf-8").read().splitlines() if os.path.exists(canp) else None
    src = os.path.join(a.manifest, "%s.source" % a.split)
    srcs = open(src, encoding="utf-8").read().splitlines() if os.path.exists(src) else None
    idx = [i for i in range(len(rows)) if not (srcs and srcs[i] != "foreign_clean" and srcs[i] != "foreign_mis")]
    # 같은 발화가 복제된 행은 한 번만(학습 매니페스트의 오발음 복제 행)
    seen, uniq = set(), []
    for i in idx:
        if rows[i] not in seen:
            seen.add(rows[i]); uniq.append(i)
    rng = random.Random(a.seed)
    if a.n and len(uniq) > a.n:
        uniq = sorted(rng.sample(uniq, a.n))
    if can is None:                       # 학습 매니페스트엔 .canon 이 없다 — 캐시에서 가져온다
        import gzip
        from shared.paths import MDD_OUTPUTS
        want = set(os.path.basename(rows[i].split("\t")[0])[:-4] for i in uniq)
        cmap = {}
        with gzip.open(os.path.join(str(MDD_OUTPUTS), "cache", "foreign_tag_lab_parsed.jsonl.gz"), "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d["stem"] in want:
                    cmap[d["stem"]] = u" ".join(d.get("canon") or [])
        can = [None] * len(rows)
        for i in uniq:
            can[i] = cmap.get(os.path.basename(rows[i].split("\t")[0])[:-4], u"")
    dec = HFDecoder(a.ckpt)
    tok2id = dict((dec.d[k], k) for k in range(len(dec.d)))
    tgt_id = [tok2id[p] for p in TARGET]
    out = {"emb_%d" % L: [] for L in a.layers}
    out.update({"lp": [], "actual": [], "canon": [], "pos": [], "utt": [], "spk": []})
    order = sorted(uniq, key=lambda i: int(rows[i].split("\t")[1]))
    t0, done, fail = time.time(), 0, 0
    batch, cur = [], 0

    def run(batch):
        nonlocal done, fail
        wavs = [read_wav(os.path.join(root, rows[i].split("\t")[0])) for i in batch]
        p = next(dec.model.parameters())
        enc = dec.fe(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.inference_mode():
            o = dec.model(input_features=enc["input_features"].to(p.device, p.dtype),
                          attention_mask=enc["attention_mask"].to(p.device), output_hidden_states=True)
        lp = torch.log_softmax(o.logits.float(), -1).cpu().numpy()
        hs = dict((L, o.hidden_states[L].float().cpu().numpy()) for L in a.layers)
        Ls = enc["attention_mask"].sum(-1).tolist()
        for k, i in enumerate(batch):
            tag = ltr[i].split(); T = int(Ls[k])
            y = [tok2id.get(t) for t in tag]
            if not tag or any(v is None for v in y):
                fail += 1; continue
            segs = viterbi_align(lp[k, :T], y, blank=dec.blank)
            if segs is None:
                fail += 1; continue
            cmap = tag_to_canon(can[i].split(), tag)
            stem = os.path.basename(rows[i].split("\t")[0])[:-4]
            for j, (t, fr) in enumerate(zip(tag, segs)):
                if t not in TARGET or not fr:
                    continue
                lo, hi = max(0, fr[0] - 1), min(T, fr[-1] + 2)
                for L in a.layers:
                    out["emb_%d" % L].append(hs[L][k, lo:hi].mean(0).astype(np.float16))
                out["lp"].append(lp[k, fr[0]:fr[-1] + 1].mean(0)[tgt_id].astype(np.float16))
                c = cmap[j] if j < len(cmap) else None
                out["actual"].append(TARGET.index(t))
                out["canon"].append(TARGET.index(c) if c in TARGET else (-2 if c is None else -1))
                nxt = tag[j + 1] if j + 1 < len(tag) else None
                out["pos"].append(2 if (nxt in VOWELS) else 3)          # 2 초성 · 3 종성
                out["utt"].append(stem); out["spk"].append(stem[:5])
            done += 1

    for i in order:
        fr = int(rows[i].split("\t")[1])
        if batch and cur + fr > a.batch_sec * 16000:
            run(batch); batch, cur = [], 0
        batch.append(i); cur += fr
    if batch:
        run(batch)
    arr = dict((k, np.asarray(v)) for k, v in out.items())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, targets=np.asarray(TARGET), layers=np.asarray(a.layers), ckpt=np.asarray(os.path.abspath(a.ckpt)),
             split=np.asarray(a.split), **arr)
    n_err = int(((arr["canon"] >= 0) & (arr["canon"] != arr["actual"])).sum())
    print("[emb] %s/%s · 발화 %d(실패 %d) · 구간 %d · 오류 자리(표준≠실제, 같은 14자모) %d · %.0fs"
          % (os.path.basename(a.manifest), a.split, done, fail, len(arr["actual"]), n_err, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
