# -*- coding: utf-8 -*-
"""외부 HuggingFace Wav2Vec2ForCTC 모델을 greedy 디코딩해 `eval_actual_ctc.py` 예측 캐시 형식으로 쓴다.

채점은 그대로 `eval_actual_ctc.py score --ckpt <이 스크립트가 쓴 model.safetensors 경로>` 로 한다
(캐시 meta 가 일치하면 디코드를 건너뛰고 채점만 한다).

출력 토큰 정리: 조합형 자모(초성 U+1100 · 중성 U+1161 · 종성 U+11A8)를 호환 자모(U+3131~)로 바꾼다.
그 밖(겹받침 · ㅙ · ㅞ 등)은 우리 모델 출력과 똑같이 두고, `|`·특수 토큰은 버린다.
원 토큰 빈도는 <out>.hf_tokens.json 에 남긴다.

transformers 는 학습 환경에 넣지 않는다(버전 핀) — 별도 폴더를 PYTHONPATH 로 얹어 실행한다:
    PYTHONPATH=/data/speech-ai/tools/hf_py38 $PY .../decode_hf_ctc.py \\
        --model slplab/wav2vec2-XLSR-300m_KoreanPhonene_spoken_by_foreigners \\
        --manifest $MDD_OUTPUTS/manifest_foreign_clean_v2 --split test_unseen --out <prefix>
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, collections, importlib.util, json, os, time, wave

import numpy as np

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JONG = {0x11A8: "ㄱ", 0x11A9: "ㄲ", 0x11AA: "ㄳ", 0x11AB: "ㄴ", 0x11AC: "ㄵ", 0x11AD: "ㄶ", 0x11AE: "ㄷ",
        0x11AF: "ㄹ", 0x11B0: "ㄺ", 0x11B1: "ㄻ", 0x11B2: "ㄼ", 0x11B3: "ㄽ", 0x11B4: "ㄾ", 0x11B5: "ㄿ",
        0x11B6: "ㅀ", 0x11B7: "ㅁ", 0x11B8: "ㅂ", 0x11B9: "ㅄ", 0x11BA: "ㅅ", 0x11BB: "ㅆ", 0x11BC: "ㅇ",
        0x11BD: "ㅈ", 0x11BE: "ㅊ", 0x11BF: "ㅋ", 0x11C0: "ㅌ", 0x11C1: "ㅍ", 0x11C2: "ㅎ"}


def to_compat(tok):
    if len(tok) != 1:
        return None
    o = ord(tok)
    if 0x1100 <= o <= 0x1112:
        return CHO[o - 0x1100]
    if 0x1161 <= o <= 0x1175:
        return chr(0x314F + o - 0x1161)
    if o in JONG:
        return JONG[o]
    if 0x3131 <= o <= 0x3163:
        return tok
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF 저장소 이름 또는 로컬 폴더")
    ap.add_argument("--cache-dir", dest="cache_dir", default=os.path.join(os.environ.get("SPEECH_AI_DATA", "."), "weights", "external"))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True, help="eval_actual_ctc.py score 에 줄 --out 과 같은 접두어")
    ap.add_argument("--batch-samples", dest="batch_samples", type=int, default=1_600_000)
    ap.add_argument("--chunk-sec", dest="chunk_sec", type=float, default=0.0,
                    help="이보다 긴 녹음은 조용한 지점에서 잘라 따로 디코드하고 결과를 이어 붙인다"
                         " (0=끄기). 학습 최대 길이(20초)를 넘는 녹음에서 출력이 줄어드는 것을 피하려면 쓸 것")
    a = ap.parse_args()

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, Wav2Vec2ForCTC

    spec = importlib.util.spec_from_file_location("eval_actual_ctc", str(_pl.Path(__file__).with_name("eval_actual_ctc.py")))
    ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    wavs, _, _, _ = ev.read_split(a.manifest, a.split)

    mdir = a.model if os.path.isdir(a.model) else snapshot_download(
        a.model, cache_dir=a.cache_dir, allow_patterns=["*.json", "*.safetensors"])
    ckpt = os.path.join(mdir, "model.safetensors")
    is_bert = AutoConfig.from_pretrained(mdir).model_type == "wav2vec2-bert"   # w2v-BERT 2.0: 필터뱅크 입력
    if os.path.exists(os.path.join(mdir, "jamo_vocab.json")):                   # train_w2vbert_ctc.py 산출물
        vocab = json.load(open(os.path.join(mdir, "jamo_vocab.json"), encoding="utf-8"))["vocab"]
    else:
        vocab = json.load(open(os.path.join(mdir, "vocab.json"), encoding="utf-8"))
    id2tok = {i: t for t, i in vocab.items()}
    pp = json.load(open(os.path.join(mdir, "preprocessor_config.json")))

    preds_path = a.out + ".preds.jsonl"
    meta = {"ckpt": os.path.abspath(ckpt), "ckpt_size": os.path.getsize(ckpt), "n": len(wavs),
            "first": wavs[0][0], "last": wavs[-1][0], "decoder": "greedy_ctc_no_lm"}
    if os.path.exists(preds_path):
        raise SystemExit(f"이미 있다(덮어쓰지 않음): {preds_path}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() and os.environ.get("MDD_FORCE_CPU") != "1" else "cpu"
    if is_bert:
        from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertForCTC
        fe = SeamlessM4TFeatureExtractor.from_pretrained(mdir)
        model = Wav2Vec2BertForCTC.from_pretrained(mdir).eval().to(dev)
    else:
        model = Wav2Vec2ForCTC.from_pretrained(mdir).eval().to(dev)
    if dev == "cuda":
        model = model.half()
    q = next(model.parameters())
    pad_id = model.config.pad_token_id
    print(f"[model] {mdir} · vocab {len(vocab)} · pad {pad_id} · {dev}", flush=True)

    # 긴 녹음 분할 — 조각을 독립 항목으로 디코드한 뒤 원래 순서로 이어 붙인다.
    # 자르는 곳은 목표 길이 ±1.5초 안에서 **가장 조용한 20ms 지점**이라 음소를 덜 자른다.
    piece_of = list(range(len(wavs)))          # 조각 index → 원래 발화 index
    span = [None] * len(wavs)                  # 조각 index → (시작, 끝) 샘플
    if a.chunk_sec and a.chunk_sec > 0:
        L = int(a.chunk_sec * 16000)
        win = int(1.5 * 16000)
        new_wavs, piece_of, span = [], [], []
        for i, (path, fr) in enumerate(wavs):
            if fr <= L:
                new_wavs.append((path, fr)); piece_of.append(i); span.append(None); continue
            with wave.open(path, "rb") as w:
                x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
            cuts, pos = [0], 0
            while len(x) - pos > L:
                tgt = pos + L
                lo, hi = max(pos + L // 2, tgt - win), min(len(x) - 1, tgt + win)
                seg = x[lo:hi]
                if len(seg) < 320:
                    cut = tgt
                else:
                    k = 320                                   # 20ms 창의 RMS 최소점
                    rms = np.sqrt(np.convolve(seg ** 2, np.ones(k) / k, mode="valid"))
                    cut = lo + int(np.argmin(rms)) + k // 2
                cuts.append(cut); pos = cut
            cuts.append(len(x))
            keep = [cuts[0]]                          # 0.3초보다 짧은 조각은 앞 조각에 붙인다
            for c in cuts[1:]:                        # (필터뱅크가 만들어지지 않아 죽는다)
                if c - keep[-1] < 4800 and len(keep) > 1:
                    keep[-1] = c
                else:
                    keep.append(c)
            if len(keep) > 1 and keep[-1] - keep[-2] < 4800:
                keep.pop(-2)
            cuts = keep
            for k in range(len(cuts) - 1):
                new_wavs.append((path, cuts[k + 1] - cuts[k]))
                piece_of.append(i); span.append((cuts[k], cuts[k + 1]))
        print(f"[chunk] {len(wavs):,} 발화 → {len(new_wavs):,} 조각 (최대 {a.chunk_sec:.0f}초)", flush=True)
        wavs = new_wavs

    order = sorted(range(len(wavs)), key=lambda i: wavs[i][1])
    batches, cur, b = [], 0, []
    for i in order:
        if b and cur + wavs[i][1] > a.batch_samples:
            batches.append(b); b, cur = [], 0
        b.append(i); cur += wavs[i][1]
    if b:
        batches.append(b)

    P, raw = [None] * len(wavs), collections.Counter()
    t0, done = time.time(), 0
    with torch.inference_mode():
        for bi, idx in enumerate(batches):
            xs = []
            for i in idx:
                with wave.open(wavs[i][0], "rb") as w:
                    assert w.getframerate() == 16000 and w.getnchannels() == 1
                    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
                if span[i] is not None:
                    x = x[span[i][0]:span[i][1]]
                if not is_bert and pp.get("do_normalize", True):
                    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)       # Wav2Vec2FeatureExtractor 와 같은 식
                xs.append(x)
            if is_bert:
                enc = fe(xs, sampling_rate=16000, return_tensors="pt", padding=True)
                logits = model(input_features=enc["input_features"].to(dev, q.dtype),
                               attention_mask=enc["attention_mask"].to(dev)).logits.float()
                olen = enc["attention_mask"].sum(-1).tolist()          # 어댑터 없음 → 입력 프레임 = 출력 프레임
            else:
                mx = max(len(x) for x in xs)
                src = torch.zeros(len(xs), mx); am = torch.zeros(len(xs), mx, dtype=torch.long)
                for k, x in enumerate(xs):
                    src[k, :len(x)] = torch.from_numpy(x); am[k, :len(x)] = 1
                logits = model(src.to(dev, q.dtype), attention_mask=am.to(dev)).logits.float()
                olen = model._get_feat_extract_output_lengths(am.sum(-1)).tolist()
            for k, i in enumerate(idx):
                toks = logits[k, : int(olen[k])].argmax(-1).unique_consecutive().tolist()
                out = []
                for t in toks:
                    if t == pad_id or (is_bert and t == 0):
                        continue
                    s = id2tok.get(t, "")
                    raw[s] += 1
                    c = to_compat(s)
                    if c:
                        out.append(c)
                P[i] = out
            done += len(idx)
            if bi % 50 == 0:
                print(f"  decode {done:,}/{len(wavs):,} ({done / max(1e-9, time.time() - t0):.0f}/s)", flush=True)

    if a.chunk_sec and a.chunk_sec > 0:                 # 조각 → 원래 발화로 합치기
        merged = collections.defaultdict(list)
        for k, out in enumerate(P):
            merged[piece_of[k]] += (out or [])
        P = [merged.get(i, []) for i in range(max(merged) + 1)]

    tmp = preds_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for i, p in enumerate(P):
            f.write(json.dumps({"i": i, "pred": p}, ensure_ascii=False) + "\n")
    os.replace(tmp, preds_path)
    json.dump({**meta, "sec": round(time.time() - t0, 1), "hf_model": a.model}, open(preds_path + ".meta.json", "w"),
              ensure_ascii=False, indent=1)
    blocks = collections.Counter()
    for s, n in raw.items():
        o = ord(s[0]) if len(s) == 1 else 0
        blocks["compat" if 0x3131 <= o <= 0x3163 else "cho" if 0x1100 <= o <= 0x1112 else
               "jung" if 0x1161 <= o <= 0x1175 else "jong" if 0x11A8 <= o <= 0x11C2 else s] += n
    json.dump({"blocks": blocks, "tokens": raw.most_common()}, open(a.out + ".hf_tokens.json", "w"), ensure_ascii=False, indent=1)
    print(f"[preds] {len(wavs):,} 발화 {time.time() - t0:.0f}s → {preds_path} · 토큰 구간 {dict(blocks)}", flush=True)


if __name__ == "__main__":
    main()
