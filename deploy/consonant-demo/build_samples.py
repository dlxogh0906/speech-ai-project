# -*- coding: utf-8 -*-
"""자음 계열별 발음 지적 샘플 데모 — 일부러 틀리게 읽은 합성 녹음으로 '지금 서비스'와 '10-18 수정 전'을 나란히 보여준다.

녹음 = `make_tts_deliberate.py` 시험셋(Qwen3-TTS · Chatterbox, Zeroth test 화자 5명, 낱말 15쌍 × 낱말/문장 × 맞게/틀리게).
채점 = API 채점 경로(`mdd.api_offline` — app.py 와 같은 계산). 섞은 모델 + canonical_fix(choose, margin 1.0).
  지금      말소리 기준 = 기준 문장 음절 수에 맞춤(10-18 18:29~) · 예사→된·거센소리 지적 예외(STRONG_MIN_PROB, 10-18 20:16~)
  수정 전   말소리 기준 1.0초 고정 · 예외 없음(점수 60 미만만 지적)
결과(data.json)와 녹음 링크를 --out 에 쓴다. 페이지는 정적 파일이라 아무 정적 서버로 연다(사내 확인용 — 밖에 열지 않는다).

    PYTHONPATH=/data/speech-ai/tools/hf_py38:$PYTHONPATH $PY deploy/consonant-demo/build_samples.py
    cd $MDD_OUTPUTS/analysis/consonant_demo_2026-10-18/site && python3 -m http.server 8937 --bind 127.0.0.1
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, collections, json, os, shutil, wave

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
SERIES = {"ㄱ": "ㄱ", "ㄲ": "ㄱ", "ㅋ": "ㄱ", "ㄷ": "ㄷ", "ㄸ": "ㄷ", "ㅌ": "ㄷ", "ㅂ": "ㅂ", "ㅃ": "ㅂ", "ㅍ": "ㅂ",
          "ㅅ": "ㅅ", "ㅆ": "ㅅ", "ㅈ": "ㅈ", "ㅉ": "ㅈ", "ㅊ": "ㅈ"}
LENIS = set("ㄱㄷㅂㅅㅈ")
BOOST = {"한국", "공부", "사과", "두부", "친구"}      # 섞은 모델의 합성 학습에 틀린 꼴이 들어간 낱말(시험이 새는 쪽)
SR = 16000


def onset(ch):
    return CHO[(ord(ch) - 0xAC00) // 588]


def token_spans(l, dec):
    """그리디 자모열(utt_features 의 pred 와 같은 순서)마다 프레임 구간 [시작, 끝)."""
    g = l.argmax(-1)
    out, start, prev = [], 0, None
    for t in range(len(g) + 1):
        v = int(g[t]) if t < len(g) else None
        if v != prev:
            if prev is not None and prev != dec.blank:
                tok = dec.d[prev]
                if len(tok) == 1 and 0x3131 <= ord(tok) <= 0x3163:
                    out.append((start, t))
            start, prev = t, v
    return out


def syllables(body):
    out = []
    for w in body["result"]["words"]:
        ph = w["phonemes"]
        st = "e" if any("errorType" in p for p in ph) else ("u" if any(p.get("uncertain") for p in ph) else "")
        out.append({"t": w["word"], "s": st, "h": [p.get("actual_jamo") or "" for p in ph],
                    "a": [p.get("alpha_jamo") or "" for p in ph]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifests", nargs="+", default=[
        os.path.join(str(MDD_OUTPUTS), "analysis", "tts_deliberate_qwen", "manifest.json"),
        os.path.join(str(MDD_OUTPUTS), "analysis", "pair_rescue_2026-10-10", "tts_deliberate", "manifest.json")])
    ap.add_argument("--ckpt", default=os.path.join(str(DATA_ROOT), "checkpoints", "w2vbert_synth_tense_v2", "interp_a0.5_step3000"))
    ap.add_argument("--scorer", default=os.path.join(str(MDD_OUTPUTS), "score_v2_w2vbert", "score_models_v2_w2vbert.joblib"))
    ap.add_argument("--out", default=os.path.join(str(MDD_OUTPUTS), "analysis", "consonant_demo_2026-10-18", "site"))
    a = ap.parse_args()

    import numpy as np
    import mdd.speechsuper as ss
    from mdd.api_offline import ApiEmulator
    from mdd import canonical_fix as cf
    rule_prob = ss.STRONG_MIN_PROB
    e = ApiEmulator(a.ckpt, scorer=a.scorer)
    B = cf.CanonicalBuilder(e.canonical)
    os.makedirs(os.path.join(a.out, "audio"), exist_ok=True)

    items = []
    for m in a.manifests:
        eng = "qwen" if "qwen" in m else "chatterbox"
        for it in json.load(open(m, encoding="utf-8")):
            it.setdefault("engine", eng)
            items.append(it)
    pairs, pair_idx, samples = [], {}, []
    for n, it in enumerate(items):
        good, bad = it["good"], it["bad"]
        k = next(i for i in range(len(good)) if good[i] != bad[i])
        gc, bc = onset(good[k]), onset(bad[k])
        if (good, bad) not in pair_idx:
            pair_idx[(good, bad)] = len(pairs)
            pairs.append({"good": good, "bad": bad, "gc": gc, "bc": bc, "series": SERIES.get(gc, gc), "boost": good in BOOST,
                          "dir": "fixed" if gc in LENIS else "old"})
        with wave.open(it["wav"], "rb") as w:
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
        ref = it["ref_text"]
        rej_now = e.check_audio(x[: SR * 20], ref)[0]
        rej_before = e.check_audio(x[: SR * 20])[0]          # 10-18 전: 1.0초 고정
        lp = e.lprobs(x)
        l = lp[0]
        base, alts = B.variants(ref)
        canon = cf.choose(l, base, alts, e.dec.d.index, blank=e.dec.blank, margin=1.0)[0] if base else None
        ss.STRONG_MIN_PROB = rule_prob
        body_now, raw = e.assess(x, ref, canon=canon, lp=lp, gate=False)
        ss.STRONG_MIN_PROB = 99.0                              # 예외 끔 = 10-18 전 화면
        body_before, _ = e.assess(x, ref, canon=canon, lp=lp, gate=False)
        ss.STRONG_MIN_PROB = rule_prob

        words = body_now["result"]["words"]
        chars = "".join(w["word"] for w in words)
        pos = chars.find(good)
        tgt = None
        if pos >= 0 and pos + k < len(words):
            off = sum(len(w["phonemes"]) for w in words[: pos + k])
            ph = words[pos + k]["phonemes"]
            q = next((j for j, p in enumerate(ph) if p["alpha_jamo"] == gc), None)
            if q is not None:
                qi = off + q
                p_now = ph[q]
                p_bef = [p for w in body_before["result"]["words"] for p in w["phonemes"]][qi]
                site = raw["sites"][qi] if qi < len(raw["sites"]) else {}
                probs = {}
                spans = token_spans(l, e.dec)
                j = site.get("j")
                if j is not None and j < len(spans):
                    s0, s1 = spans[j]
                    for ch in dict.fromkeys([gc, bc, p_now.get("actual_jamo")]):
                        if ch:
                            probs[ch] = round(float(np.exp(l[s0:s1, e.dec.d.index(ch)].max())), 3)
                tgt = {"syl": pos + k, "heard": p_now.get("actual_jamo"), "score": p_now.get("pronunciation"),
                       "now": "e" if "errorType" in p_now else ("u" if p_now.get("uncertain") else ""),
                       "before": "e" if "errorType" in p_bef else ("u" if p_bef.get("uncertain") else ""), "probs": probs}
        name = "%s_%s" % (it["engine"], os.path.basename(it["wav"]))
        dst = os.path.join(a.out, "audio", name)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(it["wav"]), dst)
        samples.append({"pair": pair_idx[(good, bad)], "engine": it["engine"], "voice": it["voice"], "form": it["form"],
                        "kind": it["kind"], "ref": ref, "audio": "audio/" + name, "target": tgt,
                        "now": {"rejected": bool(rej_now), "overall": body_now["result"]["overall"], "syl": syllables(body_now)},
                        "before": {"rejected": bool(rej_before), "overall": body_before["result"]["overall"], "syl": syllables(body_before)}})
        if (n + 1) % 100 == 0:
            print("  %d/%d" % (n + 1, len(items)), flush=True)

    # 낱말 쌍별 요약 — 거절(400)은 화면에 아무것도 안 뜬 것으로 센다
    for pi, p in enumerate(pairs):
        st = collections.Counter()
        for s in samples:
            if s["pair"] != pi or not s["target"]:
                continue
            t = s["target"]
            for when in ("now", "before"):
                shown = (t[when] == "e") and not s[when]["rejected"]
                if s["kind"] == "wrong":
                    st["wrong"] += when == "now"
                    st["wrong_shown_" + when] += shown
                    st["wrong_rej_" + when] += s[when]["rejected"]
                else:
                    st["ok"] += when == "now"
                    st["ok_flag_" + when] += shown
            if s["kind"] == "wrong":
                st["wrong_heard"] += t["heard"] == p["bc"]
        p["stats"] = dict(st)
    meta = {"model": os.path.basename(a.ckpt.rstrip("/")), "rule_prob": rule_prob, "n": len(samples),
            "engines": sorted(set(s["engine"] for s in samples))}
    json.dump({"meta": meta, "pairs": pairs, "samples": samples}, open(os.path.join(a.out, "data.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html"), os.path.join(a.out, "index.html"))
    print("[끝] %s · 녹음 %d · 낱말 쌍 %d" % (a.out, len(samples), len(pairs)))


if __name__ == "__main__":
    main()
