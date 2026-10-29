# -*- coding: utf-8 -*-
"""서비스 점검 ② — 실제 녹음 조건(폰·방·카페·배경음악)에서도 (가) 같은 억양을 같다고 하나 (나) 넣은 어긋남을 잡나.

원천: Zeroth test 원어민 낭독(공개) + 원작 대사 원본(목소리 분리본 — 내부 시험용, 저장소 밖).
바탕 2종: same(PSOLA 재합성 — 같은 사람이 똑같이) · gender(목소리를 바꿈 — 다른 사람이 억양만 똑같이 따라 한 것에 가깝다).
채널: none · opus16/32 · aac32 · reverb0.3/0.7(RT60 초) · babble20/10(다른 화자 4명 웅성거림, dB) ·
      music10(배경음악 10 dB 그대로) · music10_sep/music5_sep(배경음악을 섞었다 목소리 분리 — 원작 원본을 만들 때와 같은 처리) ·
      clip12(입력 과다로 찌그러짐) · narrowband(블루투스·전화 대역) · agc(자동 음량) ·
      phone(잔향 0.3 + 웅성 20 dB + AGC + opus 24k) · phone_bad(잔향 0.7 + 웅성 10 dB + AGC + opus 16k)
배경음악은 원작 장면에서 목소리를 뺀 나머지(배경음·효과음)다. 웅성거림은 Zeroth train 화자(원천과 다른 사람)를 겹친 것.
(가) 방해: 지적이 없어야 한다 → 잘못 지적된 음절 비율, 재녹음 판정, 항목별 측정 오차(p90).
(나) 주입: 음절 하나에 음높이 ±4 · 억양 뒤집기 · 길이 ×1.8/×0.5 · 앞 쉼 0.4초 · 세기 −10 dB 를 넣고 채널을 씌운다
     → 그 음절을 그 종류·방향으로 짚는가(type_hit), 첫 지적인가. 원작은 넣을 만한 음절마다, Zeroth 는 종류마다 하나.
변형마다 음절 특징을 rows.jsonl 에 남기고 validate_synthetic.score 로 채점한다(`--rescore` 는 디코드 없이 다시 센다).

    PYTHONPATH=/data/speech-ai/tools/hf_py38:/data/speech-ai/tools/mimic_py38 HF_HUB_OFFLINE=1 TORCH_HOME=<torch 캐시> \\
        $PY tasks/mimicry-coach/scripts/eval/audit_channel.py --clips /data/speech-ai/out/mimic/original_clips_v1 \\
        --zeroth 30 --out <출력 폴더>
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))      # 같은 폴더의 validate_synthetic

import argparse, collections, glob, json, math, os, random, re, time

import numpy as np

CHANNELS = ["none", "opus16", "opus32", "aac32", "reverb0.3", "reverb0.7", "babble20", "babble10", "music10",
            "music10_sep", "music5_sep", "clip12", "narrowband", "agc", "phone", "phone_bad"]
DETECT = ["none", "phone", "phone_bad", "music10"]
BASES = ["same", "gender"]
CHECKS = ["pitch_level", "pitch_dir", "duration", "pause", "energy"]


class Ctx:
    """채널에 섞을 소리들 — 웅성거림 원천 · 배경음 · 목소리 분리기(필요할 때 띄운다)."""

    def __init__(self, babble_wavs, bgm_wavs):
        self.babble_wavs, self.bgm_wavs, self._sep = babble_wavs, bgm_wavs, None

    def babble(self, n, rng):
        from mimicry import synth
        k = rng.choice(len(self.babble_wavs), size=4, replace=False)
        b = [synth.loop_to(self.babble_wavs[i], n, rng) for i in k]
        return np.sum([v / (np.sqrt(np.mean(v ** 2)) + 1e-9) for v in b], axis=0).astype(np.float32)

    def bgm(self, n, rng):
        from mimicry import synth
        w, hot = self.bgm_wavs[int(rng.integers(len(self.bgm_wavs)))]
        s = int(rng.choice(hot)) if len(hot) else 0                 # 배경음이 큰 1초 칸에서 시작
        return synth.loop_to(w[s:], n, rng) if len(w) - s >= n else synth.loop_to(w, n, rng)

    def separate(self, y):
        import torch
        import torchaudio
        from mimicry.separate import SR_SEP, Separator
        if self._sep is None:
            self._sep = Separator()
        up = torchaudio.functional.resample(torch.from_numpy(np.asarray(y, dtype=np.float32)), 16000, SR_SEP).numpy()
        v, _ = self._sep.split(np.stack([up, up]))
        v = v[:len(y)]
        return np.pad(v, (0, len(y) - len(v))).astype(np.float32)


def apply_channel(x, name, rng, ctx):
    from mimicry import synth
    if name == "none":
        return x
    if name.startswith("opus"):
        return synth.codec(x, "opus", int(name[4:]))
    if name.startswith("aac"):
        return synth.codec(x, "aac", int(name[3:]))
    if name.startswith("reverb"):
        rt = float(name[6:])
        return synth.reverb(x, rt, rng, drr_db=6.0 if rt < 0.5 else 0.0)
    if name.startswith("babble"):
        return synth.mix_at(x, ctx.babble(len(x), rng), float(name[6:]))
    if name.startswith("music"):
        y = synth.mix_at(x, ctx.bgm(len(x), rng), float(name[5:].split("_")[0]))
        return ctx.separate(y) if name.endswith("_sep") else y
    if name.startswith("clip"):
        return synth.clip(x, float(name[4:]))
    if name == "narrowband":
        return synth.narrowband(x)
    if name == "agc":
        return synth.agc(x)
    if name in ("phone", "phone_bad"):
        rt, drr, snr, kb = (0.3, 8.0, 20.0, 24) if name == "phone" else (0.7, 2.0, 10.0, 16)
        y = synth.reverb(x, rt, rng, drr_db=drr)
        y = synth.mix_at(y, ctx.babble(len(y), rng), snr)
        return synth.codec(synth.agc(y), "opus", kb)
    raise KeyError(name)


def load_bgm(clip_dirs):
    """원작 장면(_cache 의 mix·vocals)에서 목소리를 뺀 나머지 → [(파형, 배경음이 큰 1초 칸 시작 표본들)]."""
    from mimicry.audio import load_audio
    out = []
    for d in clip_dirs:
        for pm in sorted(glob.glob(os.path.join(d, "_cache", "*_mix.wav"))):
            m, v = load_audio(pm), load_audio(pm.replace("_mix.wav", "_vocals.wav"))
            n = min(len(m), len(v))
            rest = (m[:n] - v[:n]).astype(np.float32)
            if n < 16000 * 4 or float(np.sqrt(np.mean(rest ** 2))) < 0.01:
                continue                                   # 배경음이 거의 없는 장면은 뺀다
            sec = rest[:n // 16000 * 16000].reshape(-1, 16000)
            r = np.sqrt(np.mean(sec ** 2, axis=1))
            out.append((rest, np.flatnonzero(r >= np.median(r)) * 16000))
    return out


def originals(clip_dirs):
    from mimicry.audio import load_audio
    out = []
    for d in clip_dirs:
        meta = json.load(open(os.path.join(d, "clips.json"), encoding="utf-8"))
        for c in meta["clips"]:
            if (c.get("qc") or {}).get("ok", True):
                out.append(("o:" + c["id"], load_audio(os.path.join(d, c["files"]["ref"])), c["line"]))
    return out


def run(a):
    from mimicry import synth
    from mimicry.align import FRAME, OFFSET
    from mimicry.analyze import crosscheck
    from mimicry.audio import load_audio
    from mimicry.pipeline import Coach
    from validate_synthetic import INJECT, eligible, pack, zeroth_sources

    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    nrng = np.random.default_rng(a.seed)
    srcs = [("z%02d" % k, load_audio(p), t) for k, (p, t) in enumerate(zeroth_sources("test_data_01", a.zeroth, a.seed, 8, 32))]
    srcs = [s for s in srcs if len(s[1]) / 16000.0 <= 9.0] + originals(a.clips)
    pool = zeroth_sources("train_data_01", a.babble_pool, a.seed + 1, 8, 32)
    ctx = Ctx([load_audio(p) for p, _ in pool], load_bgm(a.clips))
    print("원천 %d · 웅성 원천 %d · 배경음 장면 %d" % (len(srcs), len(ctx.babble_wavs), len(ctx.bgm_wavs)), flush=True)
    usable = lambda chs: [c for c in chs if ctx.bgm_wavs or not c.startswith("music")]   # 장면이 없으면 배경음 채널은 뺀다
    channels, detect = usable(CHANNELS), usable(DETECT)
    coach = Coach()
    fo = open(os.path.join(a.out, "rows.jsonl"), "w", encoding="utf-8")
    t_start = time.time()

    def evaluate(sid, x, A, FA, y, text, meta):
        try:
            B, FB = coach.analyze(y, text)
            crosscheck(x, A, y, B, feats=(FA, FB), hop=FRAME, off=OFFSET)
            row = dict(meta, src=sid, syl=pack(B["syllables"]), bound_agree=B.get("bound_agree"),
                       line_jer=B.get("line_jer"), line_heard=B.get("line_heard"))
        except (ValueError, RuntimeError) as e:            # 정렬 불가 등 — 서비스에선 오류 응답
            row = dict(meta, src=sid, error=str(e)[:200])
        fo.write(json.dumps(row, ensure_ascii=False) + "\n")

    for si, (sid, x, text) in enumerate(srcs):
        A, FA = coach.analyze(x, text)
        high = A["baseline_hz"] >= 165.0
        pr = A["pitch_range"]
        wide = (max(50.0, pr[0] * 0.8), min(1100.0, pr[1] * 1.25))
        fo.write(json.dumps(dict(variant="ref", src=sid, text=text, n=len(A["syllables"]), syl=pack(A["syllables"]),
                                 set="original" if sid.startswith("o:") else "zeroth", line_jer=A.get("line_jer")),
                            ensure_ascii=False) + "\n")
        y0 = synth.psola(x, prange=wide)
        base_sig = {"same": y0, "gender": synth.change_gender(x, wide, high)}
        for base in BASES:                                  # (가) 방해
            for ch in channels:
                y = apply_channel(base_sig[base], ch, nrng, ctx)
                evaluate(sid, x, A, FA, y, text, dict(variant="nuisance", kind=ch, base=base, channel=ch))
        for kind in INJECT:                                 # (나) 주입
            ks = eligible(A, kind)
            if not ks:
                continue
            ks = ks if sid.startswith("o:") else [rng.choice(ks)]
            for k in ks:
                src_sig = y0 if kind in ("pause_add", "weaker") else x
                yi, _, truth = synth.inject(src_sig, kind, A["syllables"][k], wide, ref_delta=A["syllables"][k]["delta"])
                for base in BASES:
                    yb = synth.change_gender(yi, wide, high) if base == "gender" else yi
                    for ch in detect:
                        y = apply_channel(yb, ch, nrng, ctx)
                        evaluate(sid, x, A, FA, y, text, dict(variant="inject", kind=kind, base=base, channel=ch, k=k,
                                                              truth=truth))
        fo.flush()
        print("[%d/%d] %s %d음절 · %.0fs" % (si + 1, len(srcs), sid, len(A["syllables"]), time.time() - t_start), flush=True)
    fo.close()


def meas_err(rows, refs):
    """방해 변형에서 항목별 |원본 − 녹음| 차의 p90 — 같은 억양인데 얼마나 다르게 재나(확신 없음 음절은 뺀다)."""
    from mimicry.compare import compare
    from validate_synthetic import unpack
    errs = collections.defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        ref = refs[r["src"]]
        txt = re.sub(r"[^가-힣]", "", ref["text"])
        take = dict(syllables=unpack(r["syl"], txt), bound_agree=r.get("bound_agree"), line_jer=r.get("line_jer"),
                    line_heard=r.get("line_heard"))
        C = compare(dict(syllables=unpack(ref["syl"], txt), line_jer=ref.get("line_jer")), take)
        for row in C["rows"]:
            for k, v in row["checks"].items():
                if k == "pause" and max(v["ref"], v["take"]) < 0.2:
                    continue
                errs[k].append(abs(math.log(v["diff"])) if k == "duration" else abs(v["diff"]))
    out = {}
    for k in CHECKS:
        if errs[k]:
            p = float(np.percentile(errs[k], 90))
            out[k] = math.exp(p) if k == "duration" else p
    return out


def pct(a, b):
    return "—" if not b else "%.1f%%" % (100.0 * a / b)


def summarize(rows):
    from validate_synthetic import INJECT, TYPES, score
    refs = {r["src"]: r for r in rows if r["variant"] == "ref"}
    L = ["# 서비스 점검 ② 실제 녹음 조건 — 방해(지적 없어야 함) · 주입(잡아야 함)", "",
         "same = 같은 목소리(PSOLA 재합성) · gender = 목소리를 바꾼 것(다른 사람이 억양만 똑같이 따라 한 것에 가깝다). "
         "채널 설명은 스크립트 머리말. 오류 = 정렬 불가로 분석이 멈춘 것.", ""]
    for st, name in (("zeroth", "Zeroth 낭독"), ("original", "원작 대사")):
        R = [r for r in rows if refs.get(r["src"], {}).get("set") == st]
        nrefs = sum(1 for r in R if r["variant"] == "ref")
        L += ["## (가) 방해 — %s %d개" % (name, nrefs), "",
              "| 바탕 | 채널 | 잘못 지적된 음절 | 지적 0 발화 | 재녹음 | 오류 | 확신 없음 | 음높이 | 억양 | 길이 | 쉼 | 세기 | 빠르기 "
              "| 오차 p90 높이 / 억양 / 길이배 / 세기 |",
              "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|"]
        for base in BASES:
            for ch in CHANNELS:
                sub = [r for r in R if r["variant"] == "nuisance" and r["base"] == base and r["channel"] == ch]
                if not sub:
                    continue
                ok_rows = [r for r in sub if "error" not in r]
                nu, _, _ = score([r for r in R if r["variant"] == "ref"] + ok_rows)
                s = nu.get(ch, collections.Counter())
                err = meas_err(ok_rows, refs)
                n_err = len(sub) - len(ok_rows)
                L.append("| %s | %s | **%s** | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    base, ch, pct(s["flag_syl"], s["syl"]), pct(s["clean_utt"], len(sub)), pct(s["retake"], len(sub)),
                    pct(n_err, len(sub)), pct(s["uncertain_syl"], s["syl"]),
                    *[pct(s["flag_" + t], s["syl"]) for t in TYPES], pct(s["tempo_flag"], len(sub)),
                    " / ".join("%.2f" % err[k] if k in err else "—" for k in ("pitch_level", "pitch_dir", "duration", "energy"))))
        L += ["", "## (나) 주입 — %s · type_hit(그 음절·종류·방향) / 1순위" % name, "",
              "| 주입 | " + " | ".join("%s·%s" % (b[0], c) for b in BASES for c in DETECT) + " |",
              "|---|" + "--:|" * (len(BASES) * len(DETECT))]
        tot = collections.defaultdict(collections.Counter)
        for kind in INJECT:
            cells = []
            for base in BASES:
                for ch in DETECT:
                    sub = [r for r in R if r["variant"] == "inject" and r["kind"] == kind and r["base"] == base
                           and r["channel"] == ch]
                    ok_rows = [r for r in sub if "error" not in r]
                    _, inj, _ = score([r for r in R if r["variant"] == "ref"] + ok_rows)
                    s = inj.get((kind, base), collections.Counter())
                    n = len(sub)
                    tot[(base, ch)].update(dict(n=n, type_hit=s["type_hit"], top1=s["top1"], retake=s["retake"] + n - len(ok_rows),
                                                other=s["other_flag_syl"], other_n=s["other_syl"], unc=s["k_uncertain"]))
                    cells.append("%s / %s" % (pct(s["type_hit"], n), pct(s["top1"], n)) if n else "—")
            L.append("| %s | %s |" % (kind, " | ".join(cells)))
        L.append("| **합계** | %s |" % " | ".join(
            "**%s** / %s" % (pct(tot[(b, c)]["type_hit"], tot[(b, c)]["n"]), pct(tot[(b, c)]["top1"], tot[(b, c)]["n"]))
            for b in BASES for c in DETECT))
        L.append("| 재녹음·오류 | %s |" % " | ".join(pct(tot[(b, c)]["retake"], tot[(b, c)]["n"]) for b in BASES for c in DETECT))
        L.append("| 그 음절 확신 없음 | %s |" % " | ".join(pct(tot[(b, c)]["unc"], tot[(b, c)]["n"]) for b in BASES for c in DETECT))
        L.append("| 다른 음절 지적 | %s |" % " | ".join(pct(tot[(b, c)]["other"], tot[(b, c)]["other_n"]) for b in BASES for c in DETECT))
        L += ["", "n(채널당) = %s" % ", ".join("%s·%s %d" % (b[0], c, tot[(b, c)]["n"]) for b in BASES for c in DETECT), ""]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rescore", action="store_true", help="rows.jsonl 만 다시 센다")
    ap.add_argument("--clips", action="append", default=[], help="원작 발췌 clips.json 폴더(_cache 장면 → 배경음)")
    ap.add_argument("--zeroth", type=int, default=30)
    ap.add_argument("--babble-pool", dest="babble_pool", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not a.rescore:
        run(a)
    rows = [json.loads(l) for l in open(os.path.join(a.out, "rows.jsonl"), encoding="utf-8")]
    md = summarize(rows)
    open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8").write(md)
    print(md)


if __name__ == "__main__":
    main()
