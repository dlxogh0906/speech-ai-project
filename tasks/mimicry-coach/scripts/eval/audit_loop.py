# -*- coding: utf-8 -*-
"""서비스 점검 ③ — 교정 루프가 끝나나: "시킨 대로 고쳐 다시 녹음하면" 몇 번 만에 "원본과 거의 같아요"가 나오나.

원본 = 원작 대사(목소리 분리본), 첫 녹음 = 같은 대사 TTS 3개(낮은 목소리 2 · 높은 목소리 1 — 모두 배우와 다른 목소리).
라운드마다 compare → 보여 준 지적을 녹음에 반영(synth.follow, PSOLA) → 다시 분석. 최대 `--rounds` 번 다시 녹음.
  - ideal1: 첫 지적 하나만, 잰 차이만큼 정확히(데모 화면이 크게 보여 주는 것이 첫 지적이다)
  - ideal3: 보여 준 지적(최대 3개)을 한꺼번에 정확히
  - real1 : 첫 지적 하나를 70% 만 고치고, 다시 녹음할 때마다 음절마다 자연스러운 흔들림
            (음높이 σ0.7반음 · 길이 σ8% · 세기 σ1.5 dB — synth.retake_jitter)
잰다: 통과(지적 0 · 재녹음 아님) 비율·라운드 수 · 고친 지적이 다음 라운드에 사라진 비율 · 같은 음절·종류를 반대 방향으로
다시 짚은 횟수(왔다 갔다) · 일치 점수 변화 · 목소리 유사도(WavLM 화자 검증 임베딩 코사인 — 도구가 안 보는 음색이 그대로인지).
보정점: 원본 ↔ PSOLA 재합성(같은 사람) · 원본 ↔ 목소리 바꿈 · TTS ↔ TTS(같은 TTS 목소리).

    PYTHONPATH=/data/speech-ai/tools/hf_py38:/data/speech-ai/tools/mimic_py38 HF_HUB_OFFLINE=1 HF_HUB_CACHE=<HF 캐시> \\
        $PY tasks/mimicry-coach/scripts/eval/audit_loop.py --clips /data/speech-ai/out/mimic/original_clips_v1 \\
        --samples /data/speech-ai/out/mimic/sample_clips_v1 --out <출력 폴더>
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, collections, json, os, time

import numpy as np

MODES = {"ideal1": dict(top=1, amount=1.0, jitter=False), "ideal3": dict(top=3, amount=1.0, jitter=False),
         "real1": dict(top=1, amount=0.7, jitter=True)}
SV_MODEL = "microsoft/wavlm-base-plus-sv"


class SpeakerSim:
    """WavLM 화자 검증 임베딩(x-vector 헤드) 코사인. 모델 카드 권장 같은 사람 문턱 0.86."""

    def __init__(self, device):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.fe = AutoFeatureExtractor.from_pretrained(SV_MODEL)
        self.m = WavLMForXVector.from_pretrained(SV_MODEL).eval().to(device)
        self.device, self.torch = device, torch

    def emb(self, x):
        enc = self.fe(np.asarray(x, dtype=np.float32), sampling_rate=16000, return_tensors="pt")
        with self.torch.no_grad():
            e = self.m(input_values=enc["input_values"].to(self.device)).embeddings[0]
        return self.torch.nn.functional.normalize(e, dim=-1).cpu().numpy()

    def cos(self, a, b):
        return float(np.dot(self.emb(a), self.emb(b)))


def wide_range(res):
    pr = res["pitch_range"]
    return (max(50.0, pr[0] * 0.8), min(1100.0, pr[1] * 1.25))


def sign(kind, diff):
    """지적 방향 — 길이·빠르기는 비(1 기준), 나머지는 차(0 기준)."""
    return (1 if diff > 1 else -1) if kind in ("duration", "tempo") else (1 if diff > 0 else -1)


def flagged_signs(C):
    """{(종류, 음절): 방향} — 허용치를 넘은 항목 전부(대사 전체 빠르기는 음절 None)."""
    out = {(k, row["i"]): sign(k, v["diff"]) for row in C["rows"] for k, v in row["checks"].items() if v["flag"]}
    for iss in C["issues"]:
        if iss["type"] == "tempo":
            out[("tempo", None)] = sign("tempo", iss["diff"])
    return out


def run_loop(coach, xr, A, FA, y, text, mode, rounds, rng):
    from mimicry import synth
    from mimicry.align import FRAME, OFFSET
    from mimicry.analyze import crosscheck
    from mimicry.compare import compare
    cfg = MODES[mode]
    hist, seen = [], {}
    for r in range(rounds + 1):
        B, FB = coach.analyze(y, text)
        crosscheck(xr, A, y, B, feats=(FA, FB), hop=FRAME, off=OFFSET)
        C = compare(A, B)
        now = flagged_signs(C)
        rec = dict(round=r, match=C["match"], ok=C["quality"]["ok"], reason=C["quality"].get("reason"), agree=B.get("bound_agree"),
                   line_jer=B.get("line_jer"), uncertain=C["quality"]["uncertain"], flagged=sorted({i for _, i in now if i is not None}),
                   sel=[dict(type=s["type"], i=s.get("i"), i_end=s.get("i_end"), sign=sign(s["type"], s["diff"]),
                             ratio=round(s["ratio"], 2)) for s in C["selected"]])
        # 고친 지적이 사라졌나(그 음절들의 그 항목이 허용치 안) · 왔다 갔다(같은 음절·항목을 반대 방향으로 다시 짚음)
        if hist and hist[-1].get("fixed"):
            rec["resolved"] = [not any((t, j) in now for j in (range(i, i_end + 1) if i is not None else [None]))
                               for t, i, i_end in hist[-1]["fixed"]]
        rec["flips"] = sum(1 for k, v in now.items() if k in seen and seen[k] != v)
        seen.update(now)
        hist.append(rec)
        if not C["quality"]["ok"] or not C["selected"] or r == rounds:
            break
        todo = C["selected"][:cfg["top"]]
        # 뒤 음절부터 고친다(앞을 고치면 뒤 시각이 밀린다) · 빠르기는 맨 마지막
        todo = sorted(todo, key=lambda s: (s["type"] == "tempo", -(s.get("i") or 0)))
        cur = B
        for j, iss in enumerate(todo):
            y = synth.follow(y, cur, iss, wide_range(cur), amount=cfg["amount"])
            if j + 1 < len(todo):
                cur, _ = coach.analyze(y, text)
        if cfg["jitter"]:
            cur, _ = coach.analyze(y, text)
            y = synth.retake_jitter(y, cur, rng, wide_range(cur))
        rec["fixed"] = [(s["type"], s.get("i"), s.get("i_end", s.get("i"))) for s in todo]
    return hist, y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True, help="원작 발췌 clips.json 폴더")
    ap.add_argument("--samples", required=True, help="TTS 견본 clips.json 폴더(같은 id 의 대사)")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0, help="재녹음 흔들림 난수 + Praat PSOLA 난수(무성 구간 조각 고르기)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from mimicry import synth
    from mimicry.audio import load_audio
    from mimicry.pipeline import Coach

    os.makedirs(a.out, exist_ok=True)
    synth.PRAAT_SEED = a.seed
    coach = Coach()
    sv = SpeakerSim(coach.al.device)
    orig = json.load(open(os.path.join(a.clips, "clips.json"), encoding="utf-8"))
    samp = {c["id"]: c for c in json.load(open(os.path.join(a.samples, "clips.json"), encoding="utf-8"))["clips"]}
    out, calib = [], collections.defaultdict(list)
    t0 = time.time()
    for c in orig["clips"]:
        if not (c.get("qc") or {}).get("ok", True) or c["id"] not in samp:
            continue
        xr = load_audio(os.path.join(a.clips, c["files"]["ref"]))
        A, FA = coach.analyze(xr, c["line"])
        wr = wide_range(A)
        calib["원본 ↔ PSOLA 재합성(같은 사람)"].append(sv.cos(xr, synth.psola(xr, prange=wr)))
        calib["원본 ↔ 목소리 바꿈"].append(sv.cos(xr, synth.change_gender(xr, wr, A["baseline_hz"] >= 165.0)))
        s = samp[c["id"]]
        tts = {k: load_audio(os.path.join(a.samples, f)) for k, f in s["files"].items() if k in ("ref", "take_a", "take_b")}
        if "ref" in tts and "take_b" in tts:
            calib["TTS ↔ TTS(같은 목소리 재시도)"].append(sv.cos(tts["ref"], tts["take_b"]))
        for name, y0 in sorted(tts.items()):
            row = dict(id=c["id"], line=c["line"], take=name, sim0=sv.cos(xr, y0), modes={})
            for m, mode in enumerate(MODES):
                # 쌍·방식마다 따로 난수 — 한 쌍의 라운드 수가 바뀌어도 다른 쌍의 흔들림은 그대로(실행끼리 비교할 수 있게)
                rng = np.random.default_rng([a.seed, len(out), m])
                try:
                    hist, yf = run_loop(coach, xr, A, FA, y0, c["line"], mode, a.rounds, rng)
                except (ValueError, RuntimeError) as e:
                    row["modes"][mode] = dict(error=str(e)[:200])
                    continue
                row["modes"][mode] = dict(hist=hist, sim_end=sv.cos(xr, yf))
            out.append(row)
            last = {m: (v.get("hist") or [{}])[-1] for m, v in row["modes"].items()}
            print("[%s/%s] sim %.2f · %s · %.0fs" % (c["id"], name, row["sim0"], " ".join(
                "%s:r%s%s" % (m, l.get("round"), "✓" if l.get("ok") and not l.get("sel") else "") for m, l in last.items()),
                time.time() - t0), flush=True)
    json.dump(dict(pairs=out, calib=calib, modes=MODES), open(os.path.join(a.out, "loop.json"), "w"), ensure_ascii=False,
              indent=1)
    open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8").write(summarize(out, calib, a.rounds))
    print(summarize(out, calib, a.rounds))


def summarize(out, calib, rounds):
    L = ["# 서비스 점검 ③ 교정 루프 — 시킨 대로 고치면 통과하나", "",
         "원본 = 원작 대사 %d개, 첫 녹음 = TTS %d개(원작 배우와 다른 목소리). 최대 %d번 다시 녹음. "
         "통과 = 보여 줄 지적 0 · 재녹음 판정 아님." % (len({r["id"] for r in out}), len(out), rounds), "",
         "| 방식 | 통과 | 통과까지 다시 녹음(중앙값·최대) | 멈춤: 재녹음 판정 | 끝까지 지적 남음 | 고친 지적이 사라짐 | 왔다 갔다 | "
         "일치 점수 처음 → 끝(중앙값) | 처음 지적 음절 수(중앙값) |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for mode in MODES:
        H = [r["modes"][mode]["hist"] for r in out if "hist" in r["modes"].get(mode, {})]
        n = len(H)
        if not n:
            continue
        passed = [h for h in H if h[-1]["ok"] and not h[-1]["sel"]]
        retake = sum(1 for h in H if not h[-1]["ok"])
        why = collections.Counter(h[-1].get("reason") for h in H if not h[-1]["ok"])
        left = n - len(passed) - retake
        res = [v for h in H for rec in h for v in rec.get("resolved", [])]
        flips = sum(rec["flips"] for h in H for rec in h)
        rr = [h[-1]["round"] for h in passed]
        m0 = [h[0]["match"] for h in H if h[0]["match"] is not None]
        m1 = [h[-1]["match"] for h in H if h[-1]["match"] is not None]
        L.append("| %s | **%d/%d** | %s | %s | %d | %.0f%% | %d | %s → %s | %s |" % (
            mode, len(passed), n, ("%d · %d" % (np.median(rr), max(rr))) if rr else "—",
            "%d%s" % (retake, (" (" + " ".join("%s %d" % kv for kv in sorted(why.items())) + ")") if retake else ""), left,
            100.0 * np.mean(res) if res else float("nan"), flips,
            "%.0f" % np.median(m0) if m0 else "—", "%.0f" % np.median(m1) if m1 else "—",
            "%d" % np.median([len(h[0]["flagged"]) for h in H])))
    L += ["", "## 목소리 유사도(WavLM 화자 검증 코사인 — 모델 카드 기준 같은 사람 ≥ 0.86)", "",
          "| 쌍 | 중앙값 | 범위 |", "|---|--:|---|"]
    for k, v in calib.items():
        L.append("| %s | %.2f | %.2f ~ %.2f |" % (k, np.median(v), min(v), max(v)))
    s0 = [r["sim0"] for r in out]
    L.append("| 원본 ↔ TTS 첫 녹음 | %.2f | %.2f ~ %.2f |" % (np.median(s0), min(s0), max(s0)))
    for mode in MODES:
        s1 = [r["modes"][mode]["sim_end"] for r in out if "sim_end" in r["modes"].get(mode, {})]
        if s1:
            L.append("| 원본 ↔ 루프 끝(%s) | %.2f | %.2f ~ %.2f |" % (mode, np.median(s1), min(s1), max(s1)))
    L += ["", "## 쌍별 (방식: 라운드별 첫 지적 → 결과)", ""]
    for r in out:
        parts = []
        for mode, v in r["modes"].items():
            if "hist" not in v:
                parts.append("%s: 오류" % mode)
                continue
            seq = ["%s%s" % (s["sel"][0]["type"].replace("pitch_", "p_"), s["sel"][0]["i"] if s["sel"][0]["i"] is not None else "")
                   for s in v["hist"] if s["sel"]]
            end = v["hist"][-1]
            tag = "통과" if end["ok"] and not end["sel"] else ("재녹음" if not end["ok"] else "남음")
            parts.append("%s: %s → %s(%s)" % (mode, " ".join(seq) or "—", tag, end["match"]))
        L.append("- **%s/%s** sim %.2f · %s" % (r["id"], r["take"], r["sim0"], " · ".join(parts)))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
