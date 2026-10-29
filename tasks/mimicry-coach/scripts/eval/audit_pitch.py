# -*- coding: utf-8 -*-
"""서비스 점검 ① — 음높이 곡선을 맞게 재나: 우리 추출기(Praat) vs 독립 추적기(CREPE) 대조 + 눈으로 볼 그림.

정답 음높이(성문 신호)가 없으니 원리가 전혀 다른 두 추적기가 같은 값을 내는지로 본다 — Praat 은 자기상관,
CREPE 는 파형을 바로 보는 신경망이다(Kim 2018). 둘이 맞으면 믿고, 어긋나면 그림으로 누가 맞는지 본다.
  - 프레임: 음절 안에서 둘 다 유성인 프레임의 차(반음) — 중앙값 · 1반음/3반음 넘는 비율 · 옥타브(12±2) 차 비율.
    옥타브 오류 접기(prosody.clean_octaves) 전 Praat 값도 같이 — 접기가 CREPE 쪽으로 옮겼는지.
  - 유성 판정: 음절 안 프레임 중 한쪽만 유성인 비율
  - 음절: 같은 경계·같은 정규화(각자 기준 음높이)로 잰 음절 높이·끝 억양의 추적기 간 차 —
    지적 허용치(높이 2 · 억양 3 반음)의 절반을 넘는 음절 비율. 이게 크면 지적이 추적기에 따라 바뀐다.
대상: 원작 대사(목소리 분리본 · 배경음 포함본 — 경계는 분리본 것), TTS 견본, Zeroth 낭독(공개 원어민).
그림(원작 · 어긋남이 큰 Zeroth): 협대역 스펙트로그램(배음 줄무늬) 위에 두 곡선 · 음절 경계 / 배경음 포함본 / 파형·세기.

CREPE 는 torchcrepe 를 --no-deps 로 별도 폴더에 깔아 쓴다(librosa·resampy 는 안 깔고 viterbi 만 여기서 채운다):
    $PY -m pip install --no-deps --target /data/speech-ai/tools/audit_py38 torchcrepe==0.0.24
    PYTHONPATH=/data/speech-ai/tools/hf_py38:/data/speech-ai/tools/mimic_py38:/data/speech-ai/tools/audit_py38 \\
    HF_HUB_OFFLINE=1 $PY tasks/mimicry-coach/scripts/eval/audit_pitch.py \\
        --clips /data/speech-ai/out/mimic/original_clips_v1 --samples /data/speech-ai/out/mimic/sample_clips_v1 \\
        --zeroth 40 --font <한글 TTF> --out <출력 폴더>
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))      # 같은 폴더의 validate_synthetic(zeroth_sources)

import argparse, json, os, sys, types

import numpy as np

VOICED_P = 0.5        # CREPE 주기성(3프레임 중앙값)이 이 이상이면 유성
HALF_TOL = {"level": 1.0, "delta": 1.5}     # 지적 허용치(compare.TOL)의 절반


# ── CREPE ──────────────────────────────────────────────────────

def _viterbi(prob, transition):
    """librosa.sequence.viterbi 대신 — prob[상태, 시각] · transition[이전, 다음] → 상태 열."""
    lp, lt = np.log(prob + 1e-12), np.log(transition + 1e-12)
    S, T = lp.shape
    v = lp[:, 0] - np.log(S)
    ptr = np.zeros((T, S), dtype=np.int64)
    cols = np.arange(S)
    for t in range(1, T):
        cand = v[:, None] + lt
        ptr[t] = np.argmax(cand, axis=0)
        v = cand[ptr[t], cols] + lp[:, t]
    path = np.empty(T, dtype=np.int64)
    path[-1] = int(np.argmax(v))
    for t in range(T - 1, 0, -1):
        path[t - 1] = ptr[t, path[t]]
    return path


def load_crepe():
    """torchcrepe 를 import 한다 — 없는 librosa·resampy 는 쓰는 부분만 채운 빈 모듈로."""
    try:
        import librosa  # noqa: F401
    except ImportError:
        lib, seq = types.ModuleType("librosa"), types.ModuleType("librosa.sequence")
        seq.viterbi = _viterbi
        lib.sequence = seq
        sys.modules["librosa"], sys.modules["librosa.sequence"] = lib, seq
    try:
        import resampy  # noqa: F401
    except ImportError:
        rs = types.ModuleType("resampy")

        def _no(*a, **k):
            raise RuntimeError("16 kHz 로 넣을 것(resampy 없음)")
        rs.resample = _no
        sys.modules["resampy"] = rs
    import torchcrepe
    return torchcrepe


def crepe_track(tc, wav, device):
    """→ (시각[s], f0[Hz], 주기성). 프레임 k 의 중심 = k·10 ms."""
    import torch
    x = torch.from_numpy(np.asarray(wav, dtype=np.float32))[None]
    f, per = tc.predict(x, 16000, hop_length=160, fmin=50.0, fmax=1100.0, model="full", decoder=tc.decode.viterbi,
                        return_periodicity=True, batch_size=2048, device=device, pad=True)
    per = tc.filter.median(per, 3)
    f, per = f[0].cpu().numpy().astype(float), per[0].cpu().numpy().astype(float)
    return np.arange(len(f)) * 0.01, f, per


def at(t_src, v, t):
    idx = np.clip(np.round((np.asarray(t) - t_src[0]) / 0.01).astype(int), 0, len(v) - 1)
    return np.asarray(v)[idx]


# ── 한 녹음 ───────────────────────────────────────────────────

def measure(wav, A, tc, device):
    """A(분석 결과)의 경계로 두 추적기를 견준다 → (곡선들, 프레임 통계, 음절 통계)."""
    from mimicry import prosody
    from mimicry.analyze import baseline_st, pitch_stats
    t_p, f0, _ = prosody.pitch_track(wav)
    st_raw = prosody.semitones(f0)
    st_pr = prosody.clean_octaves(st_raw)
    t_c, f_c, per = crepe_track(tc, wav, device)
    pc = at(t_c, per, t_p)
    st_cr = np.where(pc >= VOICED_P, prosody.semitones(at(t_c, f_c, t_p)), np.nan)
    insyl = np.zeros(len(t_p), dtype=bool)
    for s in A["syllables"]:
        insyl |= (t_p >= s["start"]) & (t_p <= s["end"])
    vp, vc, vr = ~np.isnan(st_pr) & insyl, ~np.isnan(st_cr) & insyl, ~np.isnan(st_raw) & insyl
    d = (st_pr - st_cr)[vp & vc]
    d_raw = (st_raw - st_cr)[vr & vc]
    fr = dict(n=int(insyl.sum()), both=int((vp & vc).sum()), praat_only=int((vp & ~vc).sum()),
              crepe_only=int((vc & ~vp).sum()), absd=np.abs(d).round(3).tolist(), absd_raw=np.abs(d_raw).round(3).tolist())
    bp, bc = baseline_st(t_p, st_pr, A["speech"]), baseline_st(t_p, st_cr, A["speech"])
    syl = []
    for s in A["syllables"]:
        lp_, dp_, nvp, _ = pitch_stats(t_p, st_pr - bp, s["start"], s["end"])
        lc_, dc_, nvc, _ = pitch_stats(t_p, st_cr - bc, s["start"], s["end"])
        syl.append(dict(text=s["text"], level=[lp_, lc_], delta=[dp_, dc_], nv=[nvp, nvc]))
    curves = dict(t=t_p, praat_hz=100.0 * 2 ** (st_pr / 12.0), crepe_hz=100.0 * 2 ** (st_cr / 12.0), base=(bp, bc))
    return curves, fr, syl


def summarize(items):
    """측정 묶음 → 표 한 줄에 들어갈 수치."""
    absd = np.array([v for it in items for v in it["frames"]["absd"]])
    absr = np.array([v for it in items for v in it["frames"]["absd_raw"]])
    n = sum(it["frames"]["n"] for it in items)
    po = sum(it["frames"]["praat_only"] for it in items)
    co = sum(it["frames"]["crepe_only"] for it in items)
    oct_ = lambda a: float(np.mean((a >= 10) & (a <= 14))) if len(a) else float("nan")
    dl = [abs(s["level"][0] - s["level"][1]) for it in items for s in it["syl"] if None not in s["level"]]
    dd = [abs(s["delta"][0] - s["delta"][1]) for it in items for s in it["syl"] if None not in s["delta"]]
    n_syl = sum(len(it["syl"]) for it in items)
    lv_missing = sum(1 for it in items for s in it["syl"] if (s["level"][0] is None) != (s["level"][1] is None))
    return dict(files=len(items), frames=int(len(absd)), med=float(np.median(absd)) if len(absd) else None,
                gt1=float(np.mean(absd > 1)) if len(absd) else None, gt3=float(np.mean(absd > 3)) if len(absd) else None,
                octave=oct_(absd), gt3_raw=float(np.mean(absr > 3)) if len(absr) else None, octave_raw=oct_(absr),
                praat_only=po / max(1, n), crepe_only=co / max(1, n), syl=n_syl, syl_level=len(dl), syl_delta=len(dd),
                level_gt=float(np.mean(np.array(dl) > HALF_TOL["level"])) if dl else None,
                delta_gt=float(np.mean(np.array(dd) > HALF_TOL["delta"])) if dd else None,
                level_med=float(np.median(dl)) if dl else None, delta_med=float(np.median(dd)) if dd else None,
                level_one_side=lv_missing / max(1, n_syl))


# ── 그림 ─────────────────────────────────────────────────────

def spectrogram(x, sr=16000, win=0.04, hop=0.005, nfft=2048):
    n, h = int(win * sr), int(hop * sr)
    x = np.pad(np.asarray(x, dtype=float), (n // 2, n // 2))
    nf = 1 + (len(x) - n) // h
    idx = np.arange(n)[None, :] + h * np.arange(nf)[:, None]
    S = np.abs(np.fft.rfft(x[idx] * np.hanning(n)[None, :], n=nfft, axis=1)) ** 2
    db = 10 * np.log10(S + 1e-12)
    return np.arange(nf) * hop, np.fft.rfftfreq(nfft, 1.0 / sr), np.maximum(db, db.max() - 70).T


def figure(path, title, A, panels, wav, font_ok):
    """panels: [(이름, 파형, 곡선들)] — 스펙트로그램 패널 여러 개 + 마지막에 파형·세기."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mimicry import prosody
    f_all = np.concatenate([c["praat_hz"][np.isfinite(c["praat_hz"])] for _, _, c in panels] + [np.array([150.0])])
    fmax = float(np.clip(3.2 * np.percentile(f_all, 95), 400, 1200))
    fig, axs = plt.subplots(len(panels) + 1, 1, figsize=(14, 3.2 * len(panels) + 2.4), sharex=True,
                            gridspec_kw=dict(height_ratios=[3] * len(panels) + [1.6]))
    S = A["syllables"]
    for ax, (name, x, c) in zip(axs, panels):
        t, f, db = spectrogram(x)
        ax.imshow(db, origin="lower", aspect="auto", extent=[t[0], t[-1], f[0], f[-1]], cmap="gray_r")
        ax.set_ylim(0, fmax)
        ax.plot(c["t"], c["praat_hz"], color="#d62728", lw=1.8, label="Praat (우리)" if font_ok else "Praat")
        ax.plot(c["t"], c["crepe_hz"], color="#1f77b4", lw=1.2, ls="--", label="CREPE")
        for s in S:
            ax.axvline(s["start"], color="#2ca02c", lw=0.8, alpha=0.8)
            ax.text((s["start"] + s["end"]) / 2, fmax * 0.93, s["text"] if font_ok else str(s["i"]), ha="center",
                    va="top", fontsize=13, color="#2ca02c", weight="bold")
        ax.axvline(S[-1]["end"], color="#2ca02c", lw=0.8, alpha=0.8)
        ax.set_ylabel("%s · Hz" % name)
        ax.legend(loc="upper right", fontsize=9)
    ax = axs[-1]
    tt = np.arange(len(wav)) / 16000.0
    ax.plot(tt, wav / (np.max(np.abs(wav)) + 1e-9), color="0.6", lw=0.4)
    t_i, db = prosody.intensity_track(wav, min_pitch=100.0)
    ax2 = ax.twinx()
    ax2.plot(t_i, db, color="#9467bd", lw=1.2)
    ax2.set_ylabel("dB")
    for s in S:
        ax.axvline(s["start"], color="#2ca02c", lw=0.8)
        if s["pause_before"] >= 0.15:
            ax.axvspan(s["start"] - s["pause_before"], s["start"], color="#ff7f0e", alpha=0.2)
    ax.axvline(S[-1]["end"], color="#2ca02c", lw=0.8)
    ax.set_xlabel("초")
    ax.set_xlim(max(0.0, S[0]["start"] - 0.3), min(len(wav) / 16000.0, S[-1]["end"] + 0.3))
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ── 실행 ─────────────────────────────────────────────────────

def pct(v):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else "%.1f%%" % (100 * v)


def num(v, f="%.2f"):
    return "—" if v is None else f % v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", action="append", default=[], help="원작 발췌 clips.json 폴더(prepare_clips.py)")
    ap.add_argument("--samples", default="", help="TTS 견본 clips.json 폴더(make_sample_clips.py)")
    ap.add_argument("--zeroth", type=int, default=40, help="Zeroth test 낭독 수(0 이면 안 씀)")
    ap.add_argument("--worst", type=int, default=3, help="음절 어긋남이 큰 Zeroth 몇 개를 그림으로")
    ap.add_argument("--font", default=os.environ.get("MIMIC_FONT", ""), help="그림 글자용 한글 TTF(없으면 번호로)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams
    font_ok = bool(a.font and os.path.exists(a.font))
    if font_ok:
        font_manager.fontManager.addfont(a.font)
        rcParams["font.family"] = font_manager.FontProperties(fname=a.font).get_name()
        rcParams["axes.unicode_minus"] = False
    from mimicry.audio import load_audio
    from mimicry.pipeline import Coach
    from validate_synthetic import zeroth_sources

    os.makedirs(os.path.join(a.out, "fig"), exist_ok=True)
    coach = Coach()
    tc = load_crepe()
    dev = coach.al.device
    sets = {"원작 · 목소리 분리본": [], "원작 · 배경음 포함본": [], "TTS 견본": [], "Zeroth 낭독": []}
    per_clip = []

    for d in a.clips:
        meta = json.load(open(os.path.join(d, "clips.json"), encoding="utf-8"))
        for c in meta["clips"]:
            if not (c.get("qc") or {}).get("ok", True):
                continue
            xv = load_audio(os.path.join(d, c["files"]["ref"]))
            A, _ = coach.analyze(xv, c["line"])
            cv, fv, sv = measure(xv, A, tc, dev)
            sets["원작 · 목소리 분리본"].append(dict(id=c["id"], frames=fv, syl=sv))
            panels = [("분리본", xv, cv)]
            row = dict(id=c["id"], line=c["line"], vocals=summarize([dict(frames=fv, syl=sv)]))
            if "mix" in c["files"]:
                xm = load_audio(os.path.join(d, c["files"]["mix"]))
                cm, fm, sm = measure(xm, A, tc, dev)
                sets["원작 · 배경음 포함본"].append(dict(id=c["id"], frames=fm, syl=sm))
                panels.append(("배경음 포함", xm, cm))
                row["mix"] = summarize([dict(frames=fm, syl=sm)])
                # 분리 효과: 같은 추적기(Praat)로 분리본 vs 포함본 음절 높이 차
                row["sep_level"] = [None if None in (p["level"][0], q["level"][0]) else round(q["level"][0] - p["level"][0], 2)
                                    for p, q in zip(sv, sm)]
            row["syl"] = [dict(text=s["text"], level=s["level"], delta=s["delta"], nv=s["nv"]) for s in sv]
            per_clip.append(row)
            figure(os.path.join(a.out, "fig", "%s.png" % c["id"]), "%s — %s" % (c["id"], c["line"]), A, panels, xv, font_ok)
            print("[clip] %s %s" % (c["id"], json.dumps(row["vocals"], ensure_ascii=False)[:160]), flush=True)

    if a.samples:
        meta = json.load(open(os.path.join(a.samples, "clips.json"), encoding="utf-8"))
        for c in meta["clips"]:
            for name, fn in sorted(c["files"].items()):
                if not fn.endswith(".wav") or "raw" in name:
                    continue
                x = load_audio(os.path.join(a.samples, fn))
                try:
                    A, _ = coach.analyze(x, c["line"])
                except Exception as e:                      # TTS 견본 일부는 대사가 깨져 있다
                    print("[skip] %s %s: %s" % (c["id"], name, e), flush=True)
                    continue
                cv, fv, sv = measure(x, A, tc, dev)
                sets["TTS 견본"].append(dict(id="%s/%s" % (c["id"], name), frames=fv, syl=sv))
        print("[samples] %d" % len(sets["TTS 견본"]), flush=True)

    zitems = []
    if a.zeroth:
        for k, (path, text) in enumerate(zeroth_sources("test_data_01", a.zeroth, 0, 8, 32)):
            x = load_audio(path)
            if len(x) / 16000.0 > 9.0:
                continue
            A, _ = coach.analyze(x, text)
            cv, fv, sv = measure(x, A, tc, dev)
            sets["Zeroth 낭독"].append(dict(id="z%02d" % k, frames=fv, syl=sv))
            dl = [abs(s["level"][0] - s["level"][1]) for s in sv if None not in s["level"]]
            zitems.append((max(dl) if dl else 0.0, k, x, A, cv, text))
        print("[zeroth] %d" % len(sets["Zeroth 낭독"]), flush=True)
        for v, k, x, A, cv, text in sorted(zitems, key=lambda z: -z[0])[:a.worst]:
            figure(os.path.join(a.out, "fig", "zeroth_z%02d.png" % k), "Zeroth z%02d — 음절 높이 최대 차 %.1f반음 — %s"
                   % (k, v, text if font_ok else ""), A, [("낭독", x, cv)], x, font_ok)

    summ = {k: summarize(v) for k, v in sets.items() if v}
    json.dump(dict(summary=summ, clips=per_clip, voiced_p=VOICED_P), open(os.path.join(a.out, "pitch_audit.json"), "w"),
              ensure_ascii=False, indent=1)
    L = ["# 서비스 점검 ① 음높이 곡선 — Praat(우리) vs CREPE(독립 추적기)", "",
         "음절 안 프레임 기준. 차 = 둘 다 유성인 프레임의 |Praat − CREPE| 반음. 옥타브 = 차가 10~14반음. "
         "음절 높이·끝 억양 차 = 같은 경계·각자 기준으로 잰 값의 추적기 간 차, 지적 허용치의 절반(높이 1 · 억양 1.5 반음)을 넘는 비율.", "",
         "| 대상 | 파일 | 프레임 | 차 중앙값 | >1반음 | >3반음 | 옥타브 | 접기 전 >3반음 | Praat만 유성 | CREPE만 유성 "
         "| 음절 높이 차 >1 | 끝 억양 차 >1.5 | 한쪽만 높이 잼 |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for k, s in summ.items():
        L.append("| %s | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s (%d) | %s (%d) | %s |" % (
            k, s["files"], s["frames"], num(s["med"]), pct(s["gt1"]), pct(s["gt3"]), pct(s["octave"]), pct(s["gt3_raw"]),
            pct(s["praat_only"]), pct(s["crepe_only"]), pct(s["level_gt"]), s["syl_level"], pct(s["delta_gt"]),
            s["syl_delta"], pct(s["level_one_side"])))
    L += ["", "## 원작 대사별", "",
          "| id | 대사 | 분리본 >3반음 | 분리본 음절 높이 차 >1 | 포함본 >3반음 | 분리 전후 음절 높이 변화(Praat, 반음) |",
          "|---|---|--:|--:|--:|---|"]
    for r in per_clip:
        L.append("| %s | %s | %s | %s | %s | %s |" % (
            r["id"], r["line"], pct(r["vocals"]["gt3"]), pct(r["vocals"]["level_gt"]),
            pct(r.get("mix", {}).get("gt3")),
            " ".join("—" if v is None else "%+.1f" % v for v in r.get("sep_level", []))))
    L += ["", "## 원작 음절별 (Praat / CREPE)", ""]
    for r in per_clip:
        L.append("- **%s** " % r["id"] + " · ".join(
            "%s %s/%s ↗%s/%s" % (s["text"], num(s["level"][0], "%.1f"), num(s["level"][1], "%.1f"),
                                 num(s["delta"][0], "%.1f"), num(s["delta"][1], "%.1f")) for s in r["syl"]))
    md = "\n".join(L) + "\n"
    open(os.path.join(a.out, "pitch_audit.md"), "w", encoding="utf-8").write(md)
    print(md)


if __name__ == "__main__":
    main()
