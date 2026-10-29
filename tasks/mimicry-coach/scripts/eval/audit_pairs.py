# -*- coding: utf-8 -*-
"""서비스 점검 ⑤ — 원작 ↔ 따라 한 녹음: 짚은 것이 그림에서도 실제로 다른가(눈 검토용 그림).

정답이 없는 실제 쌍(원작 배우 vs 다른 목소리 TTS)에서 도구가 허용치를 넘었다고 본 항목 전부와 고른 지적(❶~❸)을
음절 단위로 맞춘 그림에 표시한다 — 그림을 보고 "정말 다른가 · 크게 다른데 안 짚은 곳은 없나 · 측정이 틀려서 생긴
지적(옥타브·경계·무성)은 없나"를 판정하는 용도다.
  1단: 음높이(각자 기준 반음) — 음절마다 같은 폭으로 늘여 겹침. 점선 = 음절 높이(중앙값). 노랑 = 허용치 넘음, 회색 = 확신 없음.
  2단: 녹음 쪽 협대역 스펙트로그램 + Praat 음높이 + 음절 경계(원본 쪽은 audit_pitch.py 그림).
  3단: 음절 길이(빠르기 맞춘 비, log2) · 허용치 ±1.4배 · 넘은 항목 글씨.

    PYTHONPATH=/data/speech-ai/tools/hf_py38:/data/speech-ai/tools/mimic_py38 HF_HUB_OFFLINE=1 \\
        $PY tasks/mimicry-coach/scripts/eval/audit_pairs.py --clips /data/speech-ai/out/mimic/original_clips_v1 \\
        --samples /data/speech-ai/out/mimic/sample_clips_v1 --font <한글 TTF> --out <출력 폴더>
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))      # 같은 폴더의 audit_pitch(spectrogram)

import argparse, json, math, os

import numpy as np

CIRCLED = "❶❷❸"
DIR_KO = {"rise": "올림", "fall": "내림", "flat": "평평"}


def label(k, v):
    if k == "pitch_level":
        return "높이 %+.1f" % v["diff"]
    if k == "pitch_dir":
        return "억양 %s→%s" % (DIR_KO.get(v["ref_dir"], "?"), DIR_KO.get(v["take_dir"], "?"))
    if k == "duration":
        return "길이 ×%.2f" % v["diff"]
    if k == "pause":
        return "쉼 %.2f→%.2f" % (v["ref"], v["take"])
    return "세기 %+.0fdB" % v["diff"]


def arr(v):
    return np.array([np.nan if a is None else a for a in v], dtype=float)


def pair_figure(path, title, A, B, C, y, font_ok):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from audit_pitch import spectrogram
    n = len(A["syllables"])
    fig, axs = plt.subplots(3, 1, figsize=(max(9.0, 1.25 * n + 3), 10.5), gridspec_kw=dict(height_ratios=[3, 2.4, 2]))
    ax, axs_, ax3 = axs
    for res, color, lab, lw in ((A, "k", "원본", 2.2), (B, "#d62728", "녹음", 1.6)):
        t, st = np.array(res["tracks"]["t_pitch"]), arr(res["tracks"]["st"])
        for i, s in enumerate(res["syllables"]):
            m = (t >= s["start"]) & (t <= s["end"])
            ax.plot(i + (t[m] - s["start"]) / max(1e-3, s["end"] - s["start"]), st[m], color=color, lw=lw,
                    label=lab if i == 0 else None)
            if s["level"] is not None:
                ax.hlines(s["level"], i + 0.25, i + 0.75, color=color, lw=1.0, ls=":")
    lo, hi = ax.get_ylim()
    for row in C["rows"]:
        i = row["i"]
        if row["uncertain"]:
            ax.axvspan(i, i + 1, color="0.8", alpha=0.5)
        elif any(v["flag"] for v in row["checks"].values()):
            ax.axvspan(i, i + 1, color="#ffcc66", alpha=0.35)
        ax.axvline(i, color="0.7", lw=0.6)
    for k, iss in enumerate(C["selected"]):
        if iss.get("i") is not None:
            ax.text(iss["i"] + 0.5, hi, CIRCLED[k], ha="center", va="bottom", fontsize=16, color="#1f77b4")
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels([s["text"] if font_ok else str(s["i"]) for s in A["syllables"]], fontsize=13)
    ax.set_xlim(0, n)
    ax.set_ylabel("반음(각자 기준)")
    ax.legend(loc="lower left", fontsize=9)
    # 2단: 녹음 스펙트로그램
    t, f, db = spectrogram(y)
    axs_.imshow(db, origin="lower", aspect="auto", extent=[t[0], t[-1], f[0], f[-1]], cmap="gray_r")
    tp, stt = np.array(B["tracks"]["t_pitch"]), arr(B["tracks"]["st"])
    hz = B["baseline_hz"] * 2 ** (stt / 12.0)
    axs_.plot(tp, hz, color="#d62728", lw=1.4)
    fmax = float(np.clip(3.2 * np.nanpercentile(hz, 95) if np.isfinite(hz).any() else 600, 400, 1200))
    axs_.set_ylim(0, fmax)
    for s in B["syllables"]:
        axs_.axvline(s["start"], color="#2ca02c", lw=0.8)
        axs_.text((s["start"] + s["end"]) / 2, fmax * 0.92, s["text"] if font_ok else str(s["i"]), ha="center", va="top",
                  color="#2ca02c", fontsize=12, weight="bold")
    axs_.axvline(B["syllables"][-1]["end"], color="#2ca02c", lw=0.8)
    axs_.set_xlim(max(0.0, B["speech"][0] - 0.2), min(len(y) / 16000.0, B["speech"][1] + 0.2))
    axs_.set_ylabel("녹음 · Hz")
    # 3단: 길이 비 + 넘은 항목
    tempo = C["tempo"]
    for i, (r, s) in enumerate(zip(A["syllables"], B["syllables"])):
        q = (s["dur"] / max(1e-3, r["dur"])) / tempo
        ax3.bar(i + 0.5, math.log2(max(q, 1e-3)), width=0.5, color="#9467bd" if abs(math.log(q)) > math.log(1.4) else "0.6")
    for v in (math.log2(1.4), -math.log2(1.4)):
        ax3.axhline(v, color="#9467bd", ls="--", lw=0.8)
    ax3.axhline(0, color="k", lw=0.6)
    ax3.set_xlim(0, n)
    ax3.set_xticks(np.arange(n) + 0.5)
    ax3.set_xticklabels([s["text"] if font_ok else str(s["i"]) for s in A["syllables"]], fontsize=12)
    ax3.set_ylabel("길이 log2(녹음/원본)")
    yl = ax3.get_ylim()
    for row in C["rows"]:
        txt = [label(k, v) for k, v in row["checks"].items() if v["flag"]]
        if txt:
            ax3.text(row["i"] + 0.5, yl[0], "\n".join(txt), ha="center", va="bottom", fontsize=9, color="#8c2d04")
    sel = " / ".join("%s %s" % (CIRCLED[k], s["say"]["what"]) for k, s in enumerate(C["selected"]))
    fig.suptitle("%s\n빠르기 %.2f배 · 일치 %s · 경계 일치 %s\n%s" % (
        title, tempo, C["match"], "—" if C["quality"]["bound_agree"] is None else "%.0f%%" % (100 * C["quality"]["bound_agree"]),
        sel or ("원본과 거의 같아요" if C["quality"]["ok"] else "다시 녹음")), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=95)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--takes", default="take_a,take_b,ref", help="견본 중 녹음으로 쓸 것")
    ap.add_argument("--font", default=os.environ.get("MIMIC_FONT", ""))
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
    os.makedirs(a.out, exist_ok=True)
    coach = Coach()
    samp = {c["id"]: c for c in json.load(open(os.path.join(a.samples, "clips.json"), encoding="utf-8"))["clips"]}
    L = []
    for c in json.load(open(os.path.join(a.clips, "clips.json"), encoding="utf-8"))["clips"]:
        if not (c.get("qc") or {}).get("ok", True) or c["id"] not in samp:
            continue
        xr = load_audio(os.path.join(a.clips, c["files"]["ref"]))
        pre = coach.analyze(xr, c["line"])
        for name in a.takes.split(","):
            f = samp[c["id"]]["files"].get(name)
            if not f:
                continue
            y = load_audio(os.path.join(a.samples, f))
            A, B, C = coach.compare(xr, y, c["line"], ref=pre)
            pair_figure(os.path.join(a.out, "%s_%s.png" % (c["id"], name)), "%s — 원작 vs TTS %s" % (c["line"], name), A, B, C, y,
                        font_ok)
            flags = ["%s:%s" % (row["text"], ",".join(label(k, v) for k, v in row["checks"].items() if v["flag"]))
                     for row in C["rows"] if any(v["flag"] for v in row["checks"].values())]
            L.append("- **%s/%s** 일치 %s · 고름: %s · 넘은 항목: %s" % (
                c["id"], name, C["match"], " / ".join(s["say"]["what"] for s in C["selected"]) or "—", " · ".join(flags) or "—"))
            print(L[-1], flush=True)
    open(os.path.join(a.out, "pairs.md"), "w", encoding="utf-8").write("# 원작 ↔ TTS 지적 목록\n\n" + "\n".join(L) + "\n")


if __name__ == "__main__":
    main()
