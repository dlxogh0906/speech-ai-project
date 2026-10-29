"""Blinded close inspection of frication release and early vowel cycles.

All spectra are displays of the original PCM. Display pre-emphasis is labeled
and never applied to the listening audio or saved acoustic measurements.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]

import argparse
import hashlib
import json
import wave

import numpy as np
from scipy import signal
from mdd.sibilant_acoustics import spectrum, partition_balance_tracks


def render(item, out, begin=None, end=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    with wave.open(item["source_wav"], "rb") as f:
        sr = f.getframerate()
        start = int(round(item["analysis_interval"][0] * sr))
        stop = min(f.getnframes(), int(round(item["analysis_interval"][1] * sr)))
        f.setpos(start)
        x = np.frombuffer(f.readframes(stop - start), np.int16).astype(float) / 32768
    offset = start / sr
    anchor = item["tag_vowel"][0]
    left = max(offset, anchor - .09) if begin is None else max(offset, begin)
    right = min(stop / sr, anchor + .12) if end is None else min(stop / sr, end)
    if not left < right:
        raise ValueError("empty viewing interval")
    times = np.arange(len(x)) / sr + offset
    visible = (times >= left) & (times <= right)
    fig, axes = plt.subplots(5, 1, figsize=(15, 12), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1.7, 1.7, 1, 1]})
    axes[0].plot(times[visible], x[visible], color="#17364c", lw=.65)
    axes[0].set_ylabel("Original PCM")
    # One-pole 50Hz pre-emphasis changes only this explicitly labeled display.
    preemphasized = signal.lfilter([1., -np.exp(-2 * np.pi * 50 / sr)], [1.], x)
    for ax, waveform, label, high in ((axes[1], x, "Raw spectrum", 8),
                                      (axes[2], preemphasized, "Display pre-emphasis\n50Hz, approx. 6dB/oct", 8)):
        f, t, sp = spectrum(waveform, sr, .005, .0005)
        sel = (t + offset >= left) & (t + offset <= right)
        top = np.max(sp[:, sel])
        ax.pcolormesh(t[sel] + offset, f / 1000, sp[:, sel], shading="auto",
                      cmap="magma", vmin=top - 50, vmax=top)
        ax.set_ylim(0, high)
        ax.set_ylabel(label + "\n(kHz)")
    balances = partition_balance_tracks(x, sr)
    for cutoff, color in ((2000, "#2b78b8"), (3000, "#88652f"), (4000, "#ca582b")):
        axes[3].plot(balances["times"] + offset, balances["upper_lower_db_%d" % cutoff], color=color, label="split %d Hz" % cutoff, lw=.8)
    axes[3].set_ylabel("Upper/lower dB\n10ms window")
    axes[3].legend(loc="upper right", fontsize=8)
    # A low-pass DISPLAY helps locate early cycles. Raw PCM above remains the
    # reference, since filtering itself can smear an acoustic onset.
    sos = signal.butter(4, 1200, fs=sr, output="sos")
    low = signal.sosfiltfilt(sos, x)
    axes[4].plot(times[visible], low[visible], color="#477259", lw=.65)
    axes[4].set_ylabel("Low-pass display\n1.2kHz; not a boundary")
    # Preserve millisecond detail for transitions, but keep word-context views
    # readable when the caller supplies a wider interval.
    major_step = next((s for s in (.01, .02, .05, .1, .2, .5, 1., 2., 5.)
                       if (right - left) / s <= 22), 10.)
    for ax in axes:
        ax.grid(axis="x", which="both", alpha=.18)
        ax.xaxis.set_major_locator(MultipleLocator(major_step))
        ax.xaxis.set_minor_locator(MultipleLocator(major_step / 5))
        ax.set_xlim(left, right)
    axes[-1].set_xlabel("Original recording time (s); no acoustic boundary asserted")
    fig.suptitle(item["id"] + " | release / vowel transition | labels and model predictions hidden", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return {"id": item["id"], "source_sha256": item["source_sha256"], "view_interval": [left, right],
            "image_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--begin", type=float)
    ap.add_argument("--end", type=float)
    a = ap.parse_args()
    if a.out.exists():
        ap.error("refusing to overwrite a viewing run")
    if (a.begin is not None or a.end is not None) and (not a.only or len(a.only) != 1):
        ap.error("explicit time range requires exactly one --only ID")
    items = [json.loads(s) for s in (a.packet / "items.private.jsonl").read_text().splitlines()]
    if a.only:
        if set(a.only) - {r["id"] for r in items}:
            ap.error("unknown opaque ID")
        items = [r for r in items if r["id"] in a.only]
    a.out.mkdir(parents=True)
    records = []
    for i, item in enumerate(items, 1):
        records.append(render(item, a.out / (item["id"] + ".png"), a.begin, a.end))
        if i % 10 == 0 or i == len(items):
            print("rendered %d/%d" % (i, len(items)), flush=True)
    sources = [Path(__file__), ROOT / "tasks/pronunciation-assessment/src/mdd/sibilant_acoustics.py"]
    (a.out / "run.json").write_text(json.dumps({"records": records,
        "packet_sha256": hashlib.sha256((a.packet / "items.private.jsonl").read_bytes()).hexdigest(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "status": "blinded_displays_not_annotations"}, indent=2) + "\n")
    (a.out / "source_snapshot").mkdir()
    for p in sources:
        (a.out / "source_snapshot" / p.name).write_bytes(p.read_bytes())


if __name__ == "__main__":
    main()
