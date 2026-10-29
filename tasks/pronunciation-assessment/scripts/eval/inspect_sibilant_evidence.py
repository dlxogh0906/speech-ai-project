"""Render blinded waveform/spectrogram evidence and time-resolved measurements.

This script produces observations, not human labels or validated boundaries.
Optional Praat tracks and OpenSauce harmonic corrections are explicitly versioned.
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
import importlib.util
import json
import subprocess
import time
import wave

import numpy as np
from mdd.sibilant_acoustics import frame_tracks, spectrum, harmonic_amplitudes, partition_balance_tracks


def finite(v):
    return float(v) if np.isfinite(v) else None


def praat_tracks(x, sr, correction):
    import parselmouth
    sound = parselmouth.Sound(x, sampling_frequency=sr)
    pitch = sound.to_pitch_cc(time_step=.005, pitch_floor=70, pitch_ceiling=500)
    times, freq, strength = pitch.xs(), pitch.selected_array["frequency"], pitch.selected_array["strength"]
    models = {ceiling: sound.to_formant_burg(time_step=.005, maximum_formant=ceiling) for ceiling in (5000, 5500, 6000)}
    rows = []
    for t, f0, st in zip(times, freq, strength):
        r = {"time": float(t), "pitch_hz": float(f0) if f0 else None, "pitch_strength": float(st), "ceilings": {}}
        if f0:
            for ceiling, fm in models.items():
                ff = [fm.get_value_at_time(i, t) for i in (1, 2, 3)]
                bb = [fm.get_bandwidth_at_time(i, t) for i in (1, 2, 3)]
                h = harmonic_amplitudes(x, sr, t, f0, ff, bb, correction)
                r["ceilings"][str(ceiling)] = {"formants": [finite(v) for v in ff], "bandwidths": [finite(v) for v in bb], "harmonics": h}
        rows.append(r)
    return rows


def render(path, item, x, sr, offset, tracks, pitch_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    f, t, sp = spectrum(x, sr)
    fig, axes = plt.subplots(6, 1, figsize=(14, 14), sharex=True, gridspec_kw={"height_ratios": [1, 2.3, 1.5, 1, 1, 1]})
    tx = np.arange(len(x)) / sr + offset
    axes[0].plot(tx, x, color="#17364c", lw=.55)
    axes[0].set_ylabel("PCM")
    top = np.max(sp)
    axes[1].pcolormesh(t + offset, f / 1000, sp, shading="auto", cmap="magma", vmin=top - 65, vmax=top)
    axes[1].set_ylim(0, 8)
    axes[1].set_ylabel("Spectrum (kHz)")
    f2, t2, sp2 = spectrum(x, sr, .005, .001)
    axes[2].pcolormesh(t2 + offset, f2 / 1000, sp2, shading="auto", cmap="Greys", vmin=np.max(sp2) - 55, vmax=np.max(sp2))
    axes[2].set_ylim(0, 2)
    axes[2].set_ylabel("Pulse detail (kHz)")
    tt = tracks["times"] + offset
    reference = max(np.max(tracks[key]) for key in ("low_db", "mid_db", "high_db"))
    for key, color in (("low_db", "#2b78b8"), ("mid_db", "#88652f"), ("high_db", "#ca582b")):
        axes[3].plot(tt, tracks[key] - reference, color=color, label=key, lw=1)
    axes[3].set_ylabel("Band energy\nrelative dB")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[4].plot(tt, tracks["periodicity"], color="#477259", label="40ms ACF peak")
    if pitch_rows:
        axes[4].plot([r["time"] + offset for r in pitch_rows], [r["pitch_strength"] if r["pitch_hz"] else np.nan for r in pitch_rows], ".", ms=2, color="#653d8a", label="Praat voiced strength")
    axes[4].set_ylim(-.05, 1.05)
    axes[4].set_ylabel("Periodicity")
    axes[4].legend(loc="upper right", fontsize=8)
    balances = partition_balance_tracks(x, sr)
    for cutoff, color in ((2000, "#2b78b8"), (3000, "#88652f"), (4000, "#ca582b")):
        axes[5].plot(balances["times"] + offset, balances["upper_lower_db_%d" % cutoff], color=color, lw=.8, label="split %d Hz" % cutoff)
    axes[5].set_ylabel("Upper/lower\nenergy dB")
    axes[5].legend(loc="upper right", fontsize=8)
    for ax in axes:
        # Phone-interval hints are explicitly marked, never acoustic ground truth.
        for v in (item["tag_consonant"][0], item["tag_vowel"][0], item["tag_vowel"][1]):
            ax.axvline(v, color="#82919a", ls="--", lw=.7, alpha=.6)
        ax.grid(axis="x", alpha=.18)
        ax.xaxis.set_minor_locator(MultipleLocator(.01))
    axes[-1].set_xlabel("Original recording time (s); dashed = existing TAG interval hints, not verified landmarks")
    axes[-1].set_xlim(offset, offset + len(x) / sr)
    fig.suptitle(item["id"] + " | unmodified PCM | transcription and model output hidden", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .975))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--praat", action="store_true")
    ap.add_argument("--opensauce-root", type=Path)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if a.out.exists():
        ap.error("refusing to overwrite measurements")
    items = [json.loads(s) for s in (a.packet / "items.private.jsonl").read_text(encoding="utf-8").splitlines()]
    if a.only:
        items = [r for r in items if r["id"] in a.only]
    correction, external = None, {}
    if a.opensauce_root:
        p = a.opensauce_root / "opensauce/harmonics.py"
        spec = importlib.util.spec_from_file_location("opensauce_harmonic_reference", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        correction = mod.correction_iseli_i
        external["opensauce"] = {"url": "https://github.com/voicesauce/opensauce-python", "commit": subprocess.check_output(["git", "-C", str(a.opensauce_root), "rev-parse", "HEAD"], text=True).strip(), "harmonics_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bandwidths": "Praat measured; not VoiceSauce default formula"}
    if a.praat:
        import parselmouth
        external["parselmouth"] = parselmouth.__version__
        external["praat"] = parselmouth.PRAAT_VERSION
    a.out.mkdir(parents=True)
    (a.out / "plots").mkdir()
    (a.out / "tracks").mkdir()
    started = time.time()
    index = []
    for n, item in enumerate(items, 1):
        with wave.open(item["source_wav"], "rb") as f:
            sr = f.getframerate()
            begin = int(round(item["analysis_interval"][0] * sr))
            end = min(f.getnframes(), int(round(item["analysis_interval"][1] * sr)))
            f.setpos(begin)
            x = np.frombuffer(f.readframes(end - begin), np.int16).astype(float) / 32768
        offset = begin / sr
        tracks = frame_tracks(x, sr)
        np.savez_compressed(a.out / "tracks" / (item["id"] + ".npz"), **tracks, offset=offset, sample_rate=sr)
        np.savez_compressed(a.out / "tracks" / (item["id"] + "_balance.npz"), **partition_balance_tracks(x, sr), offset=offset, sample_rate=sr)
        pcm_float_sha256 = hashlib.sha256(x.tobytes()).hexdigest()
        rows = praat_tracks(x, sr, correction) if a.praat else []
        if hashlib.sha256(x.tobytes()).hexdigest() != pcm_float_sha256:
            raise RuntimeError("acoustic measurement changed the source waveform")
        (a.out / "tracks" / (item["id"] + ".json")).write_text(json.dumps({"id": item["id"], "offset": offset, "rows": rows}, allow_nan=False) + "\n")
        render(a.out / "plots" / (item["id"] + ".png"), item, x, sr, offset, tracks, rows)
        index.append({"id": item["id"], "offset": offset, "end": end / sr, "frames": len(tracks["times"]), "praat_frames": len(rows)})
        if n % 10 == 0 or n == len(items):
            print("measured %d/%d %.1fs" % (n, len(items), time.time() - started), flush=True)
    (a.out / "run.json").write_text(json.dumps({"items": index, "external": external,
        "packet_sha256": hashlib.sha256((a.packet / "items.private.jsonl").read_bytes()).hexdigest(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), ROOT / "tasks/pronunciation-assessment/src/mdd/sibilant_acoustics.py")},
        "boundary_status": "no validated acoustic boundaries inferred", "waveform_immutability_checked": True,
        "seconds": time.time() - started}, indent=2) + "\n")
    (a.out / "source_snapshot").mkdir()
    for p in (Path(__file__), ROOT / "tasks/pronunciation-assessment/src/mdd/sibilant_acoustics.py"):
        (a.out / "source_snapshot" / p.name).write_bytes(p.read_bytes())


if __name__ == "__main__":
    main()
