"""Diagnostic-only CTC support and raw acoustics for specified typed cases.

TAG intervals are not closure/frication/glottal boundaries. No decisions,
new labels or classifier features are inferred from these plots.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src"),
               str(ROOT / "tasks/pronunciation-assessment/scripts/train")]
import argparse
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import eval_sibilant_directional as service
from eval_sibilant_typed_margin import app_from
from extract_sibilant_localization_diagnostic import collect
from audit_sibilant_anchor_coverage import retained_samples
from mdd.vad import trim
from mdd.sibilant_typed_margin import observations
from mdd.sibilant_acoustics import frame_tracks, spectrum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    p = json.loads(Path(args.protocol).read_text())
    for path, digest in p["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("frozen diagnostic source changed")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.protocol, out / "protocol.json")
    shutil.copy2(__file__, out / Path(__file__).name)
    wanted = {r["uid"] for r in p["cases"]}
    rows = {}
    for source in p["row_sources"]:
        for record in service.read_jsonl(source):
            row = record["row"]
            if row["uid"] in wanted:
                rows[row["uid"]] = row
    if set(rows) != wanted:
        raise ValueError("missing source rows")
    tags = collect(p["annotations"], wanted)
    app, captured = app_from(p), {}
    forward = app.S["fwd"]

    def capture(dec, wavs):
        lp, lengths = forward(dec, wavs)
        captured.update(wav=wavs[0], lp=lp[0, :lengths[0]].numpy().copy())
        return lp, lengths

    app.S["fwd"] = capture
    fig, axes = plt.subplots(len(p["cases"]), 3, figsize=(15, 3 * len(p["cases"])), squeeze=False)
    reports = []
    for case_index, case in enumerate(p["cases"]):
        row = rows[case["uid"]]
        wav = service.load_audio(row)[:320000]
        indices = retained_samples(wav)
        np.testing.assert_array_equal(wav[indices], trim(wav, 16000)[0])
        _, _, raw = app.analyze(wav, row["prompt"], "")
        np.testing.assert_array_equal(wav[indices], captured["wav"])
        tag = [t for t in tags[row["uid"]]["tag_phones"] if t[2] not in ("", ".", "-", "spn", "<unk>")]
        if [t[2] for t in tag] != row["actual"]:
            raise ValueError("TAG sequence mismatch")
        site = next(s for s in row["sites"] if s["q"] == case["q"])
        if site["alignment_ambiguous"] or site["a"] is None:
            raise ValueError("diagnostic requires uniquely mapped actual site")
        c, v = tag[site["a"]], tag[site["a"] + 1]
        runtime = service.all_optimal_mapping(row["canonical"], raw["canon"])[case["q"]]
        if len(runtime) != 1 or None in runtime:
            raise ValueError("runtime position not unique")
        q = next(iter(runtime))
        dec = app.S["dec"]
        obs = observations(captured["lp"], raw["canon"], raw["sites"], dec.d.index, dec.blank)[q]
        if obs["state"] != "available":
            raise ValueError("third-consonant observation unavailable")
        frames = obs["frames"]
        sample_indices = np.minimum(((np.array(frames) + .5) * len(indices) / len(captured["lp"])).astype(int), len(indices) - 1)
        times = indices[sample_indices] / 16000.
        support = [{"frame": f, "source_seconds": float(t), "in_target_TAG": bool(c[0] <= t < c[1]),
                    "in_following_TAG": bool(v[0] <= t < v[1])} for f, t in zip(frames, times)]
        lo, hi = max(0, int((c[0] - .12) * 16000)), min(len(wav), int((v[1] + .12) * 16000))
        clip = wav[lo:hi]
        offset = lo / 16000.
        time_axis = np.arange(lo, hi) / 16000.
        axes[case_index, 0].plot(time_axis, clip, linewidth=.6)
        frequencies, t, power = spectrum(clip, 16000)
        axes[case_index, 1].pcolormesh(t + offset, frequencies / 1000., power - power.max(), vmin=-70, vmax=0, shading="auto", cmap="magma")
        axes[case_index, 1].set_ylim(0, 8)
        tracks = frame_tracks(clip, 16000)
        for key in ("high_db", "low_db"):
            axes[case_index, 2].plot(tracks["times"] + offset, tracks[key], label=key)
        axes[case_index, 2].legend(fontsize=7)
        for ax in axes[case_index]:
            for boundary in (c[0], c[1], v[1]):
                ax.axvline(boundary, color="tab:green", linestyle="--", linewidth=.8)
            for point in times:
                ax.axvline(point, color="tab:blue", alpha=.5, linewidth=.7)
            ax.set_xlim(offset, hi / 16000.)
            ax.set_xlabel("original recording seconds")
        axes[case_index, 0].set_title(case["label"] + " | TAG=" + ("s" if site["actual"] == "ㅅ" else "ch"))
        axes[case_index, 1].set_title("relative PSD; green=TAG, blue=nominal CTC centers")
        axes[case_index, 2].set_title("raw high/low-band energy (dB)")
        reports.append({"uid": row["uid"], "prompt": row["prompt"], "q": case["q"], "TAG_actual": site["actual"],
                        "target_tag": c, "following_tag": v, "observation": obs, "mapped_nominal_centers": support,
                        "wav_sha256": service.sha(row["wav"]), "scope": "nominal model-grid centers only; encoder receptive fields are broader; TAG intervals are not measured physical event boundaries"})
    fig.tight_layout()
    fig.savefig(out / "cases.png", dpi=160)
    plt.close(fig)
    (out / "report.json").write_text(json.dumps({"scope": __doc__, "cases": reports}, ensure_ascii=False, indent=2) + "\n")
    (out / "finished.json").write_text(json.dumps({"sha256": service.sha(out / "report.json")}) + "\n")
    print(json.dumps(reports, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
