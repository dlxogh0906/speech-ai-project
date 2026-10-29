"""Compare candidate heads on NPU using previously captured validation audio.

Portable worker: copy it beside candidate heads on the NPU host. It imports the
serving implementation in a separate process and allocates its own runtimes on
explicit devices. Existing processes, model files and capture files are read
only. Each part is encoded once, then all heads use the identical pooled vector.
"""
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists() and _r.parent != _r:
    _r = _r.parent
if (_r / "pyproject.toml").exists():
    _sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-dir", type=Path, required=True)
    parser.add_argument("--main-dir", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--heads-dir", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True, help="First NPU; the gate uses the following NPU")
    parser.add_argument("--device-count", type=int, default=8)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--save-pooled", action="store_true",
                        help="Save pooled parts and gate vectors for reproducible offline head comparisons")
    parser.add_argument("--audio-dir", type=Path,
                        help="Staged evaluation WAVs; records provide audio_file instead of a captured request")
    parser.add_argument("--encoder-variant", action="append", default=[],
                        help="NAME=DEPLOY_DIR for a separately compiled encoder and its matched head")
    parser.add_argument("--profile", action="append", default=[],
                        help="NAME=LONG_HEAD,SHORT_HEAD; use short head only when actual n=1")
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error("Invalid shard")
    if not 0 <= args.device < args.device_count:
        parser.error("Invalid device")
    args.out.mkdir(parents=True, exist_ok=True)
    records_by_id = {}
    for line in args.records.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        records_by_id[record["job_id"]] = record
    records = sorted(records_by_id.values(), key=lambda r: r["job_id"])[args.shard::args.shards]
    output = args.out / ("shard_{}.jsonl".format(args.shard))
    seen = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            seen.add(json.loads(line)["job_id"])
    # The SDK interprets RBLN_DEVICES as a physical-to-logical device map.
    # Keep an identity map and initialize this worker's private allocator.
    # A singleton map such as "3" cannot accept Runtime(device=3).
    counter = args.out / ("counter_{}".format(args.shard))
    counter.write_text(str(args.device), encoding="utf-8")
    os.environ.update({"FFMPEG_BIN": str(args.ffmpeg), "NAT_BACKEND": "rbln", "KD_MODEL_DIR": str(args.main_dir),
                       "KD_GATE_DIR": str(args.gate_dir), "KD_DEVICE": "cpu", "POOL": "1", "THREADS": "2",
                       "KD_RBLN_PATH": str(args.main_dir / "encoder_pure_l17_static12s.rbln"),
                       "KD_GATE_RBLN_PATH": str(args.gate_dir / "encoder_pure_l17_static12s.rbln"),
                       "RBLN_DEVICES": ",".join(str(i) for i in range(args.device_count)), "RBLN_BUCKETS": "12",
                       "RBLN_COUNTER": str(counter),
                       "KD_SILENCE_NORM": "1", "KD_GATE_RMS": "0.0015", "KD_GATE_MIN_SEC": "1.0",
                       "KD_GATE_TAU": "0.9", "KD_GATE_INPUT": "raw", "KD_LOWCONF": "0.45",
                       "KD_KOREA_THR": "0", "NAT_RMS_NORM": "0", "NAT_WORKERS": "1",
                       "KD_LONG_PARTS": "3", "KD_LONG_MAX_SEC": "60", "KD_CAPTURE_DIR": ""})
    _sys.path.insert(0, str(args.app_dir))
    spec = importlib.util.spec_from_file_location("youtube_pilot_serving_reference", args.app_dir / "app.py")
    serving = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serving)
    heads = {"baseline": {"W": serving.MAIN.W, "b": serving.MAIN.b,
                           "mu": serving.MAIN.mu, "sd": serving.MAIN.sd}}
    for path in sorted(args.heads_dir.glob("*.npz")):
        with np.load(path, allow_pickle=True) as values:
            if [str(x) for x in values["labels"]] != serving.LABELS:
                raise ValueError("Head labels differ: " + str(path))
            if not np.array_equal(values["mu"], serving.MAIN.mu) or not np.array_equal(values["sd"], serving.MAIN.sd):
                raise ValueError("Head normalization differs: " + str(path))
            heads[path.stem] = {key: values[key].copy() for key in ("W", "b", "mu", "sd")}
    encoders = {}
    for item in args.encoder_variant:
        name, directory = item.split("=", 1)
        if name in heads or name in encoders:
            raise ValueError("Duplicate variant name: " + name)
        enc = serving.Encoder(directory, "rbln", str(Path(directory) / "encoder_pure_l17_static12s.rbln"))
        if enc.labels != serving.LABELS:
            raise ValueError("Encoder labels differ: " + name)
        encoders[name] = enc
    profiles = {}
    for item in args.profile:
        name, pair = item.split("=", 1)
        long_head, short_head = pair.split(",", 1)
        if name in heads or name in encoders or name in profiles or long_head not in heads or short_head not in heads:
            raise ValueError("Invalid routed profile: " + name)
        profiles[name] = (long_head, short_head)
    pooled_dir = args.out / "pooled"
    if args.save_pooled:
        pooled_dir.mkdir(exist_ok=True)
    start = time.monotonic()
    with output.open("a", encoding="utf-8") as stream:
        for number, source in enumerate(records, 1):
            if source["job_id"] in seen:
                continue
            rid = source["payload"]["rid"]
            if args.audio_dir:
                path = (args.audio_dir / source["audio_file"]).resolve()
                if args.audio_dir.resolve() not in path.parents:
                    raise ValueError("Evaluation audio must be inside --audio-dir")
            else:
                path = args.capture_dir / rid[:8] / (rid + ".wav")
            wave = serving.decode(path.read_bytes(), serving.LONG_MAX_SEC)
            n_requested = int(source["payload"].get("parts_requested", 3))
            n = serving.split_parts(len(wave) / serving.SR, n_requested)
            cuts = [round(j * len(wave) / n) for j in range(n + 1)]
            pms = {key: [] for key in list(heads) + list(encoders)}
            gate_parts, spans = [], []
            pooled_parts = []
            for a, b in zip(cuts[:-1], cuts[1:]):
                raw = wave[a:b]
                if len(raw) < serving.SR // 2 or serving.voiced_sec(raw) < serving.GATE_MIN_SEC:
                    continue
                normalized = serving.silence_normalize(raw, serving.GATE_RMS)
                pooled = np.clip(np.nan_to_num(serving.MAIN.pooled(normalized)), -50, 50)
                pooled_parts.append(pooled[0])
                for key, head in heads.items():
                    logits = ((pooled - head["mu"]) / head["sd"]) @ head["W"].T + head["b"]
                    prob = np.exp(logits - logits.max())
                    prob /= prob.sum()
                    pms[key].append(prob[0])
                for key, encoder in encoders.items():
                    pms[key].append(encoder.probs(normalized))
                gate_prob = serving.GATE.probs(raw)
                if serving.GATE.labels != serving.LABELS:
                    gate_prob = np.array([gate_prob[serving.GATE.labels.index(label)] for label in serving.LABELS])
                gate_parts.append(gate_prob)
                spans.append([round(a / serving.SR, 1), round(b / serving.SR, 1)])
            if "n" in source["payload"] and len(spans) != source["payload"]["n"]:
                raise ValueError("Split differs from captured NPU request: " + source["job_id"])
            if not spans:
                result = {"job_id": source["job_id"], "country": source["country"], "speaker": source["speaker"],
                          "capture_rid": rid, "n": 0, "parts_requested": n_requested,
                          "spans": [], "variants": {key: {"top5": [], "gate_korea": None} for key in pms},
                          "recorded_baseline_top5": source["payload"].get("top5", [])}
                result["variants"].update({name: {"top5": [], "gate_korea": None} for name in profiles})
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                continue
            pg = serving._mean_log(gate_parts)
            variants = {}
            for key, parts in pms.items():
                p = serving._gated(serving._mean_log(parts), pg)
                variants[key] = {"top5": serving._top5(p), "gate_korea": float(pg[serving.LABELS.index("Korea")])}
            for name, (long_head, short_head) in profiles.items():
                variants[name] = variants[short_head if len(spans) == 1 else long_head]
            result = {"job_id": source["job_id"], "country": source["country"], "speaker": source["speaker"],
                      "capture_rid": rid, "n": len(spans), "parts_requested": n_requested,
                      "spans": spans, "variants": variants,
                      "recorded_baseline_top5": source["payload"].get("top5", variants["baseline"]["top5"])}
            if args.save_pooled:
                np.savez(pooled_dir / (rid + ".npz"), pooled=np.stack(pooled_parts),
                         gate=np.stack(gate_parts), spans=np.asarray(spans), labels=np.asarray(serving.LABELS))
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            if number % 25 == 0:
                print("[shard {}] {}/{} {:.1f}s".format(args.shard, number, len(records), time.monotonic() - start), flush=True)
    print("[done] " + str(output), flush=True)


if __name__ == "__main__":
    main()
