"""Evaluate LoRA encoder/head pairs on CUDA using the current service rules.

The first 16 layers are shared and frozen. Only matched suffixes differ. The
unchanged Korean gate reuses its previous probabilities for identical request
bytes. No requests or weights are sent to the NPU server.
"""
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r:
        raise RuntimeError("Repository root not found")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
_sys.path.insert(0, str(_r / "tasks/nationality-classification/scripts/train"))

import argparse
import io
import json
import math
import time

import numpy as np

from pilot_youtube_head import read_audio, sha
from rescore_npu_heads import payload
from score_npu_cached_head import mean_log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-deploy", type=_pl.Path, required=True)
    parser.add_argument("--experiment-dir", type=_pl.Path, required=True)
    parser.add_argument("--previous-head", type=_pl.Path, required=True)
    parser.add_argument("--records", type=_pl.Path, required=True)
    parser.add_argument("--gate-cache", type=_pl.Path, required=True)
    parser.add_argument("--audio-dir", type=_pl.Path)
    parser.add_argument("--reuse-validation-prefix", type=_pl.Path)
    parser.add_argument("--out", type=_pl.Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    import torch
    import soundfile as sf
    from country_classifier.npu_encoder import load_pure, conv_out_len, _Layer
    from country_classifier.encoder_adaptation import frozen_prefix
    from country_classifier.wav_augment import silence_normalize
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the requested GPU evaluation")
    device = torch.device("cuda")
    records_by_id = {r["job_id"]: r for r in map(json.loads, args.records.read_text().splitlines())}
    records = [records_by_id[key] for key in sorted(records_by_id)]
    cached_rows = {}
    for path in sorted(args.gate_cache.glob("shard_*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["job_id"] in records_by_id:
                if row["job_id"] in cached_rows:
                    raise ValueError("Duplicate gate cache row")
                cached_rows[row["job_id"]] = row
    if set(cached_rows) != set(records_by_id):
        raise ValueError("Incomplete fixed gate cache")
    if any(cached_rows[key][field] != records_by_id[key][field]
           for key in records_by_id for field in ("country", "speaker")):
        raise ValueError("Gate cache request metadata differ")

    def load_head(path):
        with np.load(path, allow_pickle=True) as values:
            return {key: values[key].copy() for key in values.files}

    base_head = load_head(args.base_deploy / "head_np.npz")
    labels = list(map(str, base_head["labels"]))
    encoder = load_pure(args.base_deploy / "encoder_pure_l17.pt").to(device)
    encoder.requires_grad_(False)
    suffixes = {"baseline": torch.nn.ModuleList(list(encoder.encoder.layers)[16:])}
    heads = {"baseline": base_head, "previous": load_head(args.previous_head)}
    freeze = json.loads((args.experiment_dir / "candidate_freeze.json").read_text())
    for arm in ("replay", "lora_youtube", "lora_youtube_aux"):
        folder = args.experiment_dir / arm / "deploy"
        if any(sha(folder / filename) != freeze["variants"][arm][key]
               for filename, key in (("encoder_pure_l17.pt", "encoder_sha256"), ("head_np.npz", "head_sha256"))):
            raise ValueError("Model changed after selection freeze")
        state = torch.load(folder / "encoder_pure_l17.pt", map_location="cpu")
        # Verify all changes belong to the advertised final-layer Q/V weights.
        base_state = encoder.state_dict()
        for key, value in state.items():
            allowed = any(key == "encoder.layers.{}.self_attn.{}.weight".format(index, name)
                          for index in (16, 17) for name in ("q_proj", "v_proj"))
            if not allowed and not torch.equal(value, base_state[key].cpu()):
                raise ValueError("Frozen encoder tensor changed: " + key)
        layers = torch.nn.ModuleList([_Layer(), _Layer()])
        tail = {}
        for key, value in state.items():
            if key.startswith("encoder.layers."):
                parts = key.split(".")
                index = int(parts[2])
                if index >= 16:
                    tail[str(index - 16) + "." + ".".join(parts[3:])] = value
        layers.load_state_dict(tail)
        suffixes[arm] = layers.eval().requires_grad_(False).to(device)
        heads[arm] = load_head(folder / "head_np.npz")
        del state, tail
    for head in heads.values():
        if any(not np.array_equal(head[key], base_head[key]) for key in ("labels", "mu", "sd")):
            raise ValueError("Encoder/head metadata differ")
    args.out.mkdir(parents=True, exist_ok=True)
    pooled = {tag: [[] for _ in records] for tag in suffixes}
    valid_lengths = [[] for _ in records]
    started = time.monotonic()

    @torch.no_grad()
    def infer_tail(hidden, mask, group_ids):
        bias = ((mask - 1) * 1e4)[:, None, None, :]
        m = mask.unsqueeze(-1)
        for tag, suffix in suffixes.items():
            value = hidden
            for layer in suffix:
                value = layer(value, bias)
            vector = ((value * m).sum(1) / m.sum(1).clamp_min(1)).cpu().numpy()
            for index, group in enumerate(group_ids):
                pooled[tag][group].append(np.clip(np.nan_to_num(vector[index]), -50, 50))

    if args.reuse_validation_prefix:
        folder = args.reuse_validation_prefix
        meta = json.loads((folder / "meta.json").read_text())
        if (meta["conditions"]["rows"] != sha(args.records)
                or meta["conditions"]["encoder"] != sha(args.base_deploy / "encoder_pure_l17.pt")
                or meta["conditions"]["prefix_layers"] != 16 or meta["dtype"] != "float32"):
            raise ValueError("Validation prefix cache differs")
        hidden = np.load(folder / "hidden.npy", mmap_mode="r")
        masks = np.load(folder / "mask.npy", mmap_mode="r")
        groups = meta["groups"]
        for offset in range(0, len(hidden), args.batch_size):
            end = min(offset + args.batch_size, len(hidden))
            infer_tail(torch.from_numpy(np.array(hidden[offset:end])).to(device),
                       torch.from_numpy(np.array(masks[offset:end])).to(device), groups[offset:end])
    else:
        pending, pending_groups = [], []

        def flush():
            if not pending:
                return
            src = np.zeros((len(pending), 192000), np.float32)
            valid = np.zeros((len(pending), conv_out_len(192000)), np.float32)
            for index, wave in enumerate(pending):
                src[index, :len(wave)] = wave
                valid[index, :conv_out_len(len(wave))] = 1
            tm = torch.from_numpy(valid).to(device)
            tx = frozen_prefix(encoder, torch.from_numpy(src).to(device), tm, 16)
            infer_tail(tx, tm, pending_groups)
            pending.clear()
            pending_groups.clear()

        for group, record in enumerate(records):
            if args.audio_dir:
                path = args.audio_dir / record["audio_file"]
                if "audio_sha256" in record and sha(path) != record["audio_sha256"]:
                    raise ValueError("Staged audio changed")
                wave = read_audio(path)
            else:
                wave = np.concatenate([read_audio(path) for path in record["source_paths"]])
                buf = io.BytesIO()
                sf.write(buf, wave, 16000, format="WAV", subtype="PCM_16")
                buf.seek(0)
                wave = sf.read(buf, dtype="float32")[0]
            wave = wave[:960000]
            sec = len(wave) / 16000
            requested = int(record["payload"].get("parts_requested", 3))
            n = requested if sec >= requested * 3 else max(1, int(sec // 3))
            n = max(n, int(math.ceil(sec / 12)))
            cuts = [round(j * len(wave) / n) for j in range(n + 1)]
            for a, b in zip(cuts[:-1], cuts[1:]):
                raw = wave[a:b]
                frame_n = len(raw) // 480
                if len(raw) < 8000 or not frame_n:
                    continue
                rms = np.sqrt(np.square(raw[:frame_n * 480].reshape(frame_n, 480), dtype=np.float64).mean(1))
                if (rms >= .0015).sum() * .03 < 1:
                    continue
                pending.append(silence_normalize(raw)[:192000])
                pending_groups.append(group)
                valid_lengths[group].append([round(a / 16000, 1), round(b / 16000, 1)])
                if len(pending) >= args.batch_size:
                    flush()
            if len(valid_lengths[group]) != cached_rows[record["job_id"]]["n"]:
                raise ValueError("GPU split differs from fixed gate reference")
            if group % 100 == 0:
                print("[GPU eval] {}/{} {:.1f}s".format(group, len(records), time.monotonic() - started), flush=True)
        flush()
    output = []
    for group, record in enumerate(records):
        cached = cached_rows[record["job_id"]]
        if len(pooled["baseline"][group]) != cached["n"]:
            raise ValueError("GPU/group count differs")
        if cached["n"]:
            with np.load(args.gate_cache / "pooled" / (cached["capture_rid"] + ".npz")) as values:
                if list(map(str, values["labels"])) != labels:
                    raise ValueError("Gate label order differs")
                pg = mean_log(values["gate"])
            variants = {tag: payload(np.asarray(pooled[tag][group]), pg, heads[tag], labels) for tag in suffixes}
            variants["previous"] = payload(np.asarray(pooled["baseline"][group]), pg, heads["previous"], labels)
        else:
            variants = {tag: {"top5": [], "gate_korea": None} for tag in heads}
        output.append({"job_id": record["job_id"], "country": record["country"], "speaker": record["speaker"],
                       "capture_rid": cached["capture_rid"], "n": cached["n"],
                       "parts_requested": int(record["payload"].get("parts_requested", 3)),
                       "spans": cached["spans"], "variants": variants,
                       "recorded_baseline_top5": cached["variants"]["baseline"]["top5"],
                       "backend": "gpu", "gate_backend": "unchanged prior NPU cache"})
    (args.out / "shard_0.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output))
    (args.out / "evaluation_backend.json").write_text(json.dumps({"main_backend": "CUDA",
           "gpu": torch.cuda.get_device_name(device), "gate_backend": "unchanged prior NPU cache for identical audio",
           "input_records_sha256": sha(args.records), "models": freeze["variants"],
           "requests": len(records), "elapsed_seconds": time.monotonic() - started}, indent=2))
    print("[done] {} requests {:.1f}s".format(len(records), time.monotonic() - started), flush=True)


if __name__ == "__main__":
    main()
