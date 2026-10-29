"""Compare frozen pretrained speech/audio encoders on the existing dialect split."""
import argparse
import datetime
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, Subset

spec = importlib.util.spec_from_file_location("dialect_heads", Path(__file__).with_name("train_xlsr2b_dialect.py"))
heads = importlib.util.module_from_spec(spec)
spec.loader.exec_module(heads)

MODELS = {
    "w2vbert2": {
        "name": "w2v-BERT 2.0",
        "repository": "facebook/w2v-bert-2.0",
        "revision": "da985ba0987f70aaeb84a80f2851cfac8c697a7b",
        "path": "/data/speech-ai/corpus/weights/external/hub/models--facebook--w2v-bert-2.0/snapshots/da985ba0987f70aaeb84a80f2851cfac8c697a7b",
    },
    "wavlm_large": {
        "name": "WavLM Large",
        "repository": "microsoft/wavlm-large",
        "revision": "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c",
        "path": "/data/speech-ai/models/hf/hub/models--microsoft--wavlm-large/snapshots/c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c",
    },
    "eat_large": {
        "name": "EAT Large (AS-2M pretrain, epoch 20)",
        "repository": "worstchan/EAT-large_epoch20_pretrain",
        "revision": "1109aaae544915a2b82182c643c1894b1b4d9f27",
        "path": "/data/speech-ai/models/hf/hub/models--worstchan--EAT-large_epoch20_pretrain/snapshots/1109aaae544915a2b82182c643c1894b1b4d9f27",
    },
    "xlsr2b": {
        "name": "XLS-R 2B",
        "repository": "facebook/xls-r-2b",
        "revision": "local_pretrained_1000k",
        "path": "/data/speech-ai/models/xlsr2_2B_1000k.pt",
    },
}


def progress(args, stage, **extra):
    heads.write_json(Path(args.out) / "progress.json", {
        "model": args.model, "stage": stage,
        "updated_at": datetime.datetime.now().astimezone().isoformat(), **extra,
    })


class PreparedClips(Dataset):
    def __init__(self, rows, kind, processor=None):
        self.rows, self.kind, self.processor = rows, kind, processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        try:
            row = self.rows[index]
            with sf.SoundFile(row["audio_path"]) as source:
                if source.samplerate != 16000 or source.channels != 1:
                    raise ValueError("Expected 16kHz mono source")
                source.seek(round(float(row["start_sec"]) * 16000))
                wav = source.read(round(float(row["duration_sec"]) * 16000), dtype="float32")
            if len(wav) < 16000 or not np.isfinite(wav).all():
                raise ValueError("Short or nonfinite waveform")
            if self.kind == "w2vbert2":
                prepared = self.processor(wav, sampling_rate=16000, return_tensors="np", return_attention_mask=True)
                value = torch.from_numpy(prepared["input_features"][0].copy())
            elif self.kind == "eat_large":
                import torchaudio
                wave = torch.from_numpy(wav.copy())
                wave = wave - wave.mean()
                value = torchaudio.compliance.kaldi.fbank(
                    wave.unsqueeze(0), htk_compat=True, sample_frequency=16000,
                    use_energy=False, window_type="hanning", num_mel_bins=128,
                    dither=0.0, frame_shift=10,
                )
            else:
                if self.kind == "wavlm_large":
                    wav = (wav - wav.mean()) / np.sqrt(wav.var() + 1e-7)
                value = torch.from_numpy(wav.copy())
            return index, value, None
        except Exception as exc:
            return index, None, "%s: %s" % (type(exc).__name__, str(exc))


class Collator:
    def __init__(self, kind):
        self.kind = kind

    def __call__(self, items):
        errors = [(i, e) for i, _, e in items if e is not None]
        good = [(i, v) for i, v, e in items if e is None]
        if not good:
            return [], None, None, errors
        lengths = torch.tensor([len(v) for _, v in good], dtype=torch.long)
        length = int(lengths.max())
        if self.kind == "eat_large":
            length = int(math.ceil(length / 16.0) * 16)
        shape = (len(good), length) + tuple(good[0][1].shape[1:])
        values = torch.zeros(shape)
        for at, (_, value) in enumerate(good):
            values[at, :len(value)] = value
        if self.kind == "eat_large":
            values = ((values + 4.268) / (4.569 * 2)).unsqueeze(1)
        return [i for i, _ in good], values, lengths, errors


def pooled(values, valid):
    values = values.float()
    mask = valid.to(values.dtype).unsqueeze(-1)
    return (values * mask).sum(1) / mask.sum(1).clamp_min(1)


class Encoder:
    def __init__(self, kind, device):
        self.kind, self.device = kind, device
        self.info = dict(MODELS[kind])
        self.layers, self.width, self.processor = [15, 23], 1024, None
        from transformers import AutoModel, SeamlessM4TFeatureExtractor, Wav2Vec2BertModel, WavLMModel
        path = self.info["path"]
        if kind == "w2vbert2":
            self.processor = SeamlessM4TFeatureExtractor.from_pretrained(path, local_files_only=True)
            self.model = Wav2Vec2BertModel.from_pretrained(path, local_files_only=True)
        elif kind == "wavlm_large":
            self.model = WavLMModel.from_pretrained(path, local_files_only=True)
        elif kind == "eat_large":
            # Transformers 4.46 copies only immediate imports for a local custom
            # model. Cache the reviewed, pinned package's nested modules too.
            from transformers.dynamic_module_utils import get_cached_module_file
            for source in sorted(Path(path).glob("*.py")):
                get_cached_module_file(path, source.name, local_files_only=True)
            self.model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=True)
        else:
            from shared.fairseq_guard import ensure_fairseq
            ensure_fairseq()
            from fairseq import checkpoint_utils
            models, _ = checkpoint_utils.load_model_ensemble([path])
            self.model = models[0]
            self.model.remove_pretraining_modules()
            self.model.encoder.layerdrop = 0.0
            self.layers, self.width = [31, 47], 1920
        self.model.eval().to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.info["parameters"] = sum(p.numel() for p in self.model.parameters())
        self.info["precision"] = "FP32 weights, CUDA FP16 autocast"
        self.names = ["layer%d_mean" % v for v in self.layers]
        self.feature_layers = list(self.layers)
        if kind == "eat_large":
            self.names = ["layer15_cls", "layer23_cls", "layer15_mean", "layer23_mean"]
            self.feature_layers = [15, 23, 15, 23]
        self.preprocessing = {
            "audio": "Unchanged 16kHz mono windows, annotated length 3-12 seconds",
            "wavlm_large": "Per-waveform zero mean/unit variance, valid-frame attention and mean pooling",
            "w2vbert2": "Official SeamlessM4TFeatureExtractor, valid-frame attention and mean pooling",
            "eat_large": "Official Kaldi fbank128, DC removal, (mel+4.268)/(2*4.569); variable length, ceil-to-16 padding; supplied patch padding mask; no 10s truncation; CLS and valid-patch mean validated separately",
            "xlsr2b": "Original raw waveform, original frame padding mask and layer31/47 mean pooling",
        }[kind]
        print("[encoder] %s parameters=%d" % (self.info["name"], self.info["parameters"]), flush=True)

    @torch.no_grad()
    def __call__(self, values, lengths):
        values, lengths = values.to(self.device), lengths.to(self.device)
        with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            if self.kind == "eat_large":
                m = self.model.model
                x = m.local_encoder(values)
                if m.fixed_positional_encoder is not None:
                    x = x + m.fixed_positional_encoder(x, None)[:, :x.shape[1], :]
                x = torch.cat((m.extra_tokens.expand(len(x), -1, -1), x), 1)
                x = m.pos_drop(m.pre_norm(x))
                time_patches = values.shape[2] // 16
                valid_time = torch.arange(time_patches, device=self.device)[None, :] < ((lengths + 15) // 16)[:, None]
                valid = valid_time.repeat_interleave(8, dim=1)
                padding = torch.cat((torch.zeros((len(x), 1), dtype=torch.bool, device=self.device), ~valid), 1)
                captured = {}
                for layer, block in enumerate(m.blocks):
                    x, _ = block(x, padding_mask=padding)
                    if layer in self.layers:
                        captured[layer] = (x[:, 0].float(), pooled(x[:, 1:], valid))
                results = [captured[15][0], captured[23][0], captured[15][1], captured[23][1]]
            else:
                length = values.shape[1]
                valid = torch.arange(length, device=self.device)[None, :] < lengths[:, None]
                if self.kind == "xlsr2b":
                    output = self.model.extract_features(source=values, padding_mask=~valid, mask=False)
                    states = [output["layer_results"][layer][0].transpose(0, 1) for layer in self.layers]
                    mask = output.get("padding_mask")
                    valid = torch.ones(states[0].shape[:2], dtype=torch.bool, device=self.device) if mask is None else ~mask
                else:
                    kwargs = {"input_features" if self.kind == "w2vbert2" else "input_values": values,
                              "attention_mask": valid.long(), "output_hidden_states": True, "return_dict": True}
                    output = self.model(**kwargs)
                    states = [output.hidden_states[layer + 1] for layer in self.layers]
                    if self.kind == "wavlm_large":
                        valid = self.model._get_feature_vector_attention_mask(states[0].shape[1], valid.long())
                results = [pooled(state, valid) for state in states]
        result = torch.stack(results, 1)
        return torch.where(torch.isfinite(result), result.clamp(-50, 50), float('nan'))


def identity(args, encoder):
    h = hashlib.sha256()
    for split in ["train", "valid", "test"]:
        h.update((Path(args.manifests) / (split + ".csv")).read_bytes())
    p = Path(encoder.info["path"])
    files = [p] if p.is_file() else sorted(f for f in p.iterdir() if f.is_file())
    return {
        "manifest_sha256": h.hexdigest(), "model": encoder.info,
        "files": [{"name": f.name, "bytes": f.stat().st_size, "mtime_ns": f.stat().st_mtime_ns} for f in files],
        "features": encoder.names, "preprocessing": encoder.preprocessing,
        "feature_clip": [-50, 50], "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def extract(args, rows, encoder):
    cache = Path(args.out) / "features"
    cache.mkdir(parents=True, exist_ok=True)
    wanted = identity(args, encoder)
    p = cache / "identity.json"
    if p.exists() and json.loads(p.read_text()) != wanted:
        raise RuntimeError("Feature cache identity changed; use a new run directory")
    heads.write_json(p, wanted)
    torch.cuda.reset_peak_memory_stats()
    timings = {}
    for split, clips in rows.items():
        marker = cache / (split + ".done.json")
        if marker.exists():
            timings[split] = json.loads(marker.read_text())
            continue
        shape = (len(clips), len(encoder.names), encoder.width)
        feature_path, done_path = cache / (split + ".npy"), cache / (split + ".ok.npy")
        data = np.lib.format.open_memmap(str(feature_path), mode="r+" if feature_path.exists() else "w+", dtype=np.float16, shape=shape)
        done = np.lib.format.open_memmap(str(done_path), mode="r+" if done_path.exists() else "w+", dtype=bool, shape=(len(clips),))
        failures_path = cache / (split + ".errors.json")
        failures = json.loads(failures_path.read_text()) if failures_path.exists() else []
        todo = np.flatnonzero(~done)
        # Keep original cache row indices; group similar durations to reduce padded work.
        todo = np.array(sorted(todo.tolist(), key=lambda i: (float(clips[i]["duration_sec"]), i)), dtype=np.int64)
        dataset = PreparedClips(clips, args.model, encoder.processor)
        loader = DataLoader(Subset(dataset, todo.tolist()), batch_size=args.batch_size, num_workers=args.workers,
                            collate_fn=Collator(args.model), pin_memory=True)
        started = time.perf_counter()
        before = int(done.sum())

        def encode_batch(indices, values, lengths):
            oom = False
            try:
                result = encoder(values, lengths)
                finite = torch.isfinite(result).all(-1).all(-1).cpu().numpy()
                valid_indices = np.array(indices)[finite]
                data[valid_indices] = result[finite].cpu().numpy().astype(np.float16)
                done[valid_indices] = True
                for index in np.array(indices)[~finite]:
                    failures.append({"index": int(index), "row_id": clips[int(index)]["row_id"], "error": "Nonfinite encoder feature"})
            except torch.cuda.OutOfMemoryError:
                if len(indices) == 1:
                    raise
                oom = True
            if oom:
                # Drop the failed forward's exception before allocating retries.
                torch.cuda.empty_cache()
                middle = len(indices) // 2
                print("[oom] reducing batch %d -> %d" % (len(indices), middle), flush=True)
                encode_batch(indices[:middle], values[:middle], lengths[:middle])
                encode_batch(indices[middle:], values[middle:], lengths[middle:])

        for batch, (indices, values, lengths, errors) in enumerate(loader):
            failures.extend({"index": i, "row_id": clips[i]["row_id"], "error": error} for i, error in errors)
            if indices:
                encode_batch(indices, values, lengths)
            if batch % 50 == 0:
                data.flush()
                done.flush()
                successful = int(done.sum())
                elapsed = time.perf_counter() - started
                rate = (successful - before) / max(elapsed, 1e-6)
                progress(args, "extract", split=split, successful=successful, total=len(clips), elapsed_seconds=round(elapsed, 1), clips_per_second=round(rate, 2))
                print("[features %s] %d/%d elapsed=%.1fs rate=%.1f/s" % (split, successful, len(clips), elapsed, rate), flush=True)
                heads.write_json(failures_path, failures)
        data.flush()
        done.flush()
        info = {"rows": len(clips), "successful": int(done.sum()), "failed": int((~done).sum()), "elapsed_seconds": round(time.perf_counter() - started, 2), "batch_size": args.batch_size}
        heads.write_json(marker, info)
        heads.write_json(failures_path, failures)
        timings[split] = info
        del loader, dataset, data, done
        gc.collect()
    resources = {"parameters": encoder.info["parameters"], "batch_size": args.batch_size,
                 "extraction_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                 "extraction_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                 "split_extraction": timings,
                 "extraction_total_seconds": sum(v["elapsed_seconds"] for v in timings.values())}
    heads.write_json(Path(args.out) / "extraction_resources.json", resources)
    return cache, resources


def benchmark(args, rows, encoder):
    # Balanced, deterministic validation selection; identical rows across all encoders.
    selected = []
    for label in range(5):
        candidates = [i for i, row in enumerate(rows) if int(row["label_id"]) == label]
        candidates.sort(key=lambda i: hashlib.sha256(("benchmark:" + rows[i]["row_id"]).encode()).hexdigest())
        selected.extend(candidates[:24])
    dataset, collate = PreparedClips(rows, args.model, encoder.processor), Collator(args.model)
    for index in selected[:5]:
        _, values, lengths, errors = collate([dataset[index]])
        if errors:
            raise RuntimeError(errors)
        encoder(values, lengths)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times, durations, failures = [], [], []
    for index in selected:
        torch.cuda.synchronize()
        started = time.perf_counter()
        _, values, lengths, errors = collate([dataset[index]])
        if errors:
            failures.extend(errors)
            continue
        encoder(values, lengths)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - started) * 1000)
        durations.append(float(rows[index]["duration_sec"]))
    record = {"kind": "Batch1 audio-read + preprocessing + frozen encoder feature extraction; excludes classification head and HTTP",
              "n": len(times), "failures": failures, "mean_ms": float(np.mean(times)), "p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
              "mean_audio_seconds": float(np.mean(durations)),
              "real_time_factor": sum(times) / 1000 / sum(durations),
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
              "parameters": encoder.info["parameters"], "device": torch.cuda.get_device_name(),
              "row_id_sha256": hashlib.sha256("\n".join(rows[i]["row_id"] for i in selected).encode()).hexdigest()}
    heads.write_json(Path(args.out) / "benchmark.json", record)
    print("[benchmark] " + json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--manifests", default="/data/speech-ai/out/dialect/20261010/manifests")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    heads.write_json(Path(args.out) / "config.json", vars(args))
    torch.manual_seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(0.6, device=0)
    rows = {split: heads.load_rows(Path(args.manifests) / (split + ".csv")) for split in ["train", "valid", "test"]}
    if not args.benchmark_only:
        heads.require_person_disjoint_split(args, rows)
    record_sets = {split: {row["record_key"] for row in clips} for split, clips in rows.items()}
    if any(record_sets[a] & record_sets[b] for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]):
        raise RuntimeError("Recording leakage across splits")
    progress(args, "load_encoder")
    encoder = Encoder(args.model, torch.device("cuda:0"))
    args.encoder, args.backbone = encoder.info["path"], encoder.info["name"]
    args.feature_layers, args.feature_names = encoder.feature_layers, encoder.names
    heads.LAYERS = encoder.layers
    if args.model == "eat_large":
        args.feature_variants = {"layer15_cls": (0,), "layer23_cls": (1,), "concat15_23_cls": (0, 1),
                                 "layer15_mean": (2,), "layer23_mean": (3,), "concat15_23_mean": (2, 3)}
    progress(args, "benchmark")
    timing = benchmark(args, rows["valid"], encoder)
    if args.benchmark_only:
        progress(args, "complete_benchmark")
        return
    cache, resources = extract(args, rows, encoder)
    metadata, preprocessing = encoder.info, encoder.preprocessing
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    progress(args, "fit_heads")
    started = time.perf_counter()
    heads.fit(args, cache, rows, torch.device("cuda:0"))
    report_path = Path(args.out) / "report.json"
    report = json.loads(report_path.read_text())
    report.update({"model_id": args.model, "encoder_metadata": metadata, "preprocessing": preprocessing,
                   "benchmark_batch1": timing, "resources": resources,
                   "head_training_and_evaluation_seconds": time.perf_counter() - started})
    if args.model == "eat_large":
        report["limitations"].append("EAT compares both CLS and valid-patch mean pooling; six validation candidates versus three for speech encoders.")
    heads.write_json(report_path, report)
    progress(args, "complete", test_top1=report["test_single_window"]["top1"], test_macro_f1=report["test_single_window"]["macro_f1"])


if __name__ == "__main__":
    main()
