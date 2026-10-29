"""Train dialect heads on frozen XLS-R 2B features at layers 31 and 47.

The full pretrained 2B encoder is used. Linear heads are selected by validation
macro F1; the test set is scored after selection. No service is updated.
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    if _root.parent == _root:
        raise RuntimeError("Repository root not found")
    _root = _root.parent
sys.path.insert(0, str(_root))

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader

LABELS = ["Gangwon", "Gyeongsang", "Jeolla", "Jeju", "Chungcheong"]
LABELS_KO = ["강원도", "경상도", "전라도", "제주도", "충청도"]
LAYERS = [31, 47]


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(str(tmp), str(path))


def load_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


class ClipDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        try:
            with sf.SoundFile(row["audio_path"]) as source:
                if source.samplerate != 16000 or source.channels != 1:
                    raise RuntimeError("Expected 16kHz mono FLAC")
                source.seek(int(round(float(row["start_sec"]) * 16000)))
                wave = source.read(int(round(float(row["duration_sec"]) * 16000)), dtype="float32")
            if len(wave) < 16000 or not np.isfinite(wave).all():
                raise RuntimeError("Short or nonfinite audio")
            return index, torch.from_numpy(wave.copy()), None
        except Exception as exc:
            return index, None, str(exc)


def collate(items):
    good = [(index, wave) for index, wave, error in items if error is None]
    errors = [(index, error) for index, _, error in items if error is not None]
    if not good:
        return [], None, None, errors
    length = max(len(wave) for _, wave in good)
    audio = torch.zeros(len(good), length)
    padding = torch.ones(len(good), length, dtype=torch.bool)
    for i, (_, wave) in enumerate(good):
        audio[i, :len(wave)] = wave
        padding[i, :len(wave)] = False
    return [index for index, _ in good], audio, padding, errors


def extract(args, rows_by_split, device):
    cache = Path(args.out) / "features"
    cache.mkdir(parents=True, exist_ok=True)
    manifest_hash = hashlib.sha256()
    for split in ["train", "valid", "test"]:
        manifest_hash.update((Path(args.manifests) / (split + ".csv")).read_bytes())
    identity = {"manifest_sha256": manifest_hash.hexdigest(), "encoder_path": str(Path(args.encoder).resolve()), "encoder_bytes": Path(args.encoder).stat().st_size, "encoder_mtime_ns": Path(args.encoder).stat().st_mtime_ns, "layers": LAYERS, "normalize_waveform": False,
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    identity_path = cache / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("Feature cache identity differs; use a separate output directory")
    write_json(identity_path, identity)
    if all((cache / (split + ".done.json")).exists() for split in rows_by_split):
        return cache
    from shared.fairseq_guard import ensure_fairseq
    ensure_fairseq()
    from fairseq import checkpoint_utils
    print("[encoder] loading full pretrained XLS-R 2B", flush=True)
    models, _ = checkpoint_utils.load_model_ensemble([str(args.encoder)])
    encoder = models[0]
    encoder.remove_pretraining_modules()
    encoder.encoder.layerdrop = 0.0
    encoder.eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    print("[encoder] parameters=%s layers=%s" % (sum(p.numel() for p in encoder.parameters()), len(encoder.encoder.layers)), flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for split, rows in rows_by_split.items():
        if (cache / (split + ".done.json")).exists():
            continue
        shape = (len(rows), len(LAYERS), 1920)
        feature_path, ok_path = cache / (split + ".npy"), cache / (split + ".ok.npy")
        features = np.lib.format.open_memmap(str(feature_path), mode="r+" if feature_path.exists() else "w+", dtype=np.float16, shape=shape)
        done = np.lib.format.open_memmap(str(ok_path), mode="r+" if ok_path.exists() else "w+", dtype=bool, shape=(len(rows),))
        todo = np.flatnonzero(~done)
        todo = np.array(sorted(todo.tolist(), key=lambda i: (float(rows[i]["duration_sec"]), i)), dtype=np.int64)
        subset = torch.utils.data.Subset(ClipDataset(rows), todo.tolist())
        loader = DataLoader(subset, batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate, pin_memory=device.type == "cuda")
        failures_path = cache / (split + ".errors.json")
        failures = json.loads(failures_path.read_text()) if failures_path.exists() else []
        started = time.time()
        before = int(done.sum())

        def encode(indices, audio, padding):
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                result = encoder.extract_features(source=audio.to(device, non_blocking=True), padding_mask=padding.to(device, non_blocking=True), mask=False)
            feature_padding = result.get("padding_mask")
            pooled = []
            for layer in LAYERS:
                values = result["layer_results"][layer][0].transpose(0, 1).float()
                if feature_padding is None:
                    value = values.mean(1)
                else:
                    mask = (~feature_padding).to(values.dtype).unsqueeze(-1)
                    value = (values * mask).sum(1) / mask.sum(1).clamp_min(1)
                pooled.append(value)
            extracted = torch.stack(pooled, 1)
            finite = torch.isfinite(extracted).all(-1).all(-1).cpu().numpy()
            successful = np.array(indices)[finite]
            features[successful] = extracted[finite].clamp(-50, 50).cpu().numpy().astype(np.float16)
            done[successful] = True
            for index in np.array(indices)[~finite]:
                failures.append({"index": int(index), "row_id": rows[int(index)]["row_id"], "error": "Nonfinite encoder feature"})

        def encode_batch(indices, audio, padding):
            oom = False
            try:
                encode(indices, audio, padding)
            except torch.cuda.OutOfMemoryError:
                if len(indices) == 1:
                    raise
                oom = True
            if oom:
                # Release the failed forward's traceback before recursive retries.
                torch.cuda.empty_cache()
                middle = len(indices) // 2
                print("[oom] reducing XLS-R batch %d -> %d" % (len(indices), middle), flush=True)
                encode_batch(indices[:middle], audio[:middle], padding[:middle])
                encode_batch(indices[middle:], audio[middle:], padding[middle:])

        with torch.no_grad():
            for batch, (indices, audio, padding, errors) in enumerate(loader):
                failures.extend({"index": index, "row_id": rows[index]["row_id"], "error": error} for index, error in errors)
                if not indices:
                    continue
                encode_batch(indices, audio, padding)
                del audio, padding
                if batch % 100 == 0:
                    features.flush()
                    done.flush()
                    elapsed = time.time() - started
                    write_json(Path(args.out) / "progress.json", {"model": "xlsr2b", "stage": "extract", "split": split,
                               "successful": int(done.sum()), "total": len(rows), "elapsed_seconds": round(elapsed, 1),
                               "clips_per_second": round((int(done.sum()) - before) / max(elapsed, 1e-6), 2)})
                    write_json(failures_path, failures)
                    print("[features %s] %s/%s elapsed=%.0fs" % (split, int(done.sum()), len(rows), elapsed), flush=True)
        features.flush()
        done.flush()
        write_json(cache / (split + ".errors.json"), failures)
        write_json(cache / (split + ".done.json"), {"rows": len(rows), "successful": int(done.sum()), "failed": int((~done).sum()), "elapsed_seconds": round(time.time() - started, 1), "batch_size": args.batch_size})
    timings = {split: json.loads((cache / (split + ".done.json")).read_text()) for split in rows_by_split}
    write_json(Path(args.out) / "extraction_resources.json", {
        "parameters": sum(p.numel() for p in encoder.parameters()), "batch_size": args.batch_size,
        "split_extraction": timings, "extraction_total_seconds": sum(t["elapsed_seconds"] for t in timings.values()),
        "extraction_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
        "extraction_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else None})
    del encoder, models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cache


def metrics(labels, predicted):
    labels, predicted = np.asarray(labels), np.asarray(predicted)
    precision, recall, f1, counts = precision_recall_fscore_support(labels, predicted, labels=list(range(5)), zero_division=0)
    return {"n": len(labels), "top1": float(np.mean(labels == predicted)), "macro_f1": float(np.mean(f1)), "macro_recall": float(np.mean(recall)), "failed": int(np.sum(predicted < 0)), "per_region": {LABELS[i]: {"name_ko": LABELS_KO[i], "n": int(counts[i]), "recall_top1": float(recall[i]), "precision": float(precision[i]), "f1": float(f1[i])} for i in range(5)}, "confusion_labels": LABELS + ["failed"], "confusion_matrix": confusion_matrix(labels, predicted, labels=list(range(5)) + [-1]).tolist()}


def evaluate(head, x, mu, sd, device, ok, batch_size=4096):
    head.eval()
    logits = np.full((len(x), 5), np.nan, np.float32)
    with torch.no_grad():
        indices = np.flatnonzero(ok)
        for at in range(0, len(indices), batch_size):
            index = indices[at:at + batch_size]
            values = (np.asarray(x[index], np.float32) - mu) / sd
            logits[index] = head(torch.from_numpy(values).to(device)).cpu().numpy()
    predictions = np.full(len(x), -1, np.int64)
    predictions[ok] = logits[ok].argmax(1)
    return logits, predictions


def require_person_disjoint_split(args, rows_by_split):
    summary = json.loads((Path(args.manifests) / "summary.json").read_text())
    if summary.get("person_disjoint_verified") is not True or not summary.get("person_id_source"):
        raise RuntimeError("Classifier training requires a verified cross-recording person ID mapping. "
                           "Existing recording-disjoint results are a historical baseline only.")
    manifest_hash = hashlib.sha256()
    for split in ["train", "valid", "test"]:
        manifest_hash.update((Path(args.manifests) / (split + ".csv")).read_bytes())
    if summary.get("manifest_sha256") != manifest_hash.hexdigest():
        raise RuntimeError("Manifest files changed since person-split preparation")
    identities = {}
    for split, rows in rows_by_split.items():
        if any(row.get("person_id") is None or str(row["person_id"]).strip().lower() in ("", "null", "none", "nan") for row in rows):
            raise RuntimeError("Missing person_id in " + split + " manifest")
        if {int(row["label_id"]) for row in rows} != set(range(len(LABELS))):
            raise RuntimeError("All five regions need examples in the " + split + " split")
        identities[split] = {str(row["person_id"]).strip() for row in rows}
    for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        overlap = identities[a] & identities[b]
        if overlap:
            raise RuntimeError("Person leakage: %s/%s share %d person IDs" % (a, b, len(overlap)))
    return summary


def fit(args, cache, rows_by_split, device):
    data_summary = require_person_disjoint_split(args, rows_by_split)
    features = {split: np.load(cache / (split + ".npy"), mmap_mode="r") for split in rows_by_split}
    good = {split: np.load(cache / (split + ".ok.npy")) for split in rows_by_split}
    ys = {split: np.array([int(row["label_id"]) for row in rows], np.int64) for split, rows in rows_by_split.items()}
    if any(not np.any(ys["train"][good["train"]] == i) for i in range(5)):
        raise RuntimeError("All five regions need successful training examples")
    variants = getattr(args, "feature_variants", {
        "layer%d" % LAYERS[0]: (0,),
        "layer%d" % LAYERS[1]: (1,),
        "concat%d_%d" % tuple(LAYERS): (0, 1),
    })
    feature_layers = getattr(args, "feature_layers", LAYERS)
    feature_names = getattr(args, "feature_names", ["layer%d_mean" % layer for layer in LAYERS])
    results = {}
    for name, selected in variants.items():
        torch.manual_seed(args.seed)
        arrays = {split: np.asarray(value[:, selected, :], np.float32).reshape(len(value), -1) for split, value in features.items()}
        train = arrays["train"][good["train"]]
        y = ys["train"][good["train"]]
        mu = train.mean(0)
        sd = np.maximum(train.std(0), 1e-4)
        train = (train - mu) / sd
        head = nn.Linear(train.shape[1], 5).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-3)
        # Each globally identified person receives equal total weight within a region.
        keys = [row["person_id"] for row, ok in zip(rows_by_split["train"], good["train"]) if ok]
        record_counts = Counter(keys)
        region_records = {i: set(key for key, label in zip(keys, y) if label == i) for i in range(5)}
        weights = np.array([1.0 / record_counts[key] / len(region_records[int(label)]) for key, label in zip(keys, y)], np.float32)
        weights /= weights.mean()
        best = (-1.0, -1.0)
        stale = 0
        checkpoint = Path(args.out) / (name + ".pt")
        history = []
        rng = np.random.default_rng(args.seed)
        for epoch in range(args.epochs):
            head.train()
            order = rng.permutation(len(train))
            loss_sum = 0.0
            for start in range(0, len(order), 4096):
                index = order[start:start + 4096]
                inputs = torch.from_numpy(train[index]).to(device)
                labels = torch.from_numpy(y[index]).to(device)
                sample_weights = torch.from_numpy(weights[index]).to(device)
                loss = (F.cross_entropy(head(inputs), labels, reduction="none") * sample_weights).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(index)
            _, predicted = evaluate(head, arrays["valid"], mu, sd, device, good["valid"])
            report = metrics(ys["valid"], predicted)
            history.append({"epoch": epoch, "loss": loss_sum / len(train), "valid": report})
            rank = (report["macro_f1"], report["top1"])
            print("[head %s] epoch=%s valid_top1=%.4f macro_f1=%.4f" % (name, epoch, report["top1"], report["macro_f1"]), flush=True)
            if rank > best:
                best = rank
                stale = 0
                torch.save({"head": {key: value.detach().cpu() for key, value in head.state_dict().items()}, "mu": mu, "sd": sd, "labels": LABELS, "labels_ko": LABELS_KO, "layers": [feature_layers[i] for i in selected], "feature_names": [feature_names[i] for i in selected], "encoder_checkpoint": str(args.encoder), "encoder_frozen": True, "epoch": epoch, "valid": report, "manifest_summary": str(Path(args.manifests) / "summary.json")}, checkpoint)
            else:
                stale += 1
            if stale >= args.patience:
                break
        write_json(Path(args.out) / (name + ".history.json"), history)
        results[name] = {"valid_macro_f1": best[0], "valid_top1": best[1], "checkpoint": str(checkpoint)}
        del arrays, train, head
    selected_name = max(results, key=lambda name: (results[name]["valid_macro_f1"], results[name]["valid_top1"]))
    state = torch.load(results[selected_name]["checkpoint"], map_location="cpu")
    selected = variants[selected_name]
    test = np.asarray(features["test"][:, selected, :], np.float32).reshape(len(features["test"]), -1)
    head = nn.Linear(test.shape[1], 5).to(device)
    head.load_state_dict(state["head"])
    logits, predictions = evaluate(head, test, state["mu"], state["sd"], device, good["test"])
    test_report = metrics(ys["test"], predictions)
    groups = defaultdict(list)
    for index, row in enumerate(rows_by_split["test"]):
        groups[(row["speaker_key"], int(row["label_id"]))].append(index)
    triple_y, triple_p = [], []
    for (_, label), indices in groups.items():
        indices.sort(key=lambda index: float(rows_by_split["test"][index]["start_sec"]))
        for start in range(0, len(indices) - 2, 3):
            triple = indices[start:start + 3]
            triple_y.append(label)
            triple_p.append(int(logits[triple].mean(0).argmax()) if good["test"][triple].all() else -1)
    report = {"selected_variant": selected_name, "selection_rule": "validation macro F1, then validation top1", "validation_variants": results, "test_single_window": test_report, "test_three_utterance_windows_mean_logits": metrics(triple_y, triple_p) if triple_y else None, "data_summary": data_summary, "encoder": {"path": str(args.encoder), "backbone": getattr(args, "backbone", "XLS-R 2B, full 48-layer pretrained encoder"), "frozen": True}, "selected_feature_names": [feature_names[i] for i in selected], "limitations": list(data_summary.get("limitations", [])) + ["Person independence relies on the documented source identity mapping and complete person_id coverage.", "Five classes: Top-5 accuracy is not a useful metric.", "Three-window aggregation is an experiment on annotated utterances, not the NPU production segmentation rule."]}
    resources_path = Path(args.out) / "extraction_resources.json"
    if resources_path.exists():
        report['resources'] = json.loads(resources_path.read_text())
    write_json(Path(args.out) / "report.json", report)
    np.savez(Path(args.out) / "test_predictions.npz", logits=logits, predictions=predictions, labels=ys["test"], ok=good["test"])
    state["dialect_report"] = report
    torch.save(state, Path(args.out) / "best.pt")
    print("[complete] " + json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", default="/data/speech-ai/out/dialect/20261010/manifests")
    parser.add_argument("--encoder", default="/data/speech-ai/models/xlsr2_2B_1000k.pt")
    parser.add_argument("--out", default="/data/speech-ai/out/dialect/20261010/xlsr2b")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    write_json(Path(args.out) / "config.json", vars(args))
    device = torch.device(args.device)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_per_process_memory_fraction(0.5, device=device)
    rows = {split: load_rows(Path(args.manifests) / (split + ".csv")) for split in ["train", "valid", "test"]}
    require_person_disjoint_split(args, rows)
    if any(not value for value in rows.values()):
        raise RuntimeError("All data splits must contain recordings")
    record_sets = {split: {row["record_key"] for row in value} for split, value in rows.items()}
    if any(record_sets[a] & record_sets[b] for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]):
        raise RuntimeError("Recording leakage across splits")
    cache = extract(args, rows, device)
    fit(args, cache, rows, device)


if __name__ == "__main__":
    main()
