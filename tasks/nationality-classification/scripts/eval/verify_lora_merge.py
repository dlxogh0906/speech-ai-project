"""Compare adapter and merged-weight inference on real audio before NPU export."""
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r:
        raise RuntimeError("Repository root not found")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
_sys.path.insert(0, str(_r / "tasks/nationality-classification/scripts/train"))

import argparse
import json

import numpy as np

from pilot_youtube_head import read_audio, read_rows, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-deploy", type=_pl.Path, required=True)
    parser.add_argument("--experiment-dir", type=_pl.Path, required=True)
    parser.add_argument("--manifest", type=_pl.Path, required=True)
    args = parser.parse_args()
    import torch
    from country_classifier.npu_encoder import load_pure, conv_out_len
    from country_classifier.encoder_adaptation import attach_lora
    from country_classifier.wav_augment import silence_normalize
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_rows(args.manifest)
    selected = [next(row for row in rows if row["label_en"] == country)
                for country in ("Uzbekistan", "Italy", "Myanmar")]
    original = [read_audio(row["audio_path"]) for row in selected]
    durations = (2, 6, 12)
    source = np.zeros((len(selected), 192000), np.float32)
    valid = np.zeros((len(selected), conv_out_len(192000)), np.float32)
    for index, (wave, duration) in enumerate(zip(original, durations)):
        wave = np.tile(wave, int(np.ceil(duration * 16000 / len(wave))))[:duration * 16000]
        wave = silence_normalize(wave)[:192000]
        source[index, :len(wave)] = wave
        valid[index, :conv_out_len(len(wave))] = 1
    source, valid = torch.from_numpy(source).to(device), torch.from_numpy(valid).to(device)
    results = {}
    for arm in ("replay", "lora_youtube", "lora_youtube_aux"):
        folder = args.experiment_dir / arm
        adapter = torch.load(folder / "adapters.pt", map_location="cpu")
        if adapter["source"]["encoder"] != sha(args.base_deploy / "encoder_pure_l17.pt"):
            raise ValueError("Adapter base encoder differs")
        model = load_pure(args.base_deploy / "encoder_pure_l17.pt").to(device)
        attach_lora(model, adapter["from_layer"], adapter["rank"], adapter["alpha"])
        unexpected = model.load_state_dict(adapter["adapters"], strict=False).unexpected_keys
        if unexpected:
            raise ValueError("Unknown adapter tensors")
        with torch.inference_mode():
            pooled_adapter = model(source, valid).cpu().numpy()
        del model
        merged = load_pure(folder / "deploy/encoder_pure_l17.pt").to(device)
        with torch.inference_mode():
            pooled_merged = merged(source, valid).cpu().numpy()
        del merged
        with np.load(folder / "deploy/head_np.npz", allow_pickle=True) as head:
            first = ((pooled_adapter - head["mu"]) / head["sd"]) @ head["W"].T + head["b"]
            second = ((pooled_merged - head["mu"]) / head["sd"]) @ head["W"].T + head["b"]
        error = float(np.abs(pooled_adapter - pooled_merged).max())
        if error > 2e-4 or not np.array_equal(first.argmax(1), second.argmax(1)):
            raise ValueError("LoRA merge changed inference")
        results[arm] = {"max_abs_pooled_difference": error,
                        "max_abs_logit_difference": float(np.abs(first - second).max()),
                        "matching_top1": int((first.argmax(1) == second.argmax(1)).sum()), "n": len(selected)}
        print(arm, results[arm], flush=True)
    report = {"input": "real speech repeated/cropped to check 2/6/12-second masks; not an accuracy evaluation",
              "source_manifest": str(args.manifest), "variants": results}
    (args.experiment_dir / "merge_verification.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
