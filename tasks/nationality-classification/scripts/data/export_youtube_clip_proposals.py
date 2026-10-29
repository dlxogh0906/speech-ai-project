"""Export and transcribe selected guest clusters for final clip review.

Output is a proposal manifest. A reviewer must create the training manifest
after rejecting mixed speech, wrong speakers and country-name shortcuts.
"""
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r:
        raise RuntimeError("Repository root not found")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse
import csv
import json

from shared.paths import DATA_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=_pl.Path, required=True)
    parser.add_argument("--selection", type=_pl.Path, required=True)
    parser.add_argument("--holdout", type=_pl.Path, required=True)
    parser.add_argument("--out", type=_pl.Path, required=True)
    parser.add_argument("--cache-dir", type=_pl.Path, default=DATA_ROOT / "hf_cache")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    clips_dir = args.out / "clips"
    clips_dir.mkdir(exist_ok=True)
    selected = json.loads(args.selection.read_text(encoding="utf-8"))
    reserved = {r["id"] for r in json.loads(args.holdout.read_text(encoding="utf-8"))}
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo", cache_dir=args.cache_dir)
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3-turbo", cache_dir=args.cache_dir, torch_dtype=dtype).to(device).eval()
    rows, identity = [], {}
    for selection in selected:
        video_id = selection["video_id"]
        if video_id in reserved:
            raise ValueError("Reserved evaluation video: " + video_id)
        speaker = selection["speaker_id"]
        if speaker in identity and identity[speaker] != selection["country"]:
            raise ValueError("Conflicting country labels for " + speaker)
        identity[speaker] = selection["country"]
        review = json.loads((args.reviews / (video_id + ".json")).read_text(encoding="utf-8"))
        if review["source"]["country"] != selection["country"]:
            raise ValueError("Country differs from reviewed source metadata")
        wave, sr = sf.read(review["audio_path"], dtype="float32")
        if sr != 16000 or wave.ndim != 1:
            raise ValueError("Expected mono 16 kHz audio")
        proposals = []
        for index, segment in enumerate(review["segments"]):
            a, b = segment["start"], min(segment["end"], segment["start"] + 8)
            if segment["cluster_id"] not in selection["clusters"] or a < selection["min_start_sec"] or b - a < 3:
                continue
            clip = wave[round(a * sr):round(b * sr)]
            path = clips_dir / (video_id + "_{:04d}.wav".format(index))
            sf.write(path, clip, sr, subtype="PCM_16")
            proposals.append((index, a, b, path, clip))
        for offset in range(0, len(proposals), 8):
            batch = proposals[offset:offset + 8]
            feats = proc.feature_extractor([p[4] for p in batch], sampling_rate=sr,
                                           return_tensors="pt").input_features.to(device, dtype)
            with torch.inference_mode():
                texts = proc.batch_decode(model.generate(feats, language="ko", task="transcribe", max_new_tokens=128),
                                         skip_special_tokens=True)
            for (index, a, b, path, clip), text in zip(batch, texts):
                rows.append({"clip_id": video_id + "_{:04d}".format(index), "audio_path": str(path),
                             "label_en": selection["country"], "speaker": speaker, "domain": "youtube",
                             "task": "RS", "duration_sec": len(clip) / sr,
                             "video_id": video_id, "start_sec": a, "end_sec": b,
                             "transcript": text.strip(), "review_status": "pending"})
            print("[transcribe] {} {}/{}".format(video_id, min(offset + 8, len(proposals)), len(proposals)), flush=True)
    with (args.out / "clip_proposals.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("[done] {} clips / {} selected speakers".format(len(rows), len(identity)), flush=True)


if __name__ == "__main__":
    main()
