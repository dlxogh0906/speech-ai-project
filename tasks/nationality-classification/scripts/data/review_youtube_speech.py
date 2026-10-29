"""Download selected reusable sources and propose Korean speaker clusters.

This produces review material, never automatic nationality training labels.
Cluster transcripts identify guests/hosts; energy VAD and language scores are
proposals, not ground truth. Sources must provide their provenance and license.
"""
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r:
        raise RuntimeError("Repository root not found")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse
import json
import os
import subprocess

from shared.paths import DATA_ROOT, NAT_OUTPUTS


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def speech_proposals(wave, sr):
    import numpy as np
    frame = int(.03 * sr)
    count = wave.size // frame
    if count < 5:
        return []
    energy = 10 * np.log10(np.square(wave[:count * frame].reshape(count, frame)).mean(1) + 1e-10)
    threshold = max(np.percentile(energy, 15) + 10, np.percentile(energy, 95) - 35, -55)
    active = np.convolve((energy > threshold).astype(float), np.ones(5) / 5, mode="same") > .4
    spans, start = [], None
    for index, on in enumerate(np.append(active, False)):
        if on and start is None:
            start = index
        elif not on and start is not None:
            spans.append([start, index])
            start = None
    merged = []
    for start, end in spans:
        if merged and start - merged[-1][1] <= 12:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    proposals = []
    for start, end in merged:
        a, b = start * frame / sr, end * frame / sr
        while b - a > 10:
            proposals.append((a, a + 8))
            a += 8
        if b - a >= 1.5:
            proposals.append((a, b))
    return proposals


def silero_proposals(wave, sr, model):
    """Neural VAD for interviews whose music makes energy VAD unreliable."""
    import torch
    from silero_vad import get_speech_timestamps
    spans = get_speech_timestamps(torch.from_numpy(wave.copy()), model,
                                 sampling_rate=sr, threshold=.6,
                                 min_speech_duration_ms=750,
                                 min_silence_duration_ms=200,
                                 max_speech_duration_s=8,
                                 speech_pad_ms=100)
    return [(r["start"] / sr, r["end"] / sr) for r in spans
            if (r["end"] - r["start"]) / sr >= 1.5]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=_pl.Path, required=True)
    parser.add_argument("--out", type=_pl.Path, default=NAT_OUTPUTS / "youtube_pilot/speech_review")
    parser.add_argument("--cache-dir", type=_pl.Path, default=DATA_ROOT / "hf_cache")
    parser.add_argument("--max-sec", type=float, default=600)
    parser.add_argument("--cluster-distance", type=float, default=.2)
    parser.add_argument("--vad", choices=("energy", "silero"), default="energy")
    parser.add_argument("--transcribe-all", action="store_true")
    parser.add_argument("--min-korean-prob", type=float, default=.5,
                        help="Language proposal threshold; zero keeps accented speech for manual review")
    parser.add_argument("--format", default="bestaudio[ext=m4a]/bestaudio/best")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.min_korean_prob <= 1:
        parser.error("Language proposal threshold must be between zero and one")
    args.out.mkdir(parents=True, exist_ok=True)
    source_dir = args.out / "sources"
    source_dir.mkdir(exist_ok=True)
    rows = json.loads(args.sources.read_text(encoding="utf-8"))
    import yt_dlp
    import soundfile as sf
    import numpy as np
    import torch
    torch.set_num_threads(4)
    vad_model = None
    if args.vad == "silero":
        from silero_vad import load_silero_vad
        vad_model = load_silero_vad()
    from sklearn.cluster import AgglomerativeClustering
    from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                              AutoFeatureExtractor, WavLMForXVector)
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo", cache_dir=args.cache_dir)
    whisper = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3-turbo", cache_dir=args.cache_dir, torch_dtype=dtype).to(device).eval()
    frontend = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv", cache_dir=args.cache_dir)
    speaker_model = WavLMForXVector.from_pretrained(
        "microsoft/wavlm-base-plus-sv", cache_dir=args.cache_dir).to(device).eval()
    languages = list(whisper.generation_config.lang_to_id)
    language_ids = torch.tensor([whisper.generation_config.lang_to_id[lang] for lang in languages], device=device)
    ko_index = languages.index("<|ko|>")
    start_token = whisper.generation_config.decoder_start_token_id
    sr = 16000

    @torch.no_grad()
    def transcribe(clips):
        feats = processor.feature_extractor(clips, sampling_rate=sr, return_tensors="pt").input_features.to(device, dtype)
        texts = processor.batch_decode(whisper.generate(feats, language="ko", task="transcribe", max_new_tokens=128),
                                       skip_special_tokens=True)
        return [text.strip() for text in texts]

    @torch.no_grad()
    def korean_prob(clips):
        values = []
        for offset in range(0, len(clips), 8):
            feats = processor.feature_extractor(clips[offset:offset + 8], sampling_rate=sr,
                                                return_tensors="pt").input_features.to(device, dtype)
            enc = whisper.model.encoder(feats).last_hidden_state
            decoder = torch.full((feats.shape[0], 1), start_token, device=device)
            logits = whisper(encoder_outputs=(enc,), decoder_input_ids=decoder).logits[:, -1, :]
            values.append(torch.softmax(logits[:, language_ids].float(), -1)[:, ko_index].cpu().numpy())
        return np.concatenate(values) if values else np.zeros(0)

    @torch.no_grad()
    def embedding(clips):
        values = []
        for clip in clips:
            inputs = frontend([clip], sampling_rate=sr, return_tensors="pt").to(device)
            vec = speaker_model(**inputs).embeddings[0]
            values.append(torch.nn.functional.normalize(vec, dim=-1).cpu().numpy())
        return np.stack(values)

    for number, row in enumerate(rows, 1):
        video_id = row["video_id"]
        report_path = args.out / (video_id + ".json")
        if report_path.exists():
            print("[resume] " + video_id, flush=True)
            continue
        if "creative commons" not in str(row.get("license", "")).lower():
            raise ValueError("Source-use evidence needs review for " + video_id)
        print("[source {}/{}] {} {}".format(number, len(rows), video_id, row["country"]), flush=True)
        wav_path = source_dir / (video_id + ".wav")
        if not wav_path.exists():
            opts = {"quiet": True, "no_warnings": True, "noprogress": True, "format": args.format, "noplaylist": True,
                    "socket_timeout": 15, "retries": 2,
                    "outtmpl": str(source_dir / (video_id + ".%(ext)s"))}
            with yt_dlp.YoutubeDL(opts) as downloader:
                info = downloader.extract_info(row["url"], download=True)
                if "creative commons" not in str(info.get("license", "")).lower():
                    raise ValueError("License changed or unavailable for " + video_id)
                media_path = downloader.prepare_filename(info)
            env = os.environ.copy()
            env.pop("LD_LIBRARY_PATH", None)
            subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", media_path,
                            "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(wav_path)], check=True, env=env)
        wave, sample_rate = sf.read(wav_path, frames=int(args.max_sec * sr), dtype="float32")
        if sample_rate != sr or wave.ndim != 1:
            raise ValueError("Expected mono 16 kHz audio")
        spans = (silero_proposals(wave, sr, vad_model) if vad_model is not None
                 else speech_proposals(wave, sr))
        clips = [wave[int(a * sr):int(b * sr)] for a, b in spans]
        probabilities = korean_prob(clips)
        indices = [i for i, p in enumerate(probabilities) if p >= args.min_korean_prob]
        if len(indices) > 1:
            vectors = embedding([clips[i] for i in indices])
            cluster_ids = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                                 distance_threshold=args.cluster_distance).fit_predict(vectors)
            np.savez(args.out / (video_id + "_speaker_embeddings.npz"),
                     vectors=vectors, indices=np.asarray(indices))
        else:
            cluster_ids = np.zeros(len(indices), dtype=int)
        all_texts = {}
        if args.transcribe_all:
            for offset in range(0, len(indices), 8):
                current = indices[offset:offset + 8]
                all_texts.update(zip(current, transcribe([clips[i] for i in current])))
            print("  transcribed {} segments".format(len(all_texts)), flush=True)
        clustered = {}
        for index, cluster_id in zip(indices, cluster_ids):
            clustered.setdefault(int(cluster_id), []).append(index)
        summaries = []
        for cluster_id, member_indices in sorted(clustered.items()):
            duration = sum(spans[i][1] - spans[i][0] for i in member_indices)
            if duration < 5:
                continue
            # Three positions expose mixed speakers more often than a single sample.
            sample_indices = sorted({member_indices[0], member_indices[len(member_indices) // 2], member_indices[-1]})
            texts = ([all_texts[i] for i in sample_indices] if args.transcribe_all
                     else transcribe([clips[i] for i in sample_indices]))
            summary = {"cluster_id": cluster_id, "duration_sec": round(duration, 2),
                       "samples": [{"start": round(spans[i][0], 2), "end": round(spans[i][1], 2),
                                    "transcript": text} for i, text in zip(sample_indices, texts)]}
            summaries.append(summary)
            print("  cluster {} {:.0f}s: {}".format(cluster_id, duration, " / ".join(texts)), flush=True)
        segments = [{"start": round(spans[i][0], 2), "end": round(spans[i][1], 2),
                     "korean_probability": round(float(probabilities[i]), 4), "cluster_id": int(cluster_id),
                     **({"transcript": all_texts[i]} if i in all_texts else {})}
                    for i, cluster_id in zip(indices, cluster_ids)]
        save_json(report_path, {"source": row, "audio_path": str(wav_path), "analyzed_sec": len(wave) / sr,
                                "cluster_distance": args.cluster_distance, "vad": args.vad,
                                "min_korean_probability": args.min_korean_prob, "clusters": summaries,
                                "segments": segments, "review_status": "needs_guest_selection"})
        print("[saved] " + str(report_path), flush=True)


if __name__ == "__main__":
    main()
