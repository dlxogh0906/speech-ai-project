# XLS-R Nationality Inference Bundle

Inference-only bundle for the 31-class nationality/language-area classifier.

This folder does not contain training code. It runs:

```text
wav -> mono/16 kHz -> XLS-R -> mean pooling -> classifier ensemble -> top-k
```

## Files

```text
xlsr_nationality_inference_bundle/
  run_inference.py
  xlsr_nationality_inference.ipynb
  requirements.txt
  environment.yml
  labels/
    class_labels.csv
    label_map.json
    LABEL_POLICY.md
    final_label_composition.csv
    original_label_to_final_label.csv
    original_label_to_final_label.md
  checkpoints/
    linear_best_portable.pt
    mlp30_best_portable.pt
    mlp60_best_portable.pt
```

The HuggingFace model `facebook/wav2vec2-xls-r-2b` is not included. The first run
downloads it unless `--local_files_only` is used.

## Setup

Recommended:

```bash
cd $PROJECT_ROOT/xlsr_nationality_inference_bundle
conda env create -f environment.yml
conda activate xlsr_hf
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

If conda CUDA packages do not match the RTX 4060 Ti machine, install PyTorch from
the official selector, then install the remaining packages:

```bash
pip install -r requirements.txt
```

## Class Labels

The classifier has 31 output classes. The output index order is fixed by:

```text
labels/class_labels.csv
labels/label_map.json
```

Print the class order from the checkpoint:

```bash
python run_inference.py --print_labels
```

The full original-label to final-label policy is in:

```text
labels/LABEL_POLICY.md
labels/final_label_composition.csv
labels/original_label_to_final_label.csv
```

## Single Wav Inference

Full wav inference:

```bash
python run_inference.py \
  --audio /path/to/sample.wav \
  --output_dir outputs/single_full
```

Match the current trained classifier feature condition, which used the first
4 seconds:

```bash
python run_inference.py \
  --audio /path/to/sample.wav \
  --crop_seconds 4 \
  --output_dir outputs/single_crop4
```

Use chunks and average logits:

```bash
python run_inference.py \
  --audio /path/to/sample.wav \
  --chunk_seconds 4 \
  --output_dir outputs/single_chunk4
```

## Latency Benchmark

Pick samples near target durations from the final train/val/test manifests:

```bash
python run_inference.py \
  --benchmark \
  --manifest_csv \
    $PROJECT_ROOT/outputs/fine_grained_nationality_language_area31/manifests/test.csv \
    $PROJECT_ROOT/outputs/fine_grained_nationality_language_area31/manifests/val.csv \
    $PROJECT_ROOT/outputs/fine_grained_nationality_language_area31/manifests/train.csv \
  --target_seconds 5 10 15 30 45 60 120 \
  --repeats 3 \
  --output_dir outputs/latency_full
```

Benchmark first 4 seconds only:

```bash
python run_inference.py \
  --benchmark \
  --manifest_csv \
    $PROJECT_ROOT/outputs/fine_grained_nationality_language_area31/manifests/test.csv \
    $PROJECT_ROOT/outputs/fine_grained_nationality_language_area31/manifests/val.csv \
    $PROJECT_ROOT/outputs/fine_grained_nationality_language_area31/manifests/train.csv \
  --target_seconds 5 10 15 30 45 60 120 \
  --repeats 3 \
  --crop_seconds 4 \
  --output_dir outputs/latency_crop4
```

Outputs:

```text
latest_inference_result.json
latency_benchmark_runs.csv
latency_benchmark_summary.csv
latency_benchmark_summary.md
```

## Timing Meaning

The script reports:

```text
audio_load_resample_sec
feature_extractor_sec
xlsr_forward_pool_sec
classifier_ensemble_sec
request_total_sec
```

`request_total_sec` is warm inference time after the model is already loaded on
GPU. It does not include web upload, database writes, or network latency.

## Important Note

The existing final classifier was trained/evaluated on cached XLS-R features
generated from the first 4 seconds of each wav. Use `--crop_seconds 4` when you
want the inference condition to match that experiment.

Use full wav or chunk inference when testing a service-oriented policy.
