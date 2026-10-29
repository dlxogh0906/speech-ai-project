"""Cache frozen encoder features while a verified person split is unavailable.

This entry point never fits a classifier. Cache row IDs remain attached to the
historical manifests so features can later be explicitly remapped by row_id.
"""
import argparse
import gc
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("dialect_comparison", Path(__file__).with_name("train_dialect_comparison.py"))
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["w2vbert2", "wavlm_large", "eat_large"], required=True)
    parser.add_argument("--manifests", default="/data/speech-ai/out/dialect/20261010/manifests")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    engine.heads.write_json(Path(args.out) / "features_only_config.json", vars(args))
    engine.torch.cuda.set_per_process_memory_fraction(0.6, device=0)
    rows = {s: engine.heads.load_rows(Path(args.manifests) / (s + ".csv")) for s in ["train", "valid", "test"]}
    engine.progress(args, "load_encoder_for_features_only")
    encoder = engine.Encoder(args.model, engine.torch.device("cuda:0"))
    timing_path = Path(args.out) / "benchmark.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else engine.benchmark(args, rows["valid"], encoder)
    cache, resources = engine.extract(args, rows, encoder)
    engine.heads.write_json(Path(args.out) / "features_only_complete.json", {
        "classifier_trained": False, "source_manifests": str(Path(args.manifests).resolve()),
        "features": str(cache), "encoder_metadata": encoder.info,
        "preprocessing": encoder.preprocessing, "resources": resources, "benchmark": timing,
        "next_step": "Use provider person IDs to rebuild splits, explicitly remap cached rows by row_id, then fit heads.",
    })
    del encoder
    gc.collect()
    engine.torch.cuda.empty_cache()
    engine.progress(args, "features_complete_awaiting_person_split")


if __name__ == "__main__":
    main()
