"""Cache adapted XLSR features for a reproducible six-class experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

import train_dialect_comparison as engine
from train_standard_six_class import LABELS, load_rows, manifest_hash, write_json

ROOT = Path('/data/speech-ai/out/dialect')
BASE = ROOT / 'expanded_six_class_20261013'
REGIONAL = ROOT / 'finetune_speakerid_20261012/xlsr2b/best_adapted.pt'


def verify_data(manifests):
    summary = json.loads((manifests / 'summary.json').read_text())
    if manifest_hash(manifests) != summary['manifest_sha256']:
        raise RuntimeError('Manifest digest mismatch')
    rows = {s: load_rows(manifests / (s + '.csv')) for s in ['train', 'valid', 'test']}
    for split, values in rows.items():
        if len({r['row_id'] for r in values}) != len(values):
            raise RuntimeError('Duplicate row IDs: ' + split)
        if {int(r['label_id']) for r in values} != set(range(6)):
            raise RuntimeError('Expected exactly six classes')
        if any(not r['person_id'] or not Path(r['audio_path']).is_file() for r in values):
            raise RuntimeError('Missing person identity or audio')
    for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]:
        for key in ['person_id', 'record_key']:
            if {r[key] for r in rows[a]} & {r[key] for r in rows[b]}:
                raise RuntimeError('Split overlap: ' + key)
    for split in ['valid', 'test']:
        if (manifests / (split + '.csv')).read_bytes() != (BASE / 'manifests' / (split + '.csv')).read_bytes():
            raise RuntimeError('Held-out rows differ from the comparison baseline')
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'xlsr_six_class_20261014')
    parser.add_argument('--batch-size', type=int, default=48)
    args = parser.parse_args()
    args.model, args.workers = 'xlsr2b', 4
    args.manifests = BASE / 'manifests'
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status = args.out / 'extraction_status.json'
    write_json(status, {'status': 'running', 'stage': 'verify_data', 'pid': os.getpid()})
    try:
        rows, summary = verify_data(args.manifests)
        state = torch.load(REGIONAL, map_location='cpu')
        if state['data_summary']['manifest_sha256'] != summary['original_regional_manifest_sha256']:
            raise RuntimeError('Regional backbone uses another speaker split')
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.55, device=0)
        encoder = engine.Encoder('xlsr2b', torch.device('cuda:0'))
        from fairseq.modules.checkpoint_activations import unwrap_checkpoint
        unwrap_checkpoint(encoder.model)
        params = dict(encoder.model.named_parameters())
        with torch.no_grad():
            for name, value in state['encoder_delta'].items():
                params[name].copy_(value.to(encoder.device))
        encoder.info.update(adaptation_checkpoint=str(REGIONAL),
                            adaptation_sha256=hashlib.sha256(REGIONAL.read_bytes()).hexdigest())
        write_json(args.out / 'data_verified.json', {'counts': {s: len(v) for s, v in rows.items()},
                   'labels': LABELS, 'person_and_record_disjoint': True,
                   'manifest_sha256': summary['manifest_sha256'], 'summary': summary})
        # Held-out test audio is not encoded until all new model selection is finished.
        cache, train_resources = engine.extract(args, {'train': rows['train']}, encoder)
        # Match the previous cascade's evaluation batches. Padding length and
        # FP16 kernels can alter borderline predictions when batch size changes.
        valid_args = argparse.Namespace(**vars(args))
        valid_args.batch_size = 16
        marker = cache / 'valid.done.json'
        if marker.exists() and json.loads(marker.read_text())['batch_size'] != 16:
            raise RuntimeError('Archive the old validation cache before rebuilding at batch size 16')
        cache, valid_resources = engine.extract(valid_args, {'valid': rows['valid']}, encoder)
        resources = dict(train_resources)
        resources['split_extraction'].update(valid_resources['split_extraction'])
        resources['extraction_total_seconds'] = sum(v['elapsed_seconds'] for v in resources['split_extraction'].values())
        resources['validation_batch_size'] = 16
        write_json(args.out / 'extraction_resources.json', resources)
        for split in ['train', 'valid']:
            if not np.load(cache / (split + '.ok.npy')).all():
                raise RuntimeError('Incomplete extraction: ' + split)
        indices = [encoder.names.index(n) for n in state['feature_names']]
        features = np.asarray(np.load(cache / 'valid.npy')[:, indices, :], np.float32).reshape(len(rows['valid']), -1)
        logits = ((features - state['mu']) / state['sd']) @ state['head']['weight'].numpy().T + state['head']['bias'].numpy()
        previous = np.load(BASE / 'xlsr_cascade/xlsr_valid_logits.npz')
        if not np.array_equal(previous['row_id'], [r['row_id'] for r in rows['valid']]):
            raise RuntimeError('Previous prediction row mismatch')
        agreement = float(np.mean(logits.argmax(1) == previous['logits'].argmax(1)))
        difference = np.abs(logits - previous['logits'])
        diagnostics = {'agreement': agreement, 'validation_batch_size': 16,
                       'changed_row_indices': np.flatnonzero(logits.argmax(1) != previous['logits'].argmax(1)).tolist(),
                       'absolute_logit_difference_median': float(np.median(difference)),
                       'absolute_logit_difference_p99': float(np.quantile(difference, .99)),
                       'absolute_logit_difference_max': float(difference.max())}
        write_json(args.out / 'reproduction_check.json', diagnostics)
        if agreement < .999:
            raise RuntimeError('Adapted XLSR cache fails reproduction: ' + str(agreement))
        write_json(status, {'status': 'complete', 'elapsed_seconds': time.time() - started,
                   'previous_validation_prediction_agreement': agreement, 'resources': resources,
                   'reproduction_check': diagnostics})
    except Exception as exc:
        write_json(status, {'status': 'failed', 'error': type(exc).__name__ + ': ' + str(exc)})
        raise


if __name__ == '__main__':
    main()
