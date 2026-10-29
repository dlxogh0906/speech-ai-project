"""Score the sealed test split of the content-grouped full run once (2026-10-26, user request).

Rebuilds each trained model exactly as validation did (generic encoder + saved encoder_delta + head + mu/sd),
first re-scores validation and checks it against the saved validation predictions, then scores test.
The ensemble rule is the one already selected on validation (comparison_report.json 'selected'); nothing is
re-selected on test. All candidates are also scored for reference only. Outputs go to <root>/sealed_test_<date>/.
Only aggregate metrics are written or printed.

    PYTHONPATH=/data/speech-ai/tools/dialect_compare_py38:/data/speech-ai/tools/hf_py38 \
        $PY tasks/dialect-classification/scripts/eval/score_sealed_test.py --root /data/speech-ai/out/dialect/content_grouped_full_20261015
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import time

TRAIN = Path(__file__).resolve().parents[1] / 'train'
sys.path.insert(0, str(TRAIN))


def load_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def run(model, rows_by_split, ck_path, device, out, batch_size=16):
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    import train_dialect_comparison as engine
    encoder = engine.Encoder(model, device)
    if model == 'xlsr2b':
        from fairseq.modules.checkpoint_activations import unwrap_checkpoint
        unwrap_checkpoint(encoder.model)
    ck = torch.load(ck_path, map_location='cpu')
    params = dict(encoder.model.named_parameters())
    missing = [k for k in ck['encoder_delta'] if k not in params]
    assert not missing, missing[:3]
    with torch.no_grad():
        for name, value in ck['encoder_delta'].items():
            params[name].copy_(value.to(device))
    selected = [encoder.names.index(name) for name in ck['feature_names']]
    mu = torch.as_tensor(ck['mu'], device=device); sd = torch.as_tensor(ck['sd'], device=device)
    head = nn.Linear(len(ck['mu']), 6).to(device); head.load_state_dict(ck['head'])
    encoder.model.eval(); head.eval()
    result = {}
    for split, rows in rows_by_split.items():
        t0 = time.time()
        data = engine.PreparedClips(rows, model)
        order = sorted(range(len(rows)), key=lambda i: float(rows[i]['duration_sec']))
        loader = DataLoader(Subset(data, order), batch_size=batch_size, num_workers=4, collate_fn=engine.Collator(model))
        logits = np.full((len(rows), 6), np.nan, np.float32)
        with torch.no_grad():
            for idx, values, lengths, errors in loader:
                assert not errors, '%d unreadable windows' % len(errors)
                feature = encoder(values, lengths)[:, selected, :].reshape(len(values), -1).half().float()
                logits[idx] = head((feature - mu) / sd).cpu().numpy()
        assert np.isfinite(logits).all()
        np.savez_compressed(out / ('%s_%s_logits.npz' % (model, split)), logits=logits, row_id=np.array([r['row_id'] for r in rows]))
        result[split] = logits
        print('[score] %s %s %d windows %.0fs' % (model, split, len(rows), time.time() - t0), flush=True)
    del encoder, head
    torch.cuda.empty_cache()
    return result


def softmax(logits):
    import numpy as np
    p = np.exp(logits - logits.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def candidates(a, b):
    """The same candidate grid run_content_grouped_full.summarize() registered before training."""
    import numpy as np
    out = dict(eat_large=a, xlsr2b=b)
    for alpha in [.25, .5, .75]:
        out['probability_eat_' + str(alpha)] = alpha * a + (1 - alpha) * b
        gate = a.argmax(1) == 5
        conditional = alpha * a[:, :5] / a[:, :5].sum(1, keepdims=True) + (1 - alpha) * b[:, :5] / b[:, :5].sum(1, keepdims=True)
        cascade = np.zeros_like(a)
        cascade[:, :5] = conditional
        cascade[gate, :5] = 0
        cascade[gate, 5] = 1
        out['eat_standard_gate_regional_eat_' + str(alpha)] = cascade
    return out


def main(args):
    import numpy as np
    import torch
    from train_standard_six_class import metrics
    root = args.root
    report = json.loads((root / 'comparison_report.json').read_text())
    assert report['test_scored'] is False
    summary = json.loads((root / 'manifests/summary.json').read_text())
    rows = {s: load_rows(root / 'manifests' / (s + '.csv')) for s in ['valid', 'test']}
    out = root / ('sealed_test_' + datetime.now().strftime('%Y%m%d'))
    out.mkdir(exist_ok=True)
    device = torch.device('cuda:0')
    logits, checks = {}, {}
    for model in ['eat_large', 'xlsr2b']:
        logits[model] = run(model, rows, root / model / 'best.pt', device, out)
        saved = np.load(root / model / 'valid_predictions.npz')
        assert np.array_equal(saved['row_id'], np.array([r['row_id'] for r in rows['valid']]))
        checks[model] = dict(valid_argmax_agreement=float((saved['logits'].argmax(1) == logits[model]['valid'].argmax(1)).mean()),
                             valid_max_abs_logit_diff=float(np.abs(saved['logits'] - logits[model]['valid']).max()),
                             saved_valid=report['models'][model]['valid']['top1'],
                             rescored_valid=metrics(rows['valid'], logits[model]['valid'])['top1'])
        print('[check] %s %s' % (model, json.dumps(checks[model])), flush=True)
    res = {}
    for split in ['valid', 'test']:
        c = candidates(softmax(logits['eat_large'][split]), softmax(logits['xlsr2b'][split]))
        res[split] = {name: metrics(rows[split], p) for name, p in c.items()}
    selected = report['selected']
    final = dict(scored_at=datetime.now().astimezone().isoformat(), selected=selected,
                 selection_policy=report['selection_policy'] + '; selection fixed before opening test',
                 test=res['test'][selected], valid_rescored=res['valid'][selected], valid_reported=report['valid'],
                 test_all_candidates_reference_only={k: dict(top1=v['top1'], macro_f1=v['macro_f1']) for k, v in res['test'].items()},
                 single_models_test={k: res['test'][k] for k in ['eat_large', 'xlsr2b']},
                 reproduction_checks=checks, manifest_sha256=summary['manifest_sha256'], test_rows=len(rows['test']))
    (out / 'sealed_test_report.json').write_text(json.dumps(final, ensure_ascii=False, indent=2) + '\n')
    t = final['test']
    print('[test] %s top1 %.4f macro_f1 %.4f' % (selected, t['top1'], t['macro_f1']), flush=True)
    print('SEALED_TEST_DONE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args())
