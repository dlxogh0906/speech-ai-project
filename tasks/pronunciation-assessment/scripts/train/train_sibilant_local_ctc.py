"""Adapt the original phone decoder to local actual-phone sequences on TRAIN.

Same acoustic inputs and saved 300x16 sampling plan as the negative-coverage
control. The declared encoder tail and existing CTC head update (default two).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src'),
               str(ROOT / 'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import json
import time
import numpy as np
import torch
from transformers import Wav2Vec2BertForCTC
import eval_sibilant_directional as service
from train_sibilant_rank_pilot import sequence_nll, counterfactual_loss
from mdd.sibilant_local_ctc_artifact import validate_depth, make_artifact


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    p, out = json.loads(Path(a.protocol).read_text()), Path(a.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen source changed: ' + path)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'protocol.json').write_text(Path(a.protocol).read_text())
    data = dict(np.load(Path(p['cache']) / 'data.npz'))
    targets = service.read_jsonl(p['targets'])
    plan = np.load(p['batch_plan'])
    if len(targets) != len(data['x']) or plan.shape != (p['steps'], 16):
        raise ValueError('preserved input and sampling-plan identities required')
    rank_weight = p.get('rank_weight', 0.)
    rank_targets = service.read_jsonl(p['rank_targets']) if rank_weight else None
    if rank_targets is not None:
        if len(rank_targets) != len(targets):
            raise ValueError('rank/sequence target identity differs')
        for i, row in enumerate(rank_targets):
            if row['feature_index'] != i or row['uid'] != targets[i]['uid']:
                raise ValueError('rank input order differs')
            rank = row['rank']
            if rank is not None:
                target = targets[i]['target']
                if (target is None or target['ids'][rank['position']] != rank['actual_id']
                        or rank['actual_id'] == rank['alternate_id']):
                    raise ValueError('rank target does not flip the actual observed phone')
    model = Wav2Vec2BertForCTC.from_pretrained(p['checkpoint']).to('cuda').eval()
    model.requires_grad_(False)
    depth = validate_depth(p.get('encoder_train_layers', 2), len(model.wav2vec2_bert.encoder.layers))
    tail = model.wav2vec2_bert.encoder.layers[-depth:]
    tail.requires_grad_(True)
    model.lm_head.requires_grad_(True)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW([{'params': tail.parameters(), 'lr': p['encoder_lr']},
                                  {'params': model.lm_head.parameters(), 'lr': p['head_lr']}], weight_decay=.01)
    torch.set_num_threads(2)
    blank = json.loads((Path(p['checkpoint']) / 'jamo_vocab.json').read_text())['blank']
    losses, ctc_losses, rank_losses, gaps, skipped, used = [], [], [], [], 0, set()
    rank_occurrences, active_hinges = 0, 0
    tick = time.monotonic()
    for step, original in enumerate(plan, 1):
        ix = [int(i) for i in original if targets[int(i)]['target'] is not None]
        skipped += len(original) - len(ix)
        if not ix:
            raise ValueError('entire CTC batch is unavailable')
        used.update(ix)
        x = torch.as_tensor(data['x'][ix], device=device)
        length = torch.as_tensor(data['lengths'][ix], device=device)
        mask = torch.arange(x.shape[1], device=device)[None, :] < length[:, None]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_features=x, attention_mask=mask.long()).logits
        lp = logits.float().log_softmax(-1)
        sequences, clips = [], []
        for k, i in enumerate(ix):
            target = targets[i]['target']
            left, right = target['roi']
            clips.append(lp[k, left:right])
            sequences.append(target['ids'])
        padded = torch.nn.utils.rnn.pad_sequence(clips, batch_first=True)
        sizes = torch.tensor([len(c) for c in clips], device=device)
        if rank_targets is not None:
            candidates = [(k, rank_targets[i]['rank']['position'], rank_targets[i]['rank']['alternate_id'])
                          for k, i in enumerate(ix) if rank_targets[i]['rank'] is not None]
            ctc, rank, meta = counterfactual_loss(padded, sizes, sequences, candidates, blank, p['rank_margin'])
            rank_occurrences += meta['candidate_count']
            active_hinges += meta['active_count']
            if meta['mean_gap'] is not None:
                gaps.append(meta['mean_gap'])
        else:
            nll = sequence_nll(padded, sizes, sequences, blank)
            denominator = torch.tensor([len(s) for s in sequences], device=device)
            ctc = (nll / denominator).mean()
            rank = ctc * 0
        loss = ctc + rank_weight * rank
        if not bool(torch.isfinite(loss)):
            raise ValueError('nonfinite local-sequence CTC loss')
        loss.backward()
        if step == 1 and any(v.grad is not None for layer in model.wav2vec2_bert.encoder.layers[:-depth] for v in layer.parameters()):
            raise AssertionError('earlier encoder layer updated')
        torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad], 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        ctc_losses.append(float(ctc.detach()))
        rank_losses.append(float(rank.detach()))
        if step % 100 == 0:
            print('step', step, 'loss', float(np.mean(losses[-50:])),
                  'CTC', float(np.mean(ctc_losses[-50:])), 'rank', float(np.mean(rank_losses[-50:])), flush=True)
    artifact = make_artifact(tail, model.lm_head, depth, service.sha(Path(p['checkpoint']) / 'jamo_vocab.json'),
                             'CTC_plus_actual_phone_flip_rank' if rank_weight else 'local_CTC')
    torch.save(artifact, out / 'local_ctc.pt')
    report = {'scope': p['scope'], 'steps': len(plan), 'encoder_train_layers': depth,
              'unique_sampled': len(used), 'skipped_draws': skipped,
              'loss_first_last50': [float(np.mean(losses[:50])), float(np.mean(losses[-50:]))],
              'ctc_loss_first_last50': [float(np.mean(ctc_losses[:50])), float(np.mean(ctc_losses[-50:]))],
              'rank_loss_first_last50': [float(np.mean(rank_losses[:50])), float(np.mean(rank_losses[-50:]))],
              'rank_weight': rank_weight, 'rank_margin': p.get('rank_margin'),
              'rank_occurrences': rank_occurrences, 'active_hinges': active_hinges,
              'actual_vs_flipped_gap_first_last50': [float(np.mean(gaps[:50])), float(np.mean(gaps[-50:]))] if gaps else None,
              'seconds': time.monotonic() - tick, 'peak_GPU_bytes': torch.cuda.max_memory_allocated(),
              'artifact_sha256': service.sha(out / 'local_ctc.pt'), 'deployment': False}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / 'finished.json').write_text(json.dumps({'sha256': service.sha(out / 'report.json')}) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
