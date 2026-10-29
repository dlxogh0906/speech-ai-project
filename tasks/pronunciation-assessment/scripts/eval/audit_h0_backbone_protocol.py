"""Audit the existing H0 inputs before a controlled backbone comparison.

Read-only with respect to corpora, checkpoints and historical results. The new
output must not exist. Text overlap is a corpus-membership check, not proof of
optimizer exposure or of exposure during public SSL pretraining.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/scripts/train')]

import argparse
import collections
import hashlib
import json
import os
import re
import shutil
from train_w2vbert_ctc import batches_by_frames


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with open(path, encoding='utf-8') as f:
        root = f.readline().strip()
        return [(os.path.join(root, p), int(n))
                for p, n in (line.rstrip().split('\t') for line in f)]


def calibration_role(speaker):
    return ('calibration', 'selection')[int(hashlib.sha256(
        ('h0_backbone_v1:' + speaker).encode()).hexdigest()[:8], 16) % 2]


def exposure(rows, sources, checkpoints, accum=4, budget=1600000, seed=42):
    """Continuous microbatch stream; no loss/reset at an epoch boundary."""
    updates, micro, epoch = 0, 0, 0
    counts, samples = collections.Counter(), collections.Counter()
    unique = collections.defaultdict(set)
    digest, result = hashlib.sha256(), {}
    while updates < max(checkpoints):
        for batch in batches_by_frames(rows, budget, shuffle_seed=seed + epoch):
            digest.update((' '.join(map(str, batch)) + '\n').encode())
            for i in batch:
                source = sources[i]
                counts[source] += 1
                samples[source] += rows[i][1]
                unique[source].add(rows[i][0])
            micro += 1
            if micro % accum:
                continue
            updates += 1
            if updates in checkpoints:
                result[str(updates)] = dict(microbatches=micro, batch_order_sha256=digest.hexdigest(),
                    row_exposures=dict(counts), audio_samples=dict(samples),
                    hours={k: v / 16000 / 3600 for k, v in samples.items()},
                    unique_recordings={k: len(v) for k, v in unique.items()})
            if updates == max(checkpoints):
                break
        epoch += 1
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--native', required=True)
    ap.add_argument('--historical-results', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out, manifest, native = Path(a.out), Path(a.manifest), Path(a.native)
    out.mkdir(parents=True, exist_ok=False)
    bindings = {}
    for folder, names in ((manifest, ['train.tsv', 'train.ltr', 'train.source', 'dict.ltr.txt',
            'leakage_report.json', 'stats.json'] + [sp + '.' + ext
            for sp in ('valid', 'valid_unseen', 'test', 'test_unseen')
            for ext in ('tsv', 'ltr', 'canon', 'meta.jsonl')]),
            (native, ['train.tsv', 'train.ltr', 'train.meta.jsonl', 'dev.tsv', 'dev.ltr', 'dict.ltr.txt'])):
        for name in names:
            path = folder / name
            bindings[str(path)] = sha(path)
    rows = read_rows(manifest / 'train.tsv')
    sources = (manifest / 'train.source').read_text().splitlines()
    assert len(rows) == len(sources)
    train_speakers, train_items, train_l1 = set(), set(), set()
    replay = set()
    for (p, _), source in zip(rows, sources):
        stem = Path(p).stem
        if source == 'native':
            replay.add(stem)
        else:
            fields = stem.split('-')
            train_speakers.add(fields[0]); train_items.add(fields[-2]); train_l1.add(fields[3])
    evaluation, heldout, eval_speakers = {}, {}, {}
    with (out / 'evaluation_roles.jsonl').open('w', encoding='utf-8') as fw:
        for split in ('valid', 'valid_unseen', 'test', 'test_unseen'):
            speakers, items, l1, roles, tasks = set(), set(), set(), collections.Counter(), collections.Counter()
            n = 0
            for line in (manifest / (split + '.meta.jsonl')).open(encoding='utf-8'):
                r = json.loads(line); n += 1
                speakers.add(r['spk']); items.add(r['item']); tasks[r['task']] += 1
                language = r['stem'].split('-')[3]; l1.add(language)
                role = calibration_role(r['spk']) if split.startswith('valid') else 'historically_analyzed_test'
                roles[role] += 1
                strict_key = None
                if split.endswith('unseen'):
                    strict_key = r['item']
                    heldout.setdefault(strict_key, dict(prompt=r['prompt'], norm_prompt=r['norm_prompt'],
                        split=split, native_exact=0, native_contains=0, replay_exact=0, replay_contains=0))
                fw.write(json.dumps(dict(split=split, row=n-1, uid=r['stem'], speaker=r['spk'],
                    item=r['item'], l1=language, task=r['task'], role=role,
                    foreign_unseen=split.endswith('unseen'), native_overlap_key=strict_key), ensure_ascii=False) + '\n')
            assert not speakers & train_speakers, split
            if split.endswith('unseen'):
                assert not items & train_items, split
            eval_speakers[split] = speakers
            evaluation[split] = dict(utterances=n, speakers=len(speakers), items=len(items), tasks=dict(tasks),
                roles=dict(roles), l1_codes=sorted(l1), l1_absent_from_training=sorted(l1-train_l1))
    assert not (eval_speakers['valid'] | eval_speakers['valid_unseen']) & (
        eval_speakers['test'] | eval_speakers['test_unseen'])
    for line in (native / 'train.meta.jsonl').open(encoding='utf-8'):
        r = json.loads(line); text = r['norm_text']; is_replay = Path(r['rel']).stem in replay
        for h in heldout.values():
            exact, contains = text == h['norm_prompt'], h['norm_prompt'] in text
            h['native_exact'] += int(exact); h['native_contains'] += int(contains)
            h['replay_exact'] += int(is_replay and exact); h['replay_contains'] += int(is_replay and contains)
    native_rows = read_rows(native / 'train.tsv')
    schedule = dict(stage1=exposure(native_rows, ['native'] * len(native_rows), [2000, 10000]),
        stage2=exposure(rows, sources, [2000, 4000, 8000]))
    trajectory = []
    for path in sorted(Path(a.historical_results).glob('sel/step_*_valid/summary.json')):
        step = int(path.parent.name.split('_')[1])
        other = path.parent.with_name('step_%d_valid_unseen' % step) / 'summary.json'
        if not other.exists():
            continue
        pair = {}
        for name, p in (('seen', path), ('unseen', other)):
            bindings[str(p)] = sha(p)
            b = json.loads(p.read_text())['baseline']
            pair[name] = dict(normal_fpr=b['mdd']['FR'] / b['sites']['normal_sites'],
                exact_diagnosis_recall=b['mdd']['CD'] / b['sites']['mispron_sites'],
                detection_recall=b['mdd']['recall'], errors=b['sites']['mispron_sites'],
                normals=b['sites']['normal_sites'])
        trajectory.append(dict(step=step, **pair))
    storage = {}
    for p in (manifest, out):
        usage = shutil.disk_usage(str(p))
        storage[str(p)] = dict(free_bytes=usage.free, total_bytes=usage.total)
    report = dict(status='inputs_audited_training_not_started', sha256=bindings,
        train=dict(rows=len(rows), source_counts=dict(collections.Counter(sources)),
            speakers=len(train_speakers), items=len(train_items), l1_codes=sorted(train_l1)),
        evaluation=evaluation, native_text_overlap=heldout, planned_exposure=schedule,
        historical_h0_trajectory=sorted(trajectory, key=lambda x: x['step']), storage=storage,
        limitations=['H0 test was already analyzed; not a new independent final test',
            'Native overlaps are manifest membership, not measured historical optimizer exposure',
            'Public SSL pretraining text/speaker exposure is unknown',
            'Historical H0 summaries have no confidence scores for a 1% FPR policy',
            'New Stage1/Stage2 training and service-policy evaluation are not executed by this audit'])
    (out / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(out=str(out), train=report['train'], evaluation=evaluation,
        native_text_overlap=heldout, storage=storage), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
