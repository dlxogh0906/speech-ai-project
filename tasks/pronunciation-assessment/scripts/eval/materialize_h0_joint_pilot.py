"""Materialize a fixed fold's supervised lineage without training or inference."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'tasks/pronunciation-assessment/src'),
               str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import collections
import csv
import hashlib
import json
import re
import shutil
import numpy as np
from train_h0_backbone_control import file_sha
from audit_h0_p5_readiness import write


def select_file(source, destination, indices, header=False):
    """Keep original bytes/order; verify the saved view with a separate read."""
    selected = set(map(int, indices)); count = 0
    with source.open('rb') as reader, destination.open('wb') as writer:
        if header:
            writer.write(reader.readline())
        for i, line in enumerate(reader):
            if i in selected:
                writer.write(line); count += 1
    assert count == len(indices)
    lines = source.read_bytes().splitlines(keepends=True)
    offset = int(header)
    expected = (lines[0] if header else b'') + b''.join(lines[int(i)+offset] for i in indices)
    assert hashlib.sha256(expected).hexdigest() == file_sha(destination)
    return dict(rows=count, source=str(source), source_sha256=file_sha(source),
                sha256=file_sha(destination))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    a = ap.parse_args(); p = json.loads((a.root/'protocol.json').read_text())
    out = a.root/'manifests'; out.mkdir(exist_ok=False)
    design = Path(p['holdout_design']); hp = json.loads((design/'protocol.json').read_text())
    fold = p['fold']; plan = hp['folds'][fold]
    z = dict(np.load(design/('fold_%d'%fold)/'row_indices.npz'))
    original = Path(p['stage2_plan']['manifest'])
    native = Path(p['training']['stage1']['manifest'])
    for name, digest in p['bindings'].items():
        assert file_sha(name) == digest, name
    with (design/'prompt_ancestry.tsv').open() as f:
        text = {r['item']:r['text'] for r in csv.DictReader(f, delimiter='\t')}
    forbidden = set(plan['outer_items']+plan['inner_selection_items']+hp['old_diagnostics'])
    terms = set()
    for item in forbidden:
        s = re.sub('[^가-힣]', '', text[item]); terms.add(s)
        terms.update(s[j:j+12] for j in range(max(0, len(s)-11)))
    regex = re.compile('|'.join(re.escape(t) for t in sorted(terms) if t))
    bindings = {}; retained_native = set(); excluded_native = set(); native_counts = {}
    (out/'stage1').mkdir(); (out/'stage2').mkdir()
    for split, key in [('train','stage1_excluded'), ('dev','stage1_dev_excluded')]:
        excluded = []
        with (native/(split+'.tsv')).open() as ft:
            next(ft)
            for i, line in enumerate((native/(split+'.meta.jsonl')).open()):
                r = json.loads(line); uid = Path(ft.readline().split('\t')[0]).stem
                assert uid == Path(r['rel']).stem
                norm = re.sub('[^가-힣]', '', r.get('text', r['norm_text']))
                matched = bool(regex.search(norm))
                if matched:
                    excluded.append(i)
                if split == 'train':
                    (excluded_native if matched else retained_native).add(uid)
            assert not ft.readline()
        np.testing.assert_array_equal(excluded, z[key])
        selected = np.setdiff1d(np.arange(i+1), z[key])
        native_counts[split] = dict(original=i+1, retained=len(selected), excluded=len(excluded))
        for suffix in ('tsv','ltr'):
            destination = out/'stage1'/(split+'.'+suffix)
            bindings[str(destination)] = select_file(native/destination.name, destination, selected, suffix=='tsv')
    for directory, source in [('stage1',native), ('stage2',original)]:
        dest = out/directory/'dict.ltr.txt'; shutil.copyfile(source/'dict.ltr.txt', dest)
        bindings[str(dest)] = dict(sha256=file_sha(dest), source=str(source/'dict.ltr.txt'))
    np.testing.assert_array_equal(z['stage2_train'], np.load(a.root/'matrices/manifest_row_indices.npy'))
    for suffix in ('tsv','ltr','source'):
        dest = out/'stage2'/('train.'+suffix)
        bindings[str(dest)] = select_file(original/dest.name, dest, z['stage2_train'], suffix=='tsv')
    sources = (out/'stage2/train.source').read_text().splitlines()
    train_speakers = set(); train_uids = set(); replay = set(); items = set()
    with (out/'stage2/train.tsv').open() as ft:
        next(ft)
        for source, line in zip(sources, ft):
            uid = Path(line.split('\t')[0]).stem; train_uids.add(uid)
            if source == 'native':
                replay.add(uid)
            else:
                parts = uid.split('-'); train_speakers.add(parts[0]); items.add(parts[5])
    assert items <= set(plan['train_items'])
    assert not replay & excluded_native
    assert replay <= retained_native
    speakers = [train_speakers]; uids = [train_uids]; evaluation = {}
    for role, split, key, allowed in [
        ('cal','valid','cal_valid',plan['train_items']),
        ('inner','valid','inner_valid',plan['inner_selection_items']),
        ('outer','test','outer_test',plan['outer_items'])]:
        for suffix in ('tsv','ltr','canon','meta.jsonl'):
            dest = out/'stage2'/(role+'.'+suffix)
            bindings[str(dest)] = select_file(original/(split+'.'+suffix), dest, z[key], suffix=='tsv')
        records = [json.loads(s) for s in (out/'stage2'/(role+'.meta.jsonl')).open()]
        assert {r['item'] for r in records} <= set(allowed)
        speakers.append({r['spk'] for r in records}); uids.append({r['stem'] for r in records})
        evaluation[role] = dict(recordings=len(records), items=len({r['item'] for r in records}),
                               speakers=len(speakers[-1]))
    for i in range(4):
        for j in range(i):
            assert not speakers[i]&speakers[j]
            assert not uids[i]&uids[j]
    write(out/'report.json', dict(status='fixed_fold_manifests_and_supervised_text_lineage_verified',
        bindings=bindings, native=native_counts, replay_uids=len(replay), stage2_rows=len(sources),
        stage2_source_counts=dict(collections.Counter(sources)), evaluation=evaluation,
        supervised_text_and_speaker_role_checks=True, model_optimizer_updates=0,
        limitations=['public SSL text/speaker exposure unknown',
            'UID and declared speaker identities verified; cross-corpus identity and approximate audio duplication not established',
            'materialized original LTR targets preserve prior partial-label normalization; not new independent annotation',
            'audio preflight and full-denominator scoring checks remain before model training']))
    print('fold manifests and supervised text lineage verified', flush=True)


if __name__ == '__main__':
    main()
