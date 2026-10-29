"""Run the selected dialect ensemble on real recordings that have no region labels (2026-10-26, user request).

The models are rebuilt exactly as in score_sealed_test.py (same run(), same candidate rule selected on validation).
Each recording is one window: whole file, at most 12 s, recordings under 1 s are skipped (the loader rejects them).
Training windows were 3-12 s, so 1-3 s recordings are reported separately.

The list csv needs audio_path, duration_sec and group. --nat_npz optionally adds a nationality-model split:
an npz with L (rows x classes), labels and ok in list order (e.g. tta_full_encoder.py --variants orig).
Outputs per-row probabilities by row index only (no paths) and aggregate tables.

    PYTHONPATH=/data/speech-ai/tools/dialect_compare_py38:/data/speech-ai/tools/hf_py38 \
        $PY tasks/dialect-classification/scripts/eval/score_real_recordings.py \
        --root /data/speech-ai/out/dialect/content_grouped_full_20261015 --list <list.csv> --out <dir> [--nat_npz <orig.npz>]
"""
import argparse
import collections
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / 'train'))

LABELS = ['Gangwon', 'Gyeongsang', 'Jeolla', 'Jeju', 'Chungcheong', 'Standard']
KO = dict(Gangwon='강원', Gyeongsang='경상', Jeolla='전라', Jeju='제주', Chungcheong='충청', Standard='표준어')


def table(name, idx, probs, nat_note=None):
    import numpy as np
    if len(idx) == 0:
        return None
    sel = probs['selected'][idx]
    pred = sel.argmax(1)
    counts = collections.Counter(int(p) for p in pred)
    row = dict(subset=name, n=int(len(idx)),
               counts={KO[LABELS[c]]: int(counts.get(c, 0)) for c in range(6)},
               standard_rate=float((pred == 5).mean()),
               mean_top_prob=float(sel.max(1).mean()),
               confident_090=float((sel.max(1) >= 0.9).mean()),
               eat_standard_rate=float((probs['eat_large'][idx].argmax(1) == 5).mean()),
               xlsr_standard_rate=float((probs['xlsr2b'][idx].argmax(1) == 5).mean()),
               models_agree=float((probs['eat_large'][idx].argmax(1) == probs['xlsr2b'][idx].argmax(1)).mean()))
    if nat_note is not None:
        row['note'] = nat_note
    return row


def main(args):
    import numpy as np
    import torch
    from score_sealed_test import run, candidates, softmax
    report = json.loads((args.root / 'comparison_report.json').read_text())
    selected = report['selected']
    with args.list.open(encoding='utf-8') as stream:
        listed = list(csv.DictReader(stream))
    keep = [i for i, r in enumerate(listed) if float(r['duration_sec']) >= 1.0]
    rows = [dict(audio_path=listed[i]['audio_path'], start_sec='0', row_id=str(i),
                 duration_sec='%.3f' % min(float(listed[i]['duration_sec']), 12.0)) for i in keep]
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0')
    logits = {m: run(m, {'real': rows}, args.root / m / 'best.pt', device, args.out, batch_size=args.batch)['real']
              for m in ['eat_large', 'xlsr2b']}
    eat, xlsr = softmax(logits['eat_large']), softmax(logits['xlsr2b'])
    probs = dict(eat_large=eat, xlsr2b=xlsr, selected=candidates(eat, xlsr)[selected])
    np.savez_compressed(args.out / 'real_probs.npz', list_row=np.array(keep), labels=np.array(LABELS), **probs)

    group = np.array([listed[i]['group'] for i in keep], dtype=object)
    dur = np.array([float(listed[i]['duration_sec']) for i in keep])
    subsets = [('전체', np.arange(len(keep)))]
    subsets += [('묶음: ' + g, np.where(group == g)[0]) for g in sorted(set(group))]
    subsets += [('길이 1-3초', np.where(dur < 3)[0]), ('길이 3-12초', np.where(dur >= 3)[0])]
    nat = None
    if args.nat_npz:
        z = np.load(args.nat_npz, allow_pickle=True)
        labels = [str(c) for c in z['labels']]
        assert len(z['ok']) == len(listed), 'nat_npz rows != list rows'
        ok = z['ok'].astype(bool)[keep]
        top = z['L'][keep].argmax(1)
        nat = np.where(~ok, 'unread', np.where(top == labels.index('Korea'), 'Korea', 'foreign'))
        subsets += [('국적 모델이 한국인으로 본 녹음', np.where(nat == 'Korea')[0]),
                    ('국적 모델이 외국인으로 본 녹음', np.where(nat == 'foreign')[0]),
                    ('한국인으로 본 녹음 · 3-12초', np.where((nat == 'Korea') & (dur >= 3))[0])]
    result = dict(scored_at=datetime.now().astimezone().isoformat(), selected_rule=selected,
                  listed=len(listed), scored=len(keep), skipped_under_1s=len(listed) - len(keep),
                  nationality_split=None if nat is None else dict(collections.Counter(nat.tolist())),
                  tables=[t for t in (table(n, i, probs) for n, i in subsets) if t])
    (args.out / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['| 묶음 | 녹음 | ' + ' | '.join(KO[c] for c in LABELS) + ' | 표준어 비율 | 평균 확신도 | 확신도 ≥0.9 | EAT 혼자 표준어 | XLS-R 혼자 표준어 | 두 모델 일치 |',
             '|---|--:|' + '--:|' * 6 + '--:|--:|--:|--:|--:|--:|']
    for t in result['tables']:
        lines.append('| %s | %d | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |' % (
            t['subset'], t['n'], ' | '.join(str(t['counts'][KO[c]]) for c in LABELS), t['standard_rate'],
            t['mean_top_prob'], t['confident_090'], t['eat_standard_rate'], t['xlsr_standard_rate'], t['models_agree']))
    (args.out / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    print('REAL_DONE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--list', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--nat_npz', type=Path)
    parser.add_argument('--batch', type=int, default=4)
    main(parser.parse_args())
