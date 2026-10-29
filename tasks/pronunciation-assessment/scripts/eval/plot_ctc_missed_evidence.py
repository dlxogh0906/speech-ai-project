"""Standalone plots of held-fixed M3 evidence, not independent validation."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True)
    args = ap.parse_args()
    run = Path(args.run)
    if not (run / 'finished.json').exists():
        raise ValueError('full inference must finish before plotting')
    sites = [json.loads(x) for x in (run / 'sites.jsonl').open()]
    site_index = {(z['uid'], z['q']): z for z in sites}
    matched = json.loads((run / 'matched_controls.json').read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, expected, actual, title in zip(axes, ['ㅅ', 'ㅆ'], ['ㅆ', 'ㅅ'], ['S -> SS', 'SS -> S']):
        control_keys = {tuple(k) for z in matched
                        if site_index[tuple(z['target'])]['expected'] == expected
                        and site_index[tuple(z['target'])]['actual'] == actual
                        for k in z['controls']}
        groups = [([z for z in sites if z['expected'] == expected and z['label_kind'] == 'correct' and z.get('pair_delta') is not None],
                   'All scored normal sites', '#7c8a97'),
                  ([site_index[k] for k in sorted(control_keys)],
                   'Context-matched normals', '#2978a0'),
                  ([z for z in sites if z['expected'] == expected and z['actual'] == actual and z.get('target126')],
                   'Missed pair errors', '#c44b38')]
        for rows, label, color in groups:
            x = np.sort([z['pair_delta'] for z in rows])
            if len(x):
                ax.step(x, np.arange(1, len(x) + 1) / len(x), where='post', color=color,
                        label=label + ' (n=%d)' % len(x))
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.set(title=title, xlabel='log P(whole sequence with alternate) - log P(reference)')
        ax.grid(alpha=.2); ax.legend(loc='upper left', fontsize=8)
    axes[0].set_ylabel('Empirical cumulative fraction')
    fig.suptitle('Frozen M3 on reused CAL: negative scores can still retain ranking information', fontsize=11)
    fig.text(.5, .015, 'Diagnostic distributions; TAG labels are not independently adjudicated; prompts were seen by M2.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .045, 1, .94])
    fig.savefig(run / 'pair_evidence_distributions.svg')
    fig.savefig(run / 'pair_evidence_distributions.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
