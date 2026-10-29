"""Plot paired 2k->4k normal-site changes from the completed all-phone audit."""
import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit', required=True)
    a = ap.parse_args(); root = Path(a.audit)
    if not (root / 'finished.json').exists():
        raise ValueError('audit has not finished')
    with (root / 'groups.tsv').open() as f:
        groups = list(csv.DictReader(f, delimiter='\t'))
    with (root / 'changes.tsv').open() as f:
        changes = list(csv.DictReader(f, delimiter='\t'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={'width_ratios': [1, 1.7]})
    positions = ['onset', 'nucleus', 'coda']
    item_ids = ['RS012', 'RW054', 'RW058', 'RW139', 'RW023', 'RW003', 'RS036', 'RS017', 'RW040']
    payload = {}
    for i, (model, label, color) in enumerate([
            ('w2vbert', 'w2v-BERT', '#28659a'), ('xlsr300m', 'XLS-R 300M', '#cc6633')]):
        offset = (i - .5) * .36
        lookup = {(r['condition'], r['group']): int(r['normal_fp']) for r in groups
                  if r['split'] == 'valid_unseen' and r['dimension'] == 'position'}
        delta = [lookup[(model+'_eval_4000', p)]-lookup[(model+'_eval_2000', p)] for p in positions]
        item_lookup = {r['group']: int(r['delta_fp']) for r in changes if r['backbone'] == model
                       and r['split'] == 'valid_unseen' and r['dimension'] == 'item'}
        payload[model] = dict(position_delta=dict(zip(positions, delta)), prompt_delta=item_lookup)
        for ax, names, vals in [(axes[0], positions, delta), (axes[1], item_ids, [item_lookup[x] for x in item_ids])]:
            y = np.arange(len(names)) + offset
            ax.barh(y, vals, height=.34, label=label, color=color)
            for yy, v in zip(y, vals):
                ax.text(v + (12 if v >= 0 else -12), yy, '%+d' % v,
                        ha='left' if v >= 0 else 'right', va='center', fontsize=8)
    for ax, names in zip(axes, [['Onset consonants', 'Vowels', 'Coda consonants'], item_ids]):
        ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(names)
        ax.invert_yaxis(); ax.axvline(0, color='#555555', lw=.8)
        ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
        ax.set_xlabel('Change in TAG-normal false alarms (4k minus 2k)')
        ax.margins(x=.18)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    axes[0].set_title('Where the net increase occurs'); axes[0].legend(loc='lower right')
    axes[1].set_title('All 9 unseen selection prompts')
    fig.suptitle('H0: paired changes are concentrated in vowels and two prompts', fontsize=13)
    fig.text(.5, .02, 'Same 34,706 normal sites per condition. Positive = worse. Greedy canonical sites; insertion gaps excluded.\n'
             'RS012: wall / sticking sentence; RW054: suri. Reused validation data, not independent final testing.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .09, 1, .95])
    fig.savefig(root/'paired_changes.svg'); fig.savefig(root/'paired_changes.png', dpi=170)
    (root/'plot_values.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
