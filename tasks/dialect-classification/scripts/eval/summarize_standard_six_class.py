"""Summarize fixed six-class candidates, with uncertainty grouped by person."""
import argparse
from collections import defaultdict
import csv
import hashlib
import html
import json
import shutil
from pathlib import Path
import zipfile

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/data/speech-ai/out/dialect/standard_six_class_20261013'))
    args = p.parse_args()
    folder = args.root / 'eat_large'
    report = json.loads((folder / 'report.json').read_text())
    with (args.root / 'manifests/test.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    predictions = np.load(folder / 'test_predictions.npz')
    if list(predictions['row_id']) != [r['row_id'] for r in rows]:
        raise RuntimeError('Test prediction row order changed')
    truth = predictions['labels']; predicted = predictions['logits'].argmax(1)
    original_root = Path('/data/speech-ai/out/dialect/speakerid_20261011/manifests')
    original_report = json.loads(Path('/data/speech-ai/out/dialect/finetune_speakerid_20261012/eat_large/report.json').read_text())
    original_summary = json.loads((original_root / 'summary.json').read_text())
    if original_summary['manifest_sha256'] != report['data_summary']['original_regional_manifest_sha256']:
        raise RuntimeError('Original regional manifest identity changed')
    if original_report['data_summary']['manifest_sha256'] != original_summary['manifest_sha256']:
        raise RuntimeError('Original prediction identity changed')
    if hashlib.sha256((original_root / 'test.csv').read_bytes()).hexdigest() != original_summary['manifest_file_sha256']['test']:
        raise RuntimeError('Original test rows changed')
    with (original_root / 'test.csv').open(newline='') as f:
        original_rows = list(csv.DictReader(f))
    original_index = {r['row_id']: i for i, r in enumerate(original_rows)}
    original = np.load('/data/speech-ai/out/dialect/finetune_speakerid_20261012/eat_large/test_predictions.npz')
    if len(original['predictions']) != len(original_rows):
        raise RuntimeError('Original predictions length mismatch')
    regional = []
    for row, y in zip(rows, truth):
        if int(y) == 5: continue
        idx = original_index[row['row_id']]
        old_row = original_rows[idx]
        if int(original['labels'][idx]) != int(y) or any(row[k] != old_row[k] for k in ['audio_path', 'start_sec', 'duration_sec']):
            raise RuntimeError('Original audio window identity mismatch')
        regional.append((int(y), int(original['predictions'][idx])))
    original_regional_top1 = float(np.mean([y == prediction for y, prediction in regional]))
    rng = np.random.default_rng(42)
    summary = []
    for i, (label, values) in enumerate(report['test']['per_class'].items()):
        groups = defaultdict(lambda: [0, 0])
        for row, y, prediction in zip(rows, truth, predicted):
            if int(y) != i: continue
            counts = groups[row['person_id']]
            counts[0] += int(y == prediction); counts[1] += 1
        counts = np.array(list(groups.values()), dtype=np.float64)
        sampled = counts[rng.integers(0, len(counts), size=(3000, len(counts)))].sum(1)
        lower, upper = np.quantile(sampled[:, 0] / sampled[:, 1], [.025, .975])
        old = report['best_frozen_test']['per_class'][label]
        baseline = float(np.mean([y == prediction for y, prediction in regional if y == i])) if i < 5 else None
        summary.append({'class': label, **values, 'cluster_bootstrap_ci_lower': float(lower),
                        'cluster_bootstrap_ci_upper': float(upper), 'frozen_head_top1': old['recall_top1'],
                        'original_five_class_top1_same_windows': baseline,
                        'selected_minus_frozen_percentage_points': (values['recall_top1'] - old['recall_top1']) * 100})
    history = json.loads((folder / 'history.json').read_text())
    result = {'selected_candidate': report['selected_candidate'],
              'selected_additional_encoder_adaptation': report['selected_candidate'].startswith('last4_encoder:'),
              'adaptation_validation_history': [item for item in history if item['stage'] == 'encoder'],
              'test_top1': report['test']['top1'],
              'test_macro_f1': report['test']['macro_f1'], 'test_original_regions_top1': report['test']['original_five_region_top1'],
              'original_five_class_top1_same_windows': original_regional_top1,
              'regional_change_percentage_points': (report['test']['original_five_region_top1'] - original_regional_top1) * 100,
              'best_frozen_test_top1': report['best_frozen_test']['top1'],
              'per_class': summary, 'reading_reference': report.get('reading_reference'),
              'uncertainty': '95% percentile interval from 3,000 within-class source-person bootstrap samples; applies only to provider-label accuracy.',
              'additional_limitations': ['Original regional classes use speaker.residenceProvince as the source label; accents are not independently certified.',
                                         'Original regional speakers are middle/older age, while the new corpus includes broader age groups.',
                                         'A high score within these corpora does not establish app/web recording performance.'],
              'data_summary': report['data_summary']}
    (args.root / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    with (args.root / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(6); value = np.array([r['recall_top1'] for r in summary]) * 100
    errors = np.array([[r['recall_top1'] - r['cluster_bootstrap_ci_lower'] for r in summary],
                       [r['cluster_bootstrap_ci_upper'] - r['recall_top1'] for r in summary]]) * 100
    ax.bar(x - .18, [r['original_five_class_top1_same_windows'] * 100 if r['original_five_class_top1_same_windows'] is not None else np.nan for r in summary], .36, label='Original 5-class model, same regional windows')
    ax.bar(x + .18, value, .36, yerr=np.maximum(errors, 0), capsize=4, label='Final validation-selected model')
    ax.set_xticks(x); ax.set_xticklabels([r['class'] for r in summary], rotation=20)
    ax.set_ylabel('Provider-label Top-1 (%)'); ax.set_ylim(0, 105); ax.legend(loc='lower left')
    ax.set_title('Held-out source people; 95% person cluster bootstrap intervals')
    fig.tight_layout(); fig.savefig(args.root / 'class_comparison.png', dpi=180); plt.close(fig)
    table = ''.join('<tr><td>%s</td><td>%d</td><td>%d</td><td>%s</td><td>%.2f%%</td><td>%.2f%%</td><td>%.2f–%.2f%%</td></tr>' %
                    (html.escape(r['name_ko']), r['n'], r['people'], ('%.2f%%' % (r['original_five_class_top1_same_windows'] * 100)) if r['original_five_class_top1_same_windows'] is not None else '—',
                     r['recall_top1'] * 100, r['person_balanced_top1'] * 100,
                     r['cluster_bootstrap_ci_lower'] * 100, r['cluster_bootstrap_ci_upper'] * 100) for r in summary)
    limitations = report['data_summary']['limitations'] + result['additional_limitations']
    reference = report.get('reading_reference')
    reference_text = html.escape(json.dumps(reference, ensure_ascii=False, indent=2)) if reference else '별도 자료 없음'
    page = '<!doctype html><meta charset="utf-8"><title>6클래스 결과 비교</title><style>body{font-family:sans-serif;max-width:1100px;margin:40px auto;line-height:1.7}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}img{max-width:100%%}pre{white-space:pre-wrap}</style><h1>6클래스 EAT 결과 — 2026-10-13</h1><p>선택 모델: %s</p><p>테스트 Top-1 <strong>%.2f%%</strong> · Macro F1 %.4f · 기존 5지역 Top-1 %.2f%%</p><table><tr><th>클래스</th><th>녹음</th><th>화자</th><th>인코더 고정</th><th>최종 선택</th><th>화자 평균</th><th>95%% 구간</th></tr>%s</table><img src="class_comparison.png"><h2>해석 조건</h2><ul>%s</ul><h2>71349번 무감정 낭독 참고 자료</h2><p>방언 정답이 확인되지 않은 8명·96개 녹음입니다. 아래는 출력 분포이며 정확도가 아닙니다.</p><pre>%s</pre><p><a href="eat_large/report.html">모델 상세 보고서</a> · <a href="data_profile.html">데이터 구성</a> · <a href="comparison.csv">CSV</a> · <a href="comparison.json">JSON</a></p>' % (html.escape(report['selected_candidate']), result['test_top1'] * 100, result['test_macro_f1'], result['test_original_regions_top1'] * 100, table, ''.join('<li>' + html.escape(v) + '</li>' for v in limitations), reference_text)
    (args.root / 'comparison.html').write_text(page)
    page = page.replace('<th>인코더 고정</th>', '<th>기존 5클래스</th>')
    baseline_note = '<p>동일한 기존 지역 테스트 5,000개에서 원래 5클래스 모델 %.2f%% → 새 6클래스 모델 %.2f%% (%.2f%%p).</p>' % (original_regional_top1 * 100, result['test_original_regions_top1'] * 100, result['regional_change_percentage_points'])
    page = page.replace('<h2>해석 조건</h2>', baseline_note + '<h2>해석 조건</h2>')
    (args.root / 'comparison.html').write_text(page)
    review = Path('/data/speech-ai/out/dialect/standard_speaker_data_review_20261013')
    destination = args.root / 'source_review'
    destination.mkdir(exist_ok=True)
    for name in ['access.json', '537_identity_audit.json', '71349_actual_identity_audit.json', 'reading_reference_inventory.json']:
        if (review / name).is_file(): shutil.copyfile(review / name, destination / name)
    code = Path('/home/btdevgpu02/project/speech-ai-project/tasks/dialect-classification/scripts')
    provenance = {'manifest_sha256': report['data_summary']['manifest_sha256'],
                  'source_code_sha256': {str(path.resolve().relative_to(code)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [code / 'data/prepare_standard_six_class.py', code / 'train/train_standard_six_class.py', code / 'train/run_standard_six_class.sh', Path(__file__)]}}
    (args.root / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    archive = args.root / '6클래스_표준어추가_실험결과_20261013.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in args.root.rglob('*'):
            if path.is_file() and path.suffix in ['.html', '.json', '.csv', '.png'] and 'features' not in path.parts and 'manifests' not in path.parts:
                z.write(path, str(path.relative_to(args.root)))
    print(json.dumps({k: v for k, v in result.items() if k != 'data_summary'}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
