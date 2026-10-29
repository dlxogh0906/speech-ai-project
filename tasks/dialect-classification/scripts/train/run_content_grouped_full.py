"""Durable EAT -> XLSR -> validation ensemble/report queue; test stays sealed."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def summarize(root):
    import csv
    import numpy as np
    from train_standard_six_class import metrics
    with (root / 'manifests/valid.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    ids = np.array([r['row_id'] for r in rows])
    reports, probs = {}, {}
    for name in ['eat_large', 'xlsr2b']:
        report = json.loads((root / name / 'report.json').read_text())
        assert report['selected_checkpoint_verified'] and not report['test_scored']
        reports[name] = report
        predictions = np.load(root / name / 'valid_predictions.npz')
        assert np.array_equal(predictions['row_id'], ids)
        logits = predictions['logits']
        p = np.exp(logits - logits.max(1, keepdims=True))
        probs[name] = p / p.sum(1, keepdims=True)
    assert reports['eat_large']['protocol']['manifest_sha256'] == reports['xlsr2b']['protocol']['manifest_sha256']
    a, b = probs['eat_large'], probs['xlsr2b']
    candidates = dict(eat_large=a, xlsr2b=b)
    for alpha in [.25, .5, .75]:
        candidates['probability_eat_' + str(alpha)] = alpha * a + (1 - alpha) * b
        gate = a.argmax(1) == 5
        conditional = alpha * a[:, :5] / a[:, :5].sum(1, keepdims=True) + (1 - alpha) * b[:, :5] / b[:, :5].sum(1, keepdims=True)
        cascade = np.zeros_like(a)
        cascade[:, :5] = conditional
        cascade[gate, :5] = 0
        cascade[gate, 5] = 1
        candidates['eat_standard_gate_regional_eat_' + str(alpha)] = cascade
    values = {name: metrics(rows, p) for name, p in candidates.items()}
    selected = max(values, key=lambda name: (values[name]['macro_f1'], values[name]['top1']))
    baseline_path = root.parent / 'content_grouped_20261014/eat_augmented/report.json'
    baseline = json.loads(baseline_path.read_text())
    assert (root / 'manifests/valid.csv').read_bytes() == (root.parent / 'content_grouped_20261014/manifests/valid.csv').read_bytes()
    report = dict(completed_at=datetime.now().astimezone().isoformat(), selected=selected, valid=values[selected],
                  candidates=values, models={k: dict(checkpoint=v['checkpoint'], selected=v['selected'], valid=v['valid'],
                                                    optimizer_updates=v['optimizer_updates']) for k, v in reports.items()},
                  previous_small_clean_eat_valid=baseline['valid'], test=None, test_scored=False, deployed=False,
                  manifest_sha256=reports['eat_large']['protocol']['manifest_sha256'],
                  selection_policy='Fixed grid registered before training; validation macro F1 then accuracy; test untouched',
                  limitations=reports['eat_large']['protocol']['limitations'])
    write(root / 'comparison_report.json', report)
    np.savez_compressed(root / 'selected_validation_predictions.npz', scores=candidates[selected], row_id=ids)
    selected_metrics = values[selected]
    table = ''.join('<tr><td>%s</td><td>%.2f%%</td><td>%.4f</td></tr>' % (html.escape(name), val['top1'] * 100, val['macro_f1']) for name, val in values.items())
    per_class = ''.join('<tr><td>%s</td><td>%d</td><td>%.2f%%</td><td>%.2f%%</td><td>%.4f</td></tr>' %
                        (html.escape(v['name_ko']), v['n'], v['precision'] * 100, v['recall_top1'] * 100, v['f1'])
                        for v in selected_metrics['per_class'].values())
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>방언 모델 전체 학습량 재실험</title>
<style>body{font:16px system-ui;max-width:1050px;margin:40px auto;line-height:1.6}table{border-collapse:collapse;width:100%%}td,th{border:1px solid #ddd;padding:8px;text-align:left}pre{background:#f5f5f5;padding:16px;overflow:auto}</style>
<h1>방언 분류 재실험 결과</h1><p>완료: %s</p><p>선택: <b>%s</b> · 검증 정확도 <b>%.2f%%</b> · Macro F1 %.4f</p>
<p>동일 검증 3,600개에서 기존 소규모 EAT %.2f%%와 비교. 학습 데이터 597,505개, 새 EAT와 XLSR 및 앙상블. 테스트 5,820개는 미평가 상태이며 배포하지 않았습니다.</p>
<table><tr><th>모델/앙상블</th><th>검증 정확도</th><th>Macro F1</th></tr>%s</table>
<h2>선택 모델의 클래스별 검증 성능</h2><table><tr><th>클래스</th><th>구간 수</th><th>Precision</th><th>Recall</th><th>F1</th></tr>%s</table>
<h2>혼동행렬</h2><pre>%s</pre><p>과거 94.90%%는 다른 분할과 알려진 중복 문제가 있는 역사적 결과이므로 직접 비교하지 않습니다. 현재 수치는 내부 검증 성능이며, 사전학습 중복·유사 음성·서로 다른 ID의 동일 화자는 완전히 감사하지 못했습니다.</p>
<p><a href="comparison_report.json">상세 JSON</a> · <a href="manifests/summary.json">데이터 및 누수 검사</a></p></html>''' % (
        html.escape(report['completed_at']), html.escape(selected), selected_metrics['top1'] * 100, selected_metrics['macro_f1'],
        baseline['valid']['top1'] * 100, table, per_class, html.escape(json.dumps(selected_metrics['confusion_matrix'], indent=2)))
    (root / 'result_report.html').write_text(page)


def main(root):
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    script = Path(__file__).with_name('train_content_grouped_full.py')
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [script, Path(__file__),
                     script.with_name('train_dialect_comparison.py'), script.with_name('train_standard_six_class.py'), script.with_name('train_xlsr2b_dialect.py')]}
    protocol = dict(models=['eat_large', 'xlsr2b'], epochs_each=3, training='Fresh generic initialization; full training windows; same heldout CSV files',
                    ensemble_candidates=['eat_large', 'xlsr2b', 'probability blends EAT .25/.5/.75', 'EAT Standard gate + conditional regional blends EAT .25/.5/.75'],
                    selection='Validation macro F1, then accuracy, stable candidate order breaks exact ties',
                    test_scored=False, auto_report='comparison_report.json and result_report.html', source_sha256=source_hashes)
    protocol_path = root / 'experiment_protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol
    write(protocol_path, protocol)
    started = datetime.now().astimezone().isoformat()
    for model in ['eat_large', 'xlsr2b']:
        out = root / model
        out.mkdir(exist_ok=True)
        if (out / 'report.json').exists():
            assert json.loads((out / 'report.json').read_text())['selected_checkpoint_verified']
            continue
        for name, expected in source_hashes.items():
            assert hashlib.sha256(script.with_name(name).read_bytes()).hexdigest() == expected
        command = [sys.executable, '-u', str(script), '--manifests', str(root / 'manifests'), '--out', str(out), '--model', model, '--epochs', '3']
        with (out / 'training.log').open('a') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            write(root / 'queue_status.json', dict(status='running', stage=model, pid=os.getpid(), child_pid=child.pid,
                       started_at=started, updated_at=datetime.now().astimezone().isoformat(),
                       pending=['xlsr2b', 'ensemble_report'] if model == 'eat_large' else ['ensemble_report']))
            write(out / 'launch.json', dict(pid=child.pid, command=command, started_at=datetime.now().astimezone().isoformat()))
            result = child.wait()
        if result:
            raise RuntimeError('%s exited with %s; see %s' % (model, result, out / 'training.log'))
    write(root / 'queue_status.json', dict(status='running', stage='ensemble_report', pid=os.getpid()))
    summarize(root)
    write(root / 'queue_status.json', dict(status='complete', started_at=started,
               completed_at=datetime.now().astimezone().isoformat(), report=str(root / 'result_report.html'), test_scored=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.root)
    except Exception as exc:
        write(args.root / 'queue_status.json', dict(status='failed', error=type(exc).__name__ + ': ' + str(exc),
              updated_at=datetime.now().astimezone().isoformat()))
        raise
