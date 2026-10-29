"""Evaluate the expanded EAT model's North extension and write final comparisons.

Waits for the detached training pipeline; checkpoints are selected by validation.
North source-held-out checks are diagnostics, never independent test accuracy.
"""
import csv
import datetime
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

ROOT = Path('/data/speech-ai/out/dialect')
RUN = ROOT / 'expanded_six_class_20261013'
BASE = ROOT / 'standard_six_class_20261013'
OLD7 = ROOT / 'seven_class_north_pilot_20261013/feature_head'
OUT = RUN / 'seven_class'
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'tasks/dialect-classification/scripts/train'))
import train_dialect_comparison as engine

LABELS = ['Gangwon', 'Gyeongsang', 'Jeolla', 'Jeju', 'Chungcheong', 'Standard', 'North_Korean']
KO = ['강원', '경상', '전라', '제주', '충청', '표준어 후보', '북한말']


def write(path, value):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def status(stage, **extra):
    value = dict(stage=stage, updated_at=datetime.datetime.now().astimezone().isoformat(), pid=os.getpid(), **extra)
    write(RUN / 'report_status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def load(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def measure(y, pred):
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=range(7), zero_division=0)
    return {'n': len(y), 'six_class_test_top1': float(np.mean(y == pred)),
            'macro_f1_over_six_present_classes': float(f[:6].mean()),
            'north_false_positives': int(np.sum(pred == 6)),
            'per_class': [{'label': LABELS[i], 'name': KO[i], 'n': int(n[i]),
                           'precision': float(p[i]) if n[i] else None,
                           'recall': float(r[i]) if n[i] else None,
                           'f1': float(f[i]) if n[i] else None,
                           'false_north': int(np.sum((y == i) & (pred == 6)))} for i in range(7)],
            'confusion_matrix': confusion_matrix(y, pred, labels=range(7)).tolist(),
            'seven_class_accuracy': None}


def paired_ci(rows, old_pred, new_pred, repetitions=2000):
    # Sample people within each class, retaining all recordings of a person.
    rng = np.random.default_rng(42)
    groups = []
    for label in range(6):
        ids = sorted({r['person_id'] for r in rows if int(r['label_id']) == label})
        lookup = {p: i for i, p in enumerate(ids)}
        delta, counts = np.zeros(len(ids)), np.zeros(len(ids))
        for j, row in enumerate(rows):
            if int(row['label_id']) != label: continue
            i = lookup[row['person_id']]
            delta[i] += int(new_pred[j] == label) - int(old_pred[j] == label)
            counts[i] += 1
        draws = rng.integers(0, len(ids), size=(repetitions, len(ids)))
        groups.append(delta[draws].sum(1) / counts[draws].sum(1))
    samples = np.mean(groups, axis=0) * 100
    return {'method': 'paired class-stratified speaker-cluster bootstrap, 2000 draws, seed 42',
            'lower_delta_pp': float(np.quantile(samples, .025)), 'upper_delta_pp': float(np.quantile(samples, .975))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status('waiting_for_six_class', status='running')
    while True:
        pipeline = json.loads((RUN / 'pipeline_status.json').read_text())
        if pipeline.get('status') == 'failed': raise RuntimeError('Training pipeline failed: ' + str(pipeline))
        if pipeline.get('status') == 'complete': break
        time.sleep(10)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.25, device=0)
    device = torch.device('cuda:0')
    rows = {s: load(RUN / 'manifests' / (s + '.csv')) for s in ['train', 'valid', 'test']}
    for s in ['valid', 'test']:
        assert (RUN/'manifests'/(s+'.csv')).read_bytes() == (BASE/'manifests'/(s+'.csv')).read_bytes()
    variants = {'expanded_adaptation': json.loads((RUN / 'eat_large/report.json').read_text())}
    status('waiting_for_regularized_head', status='running')
    while True:
        reg_progress_path = RUN / 'regularized_head/progress.json'
        reg_progress = json.loads(reg_progress_path.read_text()) if reg_progress_path.exists() else {}
        if reg_progress.get('status') == 'failed':
            raise RuntimeError('Regularization sweep failed: ' + str(reg_progress))
        if reg_progress.get('status') == 'complete':
            variants['regularized_head'] = json.loads((RUN / 'regularized_head/report.json').read_text())
            break
        time.sleep(10)
    chosen_variant = max(variants, key=lambda n: (variants[n]['valid']['top1'], variants[n]['valid']['macro_f1']))
    candidate_dir = RUN / ('eat_large' if chosen_variant == 'expanded_adaptation' else 'regularized_head')
    ckpt_path = candidate_dir / 'best_adapted.pt'
    state = torch.load(ckpt_path, map_location='cpu')
    candidate = variants[chosen_variant]
    baseline = json.loads((BASE / 'eat_large/report.json').read_text())
    # Persist the decision before any North extension test calculation.
    best = 'expanded' if (candidate['valid']['top1'], candidate['valid']['macro_f1']) > (baseline['valid']['top1'], baseline['valid']['macro_f1']) else 'baseline'
    write(RUN / 'selection.json', {'selected_by_six_class_validation': best,
          'baseline_valid_top1': baseline['valid']['top1'], 'candidate_valid_top1': candidate['valid']['top1'],
          'expanded_variant': chosen_variant, 'candidate_dir': str(candidate_dir),
          'north_policy': 'C=.01, balanced logistic regression, validation-negative 99th percentile threshold, fixed before extension evaluation'})
    status('extract_selected_encoder_features', status='running', selected_candidate=candidate['selected_candidate'])
    encoder = engine.Encoder('eat_large', device)
    params = dict(encoder.model.named_parameters())
    with torch.no_grad():
        for name, values in state['encoder_delta'].items(): params[name].copy_(values.to(device))
    encoder.info.update(adaptation_checkpoint=str(ckpt_path), adaptation_sha256=hashlib.sha256(ckpt_path.read_bytes()).hexdigest())
    inventory = json.loads((ROOT / 'north_korean_review_20261012/media_inventory.json').read_text())
    north_rows = []
    for clip in inventory['clips']:
        duration = sf.info(clip['audio_path']).duration
        for number, start in enumerate(np.arange(0, duration, 8.)):
            length = min(8., duration-start)
            if length >= 3:
                north_rows.append({'row_id': clip['clip']+':'+str(number), 'source_clip': clip['clip'],
                                   'audio_path': clip['audio_path'], 'start_sec': float(start), 'duration_sec': float(length)})
    # Frozen candidate features can be reused by row, but adapted encoder features must be re-extracted.
    extraction = SimpleNamespace(model='eat_large', manifests=RUN/'manifests', out=OUT/'extraction', batch_size=16, workers=4)
    wanted = dict(rows, north_reference=north_rows)
    frozen = candidate['selected_candidate'] == candidate['best_frozen_candidate']
    if frozen:
        import shutil
        dest = extraction.out / 'features'; dest.mkdir(parents=True, exist_ok=True)
        for split in ['train', 'valid']:
            for suffix in ['.npy', '.ok.npy', '.done.json']:
                target = dest / (split+suffix)
                if not target.exists(): target.symlink_to(RUN/'eat_large/features'/(split+suffix))
    cache, resources = engine.extract(extraction, wanted, encoder)
    for split in wanted:
        if not np.load(cache/(split+'.ok.npy')).all(): raise RuntimeError('Failed feature rows: '+split)
    selected = [encoder.names.index(n) for n in state['feature_names']]
    feats = {}
    for split in wanted:
        arr = np.load(cache/(split+'.npy'), mmap_mode='r')
        v = np.asarray(arr[:,selected,:], dtype=np.float32).reshape(len(arr), -1)
        feats[split] = (v-np.asarray(state['mu']))/np.asarray(state['sd'])
    encoder.model.cpu(); del encoder; torch.cuda.empty_cache()
    scores = {s: v @ state['head']['weight'].numpy().T + state['head']['bias'].numpy() for s,v in feats.items()}
    original = {}
    for split in ['valid', 'test']:
        original[split] = np.load(candidate_dir/(split+'_predictions.npz'))
        if not np.array_equal(original[split]['row_id'], np.array([r['row_id'] for r in rows[split]])):
            raise RuntimeError('Prediction row IDs changed')
        agreement = np.mean(scores[split].argmax(1) == original[split]['logits'].argmax(1))
        if agreement < .999 or np.abs(scores[split]-original[split]['logits']).mean() > .02:
            raise RuntimeError('Selected feature/head mismatch on '+split)
    status('fit_north_detector', status='running')
    source = np.array([r['source_clip'] for r in north_rows])
    def fit(positive):
        x=np.vstack([feats['train'],positive]); y=np.r_[np.zeros(len(feats['train']),int),np.ones(len(positive),int)]
        model=LogisticRegression(C=.01,class_weight='balanced',solver='liblinear',max_iter=1000,random_state=42).fit(x,y)
        threshold=float(np.quantile(model.decision_function(feats['valid']),.99,method='higher'))
        return model,threshold
    folds=[]
    for held in sorted(set(source)):
        detector,threshold=fit(feats['north_reference'][source!=held])
        folds.append({'held_out_clip':held,'n':int(np.sum(source==held)),
                      'predicted_north':int(np.sum(detector.decision_function(feats['north_reference'][source==held])>threshold))})
    detector,threshold=fit(feats['north_reference'])
    # Preserve the exact selected six-way decisions from saved evaluation predictions.
    pred={s:np.where(detector.decision_function(feats[s])>threshold,6,original[s]['logits'].argmax(1)) for s in ['valid','test']}
    y=original['test']['labels']
    extension=measure(y,pred['test'])
    extension.update(valid_false_north=int(np.sum(pred['valid']==6)), valid_n=len(pred['valid']),
                     north_sources=3,north_windows=len(north_rows),north_independent_test_n=0,
                     north_inventory_sha256=hashlib.sha256((ROOT / 'north_korean_review_20261012/media_inventory.json').read_bytes()).hexdigest(),
                     source_held_out_diagnostics=folds,threshold=threshold,checkpoint_sha256=hashlib.sha256(ckpt_path.read_bytes()).hexdigest())
    torch.save({'labels':LABELS,'six_checkpoint_path':str(ckpt_path),'six_checkpoint_sha256':extension['checkpoint_sha256'],
                'north_coef':detector.coef_,'north_intercept':detector.intercept_,'north_threshold':threshold,
                'feature_names':state['feature_names'],'feature_mu':state['mu'],'feature_sd':state['sd']},OUT/'seven_option_head.pt')
    np.savez_compressed(OUT/'predictions.npz',labels=y,test_pred=pred['test'],valid_pred=pred['valid'],row_id=original['test']['row_id'])
    write(OUT/'report.json',extension)
    old6=np.load(BASE/'eat_large/test_predictions.npz')
    old7=np.load(OLD7/'diagnostic_predictions.npz')
    assert np.array_equal(old6['row_id'],original['test']['row_id']) and np.array_equal(old7['six_test_labels'],y)
    summary={'completed_at':datetime.datetime.now().astimezone().isoformat(),'selection_by_validation':best,
             'baseline_six':baseline['test'],'expanded_six':candidate['test'],
             'baseline_seven':measure(y,old7['six_test_pred']),'expanded_seven':extension,
             'six_delta_pp':100*(candidate['test']['top1']-baseline['test']['top1']),
             'seven_delta_pp':100*(extension['six_class_test_top1']-np.mean(old7['six_test_pred']==y)),
             'six_delta_ci':paired_ci(rows['test'],old6['logits'].argmax(1),original['test']['logits'].argmax(1)),
             'seven_delta_ci':paired_ci(rows['test'],old7['six_test_pred'],pred['test']),
             'data':candidate['data_summary'],'history':json.loads((RUN/'eat_large/history.json').read_text()),
             'variants': {name: {'valid': value['valid'], 'test': value['test'], 'selected_candidate': value['selected_candidate']} for name,value in variants.items()},
             'selected_expanded_variant': chosen_variant,
             'limitations':['북한말 독립 검증/테스트 없음. 7개 선택지 성능은 기존 6클래스 테스트에 한정.',
                            '데이터 확대는 같은 화자의 추가 녹음이며 새로운 화자 증가는 없음.',
                            '고정 테스트는 이전 실험에서 이미 열람한 개발 비교셋. 완전히 새로운 외부 평가셋은 아님.',
                            '표준어 후보 라벨 의미와 녹음 방식 편향, 실제 마이크 성능은 추가 검증 필요.'],
             'deployed':False}
    write(RUN/'final_report.json',summary)
    lines=[]
    for i,lab in enumerate(LABELS[:6]):
        lines.append({'class':lab,'name':KO[i], 'baseline_six_recall':baseline['test']['per_class'][lab]['recall_top1'],
                      'expanded_six_recall':candidate['test']['per_class'][lab]['recall_top1'],
                      'baseline_seven_recall':summary['baseline_seven']['per_class'][i]['recall'],
                      'expanded_seven_recall':extension['per_class'][i]['recall'],
                      'expanded_seven_precision':extension['per_class'][i]['precision'],
                      'expanded_seven_f1':extension['per_class'][i]['f1'],
                      'expanded_false_north':extension['per_class'][i]['false_north']})
    with (RUN/'final_per_class.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(lines[0]));writer.writeheader();writer.writerows(lines)
    def pct(x):return f'{x*100:.2f}%'
    page='<!doctype html><meta charset="utf-8"><title>방언 분류 확대 실험 결과</title><style>body{font:16px sans-serif;line-height:1.7;max-width:1150px;margin:40px auto;padding:20px}table{border-collapse:collapse;width:100%}th,td{padding:10px;border:1px solid #ddd;text-align:right}th{background:#edf3fa}img{max-width:100%}</style>'
    page+='<h1>방언 분류 데이터 확대 실험 결과</h1><p>완료: '+summary['completed_at']+'</p><p>검증셋 기준 선택: '+best+' / 확대 실험 내부 선택: '+candidate['selected_candidate']+'</p>'
    page+='<table><tr><th>측정</th><th>기존</th><th>확대</th><th>차이</th></tr>'
    page+=f'<tr><td>6개 선택지 / 6클래스 테스트</td><td>{pct(baseline["test"]["top1"])}</td><td>{pct(candidate["test"]["top1"])}</td><td>{summary["six_delta_pp"]:+.2f}%p</td></tr>'
    page+=f'<tr><td>7개 선택지 / 6클래스 테스트</td><td>{pct(summary["baseline_seven"]["six_class_test_top1"])}</td><td>{pct(extension["six_class_test_top1"])}</td><td>{summary["seven_delta_pp"]:+.2f}%p</td></tr></table>'
    page+='<h2>클래스별 재현율</h2><table><tr><th>클래스</th><th>기존6</th><th>확대6</th><th>기존7</th><th>확대7</th><th>북한말 오탐</th></tr>'
    for r in lines:
        page+='<tr><td>'+r['name']+'</td>'+''.join('<td>'+pct(r[k])+'</td>' for k in ['baseline_six_recall','expanded_six_recall','baseline_seven_recall','expanded_seven_recall'])+'<td>'+str(r['expanded_false_north'])+'</td></tr>'
    page+='</table><h2>통계와 학습 이력</h2><pre>'+html.escape(json.dumps({k:summary[k] for k in ['six_delta_ci','seven_delta_ci']},ensure_ascii=False,indent=2))+'</pre>'
    page+='<p>전체 학습 이력과 데이터 분할은 <a href="final_report.json">JSON</a>, 클래스별 수치는 <a href="final_per_class.csv">CSV</a>로 저장했습니다.</p><h2>해석</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in summary['limitations'])+'</ul>'
    (RUN/'final_report.html').write_text(page)
    status('complete',status='complete',six_delta_pp=summary['six_delta_pp'],seven_delta_pp=summary['seven_delta_pp'],report=str(RUN/'final_report.html'))


if __name__=='__main__':
    try: main()
    except Exception as exc:
        status('failed',status='failed',error=type(exc).__name__+': '+str(exc))
        raise
