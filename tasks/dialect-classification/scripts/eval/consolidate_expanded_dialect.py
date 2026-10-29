"""Select a small predeclared ensemble grid on validation and report paired results."""
import csv
import datetime
import html
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from report_expanded_dialect import measure,paired_ci,load,write,LABELS,KO
from sklearn.metrics import f1_score

ROOT=Path('/data/speech-ai/out/dialect');RUN=ROOT/'expanded_six_class_20261013'
BASE=ROOT/'standard_six_class_20261013/eat_large'
OLD7=ROOT/'seven_class_north_pilot_20261013/feature_head'


def status(stage,**extra):
    write(RUN/'consolidation_status.json',dict(stage=stage,pid=os.getpid(),updated_at=datetime.datetime.now().astimezone().isoformat(),**extra))


def softmax(x):
    x=x-x.max(1,keepdims=True);x=np.exp(x);return x/x.sum(1,keepdims=True)


def combine(eat,xlsr,alpha):
    regional=((1-alpha)*softmax(eat[:,:5])+alpha*softmax(xlsr)).argmax(1)
    return np.where(eat.argmax(1)==5,5,regional)


def main():
    status('waiting_for_experiments',status='running')
    for filename in ['report_status.json','xlsr_cascade/progress.json']:
        while True:
            p=RUN/filename;state=json.loads(p.read_text()) if p.exists() else {}
            if state.get('status')=='failed':raise RuntimeError(filename+': '+str(state))
            if state.get('status')=='complete':break
            time.sleep(10)
    status('select_ensemble_on_validation',status='running')
    summary=json.loads((RUN/'final_report.json').read_text())
    single_selection=json.loads((RUN/'selection.json').read_text())
    protocol=json.loads((RUN/'ensemble_protocol.json').read_text())
    expanded=Path(single_selection['candidate_dir'])
    dirs={'baseline':BASE,'expanded_validation_selected':expanded}
    valid={name:np.load(path/'valid_predictions.npz') for name,path in dirs.items()}
    xr_valid=np.load(RUN/'xlsr_cascade/xlsr_valid_logits.npz')
    y=valid['baseline']['labels'];rows=load(RUN/'manifests/test.csv')
    search=[]
    for name,z in valid.items():
        assert np.array_equal(z['row_id'],xr_valid['row_id'])
        for alpha in protocol['alpha_grid']:
            pred=combine(z['logits'],xr_valid['logits'],alpha)
            search.append({'eat_candidate':name,'alpha':alpha,'validation_top1':float(np.mean(pred==y)),
                           'validation_macro_f1':float(f1_score(y,pred,labels=range(6),average='macro',zero_division=0))})
    best=max(search,key=lambda r:(r['validation_top1'],r['validation_macro_f1'],-r['alpha'],r['eat_candidate']=='baseline'))
    write(RUN/'ensemble_selection.json',{'protocol':protocol,'validation_candidates':search,'selected':best})
    # Test scores are computed only after the above selection record is written.
    test=np.load(dirs[best['eat_candidate']]/'test_predictions.npz')
    xr_test=np.load(RUN/'xlsr_cascade/xlsr_test_logits.npz');assert np.array_equal(test['row_id'],xr_test['row_id'])
    test_y=test['labels'];six_pred=combine(test['logits'],xr_test['logits'],best['alpha'])
    if best['eat_candidate']=='baseline':
        north=np.load(OLD7/'diagnostic_predictions.npz')['six_test_pred']==6
        north_path=OLD7/'seven_option_head.pt'
    else:
        north=np.load(RUN/'seven_class/predictions.npz')['test_pred']==6
        north_path=RUN/'seven_class/seven_option_head.pt'
    seven_pred=np.where(north,6,six_pred)
    old_six=np.load(BASE/'test_predictions.npz')['logits'].argmax(1)
    old_seven=np.load(OLD7/'diagnostic_predictions.npz')['six_test_pred']
    six=measure(test_y,six_pred);seven=measure(test_y,seven_pred)
    baseline_six=measure(test_y,old_six);baseline_seven=measure(test_y,old_seven)
    report={'completed_at':datetime.datetime.now().astimezone().isoformat(),
            'selected':best,'single_model_selection':single_selection,'validation_candidates':search,
            'baseline_six':baseline_six,'baseline_seven':baseline_seven,'selected_six':six,'selected_seven':seven,
            'six_delta_pp':100*(six['six_class_test_top1']-baseline_six['six_class_test_top1']),
            'seven_delta_pp':100*(seven['six_class_test_top1']-baseline_seven['six_class_test_top1']),
            'six_delta_ci':paired_ci(rows,old_six,six_pred),'seven_delta_ci':paired_ci(rows,old_seven,seven_pred),
            'requires_two_encoders':best['alpha']>0,'expanded_variants':summary['variants'],
            'expanded_selected_seven':summary['expanded_seven'],'north_independent_test_n':0,
            'data':summary['data'],'limitations':summary['limitations'],
            'regularization_optimization':json.loads((RUN/'regularized_head/report.json').read_text())['history'],
            'latency':'Two-encoder serving latency was not benchmarked; offline shared-GPU timing is not a deployment benchmark.',
            'deployed':False}
    report['limitations'].append('기반 EAT 인코더는 이전 5방언 학습에서 596177구간을 사용함. 이번 30000→82006건 확대는 6클래스 확장 단계의 학습량임.')
    report['limitations'].append('규제 비교 3개 후보는 LBFGS 400회 제한에 도달해 수렴이 확인되지 않은 탐색 결과임.')
    write(RUN/'comparison_report.json',report)
    write(RUN/'model_bundle.json',{'method':'EAT Standard gate, weighted regional probabilities, EAT North gate',
          'alpha_xlsr':best['alpha'],'eat_checkpoint':str(dirs[best['eat_candidate']]/'best_adapted.pt'),
          'xlsr_checkpoint':str(ROOT/'finetune_speakerid_20261012/xlsr2b/best_adapted.pt'),
          'north_detector_checkpoint':str(north_path),'labels':LABELS,'protocol':protocol})
    np.savez_compressed(RUN/'selected_predictions.npz',labels=test_y,row_id=test['row_id'],six_pred=six_pred,seven_pred=seven_pred)
    per=[]
    for i in range(6):
        per.append({'class':LABELS[i],'name':KO[i],'n':1000,
                    'baseline_six_recall':baseline_six['per_class'][i]['recall'],
                    'baseline_seven_recall':baseline_seven['per_class'][i]['recall'],
                    'selected_six_recall':six['per_class'][i]['recall'],
                    'selected_seven_recall':seven['per_class'][i]['recall'],
                    'selected_seven_precision':seven['per_class'][i]['precision'],
                    'selected_seven_f1':seven['per_class'][i]['f1'],
                    'north_false_positives':seven['per_class'][i]['false_north']})
    with (RUN/'comparison_per_class.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(per[0]));w.writeheader();w.writerows(per)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=np.arange(6);fig,ax=plt.subplots(figsize=(11,5));width=.35
    for offset,label,vals,color in [(-width/2,'Baseline seven-option',[r['baseline_seven_recall']*100 for r in per],'#95a5b7'),(width/2,'Selected seven-option',[r['selected_seven_recall']*100 for r in per],'#1479bc')]:
        bars=ax.bar(x+offset,vals,width,label=label,color=color)
        for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.35,f'{v:.1f}',ha='center',fontsize=9)
    ax.set_xticks(x);ax.set_xticklabels(LABELS[:6]);ax.set_ylim(0,105);ax.set_ylabel('Recall (%)');ax.legend(loc='lower right');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(RUN/'comparison_per_class.png',dpi=170);plt.close(fig)
    def pct(v):return f'{v*100:.2f}%'
    page='<!doctype html><meta charset="utf-8"><title>방언 분류 성능 개선 실험 결과</title><style>body{font:16px sans-serif;line-height:1.7;max-width:1200px;margin:32px auto;padding:24px}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{padding:9px;border:1px solid #ddd;text-align:right}th{background:#ecf3fa}img{width:100%}pre{white-space:pre-wrap}</style>'
    page+='<h1>방언 분류 성능 개선 실험 결과</h1><p>완료: '+report['completed_at']+'</p>'
    page+='<p>검증셋 선택: EAT '+best['eat_candidate']+', 지역 분류 XLSR 혼합 비율 '+str(best['alpha'])+'. 검증 정확도 '+pct(best['validation_top1'])+'.</p>'
    page+='<table><tr><th>테스트 조건</th><th>기존</th><th>선택 모델</th><th>변화</th></tr>'
    for label,a,b,delta in [('6개 선택지',baseline_six,six,report['six_delta_pp']),('북한말 포함 7개 선택지',baseline_seven,seven,report['seven_delta_pp'])]:
        page+=f'<tr><td>{label}</td><td>{pct(a["six_class_test_top1"])}</td><td>{pct(b["six_class_test_top1"])}</td><td>{delta:+.2f}%p</td></tr>'
    page+='</table><p>모든 테스트 수치는 기존 6개 클래스 6000건 기준입니다. 북한말을 포함한 독립 7클래스 정확도는 측정할 수 없습니다.</p><img src="comparison_per_class.png">'
    page+='<h2>클래스별 7개 선택지 성능</h2><table><tr><th>클래스</th><th>기존 재현율</th><th>개선 재현율</th><th>정밀도</th><th>F1</th><th>북한말 오탐</th></tr>'
    for r in per:page+='<tr><td>'+r['name']+'</td>'+''.join('<td>'+pct(r[k])+'</td>' for k in ['baseline_seven_recall','selected_seven_recall','selected_seven_precision','selected_seven_f1'])+'<td>'+str(r['north_false_positives'])+'</td></tr>'
    page+='</table><h2>실험별 결과</h2><table><tr><th>실험</th><th>검증 정확도</th><th>테스트 정확도</th><th>선택 후보</th></tr>'
    for name,v in summary['variants'].items():page+='<tr><td>'+name+'</td><td>'+pct(v['valid']['top1'])+'</td><td>'+pct(v['test']['top1'])+'</td><td>'+html.escape(v['selected_candidate'])+'</td></tr>'
    page+='</table><h2>개선 폭의 95% 신뢰구간</h2><pre>'+html.escape(json.dumps({k:report[k] for k in ['six_delta_ci','seven_delta_ci']},ensure_ascii=False,indent=2))+'</pre>'
    page+='<h2>해석과 한계</h2><ul>'+''.join('<li>'+html.escape(v)+'</li>' for v in report['limitations'])+'</ul><p>두 모델을 사용하는 후보의 실제 서비스 지연시간은 별도로 측정해야 합니다. 앱·웹·NPU 배포는 수행하지 않았습니다.</p><p><a href="comparison_report.json">전체 JSON</a> · <a href="comparison_per_class.csv">클래스별 CSV</a> · <a href="model_bundle.json">선택 모델 구성</a></p>'
    (RUN/'comparison_report.html').write_text(page)
    status('complete',status='complete',six_top1=six['six_class_test_top1'],seven_top1=seven['six_class_test_top1'],report=str(RUN/'comparison_report.html'))
    print(json.dumps({'selected':best,'six':six['six_class_test_top1'],'seven':seven['six_class_test_top1']}),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('failed',status='failed',error=type(exc).__name__+': '+str(exc));raise
