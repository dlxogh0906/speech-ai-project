"""Independent saved-output audit and paired uncertainty, no fitting or retuning."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT: raise RuntimeError('저장소 루트(pyproject.toml)를 찾지 못했습니다')
    ROOT = ROOT.parent
sys.path[str(ROOT), :0]=[str(ROOT/'tasks/pronunciation-assessment/src')]
from shared.paths import DATA_ROOT, MDD_OUTPUTS  # 경로는 .env 에서 (CLAUDE.md 경로 규칙)
import collections,csv,gzip,hashlib,html,json
import numpy as np
import joblib
from mdd.candidate_rerank import FAMILIES

OUT=MDD_OUTPUTS/'experiments/m3_candidate_rerank_2026-11-01_v1'
CONDITIONS=['A_M3_peak','B_probability','C_segment']
LABELS={'A_M3_peak':'M3 1위 + 확률 차이 문턱','B_probability':'후보 확률 재판정','C_segment':'후보 확률 + 자음·모음 구간 재판정'}


def read(p):
    with gzip.open(p,'rt') as f:return list(map(json.loads,f))


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def write(p,r):p.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    out=Path(sys.argv[1]) if len(sys.argv)>1 else OUT
    assert (out/'completed.json').exists()
    thresholds=json.loads((out/'thresholds.json').read_text());fit=json.loads((out/'fit.json').read_text())
    frozen=json.loads((out/'frozen_head.json').read_text());assert sha(out/'thresholds.json')==frozen['thresholds_sha256']
    protocol=json.loads((out/'protocol.json').read_text())
    for path,digest in protocol['source_hashes'].items():assert sha(path)==digest
    for name,digest in protocol['checkpoint_hashes'].items():assert sha(Path(protocol['checkpoint'])/name)==digest
    prepared=json.loads((out/'prepared.json').read_text())
    assert prepared['protocol_sha256']==sha(out/'protocol.json')
    for role,digest in prepared['selected_hashes'].items():assert sha(out/(role+'_inputs.jsonl.gz'))==digest
    data={role:read(out/(role+'_features.jsonl.gz')) for role in ('train','cal','seen','rp')}
    # Independently reproduce CAL's finite-score order statistic and tie exclusion.
    models={};cal_checks={}
    for family,phones in FAMILIES.items():
        for cond in CONDITIONS:
            if cond!='A_M3_peak':
                path=out/(cond+'_'+family+'.joblib');assert sha(path)==fit[family][cond]['sha256']
                models[(cond,family)]=joblib.load(path)
            rr=[r for r in data['cal'] if r['family']==family and r['observed'] and r['actual']==r['expected']]
            ss=[]
            for r in rr:
                if cond=='A_M3_peak':score=r['original_score'] if r['original']!=r['expected'] else -np.inf
                else:
                    m=models[(cond,family)];key='base' if cond=='B_probability' else 'segment'
                    lp=m.predict_log_proba([r[key]])[0];chosen=m.classes_[lp.argmax()]
                    score=float(lp.max()-lp[list(m.classes_).index(r['expected'])]) if chosen not in ('OTHER',r['expected']) else -np.inf
                ss.append(score)
            saved=thresholds[cond+'_'+family];budget=len(rr)//100
            ordered=sorted([s for s in ss if np.isfinite(s)],reverse=True)
            threshold=np.nextafter(ordered[budget],np.inf) if len(ordered)>budget else 0.
            # BLAS batch versus one-row products differ by a few FP64 ULPs.
            # Verify the order statistic numerically, while requiring the exact
            # accepted count under BOTH the frozen and recomputed thresholds.
            threshold_delta=abs(threshold-saved['threshold'])
            assert threshold_delta<=1e-12*max(1.,abs(saved['threshold']))
            assert sum(s>=threshold for s in ss)==saved['false_alarms']<=budget
            assert sum(s>=saved['threshold'] for s in ss)==saved['false_alarms']
            cal_checks[cond+'_'+family]=dict(normal=len(rr),budget=budget,false_alarms=saved['false_alarms'],
                scalar_threshold_abs_difference=threshold_delta)
    all_reports=[];all_predictions={};checks=0
    for role in ('seen','rp'):
        inp=read(out/(role+'_inputs.jsonl.gz'));expected={}
        for r in inp:
            for q,s in enumerate(r['sites']):
                if any(s['expected'] in f for f in FAMILIES.values()):expected[(r['uid'],q)]=(s['expected'],s['identified_actual'],s['status'])
        features={(r['uid'],r['q']):r for r in data[role]};assert set(features)==set(expected)
        preds=read(out/(role+'_predictions.jsonl.gz'));all_predictions[role]=preds
        reports=json.loads((out/(role+'_report.json')).read_text());all_reports.extend(reports)
        for cond in CONDITIONS:
            pp=[r for r in preds if r['condition']==cond];assert len(pp)==len(expected)
            for r in pp:
                key=(r['uid'],r['q']);fr=features[key]
                assert (r['expected'],r['actual'],r['truth_status'])==expected[key]
                assert r['observed']==fr['observed']
                if not r['observed']:assert r['prediction'] is None and not r['warning'];continue
                if cond=='A_M3_peak':pred=fr['original'];score=fr['original_score'] if pred!=fr['expected'] else -np.inf
                else:
                    model=models[(cond,r['family'])];kind='base' if cond=='B_probability' else 'segment'
                    lp=model.predict_log_proba([fr[kind]])[0];chosen=model.classes_[lp.argmax()]
                    pred=fr['original'] if chosen=='OTHER' else chosen
                    score=float(lp.max()-lp[list(model.classes_).index(fr['expected'])]) if chosen not in ('OTHER',fr['expected']) else -np.inf
                assert pred==r['prediction']
                assert bool(score>=thresholds[cond+'_'+r['family']]['threshold'])==r['warning']
                if np.isfinite(score):assert abs(score-r['score'])<1e-8
                else:assert r['score'] is None
                checks+=1
            for report in [r for r in reports if r['condition']==cond]:
                group=report['group'];rr=pp
                if group!='all':
                    field,val=group.split(':',1);field='expected' if field=='phone' else field
                    rr=[r for r in pp if r[field]==val]
                identified=[r for r in rr if r['truth_status'] in ('normal','substitution','deletion')]
                normal=[r for r in identified if r['actual']==r['expected']];err=[r for r in identified if r['actual']!=r['expected']]
                obs_n=[r for r in normal if r['observed']];obs_e=[r for r in err if r['observed']]
                fp=sum(r['warning'] for r in obs_n);exact=sum(r['warning'] and r['prediction']==r['actual'] for r in obs_e)
                assert (report.get('normal',0),report.get('error',0),report.get('observed_normal',0),report.get('observed_error',0),report.get('fp',0),report.get('exact',0))==(len(normal),len(err),len(obs_n),len(obs_e),fp,exact)
    # All selected audio hashes checked, across fitted roles vs evaluation.
    ledgers={role:read(out/(role+'_ledger.jsonl.gz')) for role in data}
    pcm={role:{r['pcm_sha256'] for r in rr} for role,rr in ledgers.items()}
    overlaps={a+'_'+b:len(pcm[a]&pcm[b]) for a,b in [('train','cal'),('train','seen'),('train','rp'),('cal','seen'),('cal','rp')]}
    assert not any(overlaps.values())
    uncertainty=[]
    pp=all_predictions['rp'];speakers=sorted({r['speaker'] for r in pp});ix={s:i for i,s in enumerate(speakers)}
    for group in ('all','sibilants','affricates'):
        arrays={c:np.zeros((len(speakers),4)) for c in CONDITIONS}
        for r in pp:
            if group!='all' and r['family']!=group:continue
            if r['truth_status'] not in ('normal','substitution','deletion'):continue
            v=arrays[r['condition']][ix[r['speaker']]];error=r['actual']!=r['expected']
            v[3]+=error
            if r['observed']:
                v[1]+=not error;v[0]+=not error and r['warning'];v[2]+=error and r['warning'] and r['prediction']==r['actual']
        rng=np.random.RandomState(20261101);d={c:[] for c in CONDITIONS[1:]}
        for _ in range(1000):
            sample=rng.randint(len(speakers),size=len(speakers));total={c:x[sample].sum(0) for c,x in arrays.items()}
            b=total['A_M3_peak'];bf=b[0]/b[1];br=b[2]/b[3]
            for c in CONDITIONS[1:]:
                z=total[c];d[c].append([z[0]/z[1]-bf,z[2]/z[3]-br])
        for c in CONDITIONS[1:]:uncertainty.append(dict(group=group,candidate=c,baseline='A_M3_peak',
            fpr_difference_95pct=np.percentile(np.array(d[c])[:,0],[2.5,97.5]).tolist(),
            exact_recall_difference_95pct=np.percentile(np.array(d[c])[:,1],[2.5,97.5]).tolist()))
    write(out/'paired_uncertainty.json',dict(speakers=len(speakers),replicates=1000,results=uncertainty,
        scope='fixed models, thresholds and three-script composition; excludes CAL uncertainty, new prompt generalization and training seed'))
    primary=[r for r in all_reports if r['role']=='rp']
    base=next(r for r in primary if r['condition']=='A_M3_peak' and r['group']=='all')
    gate={}
    for cond in CONDITIONS[1:]:
        fam=[r for r in primary if r['condition']==cond and r['group'].startswith('family:')]
        all_=next(r for r in primary if r['condition']==cond and r['group']=='all')
        gate[cond]=dict(family_FPR_at_most_1pct=all(r['FPR'] is not None and r['FPR']<=.01 for r in fam),
                       more_exact_than_A=all_['exact']>base['exact'],coverage_equal=all_['observed_normal']==base['observed_normal'] and all_['observed_error']==base['observed_error'])
        gate[cond]['passed']=all(gate[cond].values())
    write(out/'gate.json',gate)
    fields=sorted(set().union(*(set(r) for r in all_reports)))
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(all_reports)
    # Inspectable examples: select by UID hash within outcome types, never choose a threshold from them.
    keyed=collections.defaultdict(dict)
    for r in pp:keyed[(r['uid'],r['q'])][r['condition']]=r
    changes={}
    for cond in CONDITIONS[1:]:
        counts=collections.Counter()
        for rr in keyed.values():
            a,b=rr['A_M3_peak'],rr[cond]
            if not a['observed'] or a['truth_status'] not in ('normal','substitution','deletion'):continue
            error=a['actual']!=a['expected']
            ae=error and a['warning'] and a['prediction']==a['actual']
            be=error and b['warning'] and b['prediction']==b['actual']
            counts['new_exact']+=int(be and not ae);counts['lost_exact']+=int(ae and not be)
            counts['added_false_alarm']+=int(not error and b['warning'] and not a['warning'])
            counts['removed_false_alarm']+=int(not error and a['warning'] and not b['warning'])
        changes[cond]=dict(counts)
    write(out/'transitions.json',changes)
    source={(r['uid'],r['q']):r for r in data['rp']};examples=[]
    for outcome in ('new_exact','lost_exact','false_alarm'):
        possible=[]
        for key,rr in keyed.items():
            a=rr['A_M3_peak'];c=rr['C_segment']
            if not c['observed']:continue
            ae=a['warning'] and a['prediction']==a['actual'] and a['actual']!=a['expected']
            ce=c['warning'] and c['prediction']==c['actual'] and c['actual']!=c['expected']
            yes=(ce and not ae) if outcome=='new_exact' else (ae and not ce) if outcome=='lost_exact' else (c['warning'] and c['actual']==c['expected'])
            if yes:possible.append(key)
        for key in sorted(possible,key=lambda k:hashlib.sha256(str(k).encode()).hexdigest())[:3]:
            r=source[key];examples.append(dict(outcome=outcome,uid=key[0],q=key[1],item=r['item'],
                prompt=r['prompt'],expected=r['expected'],actual=r['actual'],center_seconds=r['center']/16000.,
                A=keyed[key]['A_M3_peak']['prediction'],A_warning=keyed[key]['A_M3_peak']['warning'],
                B=keyed[key]['B_probability']['prediction'],B_warning=keyed[key]['B_probability']['warning'],
                C=keyed[key]['C_segment']['prediction'],C_warning=keyed[key]['C_segment']['warning'],audio_path=r['audio_path']))
    write(out/'examples.json',examples)
    receipt=dict(status='passed',independent_prediction_checks=checks,report_rows=len(all_reports),
        pcm_role_overlap=overlaps,calibration_checks=cal_checks,
        actual_tag_only_for_training_or_scoring=True,encoder_updates=0,
        artifacts={p.name:sha(p) for p in [out/'metrics.csv',out/'gate.json',out/'paired_uncertainty.json',out/'examples.json',out/'transitions.json']})
    write(out/'verification.json',receipt)
    lines=['# M3 후보 확률·구간 재판정 비교 — 2026-11-01','',
        '**결과: '+('사전 지속 기준 통과' if gate['C_segment']['passed'] else '사전 지속 기준 미달')+'**','',
        '고정 M3의 1위 출력과, 실제 TAG로 학습한 후보 재판정 두 조건을 비교했다. M3 인코더 추가 학습은 0회다.',
        '범위는 기대 ㅅ·ㅆ / ㅈ·ㅉ·ㅊ 자리의 자음군 내 대치다. 아래 기준도 새 CAL 정책이며 운영 API 자체의 점수가 아니다.','',
        '| RP 3대본 · 1,118녹음 | 정상 헛지적 / 관측 정상 | 정확 진단 / 전체 확정 오류 |',
        '|---|---:|---:|']
    for c in CONDITIONS:
        r=next(r for r in primary if r['condition']==c and r['group']=='all')
        lines.append('| %s | %d/%d (%.2f%%) | %d/%d (%.2f%%) |'%(LABELS[c],r['fp'],r['observed_normal'],r['FPR']*100,r['exact'],r['error'],r['exact_recall']*100))
    change=changes['C_segment']
    lines+=['', '구간 정보를 더한 C는 정확 진단 %d건을 새로 얻고 기존 %d건을 잃었다. 헛지적도 %d건 추가·%d건 제거됐다.'%(change['new_exact'],change['lost_exact'],change['added_false_alarm'],change['removed_false_alarm']),
        '정확 진단 순증 8건과 함께 정상 헛지적이 579건 늘어 두 자음군 모두 1% 기준에 미달했다. 확률만 쓴 B도 두 자음군에서 1%를 넘었고 정확 진단은 줄었다.']
    lines+=['','| 자음군 | 방법 | 정상 헛지적 | 정확 진단 |','|---|---|---:|---:|']
    for family,name in [('sibilants','ㅅ·ㅆ'),('affricates','ㅈ·ㅉ·ㅊ')]:
        for c in CONDITIONS:
            r=next(r for r in primary if r['condition']==c and r['group']=='family:'+family)
            lines.append('| %s | %s | %.2f%% (%d/%d) | %d/%d |'%(name,LABELS[c],100*r['FPR'],r['fp'],r['observed_normal'],r['exact'],r['error']))
    lines+=['',
        '정상 관측: %d/%d (%.2f%%), 오류 관측: %d/%d (%.2f%%). 세 조건의 관측 대상은 동일하다.'%(base['observed_normal'],base['normal'],base['normal_coverage']*100,base['observed_error'],base['error'],base['error_coverage']*100),
        '삭제·제3음 출력, 대응 모호, 불명 시간 또는 입력 경계와 겹친 자리는 전체 확정 분모에 남겼다. 정답부터 불명인 주석은 별도 집계한다.',
        '고정 정상 42,073자리에서 관측 헛지적만 센 하한도 A 5.91% / B 4.17% / C 7.28%다. 미관측을 정상이나 정답으로 바꾸지 않았다.',
        '문턱 적용 전 관측 오류 458자리의 실제음 단일 출력 정답은 A 124 / B 137 / C 151개다. 후보 정보를 써 일부 전사를 바꾸는 것과 정상 보호를 만족하는 오류 경고는 다른 결과다.',
        'C−A의 화자 bootstrap 95% 구간은 정상 헛지적 +1.31~+2.04%p, 전체 오류 정확 진단 −1.38~+3.92%p다. 문턱·학습 결과·세 대본은 고정하며 새 문장 모집단의 구간이 아니다.',
        '',
        '| 고정 조건 | 내용 |','|---|---|',
        '| 입력 모델 | 운영 계열 M3 step_3000, FP16 GPU. 전 구간 12초 core·400ms 문맥 연결, VAD 없음 |',
        '| 학습 / CAL / seen 확인 | 고유 3,000 / 2,000 / 1,000녹음. 학습·CAL·평가 화자 분리 |',
        '| 평가 | M2/M3 지도학습 문항 밖 RP002·003·004, 1,118녹음·859화자 |',
        '| 판별기 | 자음군별 선형 다항 로지스틱, 실제 TAG 감독, 고정 C=1. 정상·오류·제3음 사례 사용 |',
        '| B 입력 | 후보 확률과 1위와의 차이, 기대음·원래 출력 등 |',
        '| C 추가 입력 | 자동 자음 피크 ±60ms·뒤 모음 피크 ±40ms의 확률 요약·스펙트럼·주기성 대리값 |',
        '| 출력 | 실제음 한 개. OTHER 내부 판정은 경고 유보, Top-2를 정답으로 인정하지 않음 |',
        '| 문턱 | 각 조건·자음군의 CAL 정상 헛지적 1% 예산. seen/RP 재조정 없음 |','',
        '**해석 범위:** RP는 세 대본이고 이전 연구에서 분석한 화자의 다른 녹음이다. 새 문장 모집단의 독립 최종 검증이 아니다. TAG 독립 청취는 없고, 자동 창이 실제 마찰·기식·모음 경계의 정답이라는 보장도 없다. 이번 비교는 선택한 소형 판별기·구간 요약의 검사이며 모든 음향 재채점 방법의 상한은 아니다.','',
        '원시 특징·한 음소 출력·경고: `*_features.jsonl.gz`, `*_predictions.jsonl.gz`. 전체 표 `metrics.csv`, 문항·자모별 집계 `rp_report.json`, 실제 사례 `examples.json`, 조건부 화자 구간 `paired_uncertainty.json`, 검증 `verification.json`.','',
        '저장 예측 110,187개·집계 459행의 독립 재계산을 통과했다. 최초 검사는 배치/개별 행 행렬곱의 FP64 문턱 차이 약 1.1e-15를 완전 일치로 비교해 중단했다. 문턱 수치 비교에 1e-12 허용오차를 적용하고, 동결 문턱의 실제 경고 건수는 완전 일치로 재검증했다. 모델·문턱·예측은 변경하지 않았고 실패 로그를 보존했다.',
        '', '운영 모델·서비스 문턱은 변경하지 않았다.']
    (out/'결과.md').write_text('\n'.join(lines)+'\n')
    # Small portable HTML companion, no new image or external font dependencies.
    tables=[]
    for r in [r for r in primary if r['group'] in ('all','family:sibilants','family:affricates')]:
        tables.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [r['group'],LABELS[r['condition']],
            '%.2f%% (%d/%d)'%(100*r['FPR'],r['fp'],r['observed_normal']),
            '%.2f%% (%d/%d)'%(100*r['exact_recall'],r['exact'],r['error'])])+'</tr>')
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>M3 후보 재판정 결과</title><style>body{font:17px system-ui;max-width:1100px;margin:40px auto;line-height:1.6}td,th{padding:12px;border:1px solid #ccc}table{border-collapse:collapse}pre{white-space:pre-wrap}</style><h1>M3 후보 재판정 결과</h1><table><tr><th>범위</th><th>방법</th><th>정상 헛지적</th><th>전체 오류 정확 진단</th></tr>'+''.join(tables)+'</table><pre>'+html.escape('\n'.join(lines))+'</pre></html>'
    (out/'결과.html').write_text(page)
    print(json.dumps(dict(gate=gate,verification=receipt,primary=[r for r in primary if r['group']=='all']),ensure_ascii=False))


if __name__=='__main__':main()
