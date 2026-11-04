"""Independently recount completed MoE outputs and publish a compact result table."""
import argparse
import collections
import datetime
import gzip
import hashlib
import json
import time
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def recount(path,threshold):
    sites=collections.Counter();gaps=collections.Counter()
    with gzip.open(str(path),'rt') as f:
        for line in f:
            row=json.loads(line)
            for s in row['sites']:
                if s['status'] not in ('normal','substitution','deletion'):continue
                kind='normal' if s['status']=='normal' else 'error';sites[kind]+=1
                if not s['observed']:continue
                sites['observed_'+kind]+=1
                alert=s['prediction_options'][0]!=s['expected'] and (threshold is None or s['score']>=threshold)
                if alert:
                    if kind=='normal':sites['fp']+=1
                    elif s['prediction_options'][0]==s['actual']:sites['exact']+=1
            for g in row['gaps']:
                if g['actual'] is None:continue
                kind='error' if g['actual'] else 'normal';gaps[kind]+=1
                if g['predicted'] is None:continue
                gaps['observed_'+kind]+=1
                if kind=='normal' and g['predicted']:gaps['fp']+=1
                if kind=='error' and g['predicted']==g['actual']:gaps['exact']+=1
    return sites,gaps


def ratio(c,num,den):
    n,d=c.get(num,0),c.get(den,0)
    return '{:,}/{:,} ({:.2f}%)'.format(n,d,100*n/d) if d else '평가 불가'


def summarize(root):
    done=read(root/'pipeline/completion.json');report=read(root/'comparison/report.json')
    assert (root/'pipeline/run.exit').read_text().strip()=='0' and done['status']=='complete'
    assert sha(root/'comparison/report.json')==done['comparison']['report_sha256']
    p=read(root/'protocol.json');verified={}
    folders={'C0':Path(p['historical_root'])/'evaluation/C0',
             'uniform':root/'evaluation/uniform','routed':root/'evaluation/routed'}
    for arm,folder in folders.items():
        threshold=read(folder/'calibration.json')['threshold']
        for role in ('cal','inner','outer'):
            s,g=recount(folder/(role+'_records.jsonl.gz'),threshold)
            for actual,kind in ((s,'sites'),(g,'gaps')):
                expected=report['roles'][role][arm][kind]['all']['counts']
                for k in ('normal','error','observed_normal','observed_error','fp','exact'):
                    assert actual[k]==expected.get(k,0),(arm,role,kind,k)
            verified[arm+'/'+role]='independent scalar counts match'
    timestamp=datetime.datetime.fromtimestamp((root/'pipeline/completion.json').stat().st_mtime,
        datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST')
    passed=report['pilot_passed']; status='사전 기준 통과' if passed else '사전 기준 미달'
    lines=['**{} 완료. MoE 후보는 {}.** 운영 모델 변경 없음.'.format(timestamp,status),'',
        '두 조건 각각 2,000회, 총 **4,000회 본 학습**과 각 **33,657녹음 평가**를 완료했다. '
        '같은 부모·배치·오디오 노출을 확인했다. 보정 문턱은 모델별 CAL에서만 정했다.','',
        '| 평가 | 모델 | 정상 헛지적 | 전체 확정 오류의 정확 진단 |','|---|---|---:|---:|']
    labels={'C0':'기존 일반 CTC(C0)','uniform':'동일 비율 대조군','routed':'**MoE 후보**'}
    for role,label in [('inner','내부'),('outer','바깥')]:
        for arm in folders:
            c=report['roles'][role][arm]['sites']['all']['counts']
            lines.append('| {} | {} | {} | {} |'.format(label,labels[arm],ratio(c,'fp','observed_normal'),ratio(c,'exact','error')))
    lines+=['','정상 헛지적은 관측 가능한 정상 분모, 정확 진단은 미관측까지 포함한 전체 확정 오류 분모다.','',
        '| 평가 | 모델 | 정상 Coverage | 오류 Coverage |','|---|---|---:|---:|']
    for role,label in [('inner','내부'),('outer','바깥')]:
        for arm in folders:
            c=report['roles'][role][arm]['sites']['all']['counts']
            lines.append('| {} | {} | {} | {} |'.format(label,labels[arm],ratio(c,'observed_normal','normal'),ratio(c,'observed_error','error')))
    names={'normal_FPR':'정상 헛지적 1%', 'exact_recall_improves':'정확 진단 증가',
           'normal_observation_coverage':'정상 Coverage','error_observation_coverage':'오류 Coverage',
           'protected_error_recall':'음소·삭제·삽입 검출 보호','normal_gap_fpr':'정상 gap 보호'}
    lines+=['','| 내부 통과 검사 | 동일 비율 대조군 대비 | 기존 C0 대비 |','|---|---|---|']
    for key,label in names.items():
        values=['통과' if report['inner_gate_vs'][ref]['checks'][key] else '**미달**' for ref in ('uniform','C0')]
        lines.append('| {} | {} | {} |'.format(label,*values))
    lines+=['','| 바깥 평가 범주 | 기존 C0 | 동일 비율 | MoE |','|---|---:|---:|---:|']
    for group,label in [('truth:deletion','확정 삭제 정확 진단')]:
        values=[ratio(report['roles']['outer'][arm]['sites'][group]['counts'],'exact','error') for arm in folders]
        lines.append('| {} | {} | {} | {} |'.format(label,*values))
    for num,den,label in [('exact','error','확정 삽입열 정확 진단'),('fp','observed_normal','정상 gap 헛삽입')]:
        values=[ratio(report['roles']['outer'][arm]['gaps']['all']['counts'],num,den) for arm in folders]
        lines.append('| {} | {} | {} | {} |'.format(label,*values))
    lines+=['','바깥 문항 단위 paired bootstrap 95% 구간(후보−대조군):']
    for ref in ('uniform','C0'):
        ci=report['paired']['outer'][ref]['item']['ci95']
        lines.append('- {} 대비 정상 헛지적 **{:+.2f}~{:+.2f}%p**, 정확 진단 **{:+.2f}~{:+.2f}%p**.'.format(
            labels[ref],*(100*x for x in ci['FPR']),*(100*x for x in ci['full_error_exact'])))
    lines+=['','고정 모델·보정 문턱의 조건부 구간이며, 학습 seed·CAL 재표집·반복 탐색을 포함하지 않는다. '
        '이번 결과는 재사용한 개발 자료에서의 한 후보 비교다. 모든 MoE 방법의 가능·불가능을 판정하지 않는다.','',
        '| MoE 라우터: 바깥 평가 | 전문가별 평균 비중 | 평균 엔트로피(자연로그) |','|---|---|---:|']
    routing=read(root/'evaluation/routed/routing.json')['outer']
    for kind,label in [('all','전체 유효 프레임'),('nonblank','인식 blank 제외 프레임')]:
        d=routing[kind];mean=[v/d['frames'] for v in d['mass']]
        lines.append('| {} | {} | {:.4f} |'.format(label,' / '.join('{:.2f}%'.format(100*v) for v in mean),d['entropy_sum']/d['frames']))
    lines+=['','평균 비중만으로 전문가의 음소별 전문화를 증명하지 않는다. '
        '상세 프레임 분산·최대 비율 선택 횟수는 `routing.json`에 있다.','',
        '**검증:** 저장 예측의 정상·오류·관측·헛지적·정확 진단을 별도 스칼라 코드로 재계산했다. '
        '전체 음소·문항·모국어·삭제·삽입 표, 화자 단위 구간과 고정 분모의 미관측 범위는 비교 보고서에 보존했다.','',
        '[전체 비교·검사](<{}/comparison/report.json>) · [완료 기록](<{}/pipeline/completion.json>)'.format(root,root)]
    text='\n'.join(lines)+'\n'
    (root/'SUMMARY.md').write_text(text)
    verification=dict(status='passed',checks=verified,summary_sha256=sha(root/'SUMMARY.md'),
        comparison_sha256=sha(root/'comparison/report.json'),source_sha256=sha(__file__),created_unix=time.time())
    (root/'summary_verification.json').write_text(json.dumps(verification,indent=2))
    return text


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--wait',action='store_true');ap.add_argument('--document',type=Path)
    a=ap.parse_args()
    initial_hash=sha(a.document) if a.document else None
    (a.root/'summary_launch.json').write_text(json.dumps(dict(source_sha256=sha(__file__),
        document=str(a.document),initial_document_sha256=initial_hash,created_unix=time.time()),indent=2))
    while a.wait and not (a.root/'pipeline/run.exit').exists():time.sleep(10)
    if (a.root/'pipeline/run.exit').read_text().strip()!='0':
        (a.root/'summary.exit').write_text('1\n');raise RuntimeError('pipeline failed; no performance summary published')
    text=summarize(a.root)
    if a.document and sha(a.document)==initial_hash:
        content=a.document.read_text();begin='<!-- MOE_RESULT_START -->';end='<!-- MOE_RESULT_END -->'
        assert content.count(begin)==content.count(end)==1
        before,remainder=content.split(begin);_,after=remainder.split(end)
        a.document.write_text(before+begin+'\n\n'+text+'\n'+end+after)
        document_update='completed'
    else:document_update='not requested or document changed; use artifact SUMMARY.md'
    (a.root/'summary_completion.json').write_text(json.dumps(dict(status='complete',document_update=document_update,
        summary_sha256=sha(a.root/'SUMMARY.md')),indent=2))
    (a.root/'summary.exit').write_text('0\n')


if __name__=='__main__':main()
