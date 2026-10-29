"""Publish P3 completion records from existing results; never train/select a model.

Waits for the experiment's authoritative exit record. Updates only its own
progress entry, and appends a dated result to the existing experiment documents.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()


def pct(value):
    return '미측정' if value is None else '%.2f%%'%(100*value)


def append_once(path,marker,text):
    if not path.is_file(): raise FileNotFoundError(str(path))
    old=path.read_text()
    if marker in old: return
    with path.open('a') as f: f.write('\n\n'+marker+'\n\n'+text+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment',required=True); ap.add_argument('--repository',required=True)
    ap.add_argument('--progress-root',required=True)
    a=ap.parse_args(); r=Path(a.experiment); repo=Path(a.repository); progress=Path(a.progress_root)
    while not (r/'run.exit').exists(): time.sleep(30)
    code=int((r/'run.exit').read_text())
    receipt=dict(updated_at=now(),run_exit=code,service_model_changed=False)
    if code==0:
        result=json.loads((r/'comparison/report.json').read_text())
        assert result['status']=='P3_development_comparison_completed'
        receipt.update(status=result['status'],development_gate_pass={arm:v['development_gate_pass']
                       for arm,v in result['comparisons'].items()})
        table=['| 조건 | step | greedy seen FPR | greedy unseen FPR | 고정 CAL unseen FPR | 정확 진단 / 오류 |',
               '|---|--:|--:|--:|--:|---|']
        reports={}
        for arm in ('B0','B1','B2','B3'):
            for step in (3000,4000):
                x=json.loads((r/('%s_eval_%d'%(arm,step))/'report.json').read_text()); reports[(arm,step)]=x
                seen=x['results']['valid']['all']['greedy']; unseen=x['results']['valid_unseen']['all']
                c=unseen['calibrated']
                table.append('| %s | %d | %s | %s | %s | %d/%d (%s) |'%(arm,step,
                    pct(seen['normal_fpr']),pct(unseen['greedy']['normal_fpr']),pct(c['normal_fpr']),
                    c['exact'],c['errors'],pct(c['exact_recall'])))
        gate=' · '.join('%s %s'%(arm,'통과' if v else '미달') for arm,v in receipt['development_gate_pass'].items())
        diagnostic=[]
        for arm in ('B0','B1','B2','B3'):
            s=reports[(arm,4000)]['suri']
            diagnostic.append('%s ㅜ정답%d/%d·소리출력%d'%(arm,s.get('u_correct',0),s['total'],s.get('whole_sori',0)))
        body=('완료 확인: '+receipt['updated_at']+' · `run.exit=0`. 네 조건 각2,000회, 총8,000회 본 학습 및8개 checkpoint 평가를 완료했다. '
              '각 조건의 행·오디오 노출·배치 hash 일치를 확인했다.\n\n'+'\n'.join(table)+'\n\n'
              '**사전 개발 기준:** '+gate+'. 주 비교는B3 4k 대B0 4k이며 후보를 사후 교체하지 않았다. '
              '측정FPR이1%를 넘는 행은 `Recall @ FPR≤1%` 달성값이 아니다.\n\n'
              '‘수리’4k 진단: '+'; '.join(diagnostic)+'. 이는 전체 모델 채택 지표가 아니다.\n\n'
              '[조건별 판정·paired CI·검출 보존·궤적 차이](<'+str(r/'comparison/report.json')+'>), '
              '[요약 표](<'+str(r/'comparison/summary.tsv')+'>) 및 각 평가 폴더의 `all_phone_groups.tsv`, '
              '`confusions.tsv`, `sequences.jsonl`에 전체 음소·문맥·삭제·삽입 근거가 있다. '
              '개발 통과 여부와 관계없이 신규 독립 문장·화자 및 생산API/VAD/G2P/표시 검증은 남아 있다. '
              '**운영 모델은 변경하지 않았다.**')
        docs=repo/'docs'
        audit=docs/'발음평가_진단누락정보_감사_2026-10-26.md'
        append_once(audit,'### 19.5 P3 자동 완료 집계',body)
        short=('P3 네 조건의8,000회 본 학습·8개 checkpoint 평가가 '+receipt['updated_at']+
               ' 완료됐다. 사전 개발 기준은 '+gate+'. [감사 §19.5](발음평가_진단누락정보_감사_2026-10-26.md#195-p3-자동-완료-집계)에 '
               '실측 결과를 추가했다. 운영 교체·신규 독립 문장·생산 API 검증 완료를 뜻하지 않는다.')
        append_once(docs/'발음평가_모델별_데이터셋_학습명세_2026-10-26.md','### M12 P3 완료 집계',short)
        append_once(docs/'발음평가_개선작업_종합보고서_2026-10-26.md','**P3 자동 완료 집계**',short)
        # Keep the dated launch record, but make its temporal scope unambiguous.
        old=audit.read_text(); old=old.replace('**본 실험 결과는 아직 나오지 않았다.** 아래는 성능을 보기 전에 고정한 설계와 실행 점검이다.',
            '**아래 §19.1~19.4는 실행 전 기록이며 완료 결과는 §19.5에 있다.** 아래는 성능을 보기 전에 고정한 설계와 실행 점검이다.')
        audit.write_text(old)
        model_doc=docs/'발음평가_모델별_데이터셋_학습명세_2026-10-26.md'
        model_doc.write_text(model_doc.read_text().replace(
            '## 16. M12: H0 중간 CTC·층 결합 4조건 — P3 실행 중',
            '## 16. M12: H0 중간 CTC·층 결합 4조건 — P3 개발 평가 완료').replace(
            '**아직 성능 결과가 아니다.** 출발점은 통제 H0',
            '**이 문단은 실행 전 명세이며 완료 결과는 아래 M12 P3 완료 집계에 있다.** 출발점은 통제 H0'))
        summary_doc=docs/'발음평가_개선작업_종합보고서_2026-10-26.md'
        summary_doc.write_text(summary_doc.read_text().replace(
            '**본 학습 결과는 아직 미측정이며 운영 모델·라벨은 그대로다.**',
            '**시작 시점에는 본 학습 결과가 미측정이었다. 완료 집계는 아래에 있으며 운영 모델·라벨은 그대로다.**'))
    else:
        receipt.update(status='P3_pipeline_failed',performance_claim=False)
    path=progress/'progress.json'
    p=json.loads(path.read_text()); job=p.get('running_job') or {}
    if isinstance(job,dict) and job.get('experiment')==str(r):
        p['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        p['status']=receipt['status']; p['running_job']=None; p['goal_complete']=False
        p['next']='Review P3 result and separate development success from independent/API validation.' if code==0 else 'Inspect P3 failure logs; do not claim performance completion.'
        p['next_actions']=[p['next'],'Retain independent new-text/speaker and full API acceptance requirements.']
        path.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
        source=progress/'source_manifest.json'; manifest=json.loads(source.read_text())
        for path in (repo/'docs').glob('발음평가_*2026-10-26.md'):
            key=str(path.relative_to(repo))
            if key in manifest: manifest[key]=hashlib.sha256(path.read_bytes()).hexdigest()
        source.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (r/'reporting_finished.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__': main()
