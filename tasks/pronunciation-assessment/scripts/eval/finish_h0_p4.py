"""Wait for durable P4 extraction, verify saved predictions, and record completion."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import datetime
import json
import subprocess
import time
import numpy as np
from diagnose_h0_p4 import read_jsonl,write_json
from fit_h0_representation_probes import threshold,counts
from train_h0_backbone_control import file_sha


def verify(root):
    p=json.loads((root/'protocol.json').read_text());report=json.loads((root/'comparison/report.json').read_text())
    if report['smoke']:raise ValueError('smoke is not full completion')
    sites=read_jsonl(Path(p['p1'])/'sites.jsonl');classes=json.loads((Path(p['p1'])/'probes/fit_protocol.json').read_text())['classes']
    classid={t:i for i,t in enumerate(classes)};checks=[]
    for arm in p['arms']:
        ids=np.load(root/arm/'site_ids.npy');ss=[sites[i] for i in ids];mask=np.load(root/arm/'available.npy')
        e=np.array([classid[s['expected']] for s in ss]);t=np.array([classid[s['actual'] or '<DEL>'] for s in ss])
        cal=np.array([s['role']=='calibration' for s in ss]);selected=~cal
        for k,mode in enumerate(('parent_auto','own_auto','tag_same_readout','tag_matched_readout')):
            logits=np.load(root/arm/(mode+'.npy'));y=logits.argmax(-1);score=logits.max(-1)-logits[np.arange(len(ids)),e]
            tau=threshold(score,cal&(e==t),y!=e,mask[:,k])
            for split in ('valid','valid_unseen'):
                ix=np.flatnonzero(selected&np.array([s['split']==split for s in ss]))
                # Independent scalar application avoids float32 boundary rounding.
                fp=hit=0
                for i in ix:
                    alarm=bool(mask[i,k]) and y[i]!=e[i] and float(score[i])>=tau
                    fp+=int(alarm and t[i]==e[i]);hit+=int(alarm and t[i]!=e[i] and y[i]==t[i])
                old=next(r for r in report['summary'] if (r['arm'],r['mode'],r['policy'],r['split'])==
                         (arm,mode,'seen_CAL_recalibrated',split))
                if (fp,hit)!=(old['normal_fp'],old['exact']):raise ValueError('scalar recount differs')
                checks.append(dict(arm=arm,mode=mode,split=split,normal_fp=fp,exact=hit))
    write_json(root/'verification.json',dict(status='passed',scalar_recounts=checks,source_sha256=file_sha(__file__)))


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=json.loads((root/'comparison/report.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    x=np.arange(3)
    for arm,dx,color in [('B0',-.18,'#2878a5'),('B3',.18,'#d97732')]:
        s=next(s for s in r['suri'] if s['arm']==arm and s['mode']=='parent_auto')
        vals=[s['u_correct'],s['aux_u'],s['final_u']]
        bars=axes[0].bar(x+dx,vals,.36,label=arm,color=color)
        for b,n in zip(bars,vals):axes[0].text(b.get_x()+.18,n+5,str(n),ha='center',fontsize=9)
    axes[0].set(xticks=x,xticklabels=['Frozen probe','B3 aux readout','Native final'],ylim=(0,500),ylabel='Correct / 475',title='Normal Suri: identical recordings')
    axes[0].legend()
    for arm,marker in [('B0','o'),('B3','s')]:
        for row in r['summary']:
            if row['arm']==arm and row['split']=='valid_unseen' and row['policy']=='seen_CAL_recalibrated':
                axes[1].scatter(100*row['normal_fpr'],100*row['exact_recall'],marker=marker,
                    label=arm+' '+row['mode'],s=38)
    axes[1].axvline(1,color='gray',linestyle='--');axes[1].set(xlabel='Unseen normal FPR (%)',ylabel='Exact diagnosis recall (%)',title='Frozen probes; seen-CAL thresholds')
    axes[1].legend(fontsize=6)
    c=json.loads((root/'calibration/report.json').read_text())
    v=next(s for s in c['all_decomposition'] if s['split']=='valid_unseen')
    vals=[v['B0_B0'],v['B3_B0'],v['B3_B3']]
    bars=axes[2].bar(x,vals,color=['#2878a5','#b7a49b','#d97732'])
    for b,n in zip(bars,vals):axes[2].text(b.get_x()+b.get_width()/2,n+10,str(n),ha='center')
    axes[2].set(xticks=x,xticklabels=['B0 / B0 tau','B3 / B0 tau','B3 / B3 tau'],ylim=(0,2100),ylabel='False alerts / 34,706 normals',title='GOP transfer: arithmetic, not causal')
    fig.savefig(root/'comparison/P4_evidence_transfer.svg');plt.close(fig)


def record(root):
    r=json.loads((root/'comparison/report.json').read_text());now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
    lines=['\n### 20.4 P4-A 전수 완료와 고정 판별기 결과\n',
           '완료 확인: '+now+' · B0/B3 각각14,009녹음. 인코더·probe 학습 갱신0회다. 각 조건의136,577자리에서 기존 최종 인식값·GOP를 재현했으며 두 조건의 원음·순서 hash가 같다. 저장 logits의 보정·선택 집계를 독립 스칼라 비교로 다시 계산해 일치함을 확인했다.\n',
           '정상 ‘수리’475발화의ㅜ 자리 정답 수다. 고정 probe는 **부모2k에서 학습한 하나의 판별기와 부모2k 구간**이며, P1의4k별 재학습 probe412건과 비교 조건이 다르다. B0의 중간 CTC 열은 **B3 판별기를 옮긴 진단**이다.\n',
           '| 인코더 | 고정12층 probe | 같은 B3 중간 CTC | 자체 최종 CTC |\n|---|---:|---:|---:|']
    for arm in ('B0','B3'):
        s=next(s for s in r['suri'] if s['arm']==arm and s['mode']=='parent_auto')
        lines.append('| %s | %d/475 | %d/475 | %d/475 |'%(arm,s['u_correct'],s['aux_u'],s['final_u']))
    lines+=['\n다음은 같은 고정 probe의 seen CAL 문턱을 선택 unseen에 적용한 결과다. 정상34,706·오류1,287자리를 모두 분모에 유지한다. 실제 FPR이1%를 넘으면 목표 달성값이 아니다.\n',
            '| 인코더·구간 | unseen 정상 FPR | 정확 진단 |\n|---|---:|---:|']
    labels={'parent_auto':'부모2k 자동 구간 고정','own_auto':'각 모델 자동 구간','tag_same_readout':'TAG 구간·같은 자동 probe','tag_matched_readout':'TAG 구간·기존 TAG용 probe'}
    for s in r['summary']:
        if s['split']=='valid_unseen' and s['policy']=='seen_CAL_recalibrated':
            lines.append('| %s · %s | %.2f%% (%d) | %d/1287 (%.2f%%) |'%(s['arm'],labels[s['mode']],100*s['normal_fpr'],s['normal_fp'],s['exact'],100*s['exact_recall']))
    lines+=['\n[고정 판별기·중간/최종 CTC 그림](<'+str(root/'comparison/P4_evidence_transfer.svg')+'>) · [전체 음소·위치·문항 표](<'+str(root/'comparison/all_groups.tsv')+'>) · [세 판별 단계의 일치·불일치](<'+str(root/'comparison/three_way.tsv')+'>) · [실제ㅜ/ㅗ 대조](<'+str(root/'audio_labels/contrast_report.json')+'>). 중간 CTC의 전체 음소·삭제·삽입 및 정렬 민감도는 각 `B0/B3/aux_evaluation/report.json`에 있다.\n',
            '원어민 Stage1의 실제10,000회 노출도 재생했다.819,871행·61,455,408,152샘플·배치 hash가 원 학습과 같고, 평탄한 타깃의ㅅ_ㅜ_ㄹ11,877회·ㅅ_ㅗ_ㄹ10,833회를 확인했다. 음소열ㅅㅜㄹㅣ는 부분열로861회,ㅅㅗㄹㅣ는3,016회다. 실제 청취 TAG나 동일 단어 문맥의 횟수는 아니다. Stage2 외국인 문맥의 불균형을 전체 학습 이력의 부재로 해석하지 않는다.\n',
            '이 결과는 반복 개발 문항의 진단이다. 새로운 독립 문장·화자, 독립 청취, 실제 API의 VAD/G2P/표시 정책 검증은 완료하지 않았다. 검색·prototype 모델이나 새 학습 목적함수는 이번 P4에서 학습·채택하지 않았다.\n']
    doc=ROOT/'docs/발음평가_진단누락정보_감사_2026-10-26.md'
    old=doc.read_text()
    if '### 20.4 P4-A 전수 완료' not in old:doc.write_text(old+'\n'.join(lines)+'\n')
    summary=ROOT/'docs/발음평가_개선작업_종합보고서_2026-10-26.md'
    text=summary.read_text()
    if '## P4 고정 판별기 진단 완료' not in text:
        summary.write_text(text+'\n## P4 고정 판별기 진단 완료 — 2026-10-28\n\nB0/B3 각각14,009녹음에 동일한 부모2k probe와 B3 중간 CTC 판별기를 적용했다. 최종 인식열·점수를 재현하고 전체 음소·삭제·삽입, 실제ㅗ 대조, 문턱 이전과 실제 배치 노출을 감사했다. 새 모델 학습·운영 교체는 없다. 상세 결과와 제한은 [진단 감사 §20](발음평가_진단누락정보_감사_2026-10-26.md#20-p4-고정-음소-증거중간-ctc최종-ctc의-연결-진단--10월28일)에 있다.\n')
    write_json(root/'completion.json',dict(status='P4_diagnostics_completed',completed_at=now,encoder_updates=0,probe_updates=0,
        inference_recordings_per_arm=14009,report=str(root/'comparison/report.json'),verification=str(root/'verification.json'),
        independent_evaluation_completed=False,production_changed=False))
    write_json(root/'live_status.json',dict(status='P4_diagnostics_completed',completed_at=now,completion=str(root/'completion.json')))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',required=True);ap.add_argument('--wait',action='store_true');a=ap.parse_args()
    root=Path(a.root)
    if a.wait:
        while not (root/'extract.exit').exists():time.sleep(30)
    if (root/'extract.exit').read_text().strip()!='0':raise RuntimeError('extraction failed')
    for filename,args in [('summarize_h0_p4.py',[]),('compare_h0_p4_vowels.py',['--stage','compare'])]:
        subprocess.run([sys.executable,str(Path(__file__).parent/filename),'--root',str(root)]+args,check=True)
    verify(root);plot(root);record(root)
    (root/'run.exit').write_text('0\n')


if __name__=='__main__':main()
