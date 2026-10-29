"""P2 original reference provenance and frozen-GOP transfer audit, no relabeling."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),
             str(ROOT/'tasks/pronunciation-assessment/scripts/train'),
             str(ROOT/'tasks/pronunciation-assessment/scripts/data')]
import argparse
import collections
import json
import numpy as np
from shared.paths import DATA_ROOT
from build_foreign_clean_v2 import parse_one
from audit_h0_backbone_protocol import calibration_role
from train_h0_backbone_control import file_sha
from eval_h0_backbone_control import metrics
from mdd.h0_phone_audit import optimal_mapping, surface_position
from diagnose_h0_representations import write_json
from fit_h0_representation_probes import table


def canonical_flag_indices(canon,item):
    text=''.join(canon);flags=[]
    if item=='RW003':
        flags.extend((q,'sigye_allowed_e_alternative') for q,c in enumerate(canon) if c=='ㅖ')
    if item=='RS017':
        start=text.find('ㅁㅕㄴㅆ')
        if start>=0:flags.append((start+2,'myeot_coda_n_in_reference'))
        if text.startswith('ㄴㅐㅇㅣ'):flags.append((2,'naeil_extra_ieung_reference'))
    if item=='RS012':
        start=text.find('ㅣㄸㅗ')
        if start>=0:flags.append((start+1,'ittorok_single_closure_tokenization'))
    return flags


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parent-protocol',required=True);ap.add_argument('--experiments',required=True)
    ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    parent=json.loads(Path(a.parent_protocol).read_text());manifest=Path(parent['stage2']['manifest'])
    exp=Path(a.experiments)
    canonical=[s.split() for s in (manifest/'valid_unseen.canon').read_text().splitlines()]
    targets=[s.split() for s in (manifest/'valid_unseen.ltr').read_text().splitlines()]
    metadata=[json.loads(s) for s in (manifest/'valid_unseen.meta.jsonl').open()]
    inspected=[];variant=collections.defaultdict(list);flag_map={};source_hashes={};examples={}
    bases={p.name[:3]:p for p in (DATA_ROOT/'aihub').glob('04[0-3].*')}
    for row,m in enumerate(metadata):
        if m['item'] not in ('RW003','RS012','RS017') or calibration_role(m['spk'])!='selection':continue
        base=bases[m['ds']]/'01-1.정식개방데이터'/('Training' if m['part']=='TL' else 'Validation')/'02.라벨링데이터'
        tag=base/(m['part']+'_pronunciation_tag')/(m['stem']+'.TextGrid')
        lab=base/(m['part']+'_pronunciation_lab')/(m['stem']+'.json')
        record=parse_one((m['stem'],m['ds'],m['part'],str(tag),str(lab)))
        if record.get('canon')!=canonical[row] or record.get('ltr')!=targets[row]:raise ValueError('raw source and target differ')
        if record['tag_phones']!=m['tag_phones'] or record['lab_phones']!=m['lab_phones']:raise ValueError('cached timestamps differ')
        for p in (tag,lab):source_hashes[str(p)]=file_sha(p)
        key=(m['item'],''.join(canonical[row]))
        variant[key].append(m)
        if key not in examples:
            raw=json.loads(lab.read_text())['RecordingMetadata']
            examples[key]=dict(uid=m['stem'],original_lab=str(lab),original_tag=str(tag),
                lab_words=raw['phonemic'].get('words'),lab_phones=m['lab_phones'],tag_phones=m['tag_phones'],
                error_tags=m['error_tags'])
        for q,flag in canonical_flag_indices(canonical[row],m['item']):flag_map[m['stem'],q]=flag
        inspected.append(dict(uid=m['stem'],item=m['item'],canonical=''.join(canonical[row]),actual=''.join(targets[row]),
                              source_identical=True,flags=canonical_flag_indices(canonical[row],m['item'])))
    rows=[]
    for (item,text),ms in sorted(variant.items()):
        rows.append(dict(item=item,canonical=text,utterances=len(ms),
            source_datasets=dict(collections.Counter(m['ds'] for m in ms)),
            languages=dict(collections.Counter(m['language'] for m in ms)),
            example=examples[item,text]))
    write_json(out/'original_sources.json',dict(utterances=inspected,variants=rows,sha256=source_hashes,
        conclusion='All compared canonical/TAG variants already exist in raw LAB/TextGrid; not introduced by current token filtering.',
        listened=False,labels_changed=False))
    references=[
        dict(topic='sigye',url='https://www.korean.go.kr/front/onlineQna/onlineQnaView.do?mn_id=216&qna_seq=302312',
             finding='Standard pronunciation rule 5 allows both sigye and sige; this does not establish the actual sound in a recording.'),
        dict(topic='coda',url='https://korean.go.kr/front/onlineQna/onlineQnaView.do?mn_id=216&pageIndex=1&qna_seq=318280',
             finding='Rule 9 neutralizes ss/ch codas to d before consonants, including itta example.'),
        dict(topic='tensing',url='https://www.korean.go.kr/front/onlineQna/onlineQnaView.do?mn_id=261&pageIndex=1&qna_seq=318479',
             finding='Rule 23 tenses following s/d etc after stop codas. Applied inference: myeot si => myeo(t)ssi, ittorok => i(t)ttorok.'),
    ]
    write_json(out/'normative_sources.json',dict(retrieved_date='2026-10-27',sources=references,
        limitations=['Derived application to these phrases is an inference from rules, not independent listening.',
                     'Single closure versus coda+tense-onset annotation needs labeling convention review.',
                     'Extra ieung in naeil is present in original LAB; null-onset notation versus actual nasal unresolved.',
                     'Myeonssi is not explained by the cited stop-coda/tensing rules; no automatic label correction.']))
    ledgers={};taus={};distributions=[];flags=[];sigye=[];cross=[];decomposition=[]
    for kind in ('w2vbert','xlsr300m'):
        for step in (2000,4000):
            name='%s_eval_%d'%(kind,step);report=json.loads((exp/name/'report.json').read_text())
            sites=[json.loads(s) for s in (exp/name/'sites.jsonl').open()]
            tau=report['threshold'];taus[name]=tau;ledgers[name]=sites
            groups=collections.defaultdict(list)
            for s in sites:
                if s['split']=='valid_unseen' and s['role']!='selection':continue
                if s['actual']!=s['expected']:continue
                groups[s['split'],s['role'],s['expected']].append(s)
            for (sp,role,phone),ss in sorted(groups.items()):
                vals=[s['score'] for s in ss if s['predicted']!=s['expected'] and s['score'] is not None]
                distributions.append(dict(condition=name,split=sp,role=role,phone=phone,normal=len(ss),
                    mismatches=len(vals),items=len(set(s['item'] for s in ss)),
                    gop_mismatch_quantiles=np.quantile(vals,[.5,.9,.95,.99]).tolist() if vals else None,
                    threshold=tau,policy_fp=sum(v>=tau for v in vals),
                    policy_fpr=sum(v>=tau for v in vals)/len(ss)))
            selected=[s for s in sites if s['split']=='valid_unseen' and s['role']=='selection']
            byflag=collections.defaultdict(list)
            for s in selected:
                flag=flag_map.get((s['uid'],s['q']))
                if flag:byflag[flag].append(s)
            for flag,ss in sorted(byflag.items()):
                flags.append(dict(condition=name,flag=flag,**metrics(ss,tau),
                    predicted_counts=dict(collections.Counter(s['predicted'] or '<DEL>' for s in ss)),
                    tag_counts=dict(collections.Counter(s['actual'] or '<DEL>' for s in ss))))
            ss=[s for s in selected if s['item']=='RW003' and s['expected']=='ㅖ']
            normal=[s for s in ss if s['actual']=='ㅖ']
            candidates=[s for s in normal if s['predicted']=='ㅔ']
            warnings=sum(s['score'] is not None and s['score']>=tau for s in candidates)
            all_m=metrics(selected,tau)
            sigye.append(dict(condition=name,total_ye_sites=len(ss),normal_ye_sites=len(normal),
                tag_counts=dict(collections.Counter(s['actual'] or '<DEL>' for s in ss)),
                normal_prediction_counts=dict(collections.Counter(s['predicted'] or '<DEL>' for s in normal)),
                predicted_allowed_e=len(candidates),gop_warned_allowed_e=warnings,
                original_unseen_policy_fp=all_m['normal_false_alarms'],
                reference_compatibility_sensitivity_fp=all_m['normal_false_alarms']-warnings,
                reference_compatibility_sensitivity_fpr=(all_m['normal_false_alarms']-warnings)/all_m['normal'],
                interpretation='Only remove alerts predicting allowed e at normal labeled ye; no TAG correction, no new acoustic accuracy claim.'))
        for output_step in (2000,4000):
            name='%s_eval_%d'%(kind,output_step)
            for tau_step in (2000,4000):
                tau=taus['%s_eval_%d'%(kind,tau_step)]
                for split in ('valid','valid_unseen'):
                    ss=[s for s in ledgers[name] if s['split']==split and s['role']=='selection']
                    cross.append(dict(backbone=kind,output_step=output_step,threshold_step=tau_step,split=split,**metrics(ss,tau)))
        old=[s for s in ledgers[kind+'_eval_2000'] if s['split']=='valid_unseen' and s['role']=='selection']
        new=[s for s in ledgers[kind+'_eval_4000'] if s['split']=='valid_unseen' and s['role']=='selection']
        if [(s['uid'],s['q'],s['actual']) for s in old]!=[(s['uid'],s['q'],s['actual']) for s in new]:raise ValueError('paired site identity')
        for group_type in ('phone','item'):
            values=sorted(set(s['expected'] if group_type=='phone' else s['item'] for s in old))
            for value in values:
                indices=[i for i,s in enumerate(old) if (s['expected'] if group_type=='phone' else s['item'])==value]
                os=[old[i] for i in indices];ns=[new[i] for i in indices]
                aa=metrics(os,taus[kind+'_eval_2000'])['normal_false_alarms']
                ba=metrics(ns,taus[kind+'_eval_2000'])['normal_false_alarms']
                bb=metrics(ns,taus[kind+'_eval_4000'])['normal_false_alarms']
                decomposition.append(dict(backbone=kind,group_type=group_type,value=value,
                    normal=sum(s['actual']==s['expected'] for s in os),fp_2k=aa,fp_4k_fixed_2k_tau=ba,fp_4k=bb,
                    score_output_change=ba-aa,threshold_change=bb-ba,net_change=bb-aa))
    for filename,rs in [('normal_score_distributions.tsv',distributions),('reference_flag_metrics.tsv',flags),
                        ('sigye_sensitivity.tsv',sigye),('threshold_cross.tsv',cross),('gop_change_decomposition.tsv',decomposition)]:
        table(out/filename,rs)
    write_json(out/'report.json',dict(raw_source_utterances=len(inspected),raw_files=len(source_hashes),
        normalized_sequences_identical=True,normative_sources=references,variants=[{k:v for k,v in r.items() if k!='example'} for r in rows],
        sigye=sigye,flag_metrics=flags,threshold_cross=cross,labels_changed=False,model_updates=0,
        causal_decomposition=False,selection_role_only_for_reference_audit=True,
        source_sha256=file_sha(__file__)))
    write_json(out/'finished.json',dict(status='completed',labels_changed=False,model_updates=0,raw_source_utterances=len(inspected)))


if __name__=='__main__':main()
