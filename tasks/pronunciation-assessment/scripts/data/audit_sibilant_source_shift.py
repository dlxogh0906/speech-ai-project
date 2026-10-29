"""Matched source/metadata diagnostics and score-blind normal-context inventory."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),
               str(ROOT/'tasks/pronunciation-assessment/scripts/train'),
               str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import collections
import gzip
import hashlib
import json
import time
import numpy as np
import eval_sibilant_directional as service
from train_sibilant_conditional_retained import inputs,write
from mdd.sibilant_source_shift import (context,speaker_fold,matched_source_pairs,
    context_lookup,prototype_normal_targets,ridge_source_probe)


def source_inventory(p,out,features,labels,y,offset):
    rows = {z['uid']:z for z in service.read_jsonl(p['prototype_rows'])}
    prototypes = collections.defaultdict(set)
    for z in service.read_jsonl(p['prototype_runtime']):
        if z['runtime_canonical'] is not None:
            row = rows[z['uid']];prototypes[(row['item'],row['norm_prompt'])].add(tuple(z['runtime_canonical']))
    errors = collections.Counter(context(z,features[i]) for i,z in labels.items()
        if i>=offset and y[i]>=0 and z['label_kind']=='pair_error'
        and z['expected']==features[i]['canonical'] and z['following']==features[i]['expected_V'])
    if sum(errors.values())!=675:raise ValueError('source error pool changed')
    candidates = collections.defaultdict(dict);counts = collections.Counter();tick = time.monotonic()
    with gzip.open(p['source_witness_rows'],'rt') as stream:
        for n,line in enumerate(stream,1):
            row = json.loads(line);counts['all_rows']+=1
            counts['all_source_normals']+=sum(z['label_kind']=='correct' for z in row['sites'])
            counts['all_source_pair_errors']+=sum(z['label_kind']=='pair_error' for z in row['sites'])
            if row['uid'] in rows:continue
            key = (row['item'],row['norm_prompt'])
            if key not in prototypes:continue
            counts['unobserved_prototype_rows']+=1
            for canonical in sorted(prototypes[key]):
                for ctx,partial,q in prototype_normal_targets(row,canonical):
                    if ctx not in errors:continue
                    candidates[ctx][(row['uid'],partial)] = {'uid':row['uid'],
                        'partial_q':partial,'expected_runtime_q':q,'context':list(ctx)}
            if n%5000==0:print(json.dumps({'source_inventory_rows':n,'seconds':time.monotonic()-tick}),flush=True)
    if (counts['all_rows'],counts['all_source_normals'],counts['all_source_pair_errors'])!=(22829,38077,728):
        raise ValueError('source-witness denominator changed')
    selected = collections.defaultdict(list);context_report=[]
    for ctx,need in sorted(errors.items(),key=lambda z:repr(z[0])):
        pool = list(candidates.get(ctx,{}).values())
        def priority(z):
            s = 'matched-normal-inventory-v1:'+z['uid']+':'+str(z['partial_q'])
            return hashlib.sha256(s.encode()).hexdigest(),z['partial_q']
        chosen = sorted(pool,key=priority)[:2*need]
        for z in chosen:selected[z['uid']].append(z)
        context_report.append({'context':list(ctx),'source_error_features':need,
            'candidate_normal_sites':len(pool),'selected_source_normal_requests':len(chosen)})
    selected_path = out/'selected_normal_source_rows.jsonl';all_sites=collections.Counter();speakers=set();items=set()
    with gzip.open(p['source_witness_rows'],'rt') as source,selected_path.open('x') as dest:
        for line in source:
            row = json.loads(line)
            if row['uid'] not in selected:continue
            if any(z['label_kind']=='pair_error' for z in row['sites']):
                raise ValueError('unobserved normal rows unexpectedly contain source pair error')
            row['normal_selection_requests'] = selected[row['uid']]
            service.write_line(dest,row);all_sites.update(z['label_kind'] for z in row['sites'])
            speakers.add(row['speaker']);items.add(row['item'])
    covered = sum(z['source_error_features'] for z in context_report if z['candidate_normal_sites'])
    result = {'counts':dict(counts),'error_features':675,'error_contexts':len(errors),
        'error_features_with_unobserved_same_context_normal_source':covered,
        'error_contexts_with_normal_source':sum(z['candidate_normal_sites']>0 for z in context_report),
        'selected_rows':len(selected),'selected_speakers':len(speakers),'selected_items':len(items),
        'selected_source_site_kinds':dict(all_sites),'selected_source_requests':sum(map(len,selected.values())),
        'selected_rows_sha256':service.sha(selected_path),'contexts':context_report,
        'runtime_checked':False,'usable_new_training_targets':0,
        'scope':'Existing source-time-witness normal labels only; predicted q from prior runtimecanonical prototypes. No new acoustic/human truth, no model scores used in selection. Runtime mapping/support still required.'}
    write(out/'normal_source_inventory.json',result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True)
    ap.add_argument('--out',required=True);a = ap.parse_args()
    p = json.loads(Path(a.protocol).read_text());out = Path(a.out)
    data,y,features,rows,indexed,offset,_,_,_ = inputs(p)
    out.mkdir(parents=True,exist_ok=False);write(out/'protocol.json',p)
    pairs,quad,used = matched_source_pairs(y,indexed,features,offset)
    np.savez(out/'source_pairs.npz',fold0=pairs[0],fold1=pairs[1])
    with (out/'source_pair_ledger.jsonl').open('x') as stream:
        for fold,pool in enumerate(pairs):
            for old,new in pool:
                service.write_line(stream,{'fold':fold,'old_feature':int(old),'new_feature':int(new),
                    'old_uid':indexed[int(old)]['uid'],'new_uid':indexed[int(new)]['uid'],
                    'actual_class':int(y[old]),'context':list(context(indexed[int(old)],features[int(old)]))})
    eligible_new = [i for i in used if i>=offset];eligible_old = [i for i in used if i<offset]
    errors = [i for i in eligible_new if indexed[i]['label_kind']=='pair_error']
    normal = [i for i in eligible_new if indexed[i]['label_kind']=='correct']
    if len(errors)!=675 or len(normal)!=876:raise ValueError('fixed original recovered cohort differs')
    coverage = {}
    for fields in (5,7):
        normal_contexts = {context(indexed[i],features[i])[:fields] for i in normal}
        coverage[str(fields)] = {'errors':len(errors),
            'with_same_context_new_normal':sum(context(indexed[i],features[i])[:fields] in normal_contexts for i in errors),
            'missing_normal_context':sum(context(indexed[i],features[i])[:fields] not in normal_contexts for i in errors)}
    lookup = {source:{str(n):context_lookup(ids,indexed,features,n) for n in (5,7)}
              for source,ids in [('old',eligible_old),('new',eligible_new)]}
    report = {'scope':p['audit_scope'],'new_known_pairs':len(eligible_new),'old_known_pairs':len(eligible_old),
        'source_pairs_by_speaker_fold':[len(z) for z in pairs],
        'new_unmatched_known_targets_retained_in_coverage':len(eligible_new)-sum(map(len,pairs)),
        'fully_balanced_source_actual_quadruples_by_fold':[len(z) for z in quad],
        'actual_phone_cross_source_probe':'not fit: fewer than100fourcellunits per speakerfold; do notinfer intrinsic phone separability',
        'normal_context_coverage':coverage,'metadata_only_speaker_holdout_error_lookup':lookup,
        'source_probes':{},'privileged_matching_not_runtime_input':True,
        'CAL_primary_reserved_scoring':False,'production_changed':False,'service_improvement_proven':False,'goal_complete':False}
    write(out/'report.json',report)
    blocks = {'summary':data['summary'],'moments':data['moments'],
              'combined':np.concatenate([data['summary'],data['moments']],axis=1)}
    for name,x in blocks.items():
        report['source_probes'][name] = ridge_source_probe(x,pairs,p['audit_seed'],p['null_permutations'])
        write(out/'report.json',report);print(json.dumps({'source_probe':name,'results':report['source_probes'][name]}),flush=True)
    inventory = source_inventory(p,out,features,indexed,y,offset)
    report['normal_source_inventory'] = {k:v for k,v in inventory.items() if k!='contexts'}
    report['next_runtime_normal_acquisition_supported'] = bool(coverage['7']['missing_normal_context']>=.2*len(errors)
        and inventory['error_features_with_unobserved_same_context_normal_source']>=.8*len(errors)
        and inventory['selected_rows']>0)
    write(out/'report.json',report);write(out/'finished.json',{'report_sha256':service.sha(out/'report.json')})
    print(json.dumps({'finished':True,'next_runtime_normal_acquisition_supported':report['next_runtime_normal_acquisition_supported'],
        'normal_context_coverage':coverage,'inventory':report['normal_source_inventory']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
