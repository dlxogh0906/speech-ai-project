"""All-GT TRAIN proposal and availability changes for a native conditional trial.

This does not render API responses or turn reused TRAIN data into validation.
New unmapped runtime queries are explicitly separated from source GT sites.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),
    str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import collections
import hashlib
import json
import numpy as np
from train_sibilant_conditional_retained import inputs,write
from train_sibilant_extra_normals import proposed
from eval_sibilant_presence_screen import exact
import eval_sibilant_directional as service


def main():
    ap = argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True)
    ap.add_argument('--fit',required=True);ap.add_argument('--out',required=True)
    a = ap.parse_args();p = json.loads(Path(a.protocol).read_text());fit = Path(a.fit)
    data,y,features,labels,indexed,offset,_,_,_ = inputs(p)
    report = json.loads((fit/'report.json').read_text());parent = np.load(fit/'parent.scores.npz')
    parent_ps,parent_av = parent['probabilities'],parent['available'];parent_policy = report['parent_fixed_direction_TRAIN']
    result = {'scope':'Full retained TRAIN standalone proposals; not final API or independent validation',
        'protocol_sha256':service.sha(a.protocol),'fit_report_sha256':service.sha(fit/'report.json'),
        'arms':{},'case_examples':{},'source_normal_FP_target_is_not_only_supported_subset':True}
    def correct(z,ps,av,policy):
        i = z['feature_index']
        return bool(i is not None and z['label_kind']=='pair_error'
            and z['expected']==features[i]['canonical']
            and z['actual']==('ㅆ' if features[i]['canonical']=='ㅅ' else 'ㅅ')
            and proposed(z,ps,av,features,policy))
    for name,arm in report['arms'].items():
        saved = np.load(fit/(name+'.scores.npz'));ps,av = saved['probabilities'],saved['available']
        policy = arm['TRAIN_proxy'];groups = collections.defaultdict(collections.Counter)
        examples = collections.defaultdict(list)
        for index,z in enumerate(labels):
            cohort = ('original_GT' if index<14002 else 'extra_runtime_unknown'
                      if z.get('runtime_unmapped_source_query') else 'additional_source_GT')
            i = z['feature_index'];kind = z['label_kind'];counts = collections.Counter(all=1)
            counts['no_input' if i is None else 'target_unknown' if y[i]<0 else 'target_supported']+=1
            if i is not None:
                counts['available' if av[i] else 'model_unavailable']+=1
                counts['availability_lost']+=bool(parent_av[i] and not av[i])
                counts['availability_gained']+=bool(not parent_av[i] and av[i])
            pc = proposed(z,parent_ps,parent_av,features,parent_policy)
            cc = proposed(z,ps,av,features,policy)
            pe,ce = correct(z,parent_ps,parent_av,parent_policy),correct(z,ps,av,policy)
            counts['all_calls']+=cc
            if kind=='pair_error':
                counts['pair_errors_all']+=1;counts['pair_exact_proposals']+=ce
                counts['pair_missed_proposals']+=not ce
                counts['pair_added_exact']+=ce and not pe;counts['pair_lost_exact']+=pe and not ce
                counts['pair_novel_vs_existing_API']+=ce and not exact(z,z['existing'])
                counts['pair_already_exact_existing_API']+=ce and exact(z,z['existing'])
            if kind=='correct':
                counts['normal_all']+=1;counts['normal_false_calls']+=cc
                counts['normal_added_calls']+=cc and not pc;counts['normal_lost_calls']+=pc and not cc
            if kind=='ambiguous':counts['ambiguous_all']+=1;counts['ambiguous_calls']+=cc
            pairs = [('all','all'),('label_kind',kind),('following',str(z.get('following'))),
                     ('position',str(z.get('position'))),('item',str(z.get('item'))),
                     ('language',str(z.get('language')))]
            if kind=='pair_error':pairs.append(('direction',z['expected']+'→'+z['actual']))
            for key,value in pairs:groups[cohort+':'+key+':'+value].update(counts)
            kinds = []
            if kind=='pair_error' and ce and not pe:kinds.append('added_exact')
            if kind=='pair_error' and pe and not ce:kinds.append('lost_exact')
            if kind=='correct' and cc and not pc:kinds.append('added_normal_false_call')
            if kind=='correct' and pc and not cc:kinds.append('removed_normal_false_call')
            if i is not None and parent_av[i] and not av[i]:kinds.append('lost_availability')
            for change in kinds:
                case = {k:z.get(k) for k in ('uid','speaker','item','q_runtime','expected','actual','following','position','label_kind','feature_index')}
                case.update(cohort=cohort,parent_call=pc,candidate_call=cc,parent_exact=pe,candidate_exact=ce)
                if i is not None:case.update(parent_probabilities=parent_ps[i].tolist(),candidate_probabilities=ps[i].tolist(),
                    parent_available=bool(parent_av[i]),candidate_available=bool(av[i]),target_supported=bool(y[i]>=0))
                examples[change].append(case)
        values = {}
        for key,counter in groups.items():
            item = dict(counter)
            if counter['normal_all']:item['normal_false_call_rate_all_GT']=counter['normal_false_calls']/counter['normal_all']
            if counter['pair_errors_all']:item['pair_exact_rate_all_GT']=counter['pair_exact_proposals']/counter['pair_errors_all']
            values[key] = item
        if values['original_GT:all:all']['all']!=14002 or values['additional_source_GT:all:all']['all']!=1750:
            raise ValueError('original/source GT denominator differs')
        result['arms'][name] = values
        result['case_examples'][name] = {key:sorted(rows,key=lambda z:hashlib.sha256((z['uid']+str(z['q_runtime'])).encode()).hexdigest())[:8]
                                        for key,rows in examples.items()}
    out = Path(a.out)
    if out.exists():raise ValueError('preserve existing diagnostic')
    write(out,result)
    print(json.dumps({name:values['original_GT:all:all'] for name,values in result['arms'].items()},ensure_ascii=False))


if __name__=='__main__':main()
