"""Audit all H0 phonemes from frozen site ledgers and optional full sequences.

This is exploratory selection-set analysis. No threshold/model selection,
training, test-set access or production changes are performed.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src')]

import argparse
import collections
import csv
import hashlib
import json
import math
import time
import numpy as np
from mdd.h0_phone_audit import (contexts, events, metric_counts, optimal_mapping, paired_cluster_ci,
    project_gaps, rate, sequence_audit, sequence_site_audit, standardize, surface_position, token)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def tsv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w') as f:
        writer = csv.DictWriter(f, fields, delimiter='\t'); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def ckey(site, fine=False, actual=False):
    key = [site['expected'], site['position'], site['following_vowel']]
    if fine: key += [site['previous_position'], site['task']]
    if actual: key += [token(site['actual'])]
    return '|'.join(key)


def dimensions(site):
    return [('all', 'all'), ('phone', site['expected']), ('position', site['position']),
            ('phone_position', site['expected'] + '|' + site['position']),
            ('task', site['task']), ('post_coda', str(site['post_coda_onset'])),
            ('item', site['item']), ('item_phone_position', '|'.join([site['item'], site['expected'], site['position']])),
            ('context_simple', ckey(site)), ('context_fine', ckey(site, True)),
            ('error_context_fine', ckey(site, True, True))] + [('rule_candidate', r) for r in site['rule_candidates']]


def load_inputs(protocol):
    manifest = Path(protocol['stage2']['manifest'])
    bindings = json.loads(Path(protocol['input_audit']).read_text())['sha256']
    data = {}
    for split in ('valid', 'valid_unseen'):
        values = {}
        for ext in ('canon', 'ltr', 'meta.jsonl'):
            p = manifest / (split + '.' + ext)
            assert sha(p) == bindings[str(p)], p
            values[ext] = list(read(p)) if ext.endswith('jsonl') else [l.split() for l in p.read_text().splitlines()]
        data[split] = values
    return data


def enriched_sites(path, inputs):
    for s in read(path):
        if s['role'] != 'selection': continue
        data = inputs[s['split']]; row, q = s['row'], s['q']
        canon, actual, meta = data['canon'][row], data['ltr'][row], data['meta.jsonl'][row]
        assert s['uid'] == meta['stem'] and s['expected'] == canon[q]
        s.update(contexts(tuple(canon), meta['prompt'])[q])
        opts = optimal_mapping(tuple(canon), tuple(actual))[q]
        possible = {None if j is None else actual[j] for j in opts}
        assert s['actual'] in possible
        s['truth_value_ambiguous'] = len(possible) > 1
        s['truth_index_ambiguous'] = len(opts) > 1
        s['prompt'] = meta['prompt']
        yield s


def key(s):
    return s['split'], s['uid'], s['q']


def main_sites(a, out):
    exp = Path(a.experiments); protocol = json.loads(Path(a.protocol).read_text()); inputs = load_inputs(protocol)
    vocabulary = [l.split()[0] for l in (Path(protocol['stage2']['manifest'])/'dict.ltr.txt').read_text().splitlines()]
    grouped_rows, phoneme_rows, confusion_rows, item_rows, change_rows, matching_rows, transition_rows = [], [], [], [], [], [], []
    summary = dict(scope='selection-only exploratory all-phone audit', protocol_sha256=sha(a.protocol),
        models={}, source_sha256={}, identities_match=True, training_updates=0, production_changed=False)
    previous = None; previous_report = None; identity = None
    examples = []
    for name in ('w2vbert_eval_2000', 'w2vbert_eval_4000', 'xlsr300m_eval_2000', 'xlsr300m_eval_4000'):
        p = exp / name; r = json.loads((p/'report.json').read_text()); threshold = r['threshold']
        if threshold is None: threshold = -math.inf
        sites = list(enriched_sites(p/'sites.jsonl', inputs)); digest = hashlib.sha256()
        groups = collections.defaultdict(collections.Counter)
        support = collections.defaultdict(lambda: {'normal':set(), 'errors':set()})
        phone_items = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
        confusions = collections.Counter()
        policy_confusions = collections.Counter()
        for s in sites:
            digest.update(json.dumps([s[k] for k in ('split','uid','q','expected','actual')], ensure_ascii=False).encode())
            e = events(s, threshold)
            e.update(truth_value_ambiguous=int(s['truth_value_ambiguous']),
                     ambiguous_normal_fp=int(s['truth_value_ambiguous'] and e['normal_fp']))
            for dim, group in dimensions(s):
                k=(s['split'],dim,group)
                groups[k].update(e)
                support[k]['normal' if e['normal'] else 'errors'].add(s['item'])
            phone_items[(s['split'],s['expected'])][s['item']].update(e)
            if e['normal_fp']: category = 'normal_false_alarm'
            elif e['error_revert']: category = 'error_reverted_to_expected'
            elif e['error_wrong']: category = 'wrong_actual_phone'
            elif e['error_exact']: category = 'exact_error'
            else: category = 'normal_correct'
            if category != 'normal_correct':
                confusions[(s['split'],s['expected'],s['position'],token(s['actual']),token(s['predicted']),category)] += 1
                policy_confusions[(s['split'],s['expected'],s['position'],token(s['actual']),token(s['predicted']),category)] += e['policy_fp']+e['policy_exact']+e['policy_wrong']
        if identity is None: identity = digest.hexdigest()
        assert digest.hexdigest() == identity, 'selection site identities differ'
        model_summary = dict(backbone=r['backbone'], step=r['step'], threshold=r['threshold'], totals={}, matched_contexts={})
        for split in ('valid', 'valid_unseen'):
            c = groups[(split,'all','all')]
            assert c['normal'] + c['errors'] == c['total']
            assert c['error_revert'] + c['error_exact'] + c['error_wrong'] == c['errors']
            assert c['normal_fp_substitution'] + c['normal_fp_deletion'] == c['normal_fp']
            expected = r['results'][split]['all']
            for own, old, mode in [('normal_fp','normal_false_alarms','greedy'),('error_exact','exact','greedy'),
                ('error_wrong','wrong_actual','greedy'),('policy_fp','normal_false_alarms','calibrated'),
                ('policy_exact','exact','calibrated'),('policy_wrong','wrong_actual','calibrated')]:
                assert c[own] == expected[mode][old], (name, split, own)
            model_summary['totals'][split] = metric_counts(c)
            for phone in vocabulary:
                counts = groups[(split,'phone',phone)]
                vals = phone_items[(split,phone)]
                normal_rates = [v['normal_fp']/v['normal'] for v in vals.values() if v['normal']]
                phoneme_rows.append(dict(condition=name, split=split, expected=phone, **metric_counts(counts),
                    normal_items=sum(v['normal'] > 0 for v in vals.values()), error_items=sum(v['errors'] > 0 for v in vals.values()),
                    item_fpr_min=min(normal_rates) if normal_rates else None,
                    item_fpr_max=max(normal_rates) if normal_rates else None,
                    item_fpr_macro=float(np.mean(normal_rates)) if normal_rates else None))
                for item, values in vals.items():
                    item_rows.append(dict(condition=name, split=split, expected=phone, item=item, **metric_counts(values)))
        for (split, dim, group), counts in groups.items():
            grouped_rows.append(dict(condition=name, split=split, dimension=dim, group=group, **metric_counts(counts),
                normal_items=len(support[(split,dim,group)]['normal']),error_items=len(support[(split,dim,group)]['errors'])))
        for (split, expected, position, actual, pred, category), n in confusions.items():
            confusion_rows.append(dict(condition=name,split=split,expected=expected,position=position,actual=actual,
                predicted=pred,category=category,n=n,
                policy_count=policy_confusions[(split,expected,position,actual,pred,category)]))
        for dim in ('context_simple','context_fine','error_context_fine'):
            contexts_by_split = collections.defaultdict(dict)
            for (split,d,g), c in groups.items():
                if d == dim: contexts_by_split[g][split] = c
            model_summary['matched_contexts'][dim] = {}
            for numerator, denominator in [('normal_fp','normal'),('policy_fp','normal'),('error_exact','errors'),('policy_exact','errors')]:
                # Error comparison includes actual token in the stratum, normal matching does not need it.
                if (denominator == 'errors') != (dim == 'error_context_fine'): continue
                v = standardize(contexts_by_split, numerator, denominator)
                v['unseen_coverage'] = rate(v['unseen_denominator'], model_summary['totals']['valid_unseen'][denominator])
                v['seen_coverage'] = rate(v['seen_denominator'], model_summary['totals']['valid'][denominator])
                model_summary['matched_contexts'][dim][numerator] = v
                if denominator=='normal':
                    supported={k:v for k,v in contexts_by_split.items()
                        if v.get('valid',{}).get('normal',0)>=30 and v.get('valid_unseen',{}).get('normal',0)>=30
                        and len(support[('valid',dim,k)]['normal'])>=3}
                    robust=standardize(supported,numerator,denominator)
                    robust['unseen_coverage']=rate(robust['unseen_denominator'],model_summary['totals']['valid_unseen']['normal'])
                    robust['rule']='at least 30 normals in each split and at least 3 seen normal prompts per context'
                    model_summary['matched_contexts'][dim][numerator+'_supported']=robust
            for context, v in contexts_by_split.items():
                row = dict(condition=name, definition=dim, context=context)
                for split in ('valid','valid_unseen'):
                    row.update({split+'_'+k:val for k,val in metric_counts(v.get(split,{})).items()})
                    row[split+'_normal_items']=len(support[(split,dim,context)]['normal'])
                    row[split+'_error_items']=len(support[(split,dim,context)]['errors'])
                matching_rows.append(row)
        if r['step'] == 2000:
            previous = {key(s):s for s in sites}; previous_report = r
        else:
            assert previous_report['backbone'] == r['backbone'] and len(previous) == len(sites)
            deltas = collections.defaultdict(collections.Counter)
            delta_items = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
            transitions = collections.Counter()
            tau2 = previous_report['threshold']; tau4 = threshold
            for s in sites:
                before = previous[key(s)]; e2, e4 = events(before,tau2), events(s,tau4)
                fixed = events(s,tau2)
                c = dict(normal=e4['normal'], errors=e4['errors'],
                    delta_fp=e4['normal_fp']-e2['normal_fp'], delta_exact=e4['error_exact']-e2['error_exact'],
                    delta_policy_fp=e4['policy_fp']-e2['policy_fp'], delta_policy_exact=e4['policy_exact']-e2['policy_exact'],
                    newly_wrong=int(e4['normal_fp'] and not e2['normal_fp']),
                    repaired=int(e2['normal_fp'] and not e4['normal_fp']),
                    prediction_score_change_at_old_threshold=fixed['policy_fp']-e2['policy_fp'],
                    threshold_change_at_new_outputs=e4['policy_fp']-fixed['policy_fp'])
                for dim,g in dimensions(s):
                    if dim in ('all','phone','phone_position','item','context_fine','rule_candidate'):
                        k = s['split'],dim,g; deltas[k].update(c); delta_items[k][s['item']].update(c)
                if before['predicted'] != s['predicted']:
                    transitions[(s['split'],s['expected'],s['position'],s['following_vowel'],
                        token(s['actual']),token(before['predicted']),token(s['predicted']))] += 1
                    if s['split']=='valid_unseen' and c['newly_wrong'] and len(examples)<120:
                        examples.append(dict(backbone=r['backbone'], uid=s['uid'],q=s['q'],item=s['item'],prompt=s['prompt'],
                            expected=s['expected'],actual=s['actual'],before=before['predicted'],after=s['predicted'],
                            position=s['position'],previous=s['previous_phone'],following_vowel=s['following_vowel']))
            model_summary['change_from_2k'] = {}
            for (split,dim,g), c in deltas.items():
                assert c['delta_fp'] == c['newly_wrong']-c['repaired']
                assert c['delta_policy_fp'] == c['prediction_score_change_at_old_threshold']+c['threshold_change_at_new_outputs']
                row = dict(backbone=r['backbone'],split=split,dimension=dim,group=g,**dict(c),
                           delta_fpr=rate(c['delta_fp'],c['normal']),delta_exact_recall=rate(c['delta_exact'],c['errors']))
                if dim in ('all','phone','phone_position'):
                    row['paired_item_ci'] = paired_cluster_ci(list(delta_items[(split,dim,g)].values()),'delta_fp','normal')
                change_rows.append(row)
                if dim=='all': model_summary['change_from_2k'][split]=row
            for vals,n in transitions.items():
                transition_rows.append(dict(backbone=r['backbone'],split=vals[0],expected=vals[1],position=vals[2],
                    following_vowel=vals[3],actual=vals[4],pred_2k=vals[5],pred_4k=vals[6],n=n))
            model_summary['top_unseen_increases'] = sorted([r for r in change_rows if r['backbone']==model_summary['backbone']
                and r['split']=='valid_unseen' and r['dimension']=='phone_position'],key=lambda r:r['delta_fp'],reverse=True)[:12]
            total=deltas[('valid_unseen','all','all')]
            model_summary['leave_one_unseen_item_out']=[]
            item_changes=sorted([(k[2],v) for k,v in deltas.items() if k[0]=='valid_unseen' and k[1]=='item'],key=lambda pair:pair[1]['delta_fp'],reverse=True)
            for item,c in item_changes:
                model_summary['leave_one_unseen_item_out'].append(dict(item=item,normal=c['normal'],delta_fp=c['delta_fp'],
                    remaining_delta_fp=total['delta_fp']-c['delta_fp'],
                    remaining_delta_fpr=rate(total['delta_fp']-c['delta_fp'],total['normal']-c['normal'])))
            top_two=item_changes[:2]
            model_summary['exclude_top_two_increases_sensitivity']=dict(
                items=[k for k,c in top_two],post_hoc=True,
                excluded_delta_fp=sum(c['delta_fp'] for k,c in top_two),
                remaining_normal=total['normal']-sum(c['normal'] for k,c in top_two),
                remaining_delta_fp=total['delta_fp']-sum(c['delta_fp'] for k,c in top_two))
            previous = None
        summary['models'][name] = model_summary
        for filename in ('sites.jsonl','report.json'): summary['source_sha256'][str(p/filename)] = sha(p/filename)
        print(json.dumps(dict(condition=name, sites=len(sites), totals=model_summary['totals']),ensure_ascii=False),flush=True)
    summary['selection_identity_sha256'] = identity
    summary['limitations'] = ['positions inferred from canonical flat jamo, not physical boundaries',
        'phonological rule labels are spelling/expected structural candidates, not normative adjudication',
        '9 unseen prompts, re-used selection data; no independent test or new L1 claim',
        'matched contexts use unseen weights on common support, not matched recordings or causal effects',
        'separate sequence audit is required for insertion and prediction-alignment sensitivity',
        'GOP gate only filters mismatches; suppression is measurable, causal GOP alignment error is not',
        'threshold-change attribution is ordered arithmetic decomposition, not a causal intervention']
    for filename, rows in [('groups.tsv',grouped_rows),('phones.tsv',phoneme_rows),('phone_items.tsv',item_rows),
                           ('confusions.tsv',confusion_rows),('changes.tsv',change_rows),('matched_contexts.tsv',matching_rows),
                           ('transitions.tsv',transition_rows)]: tsv(out/filename,rows)
    write(out/'examples.json',examples);write(out/'report.json',summary)
    return summary


def main_sequences(a, out):
    exp, seq = Path(a.experiments), Path(a.sequences)
    receipt = json.loads((seq/'finished.json').read_text()); assert not receipt['smoke']
    summary = dict(training_updates=0, models={}, scope='selection greedy gaps and alignment sensitivity; no gap calibration')
    rows, gap_confusions, phone_rows, variant_rows = [], [], [], []
    for name, spec in receipt['conditions'].items():
        assert sha(seq/(name+'.jsonl')) == spec['sequences_sha256']
        assert sha(exp/name/'sites.jsonl') == spec['sites_sha256']
        groups = collections.defaultdict(collections.Counter); gc = collections.Counter()
        phones = collections.defaultdict(collections.Counter)
        variants = collections.defaultdict(collections.Counter)
        report = json.loads((exp/name/'report.json').read_text()); tau=report['threshold']
        ledger = {key(s):s for s in read(exp/name/'sites.jsonl') if s['role']=='selection'}
        seen_sites = set(); sequence_exact_suppressed=0
        for row in read(seq/(name+'.jsonl')):
            canon, actual, pred = (row[k] for k in ('canonical','actual','predicted'))
            diagnostics = sequence_site_audit(canon,actual,pred)
            c = sequence_audit(canon,actual,pred,diagnostics)
            p, pg = project_gaps(canon,pred);t,tg = project_gaps(canon,actual)
            variant = variants[(row['split'],row['item'],''.join(canon))]
            variant['utterances'] += 1
            for q in range(len(canon)):
                k=(row['split'],row['uid'],q);s=ledger[k];seen_sites.add(k)
                assert p[q]==s['predicted'] and t[q]==s['actual'] and canon[q]==s['expected']
                d = dict(diagnostics[q])
                d.update(events(s,tau))
                variant.update(events(s,tau))
                d['sequence_exact_but_GOP_suppressed_error'] = int(actual==pred) * d['exact_suppressed']
                phones[(row['split'],canon[q],surface_position(canon,q))].update(d)
                if actual==pred:
                    sequence_exact_suppressed += events(s,tau)['exact_suppressed']
            c['sequence_exact_but_GOP_suppressed_errors'] = sum(
                events(ledger[(row['split'],row['uid'],q)],tau)['exact_suppressed'] for q in range(len(canon))) if actual==pred else 0
            for gap,(av,pv) in enumerate(zip(tg,pg)):
                if av or pv:
                    gc[(row['split'],row['item'],canon[gap-1] if gap else '<BOS>',
                        canon[gap] if gap<len(canon) else '<EOS>',''.join(av) or '<EMPTY>',
                        ''.join(pv) or '<EMPTY>')] += 1
            for dim,g in [('all','all'),('item',row['item'])]:groups[(row['split'],dim,g)].update(c)
        assert seen_sites==set(ledger),'not all selection sites recovered'
        totals={}
        for (split,dim,g),counts in groups.items():
            z=dict(counts)
            z.update(insertion_exact_recall=rate(counts['insertion_exact_gaps'],counts['true_insertion_gaps']),
                insertion_detection_recall=rate(counts['insertion_detected_gaps'],counts['true_insertion_gaps']),
                insertion_precision=rate(counts['insertion_detected_gaps'],counts['predicted_insertion_gaps']),
                insertion_false_positive_rate=rate(counts['false_insertion_gaps'],counts['normal_gaps']))
            if dim=='all':totals[split]=z
            rows.append(dict(condition=name,split=split,dimension=dim,group=g,**z))
        for vals,n in gc.items():gap_confusions.append(dict(condition=name,split=vals[0],item=vals[1],left=vals[2],right=vals[3],actual=vals[4],predicted=vals[5],n=n))
        for (split,expected,position),counts in phones.items():
            phone_rows.append(dict(condition=name,split=split,expected=expected,position=position,**dict(counts)))
        for (split,item,canon),counts in variants.items():
            variant_rows.append(dict(condition=name,split=split,item=item,canonical=canon,**metric_counts(counts)))
        summary['models'][name]=dict(totals=totals,sequence_exact_but_GOP_suppressed_errors=sequence_exact_suppressed,
                                     original_projection_match=True,selection_sites=len(seen_sites))
        print(json.dumps(dict(condition=name,sequence_audit=totals),ensure_ascii=False),flush=True)
    summary['limitations']=['gap scores/thresholds not stored, so insertion metrics are greedy-only',
        'gap localization follows one minimum-edit path; direct TAG and all-optimal comparisons are sensitivity checks',
        'alternative possible values need not be simultaneously realizable across every site',
        'mapping-sensitive counts do not prove physical alignment or human-label errors',
        'TAG exact sequence equality cannot cause a site disagreement with identical deterministic projections']
    tsv(out/'sequences.tsv',rows);tsv(out/'gap_confusions.tsv',gap_confusions)
    tsv(out/'alignment_phones.tsv',phone_rows);tsv(out/'reference_variants.tsv',variant_rows)
    write(out/'report.json',summary)
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['sites','sequences']);ap.add_argument('--experiments',required=True)
    ap.add_argument('--protocol');ap.add_argument('--sequences');ap.add_argument('--out',required=True)
    a=ap.parse_args()
    if a.mode=='sites' and not a.protocol:ap.error('--protocol required for sites')
    if a.mode=='sequences' and not a.sequences:ap.error('--sequences required for sequences')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.time()
    write(out/'protocol.json',dict(mode=a.mode,arguments=vars(a),exploratory=True,
        source_sha256={str(p):sha(p) for p in (Path(__file__),ROOT/'tasks/pronunciation-assessment/src/mdd/h0_phone_audit.py')},
        training_updates=0,test_evaluated=False,production_changed=False))
    (main_sites if a.mode=='sites' else main_sequences)(a,out)
    write(out/'finished.json',dict(status='complete',elapsed_seconds=time.time()-start,report_sha256=sha(out/'report.json')))


if __name__=='__main__':main()
