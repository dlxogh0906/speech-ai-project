"""Post-hoc threshold diagnostic on frozen outputs; NOT a policy selection run."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT: raise RuntimeError('저장소 루트(pyproject.toml)를 찾지 못했습니다')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'tasks/pronunciation-assessment/src')]
from shared.paths import MDD_OUTPUTS  # 경로는 .env 에서 (CLAUDE.md 경로 규칙)
import collections
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

RUN = MDD_OUTPUTS/'experiments/m3_candidate_rerank_2026-11-01_v1'
CONDITIONS = ('A_M3_peak', 'B_probability', 'C_segment')
FAMILIES = ('sibilants', 'affricates')


def read(path):
    with gzip.open(path, 'rt') as f:
        return list(map(json.loads, f))


def main():
    out = RUN/'posthoc_threshold_audit_v1'
    out.mkdir(exist_ok=False)
    protocol = dict(
        purpose='Explain failures and threshold-only room on frozen predictions',
        evaluation_labels_used=True, independent_validation=False,
        deployed=False, encoder_updates=0, classifier_updates=0,
        candidate_choice='Fixed original saved single-phone prediction; no oracle relabeling',
        threshold='Per family RP-normal budget, tie exclusion; descriptive only, never replace frozen CAL threshold',
        budgets=[.005, .01, .02, .05],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    predictions = read(RUN/'rp_predictions.jsonl.gz')
    rows = []
    for cond in CONDITIONS:
        for budget in protocol['budgets']:
            for family in FAMILIES:
                rr = [r for r in predictions if r['condition']==cond and r['family']==family
                      and r['truth_status'] in ('normal', 'substitution', 'deletion')]
                normal = [r for r in rr if r['actual']==r['expected']]
                errors = [r for r in rr if r['actual']!=r['expected']]
                observed = [r for r in normal if r['observed']]
                allowed = int(np.floor(budget*len(observed)))
                finite = sorted([r['score'] for r in observed if r['score'] is not None], reverse=True)
                threshold = float(np.nextafter(finite[allowed], np.inf)) if len(finite)>allowed else 0.

                def warning(r):
                    return r['observed'] and r['score'] is not None and r['score']>=threshold

                fp = sum(warning(r) for r in observed)
                exact = sum(warning(r) and r['prediction']==r['actual'] for r in errors)
                assert fp<=allowed
                # Independent vector count using all fixed known-label rows.
                scores = np.array([-np.inf if r['score'] is None else r['score'] for r in rr])
                marked = (scores>=threshold)&np.array([r['observed'] for r in rr])
                is_error = np.array([r['actual']!=r['expected'] for r in rr])
                correct = np.array([r['prediction']==r['actual'] for r in rr])
                assert int((marked&~is_error).sum())==fp
                assert int((marked&is_error&correct).sum())==exact
                rows.append(dict(condition=cond, family=family, budget=budget, threshold=threshold,
                    normal_total=len(normal), normal_observed=len(observed), error_total=len(errors),
                    error_observed=sum(r['observed'] for r in errors), fp=fp, exact=exact,
                    FPR=fp/len(observed), exact_recall=exact/len(errors)))
    for cond in CONDITIONS:
        for budget in protocol['budgets']:
            rr=[r for r in rows if r['condition']==cond and r['budget']==budget]
            totals={k:sum(r[k] for r in rr) for k in ('normal_total','normal_observed','error_total','error_observed','fp','exact')}
            rows.append(dict(condition=cond,family='all',budget=budget,**totals,
                FPR=totals['fp']/totals['normal_observed'],exact_recall=totals['exact']/totals['error_total']))
    train = read(RUN/'train_features.jsonl.gz')
    support={family:dict(collections.Counter(r['expected']+'->'+str(r['actual']) for r in train
        if r['family']==family and r['observed'])) for family in FAMILIES}
    keyed=collections.defaultdict(dict)
    for r in predictions:keyed[(r['uid'],r['q'])][r['condition']]=r
    transitions={}
    for cond in CONDITIONS[1:]:
        counts=collections.Counter()
        for rr in keyed.values():
            a,b=rr['A_M3_peak'],rr[cond]
            if not a['observed'] or a['truth_status'] not in ('normal','substitution','deletion'):continue
            error=a['actual']!=a['expected']
            ae=error and a['warning'] and a['prediction']==a['actual']
            be=error and b['warning'] and b['prediction']==b['actual']
            if ae and not be:
                cause=('OTHER_veto' if b['veto'] else 'threshold_suppressed' if b['prediction']==b['actual']
                       else 'expected_output' if b['prediction']==b['expected'] else 'wrong_actual_output')
                counts['lost_exact:'+cause]+=1
            if be and not ae:
                counts['new_exact:'+('same_correct_output_unsuppressed' if a['prediction']==a['actual'] else 'changed_to_actual')]+=1
            if not error and b['warning'] and not a['warning']:
                counts['new_FP:'+a['expected']+'->'+b['prediction']]+=1
        transitions[cond]=dict(counts)
    report=dict(protocol=protocol,threshold_diagnostic=rows,TRAIN_observed_support=support,
        original_policy_transitions=transitions,
        input_hashes={name:hashlib.sha256((RUN/name).read_bytes()).hexdigest()
                      for name in ('rp_predictions.jsonl.gz','train_features.jsonl.gz','thresholds.json')})
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps([r for r in rows if r['family']=='all'],ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
