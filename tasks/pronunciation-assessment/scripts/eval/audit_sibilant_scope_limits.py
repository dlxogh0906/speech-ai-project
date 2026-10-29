"""Optimistic false-alert floors from observation scope, not a new policy.

Even a perfect warning classifier cannot correct sites it is forbidden to
observe/change. Retain the original full denominators. This oracle counting
diagnostic does not use TAG to decide an actual service response.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src')]
import argparse
import json
import math
import numpy as np
import eval_sibilant_directional as service


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();p=json.loads(Path(a.protocol).read_text());out=Path(a.out)
    for f,d in p['sha256'].items():
        if service.sha(f)!=d:raise ValueError('frozen input changed: '+f)
    out.mkdir(parents=True,exist_ok=False);(out/'protocol.json').write_text(Path(a.protocol).read_text())
    report={'scope':'Scope-only optimistic lower bounds; no response changes or demonstrated classifier performance. Current availability is model-specific.','cohorts':{}}
    for name,v in p['cohorts'].items():
        baseline=json.loads(Path(v['baseline_report']).read_text())['cohorts'][v['report_cohort']]['versus_prior']['baseline']
        labels=service.read_jsonl(v['warning_prefix']+'.labels.jsonl');scores=np.load(v['warning_scores'])
        normals=[r for r in labels if r['label_kind']=='correct' and (r['existing']['phone'] or {}).get('errorType')]
        measurable=[r for r in normals if r.get('feature_index') is not None]
        available=[r for r in measurable if bool(scores['available'][r['feature_index']])]
        fp,n=baseline['normal_alerts'],baseline['normal_known_accepted']
        if not len(available)<=len(measurable)<=len(normals)<=fp:raise ValueError('scope counts exceed full baseline')
        budget=math.floor(p['target_FPR']*n)
        result={'normal_all':baseline['normal_all'],'normal_known_accepted':n,'baseline_false_alerts':fp,
                'target_max_false_alerts':budget,'normal_warnings_in_selected_rows':len(normals),
                'normal_warnings_with_any_local_input':len(measurable),'normal_warnings_with_current_strict_support':len(available),
                'outside_selected_row_false_alerts':fp-len(normals),'optimistic_floors':{}}
        for mode,correctable in [('add_only',0),('all_linked_warning_inputs',len(measurable)),('current_strict_warning_inputs',len(available))]:
            floor=fp-correctable
            result['optimistic_floors'][mode]={'minimum_remaining_false_alerts':floor,'rate':floor/n if n else None,
                'target_impossible_under_this_scope':floor>budget}
        report['cohorts'][name]=result
        print(name,json.dumps(result,ensure_ascii=False),flush=True)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (out/'finished.json').write_text(json.dumps({'sha256':service.sha(out/'report.json')})+'\n')


if __name__=='__main__':main()
