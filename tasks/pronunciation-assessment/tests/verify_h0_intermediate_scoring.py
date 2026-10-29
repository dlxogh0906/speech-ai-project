"""Re-score existing H0 2k outputs through P3, without new inference/training."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT: raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),
              str(ROOT/'tasks/pronunciation-assessment/scripts/train'),
              str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import json
from train_h0_intermediate import bind
from train_h0_backbone_control import file_sha
from eval_h0_intermediate import evaluate_ledgers
from summarize_h0_intermediate import ledger, paired
from mdd.h0_phone_audit import contexts


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',required=True); ap.add_argument('--sequences',required=True)
    ap.add_argument('--out',required=True)
    a=ap.parse_args(); p=bind(a.protocol); p['_path']=a.protocol
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    old=Path(p['historical_experiment'])/'w2vbert_eval_2000'
    data={}
    for split in ('valid','valid_unseen'):
        m=Path(p['manifest'])
        data[split]=([s.split() for s in (m/(split+'.canon')).read_text().splitlines()],
                     [json.loads(s) for s in (m/(split+'.meta.jsonl')).open()])
    with (out/'sites.jsonl').open('x') as f:
        for line in (old/'sites.jsonl').open():
            s=json.loads(line)
            if s['split']=='valid_unseen' and s['role']!='selection': continue
            canon,meta=data[s['split']]; i=s['row']
            s.update(contexts(tuple(canon[i]),meta[i]['prompt'])[s['q']])
            f.write(json.dumps(s,ensure_ascii=False)+'\n')
    with (out/'sequences.jsonl').open('x') as f:
        for line in Path(a.sequences).open():
            r=json.loads(line); r['role']='selection'; f.write(json.dumps(r,ensure_ascii=False)+'\n')
    r=evaluate_ledgers(out,p,'initial_pipeline_check',2000,p['bindings'][p['parent_checkpoint']])
    expected=json.loads((old/'report.json').read_text())
    assert r['threshold']==expected['threshold'] and r['calibration']==expected['calibration']
    for split in ('valid','valid_unseen'):
        for group in ('all','sentences','strict_native_unseen'):
            assert r['results'][split][group]==expected['results'][split][group], (split,group)
    assert r['suri']['total']==475 and r['suri']['whole_sori']==86
    l=ledger(out/'sites.jsonl'); paired_result=paired(l,l,r['threshold'],r['threshold'])
    for split in ('valid','valid_unseen'):
        for axis in ('speaker','item'):
            for ci in paired_result['paired_ci'][split][axis].values(): assert ci['ci95']==[0.,0.]
    (out/'verification.json').write_text(json.dumps(dict(status='passed',
        checkpoint='historical initial 2000; no P3 training',same_original_metrics=True,
        suri=r['suri'],identity_paired_differences_zero=True,sequences=r['sequences'],
        source_sha256=file_sha(__file__)),ensure_ascii=False,indent=2))
    print(json.dumps(dict(status='passed',suri=r['suri'])),flush=True)


if __name__=='__main__': main()
