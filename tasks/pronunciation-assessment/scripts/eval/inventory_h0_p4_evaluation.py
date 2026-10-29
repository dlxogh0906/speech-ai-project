"""Metadata-only evaluation inventory; no new test inference or fresh-data claim."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import collections
import json
from diagnose_h0_p4 import read_jsonl,write_json
from train_h0_backbone_control import file_sha
from mdd.h0_phone_audit import project_gaps


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',required=True);a=ap.parse_args()
    root=Path(a.root);p=json.loads((root/'protocol.json').read_text());manifest=Path(p['manifest'])
    vocab=[line.split()[0] for line in (manifest/'dict.ltr.txt').read_text().splitlines()]
    reports={};bindings={}
    for split in ('valid_unseen','test_unseen'):
        meta=read_jsonl(manifest/(split+'.meta.jsonl'))
        canon=[s.split() for s in (manifest/(split+'.canon')).read_text().splitlines()]
        actual=[s.split() for s in (manifest/(split+'.ltr')).read_text().splitlines()]
        c=collections.Counter();errors=collections.Counter();gaps=collections.Counter()
        for exp,t in zip(canon,actual):
            truth,insert=project_gaps(exp,t)
            for x,y in zip(exp,truth):
                if x==y:c[x]+=1
                else:errors[x]+=1
            gaps['total']+=len(insert);gaps['with_actual_insertion']+=sum(bool(g) for g in insert)
        reports[split]=dict(recordings=len(meta),speakers=len({m['spk'] for m in meta}),items=sorted({m['item'] for m in meta}),
            normal_by_phone={v:c[v] for v in vocab},error_by_expected_phone={v:errors[v] for v in vocab},
            absent_normal_phones=[v for v in vocab if not c[v]],insertion_gaps=dict(gaps),
            status='reused validation development' if split=='valid_unseen' else 'historically analyzed H0 test; not a new independent acquisition')
        for suffix in ('.meta.jsonl','.canon','.ltr'):bindings[str(manifest/(split+suffix))]=file_sha(manifest/(split+suffix))
    reports['test_unseen']['items_outside_reused_valid']=sorted(set(reports['test_unseen']['items'])-set(reports['valid_unseen']['items']))
    write_json(root/'evaluation_inventory.json',dict(metadata_only=True,new_model_test_inference=0,newly_acquired_recordings=0,
        existing=reports,bindings=bindings,source_sha256=file_sha(__file__),
        future_confirmation_requirements=['new prompts and speakers independent of development selection and model tuning',
            'document original model, native replay and public pretraining exposure limitations separately',
            'predeclare normal/substitution/deletion/insertion labels, acceptable pronunciation alternatives and uncertainty',
            'report all 46 vocabulary classes with zero-support and annotation-only symbols explicitly; do not invent impossible surface contexts',
            'cover vowels, initial/final consonants, liaison/tensing/nasal/lateral contexts with multiple prompts and speakers',
            'freeze model and seen-CAL thresholds before opening independent results',
            'retain original TAG and mark no independent listening until human adjudication exists'],
        no_fresh_final_test_claim=True))


if __name__=='__main__':main()
