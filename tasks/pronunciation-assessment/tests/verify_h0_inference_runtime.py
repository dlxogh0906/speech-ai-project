"""Exercise the common evaluator on real audio with untrained public-backbone heads.

This is an execution check, never a Stage2 checkpoint or a performance result.
It verifies both split/role routes and that changing the TAG cannot change the
audio/reference prediction. Does not write model weights or touch training.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT/'tasks/pronunciation-assessment/scripts/train'),
               str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import gc
import json
import random
import numpy as np
import torch
from train_h0_backbone_control import Backbone, file_sha
from train_w2vbert_ctc import load_vocab, read_split, read_wav
from eval_h0_backbone_control import (infer_utterance_sites, calibration_role,
                                    threshold_for_normal_budget, metrics, cluster_intervals)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',required=True)
    ap.add_argument('--out',required=True)
    a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(Path(a.protocol).read_text())
    audit=json.loads(Path(protocol['input_audit']).read_text())
    manifest=Path(protocol['stage2']['manifest'])
    vocab=load_vocab(str(manifest/'dict.ltr.txt'))
    examples=[]
    for split in ['valid','valid_unseen']:
        rows,actual=read_split(str(manifest),split)
        canonical=[s.split() for s in (manifest/(split+'.canon')).read_text().splitlines()]
        metadata=[json.loads(s) for s in (manifest/(split+'.meta.jsonl')).read_text().splitlines()]
        selected={}
        for i,m in enumerate(metadata):
            # One real word and sentence in each speaker role from each split.
            key=(calibration_role(m['spk']),m['task'])
            if m['task'] in ('RS','RW') and key not in selected:
                selected[key]=i
            if len(selected)==4:break
        assert len(selected)==4
        for i in selected.values():
            examples.append((split,i,rows[i],canonical[i],actual[i],metadata[i]))
    report=dict(claim='execution_check_only_no_performance_result',
                protocol_sha256=file_sha(a.protocol),examples=len(examples),backbones={})
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for kind in ('w2vbert','xlsr300m'):
        torch.manual_seed(42);random.seed(42);np.random.seed(42)
        model=Backbone(kind,protocol['backbones'][kind]['path'],max(vocab.values())+1,device).eval()
        sites=[]
        for split,i,(path,frames),expected,actual,m in examples:
            x=read_wav(path,20.);assert len(x)==frames
            overlap=audit['native_text_overlap'].get(m['item'])
            got=infer_utterance_sites(model,x,expected,actual,vocab,m,overlap,split,i)
            alternative=infer_utterance_sites(model,x,expected,list(reversed(actual)),vocab,m,overlap,split,i)
            assert [(s['predicted'],s['score']) for s in got] == [(s['predicted'],s['score']) for s in alternative]
            assert len(got)==len(expected)
            sites.extend(got)
        calibration=[s for s in sites if s['split']=='valid' and s['role']=='calibration']
        selected=[s for s in sites if s['role']=='selection']
        threshold=threshold_for_normal_budget(calibration)
        m=metrics(selected,threshold)
        ci=cluster_intervals(selected,threshold,'speaker',repeats=100)
        assert m['total_sites']==len(selected)
        report['backbones'][kind]=dict(utterances=len(examples),sites=len(sites),
            calibration_sites=len(calibration),selection_sites=len(selected),
            tag_independent_prediction=True,canonical_site_count_preserved=True,
            bootstrap_status=ci['status'])
        print(json.dumps(report['backbones'][kind]),flush=True)
        del model
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    (out/'finished.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
