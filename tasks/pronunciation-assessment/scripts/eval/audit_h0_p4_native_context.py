"""Check inherited Stage1 actual exposure before attributing Stage2 context bias."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import collections
import hashlib
import json
from train_h0_backbone_control import stream,file_sha
from train_w2vbert_ctc import read_split
from diagnose_h0_p4 import write_json


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',required=True);a=ap.parse_args()
    root=Path(a.root);p=json.loads((root/'protocol.json').read_text())
    p3=json.loads((Path(p['p3'])/'design_v2/protocol.json').read_text())
    parent=json.loads(Path(p3['parent_protocol']).read_text())
    manifest=Path(parent['stage1']['manifest']);rows,labels=read_split(str(manifest),'train')
    batches=stream(rows,p3['batch_samples'],p3['seed']);h=hashlib.sha256();draws=samples=0;exposure=collections.Counter()
    for _ in range(parent['stage1']['stop_at']*p3['accum']):
        ids=next(batches);exposure.update(ids);h.update((' '.join(map(str,ids))+'\n').encode())
        draws+=len(ids);samples+=sum(rows[i][1] for i in ids)
    observed=dict(row_draws=draws,audio_samples=samples,batch_order_sha256=h.hexdigest())
    receipt=Path(p3['historical_experiment'])/'w2vbert_stage1/finished.json'
    old=json.loads(receipt.read_text())['final_exposure']
    if any(observed[k]!=old[k] for k in observed):raise ValueError('Stage1 stream mismatch')
    c=collections.Counter();unique=collections.defaultdict(set)
    for i,n in exposure.items():
        tokens=labels[i];text=''.join(tokens)
        for j,t in enumerate(tokens):
            if t in ('ㅜ','ㅗ'):
                c[t+'_all_draws']+=n;unique[t+'_all_draws'].add((rows[i][0],j))
                if j>0 and j+1<len(tokens) and tokens[j-1]=='ㅅ' and tokens[j+1]=='ㄹ':
                    c[t+'_s_l_draws']+=n;unique[t+'_s_l_draws'].add((rows[i][0],j))
        for name,s in [('suri','ㅅㅜㄹㅣ'),('sori','ㅅㅗㄹㅣ')]:
            c[name+'_whole_target_draws']+=n*(text==s)
            c[name+'_substring_token_occurrences']+=n*text.count(s)
    write_json(root/'exposure/native_stage1.json',dict(actual_exposure=observed,counts=dict(c),
        unique_recording_positions={k:len(v) for k,v in unique.items()},source_sha256=file_sha(__file__),
        bindings={str(f):file_sha(f) for f in (manifest/'train.tsv',manifest/'train.ltr',receipt)},
        limitations=['native transcript/G2P targets, not independently annotated actual pronunciation',
                    'phone substring is not necessarily the written word or same coarticulation',
                    'public SSL pretraining exposure remains unknown']))


if __name__=='__main__':main()
