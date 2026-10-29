"""Original-PCM inference and full CAL screening for one frozen native model.

TRAIN-selected exploratory follow-ups must identify that selection explicitly;
this entry point never chooses another model or changes a trained parameter.
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
import functools
import json
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from train_sibilant_conditional_retained import write
from eval_sibilant_presence_screen import calibrate,changed,flagged,exact,counts,row_changes
from mdd.sibilant_presence_policy import decision
from mdd.sibilant_hubert_adaptation import load_native,tensor_digest
from mdd.sibilant_native_conditional import prefix_digest,restore_native,runtime_logits


def screen(features,labels,ps,av):
    result = calibrate(features,labels,ps,av)
    strict = result['alternatives']['one_percent']
    result['strict_followup_qualified'] = False
    if strict['feasible']:
        actions = [decision(f,ps[i],av[i],strict['policy']) for i,f in enumerate(features)]
        by = collections.defaultdict(list)
        for z in labels:
            if z['feature_index'] is not None:by[z['feature_index']].append(z)
        unknown_actions = sum(action is not None and (len(by[i])!=1 or by[i][0]['label_kind']=='ambiguous')
                              for i,action in enumerate(actions))
        before = [z['existing'] for z in labels]
        after = [changed(v,actions[z['feature_index']]) if z['feature_index'] is not None else v
                 for z,v in zip(labels,before)]
        lost_warnings = sum(z['label_kind'] not in ('correct','ambiguous') and flagged(a) and not flagged(b)
                            for z,a,b in zip(labels,before,after))
        lost_exact = sum(exact(z,a) and not exact(z,b) for z,a,b in zip(labels,before,after))
        result['strict_pairwise_preservation'] = {'lost_real_error_warnings':lost_warnings,
            'lost_exact':lost_exact,'actions_with_unknown_or_ambiguous_GT':unknown_actions}
        result['strict_followup_qualified'] = not (lost_warnings or lost_exact or unknown_actions)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True)
    ap.add_argument('--out',required=True);a = ap.parse_args()
    p = json.loads(Path(a.protocol).read_text());out = Path(a.out)
    for path,digest in p['sha256'].items():
        if service.sha(path)!=digest:raise ValueError('frozen followup input changed: '+path)
    if p.get('require_training_audit'):
        if not json.loads(Path(p['require_training_audit']).read_text())['passed']:
            raise ValueError('training reexecution audit must finish before followup')
    cache = Path(p['cache']);meta = json.loads((cache/'report.json').read_text())
    if meta['partition']!=p['partition']:raise ValueError('declared followup partition differs')
    features = service.read_jsonl(p['native_features']);original = service.read_jsonl(cache/'data.features.jsonl')
    if len(features)!=len(original) or len(features)!=p['expected_inputs']:raise ValueError('full input population changed')
    for i,(f,g) in enumerate(zip(features,original)):
        if f['feature_index']!=i or any(f[k]!=g[k] for k in ('uid','q_runtime','left','right','clip_sha256','canonical','expected_V')):
            raise ValueError('automatic runtime feature identity changed')
    rows = {}
    with (cache/'data.responses.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)['row'];rows[row['uid']] = {k:row[k] for k in ('uid','wav','frames')}
    @functools.lru_cache(maxsize=128)
    def source(uid):return service.load_audio(rows[uid])[:320000]
    torch.set_num_threads(2);device = torch.device(p.get('device','cuda'))
    encoder,processor,native = load_native(p['native_manifest'],device)
    binding = {'native':native,'prefix_sha256':prefix_digest(encoder),
        'original_tail_sha256':tensor_digest(encoder.encoder.layers[-2:].state_dict())}
    model = restore_native(torch.load(p['artifact'],map_location='cpu'),encoder,binding,p['head_binding'],device)
    before = tensor_digest(model.state_dict())
    out.mkdir(parents=True,exist_ok=False);write(out/'protocol.json',p)
    ps,av = [],[]
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize()
    tick = time.monotonic()
    with torch.no_grad():
        for i,f in enumerate(features):
            logits,available = runtime_logits(model,processor,source(f['uid']),f)
            ps.append(logits.softmax(-1)[0].cpu().numpy());av.append(bool(available[0]))
            if (i+1)%500==0 or i+1==len(features):
                progress = {'partition':p['partition'],'done':i+1,'total':len(features),'seconds':time.monotonic()-tick}
                write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    if device.type=='cuda':torch.cuda.synchronize()
    elapsed = time.monotonic()-tick
    if tensor_digest(model.state_dict())!=before:raise ValueError('scoring changed frozen trained model')
    ps,av = np.asarray(ps,np.float32),np.asarray(av,bool)
    np.savez(out/'scores.npz',probabilities=ps,available=av)
    report = {'scope':p['scope'],'selection':p['selection'],'partition':p['partition'],
        'all_inputs':len(features),'available':int(av.sum()),'parameter_updates':0,
        'original_PCM_native_forward':True,'TAG_actual_or_times_in_inference':False,
        'seconds_scoring':elapsed,'mean_seconds_per_site':elapsed/len(features),
        'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        'cost_scope':'Offline singleclip native encoder+head/WAV IO; excludes service decoder, API/UI/NPU/model load. Not isolated service benchmark.',
        'scores_sha256':service.sha(out/'scores.npz'),'production_changed':False,'goal_complete':False}
    # First GT access occurs after all original-PCM model outputs are fixed.
    labels = service.read_jsonl(cache/'data.labels.jsonl')
    if len(labels)!=p['expected_sites']:raise ValueError('full GT denominator changed')
    report['all_GT_sites'] = len(labels);report['no_input_GT_sites'] = sum(z['feature_index'] is None for z in labels)
    if p['partition']=='cal':
        report['CAL_screen'] = screen(original,labels,ps,av)
    elif p.get('policy'):
        actions = [decision(f,ps[i],av[i],p['policy']) for i,f in enumerate(original)]
        before = [z['existing'] for z in labels]
        after = [changed(v,actions[z['feature_index']]) if z['feature_index'] is not None else v for z,v in zip(labels,before)]
        report['frozen_policy_counts'] = counts(labels,after)
        report['frozen_policy_changes'] = row_changes(labels,before,after)
    write(out/'report.json',report);write(out/'finished.json',{'report_sha256':service.sha(out/'report.json')})
    print(json.dumps({'finished':True,'partition':p['partition'],'CAL_screen':report.get('CAL_screen')}),flush=True)


if __name__=='__main__':main()
