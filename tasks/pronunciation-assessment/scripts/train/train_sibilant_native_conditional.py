"""Matched native frozen/tail2 by retained old/new balanced source data."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),
               str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import collections
import functools
import json
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from train_sibilant_conditional_retained import inputs,score as score_summary,write
from train_sibilant_extra_normals import proposed
from train_sibilant_hubert_adaptation import retained_subgroups
from mdd.sibilant_cohort_exposure import cohort_plans
from mdd.sibilant_conditional_retained import TOKENS,new_conditional,fixed_direction_proxy
from mdd.sibilant_hubert_adaptation import load_native,tensor_digest
from mdd.sibilant_native_conditional import (capture_native,relative_grid,NativeConditional,
    prefix_digest,export_native,restore_native,runtime_logits)
from mdd.sibilant_vowel_presence import observed_vowel_support

ARMS = ('frozen_old_balanced_control','frozen_new_balanced_control',
        'tail2_old_balanced_control','tail2_new_balanced_candidate')


class Sources:
    def __init__(self,p,features):
        wanted = {f['uid'] for f in features};self.rows = {}
        for path in (p['old_responses'],p['new_rows']):
            with Path(path).open() as stream:
                for line in stream:
                    record = json.loads(line);row = record.get('row',record)
                    if row['uid'] in wanted:
                        self.rows[row['uid']] = {k:row[k] for k in ('uid','wav','frames')}
        if set(self.rows)!=wanted:raise ValueError('source runtime row missing')
        self.wave = functools.lru_cache(maxsize=128)(self._wave)
    def _wave(self,uid):return service.load_audio(self.rows[uid])[:320000]


def initialize(p,encoder,base,mode,device,tail):
    encoder.encoder.layers[-2:].load_state_dict(tail,strict=True)
    torch.manual_seed(p['seed'])
    return NativeConditional(encoder,new_conditional(base,p['binding'],device),mode).to(device).eval()


def preflight(p,out,protocol,data,y,features,offset,encoder,processor,base,binding,sources,device):
    out.mkdir(parents=True,exist_ok=False)
    model = NativeConditional(encoder,new_conditional(base,p['binding'],device),'frozen').to(device).eval()
    ix = sorted(set(np.linspace(0,offset-1,32,dtype=int).tolist()+
                    np.linspace(offset,len(features)-1,32,dtype=int).tolist()))
    maxima = collections.Counter();raw = np.load(p['old_native_observations'])
    with torch.no_grad():
        for i in ix:
            f = features[i];prefix,early,hidden,skip,detail = capture_native(encoder,processor,sources.wave(f['uid']),f)
            n = len(hidden);x = torch.nn.functional.pad(hidden,(0,0,0,32-n))[None]
            mask = torch.arange(32,device=device)[None]<n
            rel = torch.as_tensor(relative_grid(f)[None],device=device)
            s,m,phase = model.heads.heads[0].summarize(x,mask,rel)
            np.testing.assert_allclose(s.cpu().numpy()[0],data['summary'][i],atol=2e-5,rtol=2e-5)
            np.testing.assert_allclose(m.cpu().numpy()[0],data['moments'][i],atol=1e-6,rtol=1e-6)
            if bool(observed_vowel_support(phase,mask)[0])!=bool(data['available'][i]):
                raise ValueError('raw-control availability changed')
            if i<offset:
                np.testing.assert_array_equal(hidden.cpu().numpy().astype(np.float16),raw['x'][i,:n])
            maxima['summary'] = max(maxima['summary'],float(abs(s.cpu().numpy()[0]-data['summary'][i]).max()))
            maxima['moments'] = max(maxima['moments'],float(abs(m.cpu().numpy()[0]-data['moments'][i]).max()))
            maxima['tail_replay'] = max(maxima['tail_replay'],detail['native_full_vs_replay_max_difference'])
    # A real source clip must backpropagate identically through the full encoder
    # and its detached frozen-prefix replay. No optimizer update in preflight.
    i = next(i for i in ix if y[i]>=0);f = features[i];source = sources.wave(f['uid'])
    model.mode = 'tail2';encoder.encoder.layers[-2:].requires_grad_(True)
    params = [v for v in model.parameters() if v.requires_grad]
    target = torch.tensor([int(y[i])],device=device)
    logits,valid = runtime_logits(model,processor,source,f)
    direct = torch.autograd.grad(torch.nn.functional.cross_entropy(logits,target),params)
    with torch.no_grad():prefix,early,_,skip,_ = capture_native(encoder,processor,source,f)
    n = int(f['input_frames']);size = len(prefix)
    cache = {'prefix':prefix.cpu().numpy()[None],'early':early.cpu().numpy()[None],
        'lengths':np.asarray([n]),'source_lengths':np.asarray([size]),'skips':np.asarray([skip]),
        'relative':relative_grid(f)[None]}
    replay,available = model.from_cache(cache,np.asarray([0]),torch.tensor([TOKENS.index(f['canonical'])],device=device))
    cached = torch.autograd.grad(torch.nn.functional.cross_entropy(replay,target),params)
    torch.testing.assert_close(logits,replay,atol=2e-6,rtol=2e-6)
    grad_max = 0.
    for a,b in zip(direct,cached):
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-4)
        grad_max = max(grad_max,float((a-b).abs().max()))
    count = len(list(encoder.encoder.layers[-2:].parameters()))
    tail_grads = cached[:count]
    if not tail_grads or not sum(float(z.square().sum()) for z in tail_grads)>0:
        raise ValueError('no actual encoder tail gradient')
    if prefix_digest(encoder)!=binding['prefix_sha256']:raise ValueError('preflight prefix changed')
    write(out/'report.json',{'passed':True,'protocol_sha256':service.sha(protocol),
        'raw_controls':len(ix),'old_controls_native_hidden_fp16_exact':sum(i<offset for i in ix),
        'raw_summary_max_differences':dict(maxima),'full_vs_replay_gradient_max_difference':grad_max,
        'tail_gradient_squared_sum':sum(float(z.square().sum()) for z in tail_grads),
        'prefix_unchanged':True,'actual_optimizer_updates':0,'binding':binding})


def prepare_cache(p,out,protocol,features,data,offset,encoder,processor,base,sources,device):
    folder = Path(p['ram_cache']);n = len(features)
    if folder.exists():
        meta = json.loads((folder/'finished.json').read_text())
        if meta['protocol_sha256']!=service.sha(protocol):raise ValueError('RAM cache belongs to another protocol')
        for name,digest in meta['sha256'].items():
            if service.sha(folder/name)!=digest:raise ValueError('RAM cache contents changed')
        return {name:np.load(folder/(name+'.npy'),mmap_mode='r') for name in meta['arrays']},meta
    folder.mkdir(parents=True,exist_ok=False)
    shapes = {'prefix':((n,34,768),np.float32),'early':((n,32,768),np.float16),
        'source_lengths':((n,),np.int64),'lengths':((n,),np.int64),'skips':((n,),np.int64),
        'relative':((n,32),np.float32),'summary':((n,768),np.float32),
        'moments':((n,9),np.float32),'available':((n,),bool)}
    cache = {k:np.lib.format.open_memmap(str(folder/(k+'.npy')),mode='w+',dtype=dtype,shape=shape)
             for k,(shape,dtype) in shapes.items()}
    head = new_conditional(base,p['binding'],device).heads[0]
    maxima = collections.Counter();tick = time.monotonic()
    with torch.no_grad():
        for i,f in enumerate(features):
            prefix,early,hidden,skip,detail = capture_native(encoder,processor,sources.wave(f['uid']),f)
            count,size = len(hidden),len(prefix)
            cache['prefix'][i] = 0.;cache['early'][i] = 0.
            cache['prefix'][i,:size] = prefix.cpu().numpy();cache['early'][i,:count] = early.cpu().numpy()
            cache['lengths'][i],cache['source_lengths'][i],cache['skips'][i] = count,size,skip
            relative = relative_grid(f);cache['relative'][i] = relative
            x = torch.nn.functional.pad(hidden,(0,0,0,32-count))[None]
            mask = torch.arange(32,device=device)[None]<count
            s,m,phase = head.summarize(x,mask,torch.as_tensor(relative[None],device=device))
            cache['summary'][i] = s.cpu().numpy()[0];cache['moments'][i] = m.cpu().numpy()[0]
            cache['available'][i] = bool(observed_vowel_support(phase,mask)[0])
            np.testing.assert_allclose(cache['summary'][i],data['summary'][i],atol=2e-5,rtol=2e-5)
            np.testing.assert_allclose(cache['moments'][i],data['moments'][i],atol=1e-6,rtol=1e-6)
            if cache['available'][i]!=data['available'][i]:raise ValueError('original model availability differs')
            maxima['summary'] = max(maxima['summary'],float(abs(cache['summary'][i]-data['summary'][i]).max()))
            maxima['moment'] = max(maxima['moment'],float(abs(cache['moments'][i]-data['moments'][i]).max()))
            maxima['native_replay'] = max(maxima['native_replay'],detail['native_full_vs_replay_max_difference'])
            if (i+1)%500==0 or i+1==n:
                progress = {'stage':'original_native_prefix_capture','done':i+1,'total':n,'seconds':time.monotonic()-tick}
                write(out/'cache_progress.json',progress);print(json.dumps(progress),flush=True)
    for array in cache.values():array.flush()
    meta = {'protocol_sha256':service.sha(protocol),'arrays':list(cache),'observations':n,
        'source_prefix_precision':'float32_unpadded','max_differences':dict(maxima),
        'seconds_extraction':time.monotonic()-tick,
        'sha256':{name+'.npy':service.sha(folder/(name+'.npy')) for name in cache}}
    write(folder/'finished.json',meta);write(out/'cache_report.json',meta)
    return cache,meta


def train_arm(name,p,base,encoder,original_tail,data,cache,plans,y,device):
    mode = 'tail2' if name.startswith('tail2') else 'frozen'
    model = initialize(p,encoder,base,mode,device,original_tail)
    groups = [{'params':[v for v in head.parameters() if v.requires_grad],'lr':p['head_lr']}
              for head in model.heads.heads]
    if any(sum(v.numel() for v in group['params'])!=50117 for group in groups):raise ValueError('head capacity differs')
    if mode=='tail2':groups.append({'params':list(encoder.encoder.layers[-2:].parameters()),'lr':p['encoder_lr']})
    parameters = [v for g in groups for v in g['params']]
    optimizer = torch.optim.AdamW(groups,weight_decay=.01)
    arm = 'new_errors_new_normals' if '_new_' in name else 'old_errors_old_normals'
    losses,first_grad = [[],[]],None;tick = time.monotonic()
    for step in range(p['cohort_steps']):
        for branch,token in enumerate(TOKENS):
            ix = plans[arm][token][step];optimizer.zero_grad(set_to_none=True)
            if (y[ix]<0).any():raise ValueError('unknown class reached optimizer')
            if mode=='frozen':
                summary = torch.as_tensor(np.asarray(cache['summary'][ix]),device=device)
                moments = torch.as_tensor(np.asarray(cache['moments'][ix]),device=device)
            else:
                summary,moments,_ = model.summary_batch(cache,ix)
            logits = model.heads.heads[branch].from_summary(summary,moments)
            loss = torch.nn.functional.cross_entropy(logits,torch.as_tensor(y[ix],device=device))
            if not bool(torch.isfinite(loss)):raise ValueError('nonfinite acoustic actual5 loss')
            loss.backward()
            if first_grad is None:
                forbidden = [key for key,v in encoder.named_parameters() if v.grad is not None
                    and not (mode=='tail2' and key.startswith(('encoder.layers.10.','encoder.layers.11.')))]
                norms = [sum(float(v.grad.square().sum()) for v in layer.parameters() if v.grad is not None)**.5
                         for layer in encoder.encoder.layers[-2:]]
                if forbidden or (mode=='tail2' and not all(value>0 for value in norms)):
                    raise ValueError('native trainable boundary or backward differs')
                first_grad = {'forbidden_gradients':forbidden,'last2_gradient_norms':norms}
            torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();losses[branch].append(float(loss.detach()))
        if (step+1)%100==0:print(json.dumps({'arm':name,'updates_per_head':step+1,'total_updates':2*(step+1),'seconds':time.monotonic()-tick}),flush=True)
    updates = {token:sorted(set(int(optimizer.state[v]['step']) for v in group['params']))
               for token,group in zip(TOKENS,groups[:2])}
    if any(value!=[400] for value in updates.values()):raise ValueError('actual head optimizer step count differs')
    if mode=='tail2':
        tail_updates = sorted(set(int(optimizer.state[v]['step']) for v in groups[-1]['params']))
        if tail_updates!=[800]:raise ValueError('shared tail optimizer state did not receive800steps')
    else:tail_updates = []
    return model,{'actual_optimizer_updates':800,'head_optimizer_steps':updates,'tail_optimizer_steps':tail_updates,
        'head_trainable_parameters':100234,'encoder_trainable_parameters':sum(v.numel() for v in groups[-1]['params']) if mode=='tail2' else 0,
        'loss_first_last50':[[float(np.mean(z[:50])),float(np.mean(z[-50:]))] for z in losses],
        'first_gradient_audit':first_grad,'seconds_training':time.monotonic()-tick}


def score_native(model,cache,features,offset,device,name):
    if model.mode=='frozen':
        return score_summary(model.heads,{k:cache[k] for k in ('summary','moments','available')},features,offset,device,True)
    ps,av = [],[]
    with torch.no_grad():
        for left,right in ((0,offset),(offset,len(features))):
            for start in range(left,right,64):
                ix = np.arange(start,min(start+64,right))
                route = torch.tensor([TOKENS.index(features[i]['canonical']) for i in ix],device=device)
                logits,available = model.from_cache(cache,ix,route)
                ps.extend(logits.softmax(-1).cpu().numpy());av.extend(available.cpu().numpy())
                if (start-left)%2048==0 or start+64>=right:
                    print(json.dumps({'scoring':name,'done':min(start+64,right),'total':len(features)}),flush=True)
    return np.asarray(ps,np.float32),np.asarray(av,bool)


def main():
    ap = argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--stage',required=True,choices=['preflight','fit','audit'])
    a = ap.parse_args();p = json.loads(Path(a.protocol).read_text());out = Path(a.out)
    if p['native_candidate']!=ARMS[-1] or p['cohort_steps']!=400:raise ValueError('frozen native candidate/design differs')
    data,y,features,labels,indexed,offset,_,_,_ = inputs(p)
    plans,plan_report = cohort_plans(np.load(p['old_plan']),y,indexed,features,
        service.read_jsonl(p['old_ctc_targets']),offset,p['seed'],steps=p['cohort_steps'])
    for arm in ('old_errors_old_normals','new_errors_new_normals'):
        for branch,token in enumerate(TOKENS):
            np.testing.assert_array_equal(plans[arm][token],np.load(Path(p['cohort_preflight'])/(arm+'.'+str(branch)+'.plan.npy')))
    torch.set_num_threads(2);device = torch.device(p['device'])
    encoder,processor,native_binding = load_native(p['native_manifest'],device)
    original_tail = {k:v.detach().cpu().clone() for k,v in encoder.encoder.layers[-2:].state_dict().items()}
    base = torch.load(p['control_artifact'],map_location='cpu')
    binding = {'native':native_binding,'prefix_sha256':prefix_digest(encoder),'original_tail_sha256':tensor_digest(original_tail)}
    sources = Sources(p,features)
    if a.stage=='preflight':
        preflight(p,out,a.protocol,data,y,features,offset,encoder,processor,base,binding,sources,device);return
    check = json.loads((Path(p['native_preflight'])/'report.json').read_text())
    if not check['passed'] or check['protocol_sha256']!=service.sha(a.protocol) or check['binding']!=binding:
        raise ValueError('native prefix/gradient preflight missing or stale')
    if a.stage=='fit':out.mkdir(parents=True,exist_ok=False)
    cache,cache_report = prepare_cache(p,out,a.protocol,features,data,offset,encoder,processor,base,sources,device)
    parent = initialize(p,encoder,base,'frozen',device,original_tail)
    parent_ps,parent_av = score_native(parent,cache,features,offset,device,'parent')
    reference = np.concatenate([np.load(p['old_scores'])['probabilities'],np.load(Path(p['new_summary'])/'summary.npz')['probabilities']])
    np.testing.assert_allclose(parent_ps,reference,atol=2e-6,rtol=2e-6)
    np.testing.assert_array_equal(parent_av,data['available'])
    by_old = {i:z for i,z in indexed.items() if i<offset}
    policy = fixed_direction_proxy(parent_ps[:offset],parent_av[:offset],y[:offset],features[:offset],by_old)
    if (policy['novel_TP'],policy['normal_FP'],policy['wrong_actual'])!=(564,87,3):
        raise ValueError('initial full-population native decision counts changed')
    if a.stage=='audit':
        audit = {'passed':True,'arms':{},'production_changed':False}
        for name in ARMS:
            model,training = train_arm(name,p,base,encoder,original_tail,data,cache,plans,y,device)
            saved = torch.load(out/(name+'.pt'),map_location='cpu')
            current = export_native(model,base,binding)
            if tensor_digest(model.encoder.encoder.layers[-2:].state_dict())!=saved['binding']['tail_sha256']:
                raise ValueError('reexecuted native tail weights differ')
            for token in TOKENS:
                if current['conditional']['heads'][token]['binding']['head_sha256']!=saved['conditional']['heads'][token]['binding']['head_sha256']:
                    raise ValueError('reexecuted conditional head weights differ')
            audit['arms'][name] = {'reexecuted_tail_and_heads_bitexact':True,'actual_reexecution_updates':training['actual_optimizer_updates']}
        write(out/'final_training_audit.json',audit);return
    write(out/'protocol.json',p);np.savez(out/'parent.scores.npz',probabilities=parent_ps,available=parent_av)
    report = {'scope':p['native_scope'],'plans':plan_report,'arms':{},'parent_fixed_direction_TRAIN':policy,
        'original_all_GT_sites':14002,'original_unknown_inputs':int((y[:offset]<0).sum()),
        'original_no_input_sites':sum(z['feature_index'] is None for z in labels[:14002]),
        'new_source_errors_retained':728,'new_supported_pair_errors':675,'new_supported_normals':876,
        'cache_report':cache_report,'production_changed':False,'service_improvement_proven':False,
        'reserved_scored':False,'goal_complete':False,'binding':binding}
    all_ps,all_av,policies = {'parent':parent_ps},{'parent':parent_av},{'parent':policy}
    sample_ix = sorted(set(np.linspace(0,offset-1,32,dtype=int).tolist()+np.linspace(offset,len(features)-1,32,dtype=int).tolist()))
    for name in ARMS:
        tick = time.monotonic();torch.cuda.reset_peak_memory_stats(device) if device.type=='cuda' else None
        model,training = train_arm(name,p,base,encoder,original_tail,data,cache,plans,y,device)
        artifact = export_native(model,base,binding)
        artifact['training_provenance'] = dict(training,protocol_sha256=service.sha(a.protocol),arm=name)
        torch.save(artifact,out/(name+'.pt'))
        ps,av = score_native(model,cache,features,offset,device,name)
        # Reload the coupled native-tail/scoped-head artifact, then use direct
        # original PCM inference rather than the replay/cache path on64controls.
        encoder.encoder.layers[-2:].load_state_dict(original_tail,strict=True)
        restored = restore_native(torch.load(out/(name+'.pt'),map_location='cpu'),encoder,binding,p['binding'],device)
        raw_max = 0.
        with torch.no_grad():
            for i in sample_ix:
                logits,valid = runtime_logits(restored,processor,sources.wave(features[i]['uid']),features[i])
                probability = logits.softmax(-1)[0].cpu().numpy()
                np.testing.assert_allclose(probability,ps[i],atol=2e-6,rtol=2e-6)
                if bool(valid[0])!=bool(av[i]):raise ValueError('raw reload availability differs')
                raw_max = max(raw_max,float(abs(probability-ps[i]).max()))
        np.savez(out/(name+'.scores.npz'),probabilities=ps,available=av)
        policy = fixed_direction_proxy(ps[:offset],av[:offset],y[:offset],features[:offset],by_old)
        write(out/(name+'.original.subgroups.json'),retained_subgroups(ps[:offset],av[:offset],y[:offset],features[:offset],labels[:14002],policy))
        write(out/(name+'.all.subgroups.json'),retained_subgroups(ps,av,y,features,labels,policy))
        report['arms'][name] = dict(training,TRAIN_proxy=policy,raw_reload_controls=64,
            raw_reload_probability_max_difference=raw_max,prefix_PCA_roles_unchanged=True,
            available_original=int(av[:offset].sum()),available_new=int(av[offset:].sum()),
            availability_lost_original=int((parent_av[:offset]&~av[:offset]).sum()),
            availability_gained_original=int((~parent_av[:offset]&av[:offset]).sum()),
            seconds_training_scoring_reload=time.monotonic()-tick,
            peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            artifact_sha256=service.sha(out/(name+'.pt')))
        all_ps[name],all_av[name],policies[name] = ps,av,policy
        write(out/'report.json',report)
    candidate = policies[ARMS[-1]]
    report['proceed_CAL'] = bool(candidate['novel_TP']>max(z['novel_TP'] for name,z in policies.items() if name!=ARMS[-1])
        and candidate['S_to_SS']['novel_TP']>=240 and candidate['SS_to_S']['novel_TP']>=324 and candidate['wrong_actual']<=3)
    changes = {name:collections.Counter() for name in ARMS}
    with (out/'paired_proposal_ledger.jsonl').open('x') as stream:
        for z in labels:
            calls = {name:proposed(z,ps,all_av[name],features,policies[name]) for name,ps in all_ps.items()}
            prefix = 'normal' if z['label_kind']=='correct' else 'error_or_unknown'
            for name,counter in changes.items():
                counter[prefix+'_added_calls'] += calls[name] and not calls['parent']
                counter[prefix+'_lost_calls'] += calls['parent'] and not calls[name]
            service.write_line(stream,{'uid':z['uid'],'q_runtime':z.get('q_runtime'),
                'feature_index':z['feature_index'],'label_kind':z['label_kind'],'proposals':calls})
    report['all_retained_call_changes'] = {k:dict(v) for k,v in changes.items()}
    write(out/'report.json',report);write(out/'finished.json',{'report_sha256':service.sha(out/'report.json')})
    print(json.dumps({'finished':True,'proceed_CAL':report['proceed_CAL'],
        'results':{k:v['TRAIN_proxy'] for k,v in report['arms'].items()}}),flush=True)


if __name__=='__main__':main()
