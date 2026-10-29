"""Matched XLS-R300M/2B original-audio caching and supervised dense-tail comparison."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),
               str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import collections
import copy
import gc
import hashlib
import json
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from train_sibilant_hubert_adaptation import RawDataset,novel_proxy,retained_subgroups
from train_sibilant_frozen_probe import write,pair_proxy
from mdd.sibilant_frozen_probe import sampled_frames,fit_projection,projection_digest
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_joint_vowel import joint_vowel_loss
from mdd.sibilant_vowel_presence import observed_vowel_support
from mdd.sibilant_xlsr_scale import (load_original,capture_prefix,ScaleReadout,
                                   scale_artifact,restore_scale,PRECISION)


class PrefixDataset:
    def __init__(self,path,relative,lengths):
        self.prefix = np.load(path/'prefix.npy',mmap_mode='r')
        meta = np.load(path/'coordinates.npz')
        self.source_lengths,self.skips = meta['source_lengths'],meta['skips']
        np.testing.assert_array_equal(meta['lengths'],lengths)
        self.lengths,self.relative = lengths,relative
        if self.prefix.shape[:2] != (len(lengths),34):
            raise ValueError('complete fixed original prefix cache required')

    def batch(self,ix,device):
        return (torch.as_tensor(np.array(self.prefix[ix]),device=device).float(),
                torch.as_tensor(self.source_lengths[ix],device=device),
                torch.as_tensor(self.skips[ix],device=device),
                torch.as_tensor(self.lengths[ix],device=device),
                torch.as_tensor(self.relative[ix],device=device))


def extract(model,meta,features,raw,frame_plan,out,device,binding):
    out.mkdir(parents=True,exist_ok=False)
    width = meta['width']
    prefix = np.lib.format.open_memmap(out/'prefix.npy',mode='w+',dtype=np.float16,
                                     shape=(len(features),34,width))
    samples = np.full((len(frame_plan),width),np.nan,np.float32)
    selected = collections.defaultdict(list)
    for destination,(i,frame) in enumerate(frame_plan):
        selected[int(i)].append((destination,int(frame)))
    source_lengths,skips = np.zeros(len(features),np.int64),np.zeros(len(features),np.int64)
    native_max,quantization_max = 0.,0.
    tick = time.monotonic()
    with torch.no_grad():
        for i,feature in enumerate(features):
            x,n,skip,hidden,info = capture_prefix(model,raw.source(feature['uid']),feature)
            prefix[i] = 0
            prefix[i,:len(x)] = x.cpu().numpy()
            source_lengths[i],skips[i] = n,skip
            # Final hidden roundtrip is part of every arm's runtime contract.
            h = hidden.to(torch.float16).float().cpu().numpy()
            for destination,frame in selected[i]:
                samples[destination] = h[frame]
            native_max = max(native_max,info['original_replay_max_difference'])
            quantization_max = max(quantization_max,info['prefix_roundtrip_hidden_max_difference'])
            if (i+1)%128 == 0 or i+1 == len(features):
                prefix.flush()
                progress = {'stage':'extract','done':i+1,'total':len(features),
                            'seconds':time.monotonic()-tick}
                write(out/'progress.json',progress)
                print(json.dumps(dict(size=meta['width'],**progress)),flush=True)
    prefix.flush()
    if not np.isfinite(samples).all():
        raise ValueError('TRAIN-only PCA frames missing/nonfinite')
    np.savez(out/'coordinates.npz',source_lengths=source_lengths,skips=skips,lengths=raw.lengths)
    projection = fit_projection(samples,256,device)
    torch.save(projection,out/'projection.pt')
    report = {'binding':binding,'metadata':meta,'complete':True,'source_sites':len(features),
              'prefix_shape':list(prefix.shape),'native_full_forward_max_difference':native_max,
              'prefix_roundtrip_hidden_max_difference':quantization_max,
              'PCA_sample_frames':len(frame_plan),'PCA_frame_plan_sha256':hashlib.sha256(frame_plan.tobytes()).hexdigest(),
              'projection_sha256':projection_digest(projection),'explained_variance_ratio':projection['explained_variance_ratio'],
              'seconds':time.monotonic()-tick,'supervised_optimizer_updates':0,
              'sha256':{name:service.sha(out/name) for name in ('prefix.npy','coordinates.npz','projection.pt')}}
    write(out/'report.json',report)
    del prefix,samples
    return projection,report


def score(model,data,device,batch=64,count=None):
    ps,av = [],[]
    model.eval()
    size = len(data.lengths) if count is None else count
    with torch.no_grad():
        for start in range(0,size,batch):
            ix = np.arange(start,min(size,start+batch))
            args = data.batch(ix,device)
            logits,role = model(*args)
            mask = torch.arange(32,device=device)[None] < args[3][:,None]
            ps.append(logits.softmax(-1).cpu().numpy())
            av.append(observed_vowel_support(role,mask).cpu().numpy())
    return np.concatenate(ps),np.concatenate(av)


def calls(ps,av,features,proxy):
    result = []
    for i,f in enumerate(features):
        ci = 1 if f['canonical'] == 'ㅅ' else 0
        direction = 'S_to_SS' if ci == 1 else 'SS_to_S'
        threshold = proxy[direction]['threshold']
        result.append(threshold is not None and bool(av[i]) and ps[i].argmax() == ci and ps[i,ci] >= threshold)
    return np.asarray(result,bool)


def fit_arm(encoder,projection,data,features,labels,y,phase,plan,eligibility,p,out,size,mode,binding):
    device = next(encoder.parameters()).device
    encoder.eval().requires_grad_(False)
    torch.manual_seed(p['seed'])
    model = ScaleReadout(encoder.encoder.layers[-2:],encoder.encoder.layer_norm,projection,mode).to(device).eval()
    initial = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.probe.head.parameters())).hexdigest()
    if initial != p['initial_head_sha256'] or sum(v.numel() for v in model.probe.head.parameters()) != 66889:
        raise ValueError('matched fresh head initialization/capacity differs')
    depth = len(encoder.encoder.layers)
    tail_keys = tuple('encoder.layers.'+str(i)+'.' for i in (depth-2,depth-1))
    prefix_before = tensor_digest({k:v for k,v in encoder.state_dict().items() if not k.startswith(tail_keys)})
    tail_before = tensor_digest(model.layers.state_dict())
    binding = dict(binding,original_tail_sha256=tail_before)
    groups = [{'params':list(model.probe.head.parameters()),'lr':p['head_lr']}]
    if mode == 'tail2':
        groups.append({'params':list(model.layers.parameters()),'lr':p['encoder_lr']})
    trainable = [v for group in groups for v in group['params']]
    optimizer = torch.optim.AdamW(groups,weight_decay=.01)
    losses,phase_losses,used,skipped,first_grad = [],[],set(),0,None
    tick = time.monotonic()
    for step,requested in enumerate(plan,1):
        ix = np.asarray([int(i) for i in requested if eligibility[int(i)]['target'] is not None],np.int64)
        if not len(ix) or (y[ix]<0).any():
            raise ValueError('empty/unknown original sampled targets')
        used.update(map(int,ix));skipped += len(requested)-len(ix)
        optimizer.zero_grad(set_to_none=True)
        logits,role = model(*data.batch(ix,device))
        cls = torch.nn.functional.cross_entropy(logits,torch.as_tensor(y[ix],device=device))
        aux = joint_vowel_loss(role,torch.as_tensor(phase[ix],device=device))
        loss = cls+p['phase_weight']*aux
        if not bool(torch.isfinite(loss)):
            raise ValueError('nonfinite acoustic adaptation loss')
        loss.backward()
        if step == 1:
            forbidden = [k for k,v in encoder.named_parameters() if v.grad is not None
                         and not (mode == 'tail2' and k.startswith(tail_keys))]
            norms = [float(sum(v.grad.square().sum() for v in layer.parameters() if v.grad is not None).sqrt())
                     for layer in model.layers] if mode == 'tail2' else []
            if forbidden or (mode == 'tail2' and not all(z>0 and np.isfinite(z) for z in norms)):
                raise ValueError('dense acoustic gradient/isolation failed')
            first_grad = {'tail_norms':norms,'forbidden_gradients':forbidden}
        torch.nn.utils.clip_grad_norm_(trainable,1.)
        optimizer.step()
        losses.append(float(cls.detach()));phase_losses.append(float(aux.detach()))
        if step%20 == 0:
            progress = {'arm':size+'_'+mode,'actual_optimizer_updates':step,'class_loss':losses[-1],
                        'phase_loss':phase_losses[-1],'seconds':time.monotonic()-tick}
            write(out/(size+'_'+mode+'.progress.json'),progress)
            print(json.dumps(progress),flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    seconds = time.monotonic()-tick
    prefix_after = tensor_digest({k:v for k,v in encoder.state_dict().items() if not k.startswith(tail_keys)})
    tail_changed = tensor_digest(model.layers.state_dict()) != tail_before
    if prefix_before != prefix_after or tail_changed != (mode == 'tail2') or skipped != 76:
        raise ValueError('frozen original prefix/dense update/replay changed')
    artifact = scale_artifact(model,binding)
    arm = size+'_'+mode
    torch.save(artifact,out/(arm+'.pt'))
    del optimizer
    model.eval().requires_grad_(False)
    ps,av = score(model,data,device)
    np.savez(out/(arm+'.scores.npz'),probabilities=ps,available=av)
    # Restore an independent tail, deliberately corrupting it in the tuned arm
    # so serialized-tail restoration cannot pass by reusing live trained weights.
    layers = copy.deepcopy(model.layers).requires_grad_(False)
    if mode == 'tail2':
        with torch.no_grad():
            next(layers.parameters()).add_(1.)
    restored = restore_scale(torch.load(out/(arm+'.pt'),map_location='cpu'),binding,layers,
                             copy.deepcopy(model.final_norm),device)
    check,valid = score(restored,data,device,count=64)
    np.testing.assert_array_equal(check,ps[:64]);np.testing.assert_array_equal(valid,av[:64])
    indexed = {z['feature_index']:z for z in labels if z['feature_index'] is not None}
    proxy = novel_proxy(ps,av,y,features,indexed)
    subgroups = retained_subgroups(ps,av,y,features,labels,proxy)
    write(out/(arm+'.subgroups.json'),subgroups)
    report = {'actual_optimizer_updates':300,'initial_head_sha256':initial,'head_parameters':66889,
              'trainable_parameters':sum(v.numel() for v in trainable),'prefix_unchanged':True,
              'tail_changed':tail_changed,'first_gradient':first_grad,'skipped_draws':skipped,
              'unique_sampled':len(used),'training_seconds':seconds,'reload_first64_exact':True,
              'class_loss_first_last50':[float(np.mean(losses[:50])),float(np.mean(losses[-50:]))],
              'phase_loss_first_last50':[float(np.mean(phase_losses[:50])),float(np.mean(phase_losses[-50:]))],
              'TRAIN_novel_proxy':proxy,'TRAIN_total_proxy':pair_proxy(ps,av,y,features,indexed),
              'all_retained_counts':subgroups['counts'],'available':int(av.sum()),
              'artifact_sha256':service.sha(out/(arm+'.pt')),
              'score_sha256':service.sha(out/(arm+'.scores.npz'))}
    write(out/(arm+'.report.json'),report)
    print(arm,json.dumps(report,ensure_ascii=False),flush=True)
    del model,restored,layers,artifact,trainable,groups
    gc.collect()
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',required=True)
    ap.add_argument('--out',required=True)
    a = ap.parse_args()
    p,out = json.loads(Path(a.protocol).read_text()),Path(a.out)
    for path,digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen experiment source/input changed: '+path)
    preflight = json.loads(Path(p['preflight_report']).read_text())
    if not preflight['passed'] or set(preflight['models']) != {'300m','2b'}:
        raise ValueError('both original native/gradient checks must pass')
    cache = Path(p['train_cache'])
    if json.loads((cache/'report.json').read_text())['partition'] != 'train':
        raise ValueError('only TRAIN may fit scale experiment')
    features = service.read_jsonl(p['features'])
    reference = service.read_jsonl(cache/'data.features.jsonl')
    labels = service.read_jsonl(cache/'data.labels.jsonl')
    target = np.load(p['targets']);y,phase = target['y'],target['phase']
    native = np.load(p['data']);lengths,relative = native['lengths'],native['relative_vowel']
    np.testing.assert_array_equal(y,np.load(cache/'targets.npz')['y'])
    if len(features) != 13741 or len(labels) != 14002 or phase.shape != relative.shape:
        raise ValueError('original retained TRAIN coverage differs')
    for i,(f,ref) in enumerate(zip(features,reference)):
        if (any(f[k] != ref[k] for k in ('uid','q_runtime','left','right','clip_sha256'))
                or f['feature_index'] != i or f['input_frames'] != lengths[i]):
            raise ValueError('original runtime frame order/source differs')
    plan = np.load(p['batch_plan']);eligibility = service.read_jsonl(p['ctc_targets'])
    if plan.shape != (300,16) or len(eligibility) != len(features):
        raise ValueError('fixed original 300x16 replay plan required')
    frame_plan = sampled_frames(features,lengths,32768)
    device = torch.device(p.get('device','cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',p)
    np.save(out/'PCA_frame_plan.npy',frame_plan)
    raw = RawDataset(cache,features,relative,lengths)
    report = {'scope':p['scope'],'arms':{},'deployment':False,'goal_complete':False,
              'retained_sites':len(labels),'runtime_inputs':len(features),'TRAIN_target_unknown':int((y<0).sum()),
              'TRAIN_no_input':sum(z['feature_index'] is None for z in labels)}
    for size,spec in p['models'].items():
        tick = time.monotonic()
        encoder,meta = load_original(spec['path'],device)
        binding = {'checkpoint_sha256':spec['sha256'],'feature_sha256':p['sha256'][p['features']],
                   'coordinate_schema':'original_global400_hop320_observedV',
                   'PCA_frame_plan_sha256':hashlib.sha256(frame_plan.tobytes()).hexdigest(),
                   'precision':PRECISION}
        projection,extraction = extract(encoder,meta,features,raw,frame_plan,out/(size+'_prefix'),device,binding)
        data = PrefixDataset(out/(size+'_prefix'),relative,lengths)
        # Before fitting, quantify batched cache replay against the extraction's
        # original-source outputs through the very same native model.
        torch.manual_seed(p['seed'])
        check_model = ScaleReadout(encoder.encoder.layers[-2:],encoder.encoder.layer_norm,projection,'frozen').to(device).eval()
        with torch.no_grad():
            selected,_ = check_model.observed_hidden(*data.batch(np.arange(16),device)[:4])
            differences = []
            for i in range(16):
                _,_,_,h,_ = capture_prefix(encoder,raw.source(features[i]['uid']),features[i])
                expected = h.to(torch.float16).float()
                torch.testing.assert_close(selected[i,:lengths[i]],expected,rtol=.01,atol=.01)
                differences.append(float((selected[i,:lengths[i]]-expected).abs().max()))
        extraction['batch_replay_first16_max_difference'] = max(differences)
        write(out/(size+'_prefix/report.json'),extraction)
        del check_model,selected,h,expected
        for mode in ('frozen','tail2'):
            report['arms'][size+'_'+mode] = fit_arm(encoder,projection,data,features,labels,y,phase,
                plan,eligibility,p,out,size,mode,binding)
            write(out/'report.json',report)
        report.setdefault('cost',{})[size] = {'seconds_load_extract_fit_score':time.monotonic()-tick,
            'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
            'prefix_bytes':(out/(size+'_prefix/prefix.npy')).stat().st_size}
        del encoder,data,projection
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
    candidate = report['arms']['2b_tail2']['TRAIN_novel_proxy']
    others = [report['arms'][k]['TRAIN_novel_proxy']['novel_TP'] for k in ('300m_frozen','300m_tail2','2b_frozen')]
    report['proceed_CAL'] = bool(candidate['novel_TP'] > max([564]+others)
        and candidate['wrong_actual'] <= 3 and candidate['S_to_SS']['novel_TP'] >= 240
        and candidate['SS_to_S']['novel_TP'] >= 324)
    # Keep exact call transitions on the complete retained TRAIN population.
    candidate_scores = np.load(out/'2b_tail2.scores.npz')
    candidate_calls = calls(candidate_scores['probabilities'],candidate_scores['available'],features,candidate)
    transitions = {}
    for arm,result in report['arms'].items():
        old = np.load(out/(arm+'.scores.npz'))
        old_calls = calls(old['probabilities'],old['available'],features,result['TRAIN_novel_proxy'])
        counts = collections.Counter()
        for z in labels:
            i = z['feature_index']
            if i is not None and candidate_calls[i] != old_calls[i]:
                counts[('added_' if candidate_calls[i] else 'lost_')+z['label_kind']] += 1
        transitions[arm] = dict(counts)
    report['candidate_vs_arm_call_changes_TRAIN_only'] = transitions
    write(out/'report.json',report)
    write(out/'finished.json',{'report_sha256':service.sha(out/'report.json')})
    print(json.dumps({'finished':True,'proceed_CAL':report['proceed_CAL']}),flush=True)


if __name__ == '__main__':
    main()
