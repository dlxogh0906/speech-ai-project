"""Preregister, check, train and evaluate one dense residual-MoE pilot.

Historical assets remain read-only. The learned-router arm is compared against
new equal-expert mixing and the saved C0. No automatic follow-up/deployment.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
TASK = ROOT / 'tasks/pronunciation-assessment'
sys.path[:0] = [str(ROOT), str(TASK/'src'), str(TASK/'scripts/train'), str(TASK/'scripts/eval')]
import argparse
import collections
import copy
import gc
import gzip
import hashlib
import json
import os
import random
import shutil
import subprocess
import time
import numpy as np
import torch
from train_h0_backbone_control import Backbone, ctc_loss, file_sha, stream
from train_w2vbert_ctc import read_split, read_wav, load_vocab, tri_stage
from train_h0_intermediate import replay_prefix
from mdd.h0_intermediate import restore_rng, rng_state
from mdd.h0_moe import ResidualExperts, routing_counts
from mdd.h0_joint_evaluation import score_record, calibrate, aggregate, pilot_gate
from eval_h0_joint_pilot import records, paired
from eval_h0_rp_stress import read_audio, forward
from audit_h0_p5_readiness import write


class MoEModel(Backbone):
    def __init__(self, source, vocab, device, arm, parent=None):
        super().__init__('w2vbert', source, max(vocab.values())+1, device)
        if parent is not None:
            self.load_state_dict(parent, strict=True)
        self.added = ResidualExperts(routed=arm == 'routed').to(device)
        self.force_uniform = False

    def forward(self, xs, details=False):
        enc = self.fe(xs, sampling_rate=16000, return_tensors='pt', padding=True)
        am = enc['attention_mask'].to(self.device)
        outputs = self.net.wav2vec2_bert(input_features=enc['input_features'].to(self.device),
            attention_mask=am, output_hidden_states=True, return_dict=True)
        lengths = self.net._get_feat_extract_output_lengths(am.sum(-1)).long()
        fused, balance, gates = self.added(outputs.hidden_states[12],
            outputs.last_hidden_state, lengths, force_uniform=self.force_uniform)
        logits = self.net.lm_head(self.net.dropout(fused))
        self.last_gates = gates.detach()
        self.last_lengths = lengths.detach()
        self.last_predictions = logits.detach().argmax(-1)
        return (logits, lengths, balance, gates) if details else (logits, lengths)


def read(path):
    return json.loads(Path(path).read_text())


def bind(root):
    p = read(root/'protocol.json')
    for path, digest in p['bindings'].items():
        if file_sha(path) != digest:
            raise ValueError('changed bound asset: '+path)
    return p


def seed_all(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)


def prepare(root, historical):
    root.mkdir(parents=True, exist_ok=False)
    old = read(historical/'protocol.json')
    runtime = read(historical/'runtime_check.json')
    assert runtime['status'] == 'passed'
    for path, digest in runtime['source_bindings'].items():
        assert file_sha(path) == digest, path
    preflight = read(historical/'audio_annotation_preflight/report.json')
    assert preflight['status'] == 'passed'
    parent = historical/'training/common/step_2000.pt'
    parent_receipt = read(parent.parent/'finished.json')
    assert file_sha(parent) == parent_receipt['checkpoint_sha256']
    manifest = historical/'manifests/stage2'
    bindings = dict(runtime['source_bindings'])
    sources = [Path(__file__), TASK/'src/mdd/h0_moe.py', TASK/'src/mdd/h0_intermediate.py',
               TASK/'scripts/train/train_h0_intermediate.py', TASK/'tests/test_h0_moe.py',
               TASK/'scripts/eval/audit_h0_p5_readiness.py']
    bindings.update({str(f):file_sha(f) for f in sources})
    for path, info in read(historical/'manifests/report.json')['bindings'].items():
        if Path(path).parent == manifest:
            assert file_sha(path) == info['sha256'], path
            bindings[path] = info['sha256']
    for name in ('protocol.json', 'runtime_check.json', 'manifests/report.json',
                 'audio_annotation_preflight/report.json', 'audio_annotation_preflight/annotation_support.json',
                 'audio_annotation_preflight/audio_bindings.jsonl.gz', 'speaker_alias_verification.json'):
        f = historical/name; bindings[str(f)] = file_sha(f)
    for role in ('cal','inner','outer'):
        f = historical/'audio_annotation_preflight'/(role+'_annotations.jsonl.gz')
        digest = runtime['input_bindings'][str(f)]
        assert file_sha(f) == digest
        bindings[str(f)] = digest
        for suffix in ('_records.jsonl.gz', '_report.json'):
            f = historical/'evaluation/C0'/(role+suffix)
            bindings[str(f)] = file_sha(f)
    for name in ('calibration.json', 'finished.json'):
        f = historical/'evaluation/C0'/name; bindings[str(f)] = file_sha(f)
    bindings[str(parent)] = parent_receipt['checkpoint_sha256']
    bindings.update(old['public_SSL']['sha256'])
    for path, digest in old['public_SSL']['sha256'].items():
        assert file_sha(path) == digest
    exposures = preflight['planned_exposures']['stage2']['receipts']
    c0 = read(historical/'training/C0/finished.json')
    assert {k:c0['final_exposure'][k] for k in exposures[-1]} == exposures[-1]
    assert c0['parent_sha256'] == parent_receipt['checkpoint_sha256']
    p = dict(version=3, created_unix=time.time(), hypothesis='audio-dependent expert mixing improves exact diagnosis while protecting normal speech',
        implementation_revision='centered CV squared; absolute 1e-12 tolerance for uniform-point FP32 reduction residual; v1/v2 stopped before main training',
        authorization='user explicitly requested trying MoE on 2026-10-31; separate from closed weighting family',
        historical_root=str(historical), source=old['public_SSL']['path'], manifest=str(manifest),
        parent_checkpoint=str(parent), parent_exposure=exposures[0], expected_final_exposure=exposures[-1],
        seed=42, start_step=2000, stop_step=4000, accum=4, batch_samples=1600000,
        schedule_steps=40000, lr=1e-5, new_parameter_lr_factor=100., balance_weight=.01,
        architecture=dict(experts=4, width=1024, bottleneck=256, activation='GELU',
            expert_input='LayerNorm(hidden_states[12])',
            router_input='concatenate normalized layer12 and final layer24 audio features',
            output='h24 + sum(softmax(router)_k * expert_k(h12)); original shared CTC head',
            initialization='independent expert down projections, zero up projections and zero router; exact parent initial logits',
            sparse=False, all_experts_execute=True, expected_phone_or_TAG_input=False),
        arms=['uniform','routed'], main_comparison='routed minus uniform; historical C0 is secondary reference',
        objective='mean(CTC_NLL/target_length) + .01*4*sum((mean_valid_frame_soft_gates-.25)**2)',
        optimizer='fresh AdamW each arm; betas .9/.98 eps1e-8 WD0 clip5; parent RNG/AMP restored after initialization',
        masking='original SpecAugment off, LayerDrop .1, activation/head dropout .1',
        smoke_updates_per_arm=3, main_new_updates=4000,
        evaluation=dict(roles=['cal','inner','outer'], counts={'cal':2820,'inner':1016,'outer':29821},
            calibration='per-model 1% budget from observed CAL normals only; frozen for inner/outer',
            scoring='unchanged historical scorer; full confirmed denominators, coverage, ambiguity, deletion and insertion',
            gate='inner FPR<=1%; full-error exact recall>both uniform and historical C0; both coverages>=both references; protected recall drop<=5pp for >=30 errors; normal gap FPR rise<=.2pp',
            outer='report regardless of inner gate; never fit or select using outer',
            uncertainty='paired 2000 bootstrap replicates, speaker and item separately; fixed model/CAL',
            reused_development=True, independent_final=False),
        storage='two final FP32 model/RNG/AMP snapshots, no Adam state; compressed scored records and raw FP32 log-probabilities',
        no_automatic_extension=True, no_deployment=True, weighting_branch_stays_closed=True,
        minimum_free_before_training_bytes=8*1024**3, bindings=bindings)
    write(root/'protocol.json',p)
    snap = root/'source_snapshot'; snap.mkdir()
    for path in bindings:
        f=Path(path)
        if f.suffix=='.py' and str(f).startswith(str(ROOT)):
            dest=snap/f.relative_to(ROOT); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(f,dest)
    write(root/'live_status.json',dict(status='protocol_frozen_not_trained',protocol_sha256=file_sha(root/'protocol.json')))
    print(json.dumps(dict(status='prepared',free_bytes=shutil.disk_usage(root).free)),flush=True)


def init_model(p, arm):
    seed_all(p['seed'])
    ck = torch.load(p['parent_checkpoint'], map_location='cpu')
    assert ck['step']==p['start_step'] and ck['optimizer'] is None
    vocab = load_vocab(str(Path(p['manifest'])/'dict.ltr.txt'))
    assert ck['vocab']==vocab
    device=torch.device('cuda')
    model=MoEModel(p['source'],vocab,device,arm,ck['model'])
    return model,ck,vocab


def same_rng(a,b):
    return torch.equal(a['torch'],b['torch']) and all(torch.equal(x,y) for x,y in zip(a['cuda'],b['cuda']))


def check_identity(root):
    p=bind(root); rows,labels=read_split(p['manifest'],'train')
    batches,_,_,_=replay_prefix(rows,p); ids=next(batches)[:4]
    xs=[read_wav(rows[i][0],20.) for i in ids]
    model,ck,vocab=init_model(p,'routed')
    checks=[]
    for training in (False,True):
        model.train(training); state=ck['rng']
        restore_rng(state)
        with torch.no_grad(),torch.cuda.amp.autocast():
            native,lengths=Backbone.forward(model,xs)
        native_rng=rng_state()
        restore_rng(state)
        with torch.no_grad(),torch.cuda.amp.autocast():
            logits,lens,balance,gates=model(xs,details=True)
        assert torch.equal(native,logits) and torch.equal(lengths,lens)
        assert same_rng(native_rng,rng_state()) and abs(float(balance))<=1e-12
        checks.append(dict(training=training,max_logit_difference=float((native-logits).abs().max()),rng_equal=True))
    # Reproduce historical C0 on actual CAL audio with the unchanged scoring path.
    historical=Path(p['historical_root']); c0path=historical/'training/C0/step_4000.pt'
    assert file_sha(c0path)==read(c0path.parent/'finished.json')['checkpoint_sha256']
    del model,ck; gc.collect(); torch.cuda.empty_cache()
    c0=torch.load(c0path,map_location='cpu')
    native=Backbone('w2vbert',p['source'],max(vocab.values())+1,torch.device('cuda'))
    native.load_state_dict(c0['model'],strict=True); native.eval(); del c0
    ann=records(historical/'audio_annotation_preflight/cal_annotations.jsonl.gz')[:4]
    calrows,_=read_split(p['manifest'],'cal')
    oldrecords=records(historical/'evaluation/C0/cal_records.jsonl.gz')[:4]
    for r,(path,n),old in zip(ann,calrows,oldrecords):
        x,h=read_audio(path); assert len(x)==n
        lp=forward(native,x)
        saved=np.load(historical/'evaluation/C0/cal_log_probs'/(r['uid']+'.npz'))['lp']
        assert np.array_equal(lp,saved)
        value=score_record(r,lp,vocab); value.update(pcm_sha256=h,frames=len(lp))
        assert value==old
        altered=copy.deepcopy(r)
        for s in altered['sites']:
            s['actual']='ㄱ'
        changed=score_record(altered,lp,vocab)
        assert changed['predicted']==value['predicted']
        assert [s['score'] for s in changed['sites']]==[s['score'] for s in value['sites']]
    write(root/'identity_check.json',dict(status='passed',identity=checks,real_train_audio=len(xs),
        historical_C0_CAL_reproduced=4,tag_does_not_change_predictions_or_scores=True,
        protocol_sha256=file_sha(root/'protocol.json')))


def train(root,arm,smoke=False):
    p=bind(root); assert read(root/'identity_check.json')['status']=='passed'
    if not smoke: assert read(root/'runtime_check.json')['status']=='passed'
    out=root/('smoke' if smoke else 'training')/arm; out.mkdir(parents=True,exist_ok=False)
    if not smoke and shutil.disk_usage(root).free < (8 if arm=='uniform' else 5)*1024**3:
        raise RuntimeError('insufficient reserved storage; no cleanup performed')
    rows,labels=read_split(p['manifest'],'train'); batches,order,draws,samples=replay_prefix(rows,p)
    model,ck,vocab=init_model(p,arm)
    opt=torch.optim.AdamW([
        dict(params=model.net.parameters(),lr=p['lr'],lr_factor=1.),
        dict(params=[v for v in model.added.parameters() if v.requires_grad],
             lr=p['lr']*p['new_parameter_lr_factor'],lr_factor=p['new_parameter_lr_factor'])],
        betas=(.9,.98),eps=1e-8,weight_decay=0.)
    scaler=torch.cuda.amp.GradScaler(); scaler.load_state_dict(ck['scaler'])
    restore_rng(ck['rng']); del ck
    initial={n:v.detach().cpu().clone() for n,v in model.added.named_parameters()}
    base_name,base=next((n,v) for n,v in model.encoder().named_parameters() if v.ndim>=2)
    before=base.detach().clone() if smoke else None
    stop=p['start_step']+p['smoke_updates_per_arm'] if smoke else p['stop_step']
    started=time.time(); model.train()
    write(out/'execution.json',dict(arm=arm,smoke=smoke,protocol_sha256=file_sha(root/'protocol.json'),
        trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
        added_parameters=sum(v.numel() for v in model.added.parameters()),start_step=p['start_step'],stop_step=stop))
    with (out/'updates.jsonl').open('x') as log:
        for step in range(p['start_step']+1,stop+1):
            opt.zero_grad(set_to_none=True); losses=[]; balances=[]; route_mass=[]
            for _ in range(p['accum']):
                ids=next(batches); order.update((' '.join(map(str,ids))+'\n').encode())
                xs=[read_wav(rows[i][0],20.) for i in ids]
                if any(len(x)!=rows[i][1] for x,i in zip(xs,ids)):
                    raise ValueError('changed/truncated audio')
                ys=[[vocab[t] for t in labels[i]] for i in ids]
                draws+=len(ids); samples+=sum(map(len,xs))
                with torch.cuda.amp.autocast():
                    logits,lengths,balance,gates=model(xs,details=True)
                    ctc=ctc_loss(logits,lengths,ys)
                    total=ctc+p['balance_weight']*balance
                scaler.scale(total/p['accum']).backward()
                losses.append(float(ctc.detach())); balances.append(float(balance.detach()))
                route_mass.append(routing_counts(gates.detach(),lengths))
            lr=tri_stage(step-1,p['schedule_steps'],p['lr'])
            for group in opt.param_groups: group['lr']=lr*group['lr_factor']
            scaler.unscale_(opt); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not torch.isfinite(norm): raise ValueError('nonfinite gradient; no silent skipped update')
            scaler.step(opt); scaler.update()
            mass=np.sum([x['mass'] for x in route_mass],axis=0)/sum(x['frames'] for x in route_mass)
            row=dict(arm=arm,step=step,ctc_loss=float(np.mean(losses)),balance=float(np.mean(balances)),
                mean_routing_mass=mass.tolist(),lr=lr,gradient_norm=float(norm),row_draws=draws,
                audio_samples=samples,batch_order_sha256=order.hexdigest(),elapsed_seconds=time.time()-started)
            log.write(json.dumps(row)+'\n'); log.flush(); write(out/'live_status.json',row)
            if smoke or step%100==0: print(json.dumps(row),flush=True)
    changes={n:float((v.detach().cpu()-initial[n]).abs().max()) for n,v in model.added.named_parameters()}
    if arm=='routed': assert changes['router.weight']>0
    for k in range(4): assert changes['experts.%d.0.weight'%k]>0 and changes['experts.%d.2.weight'%k]>0
    receipt=dict(status='smoke_complete' if smoke else 'trained_not_evaluated',arm=arm,updates=stop-p['start_step'],
        final_exposure=row,added_parameter_changes=changes,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        protocol_sha256=file_sha(root/'protocol.json'),parent_sha256=p['bindings'][p['parent_checkpoint']])
    if smoke:
        assert not torch.equal(base,before)
        model.eval()
        with torch.no_grad(),torch.cuda.amp.autocast():
            dynamic,lens,_,gates=model(xs[:4],details=True)
            model.force_uniform=True; fixed,_=model(xs[:4]); model.force_uniform=False
        difference=float((dynamic-fixed).abs().max())
        if arm=='routed': assert difference>0 and float(gates.var(dim=1).sum())>0
        receipt.update(base_matrix_changed=base_name,dynamic_vs_uniform_logit_max=difference,
                       smoke_model_discarded=True)
    else:
        assert {k:row[k] for k in p['expected_final_exposure']}==p['expected_final_exposure']
        payload=dict(arm=arm,step=stop,model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            vocab=vocab,rng=rng_state(),scaler=scaler.state_dict(),optimizer=None,exposure=row,
            protocol_sha256=file_sha(root/'protocol.json'))
        dest=out/'step_4000.pt'; torch.save(payload,str(dest)+'.partial'); Path(str(dest)+'.partial').rename(dest)
        receipt['checkpoint_sha256']=file_sha(dest)
    write(out/'finished.json',receipt)


def close_checks(root):
    p=bind(root); reports=[read(root/'smoke'/arm/'finished.json') for arm in p['arms']]
    for d in reports: assert d['status']=='smoke_complete' and d['updates']==3
    a,b=[[json.loads(s) for s in (root/'smoke'/arm/'updates.jsonl').read_text().splitlines()]
         for arm in p['arms']]
    assert a[0]['ctc_loss']==b[0]['ctc_loss']
    assert a[0]['balance']==b[0]['balance'] and abs(a[0]['balance'])<=1e-12
    for x,y in zip(a,b):
        for key in p['expected_final_exposure']: assert x[key]==y[key]
    write(root/'runtime_check.json',dict(status='passed',identity=read(root/'identity_check.json'),
        smoke=reports,smoke_optimizer_updates=6,main_optimizer_updates=0,
        first_CTC_loss_equal=True,matched_smoke_exposure=True,protocol_sha256=file_sha(root/'protocol.json')))


def add_routing(acc,new):
    for k,v in new.items():
        if isinstance(v,list):
            acc[k]=[x+y for x,y in zip(acc.get(k,[0]*len(v)),v)]
        else: acc[k]=acc.get(k,0)+v


def evaluate(root,arm):
    p=bind(root); historical=Path(p['historical_root'])
    out=root/'evaluation'/arm; out.mkdir(parents=True,exist_ok=False)
    ckpath=root/'training'/arm/'step_4000.pt'; receipt=read(ckpath.parent/'finished.json')
    assert file_sha(ckpath)==receipt['checkpoint_sha256']
    ck=torch.load(ckpath,map_location='cpu')
    assert ck['arm']==arm and ck['protocol_sha256']==file_sha(root/'protocol.json')
    vocab=ck['vocab'];model=MoEModel(p['source'],vocab,torch.device('cuda'),arm)
    model.load_state_dict(ck['model'],strict=True); model.eval(); del ck
    audio={}
    with gzip.open(str(historical/'audio_annotation_preflight/audio_bindings.jsonl.gz'),'rt') as f:
        for line in f:
            x=json.loads(line)
            if any(role!='train' for role in x['roles']): audio[x['path']]=x
    reports={}; started=time.time(); diagnostics={}
    for role in p['evaluation']['roles']:
        annotations=records(historical/'audio_annotation_preflight'/(role+'_annotations.jsonl.gz'))
        rows,_=read_split(p['manifest'],role); assert len(rows)==len(annotations)==p['evaluation']['counts'][role]
        result=[]; lpdir=out/(role+'_log_probs'); lpdir.mkdir(); stats={'all':{},'nonblank':{}}
        with gzip.open(str(out/(role+'_records.jsonl.gz')),'xt') as dest:
            for i,(r,(path,n)) in enumerate(zip(annotations,rows)):
                assert r['row']==i and r['uid']==Path(path).stem
                x,h=read_audio(path); assert len(x)==n==audio[path]['samples'] and h==audio[path]['pcm_sha256']
                lp=forward(model,x); np.savez_compressed(lpdir/(r['uid']+'.npz'),lp=lp)
                for kind,pred in [('all',None),('nonblank',model.last_predictions)]:
                    add_routing(stats[kind],routing_counts(model.last_gates,model.last_lengths,pred))
                value=score_record(r,lp,vocab); value.update(pcm_sha256=h,frames=len(lp))
                dest.write(json.dumps(value,ensure_ascii=False)+'\n'); result.append(value)
                if (i+1)%500==0:
                    status=dict(arm=arm,role=role,completed=i+1,total=len(rows),elapsed_seconds=time.time()-started)
                    write(out/'live_status.json',status);print(json.dumps(status),flush=True)
        if role=='cal':
            calibration=calibrate(result); threshold=calibration['threshold']
            write(out/'calibration.json',dict(calibration,checkpoint_sha256=receipt['checkpoint_sha256']))
        reports[role]=aggregate(result,threshold,vocab);write(out/(role+'_report.json'),reports[role])
        diagnostics[role]=stats
        write(out/'routing.json',diagnostics)
    write(out/'finished.json',dict(status='fixed_arm_evaluated',arm=arm,vocab=vocab,
        checkpoint_sha256=receipt['checkpoint_sha256'],calibration=calibration,
        recordings=sum(d['recordings'] for d in reports.values()),elapsed_seconds=time.time()-started,
        output_hashes={str(f):file_sha(f) for f in out.glob('*.json*')}))


def compare(root):
    p=bind(root); historical=Path(p['historical_root']); out=root/'comparison';out.mkdir(exist_ok=False)
    folders={'C0':historical/'evaluation/C0','uniform':root/'evaluation/uniform','routed':root/'evaluation/routed'}
    thresholds={k:read(d/'calibration.json')['threshold'] for k,d in folders.items()}
    roles={}; uncertainty={}
    support=read(historical/'audio_annotation_preflight/annotation_support.json')
    for role in p['evaluation']['roles']:
        outputs={k:records(d/(role+'_records.jsonl.gz')) for k,d in folders.items()}; rebuilt={}
        for k,rs in outputs.items():
            vocab=read(folders[k]/'finished.json')['vocab']
            rebuilt[k]=aggregate(rs,thresholds[k],vocab)
            assert rebuilt[k]==read(folders[k]/(role+'_report.json'))
            c=rebuilt[k]['sites']['all']['counts'];g=rebuilt[k]['gaps']['all']['counts'];s=support[role]
            assert c['normal']==s['normal'] and c['error']==s['substitution']+s['deletion']
            assert g['normal']==s['normal_gap'] and g['error']==s['insertion_gap']
        for ref in ('C0','uniform'):
            assert len(outputs[ref])==len(outputs['routed'])
            for a,b in zip(outputs[ref],outputs['routed']):
                assert (a['uid'],a['pcm_sha256'])==(b['uid'],b['pcm_sha256'])
        roles[role]=rebuilt
        if role!='cal':
            uncertainty[role]={ref:{axis:paired(outputs[ref],outputs['routed'],thresholds[ref],thresholds['routed'],axis)
                for axis in ('speaker','item')} for ref in ('C0','uniform')}
    gates={ref:pilot_gate(roles['inner'][ref],roles['inner']['routed']) for ref in ('C0','uniform')}
    passed=all(g['continue_folds'] for g in gates.values())
    for g in gates.values(): g['passed']=g.pop('continue_folds')
    for arm in p['arms']:
        training=read(root/'training'/arm/'finished.json')
        assert training['updates']==2000
        assert {k:training['final_exposure'][k] for k in p['expected_final_exposure']}==p['expected_final_exposure']
    write(out/'report.json',dict(status='pilot_compared',pilot_passed=passed,inner_gate_vs=gates,roles=roles,
        paired=uncertainty,thresholds=thresholds,main_new_optimizer_updates=4000,smoke_optimizer_updates=6,
        per_arm_evaluation_recordings=33657,independent_final=False,production_changed=False,
        no_automatic_extension=True,verification='all reports recounted from saved predictions; fixed annotation denominators and PCM identity checked'))
    write(out/'finished.json',dict(status='comparison_complete',pilot_passed=passed,report_sha256=file_sha(out/'report.json')))
    print(json.dumps(dict(pilot_passed=passed,inner_gate_vs=gates)),flush=True)


def pipeline(root):
    p=bind(root); pipe=root/'pipeline';pipe.mkdir(exist_ok=False)
    steps=[('identity','check',None),('smoke_uniform','smoke','uniform'),('smoke_routed','smoke','routed'),
           ('close_checks','close_checks',None),('train_uniform','train','uniform'),('train_routed','train','routed'),
           ('evaluate_uniform','evaluate','uniform'),('evaluate_routed','evaluate','routed'),('compare','compare',None)]
    commands=[]
    for name,mode,arm in steps:
        cmd=[sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--mode',mode]
        if arm:cmd+=['--arm',arm]
        commands.append((name,cmd))
    write(pipe/'launch.json',dict(commands=commands,protocol_sha256=file_sha(root/'protocol.json'),created_unix=time.time()))
    started=time.time()
    for name,cmd in commands:
        with (pipe/(name+'.log')).open('x') as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                write(root/'live_status.json',dict(status='running',stage=name,pid=process.pid,
                    elapsed_seconds=time.time()-started,free_bytes=shutil.disk_usage(root).free))
                time.sleep(10)
        code=process.returncode;(pipe/(name+'.exit')).write_text(str(code)+'\n')
        if code:
            write(root/'live_status.json',dict(status='failed',stage=name,exit_code=code))
            (pipe/'run.exit').write_text(str(code)+'\n');raise SystemExit(code)
    result=read(root/'comparison/finished.json')
    write(pipe/'completion.json',dict(status='complete',comparison=result,elapsed_seconds=time.time()-started,
        protocol_sha256=file_sha(root/'protocol.json')))
    (pipe/'run.exit').write_text('0\n')
    write(root/'live_status.json',dict(status='pilot_complete',pilot_passed=result['pilot_passed'],production_changed=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--historical',type=Path)
    ap.add_argument('--mode',choices=['prepare','check','smoke','close_checks','train','evaluate','compare','run'],required=True)
    ap.add_argument('--arm',choices=['uniform','routed'])
    a=ap.parse_args();root=a.root.resolve()
    if a.mode=='prepare':prepare(root,a.historical.resolve())
    elif a.mode=='check':check_identity(root)
    elif a.mode=='smoke':train(root,a.arm,True)
    elif a.mode=='close_checks':close_checks(root)
    elif a.mode=='train':train(root,a.arm)
    elif a.mode=='evaluate':evaluate(root,a.arm)
    elif a.mode=='compare':compare(root)
    elif a.mode=='run':pipeline(root)


if __name__=='__main__':main()
