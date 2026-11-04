"""One preregistered auxiliary-feature CTC arm versus the verified fold0 C0.

Reuses the historical equal-data/equal-parent/equal-update baseline, never its
weights as candidate initialization. No deployment or automatic next trial.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
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
from train_h0_backbone_control import Backbone, file_sha
from train_w2vbert_ctc import read_split, read_wav, load_vocab, tri_stage
from train_h0_intermediate import replay_prefix
from mdd.h0_intermediate import rng_state, restore_rng
from mdd.h0_joint_weighting import weighted_ctc_loss
from mdd.phonological_aux import schema, feature_targets, minimum_frames, FeatureHeads, auxiliary_loss
from mdd.h0_joint_evaluation import score_record, calibrate, aggregate, pilot_gate, site_counts
from mdd.h0_phone_audit import surface_position
from eval_h0_joint_pilot import records, paired
from eval_h0_rp_stress import read_audio, forward
from audit_h0_p5_readiness import write
from summarize_h0_moe_pilot import recount, ratio


def read(path):
    return json.loads(Path(path).read_text())


class FeatureModel(Backbone):
    def __init__(self, source, vocab, device, parent=None):
        super().__init__('w2vbert', source, max(vocab.values())+1, device)
        if parent is not None:
            self.load_state_dict(parent, strict=True)
        self.spec = schema(vocab)
        self.added = FeatureHeads(1024, self.spec).to(device)
        # Capture exactly the existing final-dropout output. No second encoder
        # forward, extra stochastic layer, or reference/TAG inference input.
        self.net.dropout.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self.features = output

    def forward(self, xs, auxiliary=False):
        logits, lengths = super().forward(xs)
        if auxiliary:
            return logits, lengths, self.added(self.features)
        return logits, lengths


def bind(root):
    p = read(root/'protocol.json')
    for path, digest in p['bindings'].items():
        if file_sha(path) != digest:
            raise ValueError('changed frozen input/code: '+path)
    return p


def init_model(p):
    torch.manual_seed(p['seed']); np.random.seed(p['seed']); random.seed(p['seed'])
    ck = torch.load(p['parent_checkpoint'], map_location='cpu')
    assert ck['step'] == p['start_step'] and ck['optimizer'] is None
    vocab = load_vocab(str(Path(p['manifest'])/'dict.ltr.txt'))
    assert ck['vocab'] == vocab
    model = FeatureModel(p['source'], vocab, torch.device('cuda'), ck['model'])
    return model, ck, vocab


def prepare(root, historical):
    root.mkdir(parents=True, exist_ok=False)
    old = read(historical/'protocol.json'); runtime = read(historical/'runtime_check.json')
    preflight = read(historical/'audio_annotation_preflight/report.json')
    assert runtime['status'] == preflight['status'] == 'passed'
    bindings = dict(runtime['source_bindings'])
    for path, digest in bindings.items():
        assert file_sha(path) == digest, path
    parent = historical/'training/common/step_2000.pt'
    parent_done = read(parent.parent/'finished.json')
    assert file_sha(parent) == parent_done['checkpoint_sha256']
    c0 = historical/'training/C0/step_4000.pt'; c0done = read(c0.parent/'finished.json')
    assert file_sha(c0) == c0done['checkpoint_sha256']
    assert c0done['parent_sha256'] == parent_done['checkpoint_sha256']
    manifest = historical/'manifests/stage2'
    for path, info in read(historical/'manifests/report.json')['bindings'].items():
        if Path(path).parent == manifest:
            assert file_sha(path) == info['sha256'], path
            bindings[path] = info['sha256']
    for name in ('protocol.json', 'runtime_check.json', 'manifests/report.json',
                 'audio_annotation_preflight/report.json', 'audio_annotation_preflight/annotation_support.json',
                 'audio_annotation_preflight/audio_bindings.jsonl.gz', 'speaker_alias_verification.json',
                 'training/C0/finished.json', 'training/C0/updates.jsonl', 'training/common/finished.json'):
        f = historical/name; bindings[str(f)] = file_sha(f)
    for role in ('cal', 'inner', 'outer'):
        f = historical/'audio_annotation_preflight'/(role+'_annotations.jsonl.gz')
        assert file_sha(f) == runtime['input_bindings'][str(f)]
        bindings[str(f)] = file_sha(f)
        for suffix in ('_records.jsonl.gz', '_report.json'):
            f = historical/'evaluation/C0'/(role+suffix); bindings[str(f)] = file_sha(f)
    for name in ('calibration.json', 'finished.json'):
        f = historical/'evaluation/C0'/name; bindings[str(f)] = file_sha(f)
    sources = [Path(__file__), TASK/'src/mdd/phonological_aux.py', TASK/'tests/test_phonological_aux.py',
               TASK/'src/mdd/h0_intermediate.py', TASK/'scripts/train/train_h0_intermediate.py',
               TASK/'scripts/eval/summarize_h0_moe_pilot.py']
    bindings.update({str(f): file_sha(f) for f in sources})
    bindings[str(parent)] = parent_done['checkpoint_sha256']
    bindings[str(c0)] = c0done['checkpoint_sha256']
    bindings.update(old['public_SSL']['sha256'])
    for path, digest in old['public_SSL']['sha256'].items(): assert file_sha(path) == digest
    exposures = preflight['planned_exposures']['stage2']['receipts']
    assert {k:c0done['final_exposure'][k] for k in exposures[-1]} == exposures[-1]
    vocab = load_vocab(str(manifest/'dict.ltr.txt')); spec = schema(vocab)
    p = dict(version=1, created_unix=time.time(), authorization='user requested fixed CTC versus phonological auxiliary learning on 2026-11-01',
        historical_root=str(historical), source=old['public_SSL']['path'], manifest=str(manifest),
        parent_checkpoint=str(parent), baseline_checkpoint=str(c0), seed=42, start_step=2000, stop_step=4000,
        accum=4, batch_samples=1600000, schedule_steps=40000, lr=1e-5, new_parameter_lr_factor=100.,
        auxiliary_weight=.3, smoke_updates=3, parent_exposure=exposures[0], expected_final_exposure=exposures[-1],
        hypothesis='sharing actual-label manner, laryngeal-series and place supervision improves held-out-prompt exact diagnosis',
        architecture='three independent linear CTC heads on existing final-layer post-dropout audio features; unchanged original phone head for all scoring',
        feature_schema=spec, auxiliary_sequence='one mapped attribute per actual target token; adjacent identical attributes retained, blank=0',
        loss='mean(phone_CTC_NLL/phone_length) + .3 * mean_axes(mean(attribute_CTC_NLL/phone_length))',
        optimizer='fresh AdamW net LR1e-5, auxiliary LR1e-3, shared 40k schedule, betas(.9,.98) eps1e-8 WD0 clip5; parent RNG/AMP restored',
        baseline_reuse='existing C0 has identical parent/batches/2000 updates/optimizer; require original first 3 update losses/norms and 4 CAL outputs to reproduce',
        labels='foreign actual TAG, native replay G2P as original; no LAB-derived or pseudo-label new targets; original unknown-removal convention inherited',
        feature_limitations=['categorical properties of saved labels, not measured VOT/F0 or independent articulation',
            'no context-sensitive allophone mapping; h and composite labels retain explicit separate classes',
            'vowels retain a generic vowel class in each auxiliary head; main phone CTC unchanged',
            'not a reproduction of the cited multi-label CTC paper; no claim of new ground truth'],
        sources=['https://www.isca-archive.org/interspeech_2024/shahin24_interspeech.html',
                 'https://arxiv.org/abs/2311.07037', 'https://encykorea.aks.ac.kr/Article/E0078688'],
        evaluation=dict(roles=['cal','inner','outer'], counts=dict(cal=2820,inner=1016,outer=29821),
            calibration='each model observed CAL normals only, one percent budget; freeze on inner/outer',
            gate='inner FPR<=1%, full-error exact recall>C0, normal/error coverage>=C0, protected >=30-error categories drop<=5pp, normal gap FPR increase<=.2pp',
            diagnostics='all phones, insertion/deletion, L1/items plus onset s/ss and j/jj/ch family/direction counts',
            outer='report regardless of inner gate; no candidate/checkpoint/threshold reselection',
            reused_development=True, independent_final=False),
        storage='one FP32 model/RNG/AMP, optimizer absent; existing C0 reused; compressed scored records, no full new raw-logit archive',
        no_automatic_extension=True, production_changed=False, closed_weighting_branch_unchanged=True, bindings=bindings)
    write(root/'protocol.json', p)
    snap = root/'source_snapshot'; snap.mkdir()
    for path in bindings:
        f = Path(path)
        if f.suffix == '.py' and str(f).startswith(str(ROOT)):
            dest = snap/f.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(f, dest)
    rows, labels = read_split(str(manifest), 'train')
    counts = {axis: collections.Counter() for axis in spec}; invalid=[]
    # Conservative native front-end frame count; runtime also checks actual lengths.
    for i, ((_, n), y) in enumerate(zip(rows, labels)):
        frames = max(0, ((n-400)//160+1)//2)
        for axis, info in spec.items():
            mapped = [info['phone_to_id'][t] for t in y]
            counts[axis].update(mapped)
            if not mapped or minimum_frames(mapped)>frames:
                invalid.append(dict(row=i,axis=axis,frames=frames,needed=minimum_frames(mapped)))
    write(root/'training_support.json', dict(rows=len(rows), axes=counts, invalid=invalid,
        labels_sha256=file_sha(manifest/'train.ltr'), scope='all manifest rows including repeats; static categorical labels'))
    if invalid: raise ValueError('auxiliary target preflight failed; no training started')
    # Preallocate only this experiment's final checkpoint, protecting its space.
    reserve = c0.stat().st_size + 16*1024**2
    if shutil.disk_usage(root).free < reserve+256*1024**2:
        raise RuntimeError('need checkpoint plus 256MiB free; no existing asset removed')
    path = root/'checkpoint_reserved.pt.partial'
    with path.open('xb') as f: os.posix_fallocate(f.fileno(), 0, reserve)
    write(root/'storage_reservation.json', dict(path=str(path),bytes=reserve,free_after=shutil.disk_usage(root).free,
        existing_assets_deleted=0,persistent=True))
    write(root/'live_status.json', dict(status='prepared_not_trained',protocol_sha256=file_sha(root/'protocol.json')))
    print('prepared', flush=True)


def check(root):
    p=bind(root); model,ck,vocab=init_model(p)
    rows,_=read_split(p['manifest'],'train'); batches,_,_,_=replay_prefix(rows,p)
    xs=[read_wav(rows[i][0],20.) for i in next(batches)[:4]]
    checks=[]
    for mode in (False,True):
        model.train(mode);restore_rng(ck['rng'])
        with torch.no_grad(),torch.cuda.amp.autocast(): native,lens=Backbone.forward(model,xs)
        state=rng_state();restore_rng(ck['rng'])
        with torch.no_grad(),torch.cuda.amp.autocast(): new,nlens,aux=model(xs,auxiliary=True)
        after=rng_state()
        assert torch.equal(native,new) and torch.equal(lens,nlens)
        assert torch.equal(state['torch'],after['torch'])
        assert all(torch.equal(a,b) for a,b in zip(state['cuda'],after['cuda']))
        checks.append(dict(training=mode,logit_difference=0,rng_equal=True))
    del model,ck;gc.collect();torch.cuda.empty_cache()
    c0=torch.load(p['baseline_checkpoint'],map_location='cpu')
    model=Backbone('w2vbert',p['source'],max(vocab.values())+1,torch.device('cuda'))
    model.load_state_dict(c0['model'],strict=True);model.eval();del c0
    historical=Path(p['historical_root']);ann=records(historical/'audio_annotation_preflight/cal_annotations.jsonl.gz')[:4]
    oldrecords=records(historical/'evaluation/C0/cal_records.jsonl.gz')[:4]
    rows,_=read_split(p['manifest'],'cal')
    for r,(path,n),old in zip(ann,rows,oldrecords):
        x,h=read_audio(path);assert len(x)==n
        lp=forward(model,x)
        assert np.array_equal(lp,np.load(historical/'evaluation/C0/cal_log_probs'/(r['uid']+'.npz'))['lp'])
        value=score_record(r,lp,vocab);value.update(pcm_sha256=h,frames=len(lp));assert value==old
        changed=copy.deepcopy(r)
        for s in changed['sites']:s['actual']='ㄱ'
        v2=score_record(changed,lp,vocab)
        assert v2['predicted']==value['predicted']
        assert [s['score'] for s in v2['sites']]==[s['score'] for s in value['sites']]
    write(root/'identity_check.json',dict(status='passed',checks=checks,historical_C0_CAL_reproduced=4,TAG_independent_inference=True))


def train(root, arm, smoke=False):
    p=bind(root);assert read(root/'identity_check.json')['status']=='passed'
    if not smoke: assert arm=='phonological' and read(root/'runtime_check.json')['status']=='passed'
    out=root/('smoke' if smoke else 'training')/arm;out.mkdir(parents=True,exist_ok=False)
    rows,labels=read_split(p['manifest'],'train');batches,order,draws,samples=replay_prefix(rows,p)
    model,ck,vocab=init_model(p); active=arm=='phonological';model.added.requires_grad_(active)
    groups=[dict(params=model.net.parameters(),lr=p['lr'],lr_factor=1.)]
    if active:groups.append(dict(params=model.added.parameters(),lr=p['lr']*100,lr_factor=100.))
    opt=torch.optim.AdamW(groups,betas=(.9,.98),eps=1e-8,weight_decay=0.)
    scaler=torch.cuda.amp.GradScaler();scaler.load_state_dict(ck['scaler']);restore_rng(ck['rng']);del ck
    initial={n:v.detach().cpu().clone() for n,v in model.added.named_parameters()}
    base_name,base=next((n,v) for n,v in model.encoder().named_parameters() if v.ndim>=2)
    before=base.detach().clone() if smoke else None
    stop=p['start_step']+p['smoke_updates'] if smoke else p['stop_step'];started=time.time();model.train()
    write(out/'execution.json',dict(arm=arm,smoke=smoke,start_step=p['start_step'],stop_step=stop,
        parent_sha256=p['bindings'][p['parent_checkpoint']],protocol_sha256=file_sha(root/'protocol.json'),
        added_parameters=sum(v.numel() for v in model.added.parameters()),optimizer_saved=False))
    with (out/'updates.jsonl').open('x') as log:
        for step in range(p['start_step']+1,stop+1):
            opt.zero_grad(set_to_none=True);losses=[];auxes=collections.defaultdict(list)
            for _ in range(p['accum']):
                ids=next(batches);order.update((' '.join(map(str,ids))+'\n').encode())
                xs=[read_wav(rows[i][0],20.) for i in ids]
                if any(len(x)!=rows[i][1] for x,i in zip(xs,ids)):raise ValueError('changed or truncated training audio')
                ys=[[vocab[t] for t in labels[i]] for i in ids]
                draws+=len(ids);samples+=sum(map(len,xs))
                with torch.cuda.amp.autocast():
                    if active: logits,lengths,outputs=model(xs,auxiliary=True)
                    else: logits,lengths=model(xs)
                    main=weighted_ctc_loss(logits,lengths,ys,np.ones(len(ids)))
                    total=main
                    if active:
                        aux,parts=auxiliary_loss(outputs,lengths,feature_targets(ys,vocab,model.spec))
                        total=main+p['auxiliary_weight']*aux
                        for axis,value in parts.items():auxes[axis].append(float(value.detach()))
                scaler.scale(total/p['accum']).backward();losses.append(float(main.detach()))
            lr=tri_stage(step-1,p['schedule_steps'],p['lr'])
            for group in opt.param_groups:group['lr']=lr*group['lr_factor']
            scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not torch.isfinite(norm):raise ValueError('nonfinite gradient; no silent skipped update')
            scaler.step(opt);scaler.update()
            row=dict(step=step,loss=float(np.mean(losses)),auxiliary_losses={k:float(np.mean(v)) for k,v in auxes.items()},
                lr=lr,gradient_norm=float(norm),row_draws=draws,audio_samples=samples,
                batch_order_sha256=order.hexdigest(),elapsed_seconds=time.time()-started)
            log.write(json.dumps(row)+'\n');log.flush();write(out/'live_status.json',row)
            if smoke or step%100==0:print(json.dumps(row),flush=True)
    receipt=dict(status='smoke_complete' if smoke else 'trained_not_evaluated',arm=arm,updates=stop-p['start_step'],
        final_exposure=row,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        parent_sha256=p['bindings'][p['parent_checkpoint']],protocol_sha256=file_sha(root/'protocol.json'))
    if smoke:
        assert not torch.equal(base,before)
        changes={n:float((v.detach().cpu()-initial[n]).abs().max()) for n,v in model.added.named_parameters()}
        assert all(v>0 for v in changes.values()) if active else all(v==0 for v in changes.values())
        receipt.update(base_matrix_changed=base_name,auxiliary_parameter_changes=changes,smoke_discarded=True)
    else:
        assert {k:row[k] for k in p['expected_final_exposure']}==p['expected_final_exposure']
        model.eval();probe_paths=[rows[i][0] for i in ids[:4]]
        probes=[forward(model,read_wav(path,20.)) for path in probe_paths]
        payload=dict(arm=arm,step=stop,model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            vocab=vocab,rng=rng_state(),scaler=scaler.state_dict(),optimizer=None,exposure=row,
            protocol_sha256=file_sha(root/'protocol.json'),reproduction_paths=probe_paths,reproduction_log_probs=probes)
        reserved=root/'checkpoint_reserved.pt.partial';reservation_size=reserved.stat().st_size
        with reserved.open('r+b') as f:
            torch.save(payload,f);length=f.tell()
            if length>reservation_size:raise RuntimeError('checkpoint exceeded reservation')
            f.truncate(length);f.flush();os.fsync(f.fileno())
        dest=out/'step_4000.pt';reserved.rename(dest);receipt['checkpoint_sha256']=file_sha(dest)
    write(out/'finished.json',receipt)


def close_checks(root):
    p=bind(root);hist=Path(p['historical_root'])
    baseline=[json.loads(x) for x in (root/'smoke/C0/updates.jsonl').read_text().splitlines()]
    prior=[json.loads(x) for x in (hist/'training/C0/updates.jsonl').read_text().splitlines()][:p['smoke_updates']]
    cand=[json.loads(x) for x in (root/'smoke/phonological/updates.jsonl').read_text().splitlines()]
    for a,b,c in zip(baseline,prior,cand):
        for k in ('step','loss','lr','gradient_norm','row_draws','audio_samples','batch_order_sha256'):assert a[k]==b[k],(k,a[k],b[k])
        for k in p['expected_final_exposure']:assert a[k]==c[k]
    assert len(baseline)==len(cand)==p['smoke_updates']
    assert baseline[0]['loss']==cand[0]['loss']
    for arm in ('C0','phonological'):assert read(root/'smoke'/arm/'finished.json')['status']=='smoke_complete'
    write(root/'runtime_check.json',dict(status='passed',historical_first_updates_exactly_reproduced=len(baseline),
        first_main_loss_equal=True,matched_batch_exposure=True,identity=read(root/'identity_check.json'),
        smoke_updates_total=2*p['smoke_updates'],main_new_updates=0))


def evaluate(root):
    p=bind(root);hist=Path(p['historical_root']);out=root/'evaluation/phonological';out.mkdir(parents=True,exist_ok=False)
    ckpath=root/'training/phonological/step_4000.pt';done=read(ckpath.parent/'finished.json')
    assert file_sha(ckpath)==done['checkpoint_sha256']
    ck=torch.load(ckpath,map_location='cpu');assert ck['protocol_sha256']==file_sha(root/'protocol.json')
    vocab=ck['vocab'];model=FeatureModel(p['source'],vocab,torch.device('cuda'))
    model.load_state_dict(ck['model'],strict=True);model.eval()
    for path,old in zip(ck['reproduction_paths'],ck['reproduction_log_probs']):
        assert np.array_equal(forward(model,read_wav(path,20.)),old)
    del ck;audio={}
    with gzip.open(str(hist/'audio_annotation_preflight/audio_bindings.jsonl.gz'),'rt') as f:
        for line in f:
            x=json.loads(line)
            if any(role!='train' for role in x['roles']):audio[x['path']]=x
    reports={};started=time.time()
    for role in p['evaluation']['roles']:
        ann=records(hist/'audio_annotation_preflight'/(role+'_annotations.jsonl.gz'));rows,_=read_split(p['manifest'],role)
        assert len(rows)==len(ann)==p['evaluation']['counts'][role]
        result=[];raw_hash=hashlib.sha256()
        with gzip.open(str(out/(role+'_records.jsonl.gz')),'xt') as dest:
            for i,(r,(path,n)) in enumerate(zip(ann,rows)):
                assert r['row']==i and r['uid']==Path(path).stem
                x,h=read_audio(path);assert len(x)==n==audio[path]['samples'] and h==audio[path]['pcm_sha256']
                lp=forward(model,x);raw_hash.update(r['uid'].encode());raw_hash.update(lp.tobytes())
                value=score_record(r,lp,vocab);value.update(pcm_sha256=h,frames=len(lp))
                dest.write(json.dumps(value,ensure_ascii=False)+'\n');result.append(value)
                if (i+1)%500==0:
                    status=dict(role=role,completed=i+1,total=len(rows),elapsed_seconds=time.time()-started)
                    write(out/'live_status.json',status);print(json.dumps(status),flush=True)
        if role=='cal':
            calibration=calibrate(result);threshold=calibration['threshold'];write(out/'calibration.json',calibration)
        reports[role]=aggregate(result,threshold,vocab);write(out/(role+'_report.json'),reports[role])
        write(out/(role+'_raw_log_probs_digest.json'),dict(sha256=raw_hash.hexdigest(),raw_arrays_not_archived=True))
    write(out/'finished.json',dict(status='fixed_arm_evaluated',vocab=vocab,checkpoint_sha256=done['checkpoint_sha256'],
        calibration=calibration,recordings=sum(v['recordings'] for v in reports.values()),elapsed_seconds=time.time()-started,
        checkpoint_reproduction_exact=4,output_hashes={str(f):file_sha(f) for f in out.glob('*.json*')}))


def family_counts(rs, threshold):
    groups=collections.defaultdict(collections.Counter)
    for r in rs:
        ref=[s['expected'] for s in r['sites']]
        for i,s in enumerate(r['sites']):
            if surface_position(ref,i) not in ('onset','onset_fallback'):continue
            for family,phones in [('s_ss',set('ㅅㅆ')),('j_jj_ch',set('ㅈㅉㅊ'))]:
                if s['expected'] not in phones:continue
                groups[family+':all_expected'].update(site_counts([s],threshold))
                if s['status']=='substitution' and s['actual'] in phones:
                    c=site_counts([s],threshold);groups[family+':within_family_error'].update(c)
                    groups[s['expected']+'->'+s['actual']].update(c)
    return {k:dict(v) for k,v in groups.items()}


def compare(root):
    p=bind(root);hist=Path(p['historical_root']);out=root/'comparison';out.mkdir(exist_ok=False)
    folders={'C0':hist/'evaluation/C0','phonological':root/'evaluation/phonological'}
    thresholds={k:read(d/'calibration.json')['threshold'] for k,d in folders.items()}
    roles={};uncertainty={};families={};checks={};support=read(hist/'audio_annotation_preflight/annotation_support.json')
    for role in p['evaluation']['roles']:
        outputs={k:records(d/(role+'_records.jsonl.gz')) for k,d in folders.items()};rebuilt={};families[role]={}
        for k,rs in outputs.items():
            rebuilt[k]=aggregate(rs,thresholds[k],read(folders[k]/'finished.json')['vocab'])
            assert rebuilt[k]==read(folders[k]/(role+'_report.json'))
            # Independent scalar implementation, not the aggregation helper.
            sc,gc_=recount(folders[k]/(role+'_records.jsonl.gz'),thresholds[k])
            for actual,kind in ((sc,'sites'),(gc_,'gaps')):
                expected=rebuilt[k][kind]['all']['counts']
                for field in ('normal','error','observed_normal','observed_error','fp','exact'):
                    assert actual[field]==expected.get(field,0)
            c=rebuilt[k]['sites']['all']['counts'];g=rebuilt[k]['gaps']['all']['counts'];s=support[role]
            assert c['normal']==s['normal'] and c['error']==s['substitution']+s['deletion']
            assert g['normal']==s['normal_gap'] and g['error']==s['insertion_gap']
            families[role][k]=family_counts(rs,thresholds[k]);checks[role+'/'+k]='independent recount and original denominator preserved'
        assert len(outputs['C0'])==len(outputs['phonological'])
        for a,b in zip(outputs['C0'],outputs['phonological']):assert (a['uid'],a['pcm_sha256'])==(b['uid'],b['pcm_sha256'])
        roles[role]=rebuilt
        if role!='cal':uncertainty[role]={axis:paired(outputs['C0'],outputs['phonological'],thresholds['C0'],thresholds['phonological'],axis) for axis in ('speaker','item')}
    gate=pilot_gate(roles['inner']['C0'],roles['inner']['phonological']);passed=gate.pop('continue_folds')
    receipt=read(root/'training/phonological/finished.json')
    assert receipt['updates']==2000
    assert {k:receipt['final_exposure'][k] for k in p['expected_final_exposure']}==p['expected_final_exposure']
    report=dict(status='pilot_compared',pilot_passed=passed,inner_gate=gate,roles=roles,paired=uncertainty,
        onset_families=families,thresholds=thresholds,verification=checks,main_new_updates=2000,
        historical_C0_updates=2000,smoke_updates=6,independent_final=False,production_changed=False,no_automatic_extension=True)
    write(out/'report.json',report)
    lines=['# 음운 특징 보조 학습 비교 결과','', '**사전 기준 {}.** 운영 변경 없음.'.format('통과' if passed else '미달'),'',
        '| 평가 | 모델 | 관측 정상 헛지적 | 전체 확정 오류의 정확 진단 |','|---|---|---:|---:|']
    for role,label in [('inner','내부 선택'),('outer','바깥 평가')]:
        for arm,name in [('C0','기존 CTC'),('phonological','음운 특징 추가')]:
            c=roles[role][arm]['sites']['all']['counts']
            lines.append('| {} | {} | {} | {} |'.format(label,name,ratio(c,'fp','observed_normal'),ratio(c,'exact','error')))
    lines+=['','| 초성 대립 (바깥 평가) | 기존 CTC 정확 진단 | 음운 특징 추가 정확 진단 |','|---|---:|---:|']
    for key,name in [('s_ss:within_family_error','ㅅ↔ㅆ'),('j_jj_ch:within_family_error','ㅈ↔ㅉ↔ㅊ')]:
        lines.append('| {} | {} | {} |'.format(name,ratio(families['outer']['C0'].get(key,{}),'exact','error'),ratio(families['outer']['phonological'].get(key,{}),'exact','error')))
    lines+=['','기존 fold0 공통 부모에서 각각 2,000회 추가 학습. 기존 C0는 hash·초기 3회 갱신·실제 CAL 출력 재현 후 재사용했다.',
        '문턱은 모델별 CAL 정상에서만 결정했다. 기존 자료를 다시 사용한 개발 비교이며 새 독립 검증이나 운영 M3 성적이 아니다.',
        '전체 음소·삭제·삽입·Coverage·고정 분모 범위와 paired 구간은 comparison/report.json에 보존했다.']
    (root/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    write(out/'finished.json',dict(status='comparison_complete',pilot_passed=passed,report_sha256=file_sha(out/'report.json')))
    print(json.dumps(dict(pilot_passed=passed,inner_gate=gate)),flush=True)


def pipeline(root):
    p=bind(root);pipe=root/'pipeline';pipe.mkdir(exist_ok=False);started=time.time()
    steps=[('identity','check',None),('smoke_C0','smoke','C0'),('smoke_phonological','smoke','phonological'),
           ('runtime_check','close_checks',None),('train','train','phonological'),('evaluation','evaluate',None),('comparison','compare',None)]
    commands=[]
    for name,mode,arm in steps:
        cmd=[sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--mode',mode]
        if arm:cmd+=['--arm',arm]
        commands.append((name,cmd))
    write(pipe/'launch.json',dict(commands=commands,protocol_sha256=file_sha(root/'protocol.json')))
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
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--historical',type=Path);ap.add_argument('--arm',choices=['C0','phonological'])
    ap.add_argument('--mode',choices=['prepare','check','smoke','close_checks','train','evaluate','compare','run'],required=True)
    a=ap.parse_args();root=a.root.resolve()
    if a.mode=='prepare':prepare(root,a.historical.resolve())
    elif a.mode=='check':check(root)
    elif a.mode=='smoke':train(root,a.arm,True)
    elif a.mode=='close_checks':close_checks(root)
    elif a.mode=='train':train(root,a.arm)
    elif a.mode=='evaluate':evaluate(root)
    elif a.mode=='compare':compare(root)
    elif a.mode=='run':pipeline(root)


if __name__=='__main__':main()
