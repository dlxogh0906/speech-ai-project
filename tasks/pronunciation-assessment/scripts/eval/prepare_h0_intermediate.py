"""Freeze P3 inputs, exact continuation exposure and the decision rules before training."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT: raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),
              str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import collections
import hashlib
import json
import shutil
import torch
from train_h0_backbone_control import file_sha, stream
from train_w2vbert_ctc import read_split, load_vocab


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parent-protocol',required=True); ap.add_argument('--parent-checkpoint',required=True)
    ap.add_argument('--historical-experiment',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    parent=json.loads(Path(a.parent_protocol).read_text()); audit=json.loads(Path(parent['input_audit']).read_text())
    ck=torch.load(a.parent_checkpoint,map_location='cpu')
    assert ck['optimizer'] is None and ck['stage']=='stage2' and ck['backbone']=='w2vbert' and ck['step']==2000
    assert ck['protocol_sha256']==file_sha(a.parent_protocol)
    manifest=Path(parent['stage2']['manifest']); rows,labels=read_split(str(manifest),'train')
    vocab=load_vocab(str(manifest/'dict.ltr.txt')); assert ck['vocab']==vocab
    sources=(manifest/'train.source').read_text().splitlines(); assert len(sources)==len(rows)
    all_counts=collections.defaultdict(collections.Counter)
    for kind,lab in zip(sources,labels): all_counts[kind].update(lab)
    draws=samples=0; h=hashlib.sha256(); snapshots={}; continuation=collections.defaultdict(collections.Counter)
    unique=set(); draw_kinds=collections.Counter(); unique_by_kind=collections.defaultdict(set)
    for batch_number,ids in enumerate(stream(rows,parent['batch_samples'],parent['seed']),1):
        h.update((' '.join(map(str,ids))+'\n').encode())
        draws+=len(ids); samples+=sum(rows[i][1] for i in ids)
        if batch_number>8000:
            for i in ids:
                unique.add(i); unique_by_kind[sources[i]].add(rows[i][0]); draw_kinds[sources[i]]+=1
                continuation[sources[i]].update(labels[i])
        if batch_number in (8000,12000,16000):
            snapshots[str(batch_number//4)]=dict(row_draws=draws,audio_samples=samples,batch_order_sha256=h.hexdigest())
        if batch_number==16000: break
    assert snapshots['2000']=={k:ck['exposure'][k] for k in snapshots['2000']}
    historical=json.loads((Path(a.historical_experiment)/'w2vbert_stage2/finished.json').read_text())
    assert snapshots['4000']=={k:historical['final_exposure'][k] for k in snapshots['4000']}
    bindings={str(Path(a.parent_protocol)):file_sha(a.parent_protocol),
              str(Path(a.parent_checkpoint)):file_sha(a.parent_checkpoint),
              parent['input_audit']:file_sha(parent['input_audit'])}
    for name in ['train.tsv','train.ltr','train.source','dict.ltr.txt']+[s+'.'+e
            for s in ('valid','valid_unseen') for e in ('tsv','ltr','canon','meta.jsonl')]:
        path=manifest/name; digest=file_sha(path)
        if str(path) in audit['sha256']: assert digest==audit['sha256'][str(path)]
        bindings[str(path)]=digest
    bindings.update(parent['backbones']['w2vbert']['sha256'])
    code_paths=[ROOT/'tasks/pronunciation-assessment/src/mdd/h0_intermediate.py']
    for folder,names in [('train',['train_h0_intermediate.py','train_h0_backbone_control.py','train_w2vbert_ctc.py',
                                   'run_h0_intermediate.sh']),
                          ('eval',['prepare_h0_intermediate.py','eval_h0_intermediate.py','summarize_h0_intermediate.py',
                                   'eval_h0_backbone_control.py'])]:
        code_paths += [ROOT/'tasks/pronunciation-assessment/scripts'/folder/n for n in names]
    code_paths += [ROOT/'tasks/pronunciation-assessment/src/mdd/h0_phone_audit.py']
    code_paths += [ROOT/'tasks/pronunciation-assessment/tests'/name for name in
                   ('test_h0_intermediate.py','verify_h0_intermediate_runtime.py','verify_h0_intermediate_scoring.py')]
    code_paths += [ROOT/'tasks/pronunciation-assessment/scripts/train/watch_h0_intermediate.py']
    snapshot=out/'source_snapshot'; snapshot.mkdir()
    for path in code_paths:
        bindings[str(path)]=file_sha(path); shutil.copyfile(path,snapshot/path.name)
    report=dict(row_count=len(rows),source_rows=dict(collections.Counter(sources)),
        actual_TAG_phone_tokens_by_source=dict(all_counts),continuation_phone_tokens_by_source=dict(continuation),
        continuation_draws_by_source=dict(draw_kinds),continuation_unique_row_indices=len(unique),
        continuation_unique_audio_by_source={k:len(v) for k,v in unique_by_kind.items()},
        snapshots=snapshots,parent_optimizer_present=False,parent_scaler=ck['scaler'],
        parent_rng_keys=list(ck['rng']),label_quality='existing TAG labels; no listening or relabeling',
        unseen_prompt_count=9,independent_new_sentence_confirmation=False,
        full_vocabulary=list(vocab),missing_train_tokens=[t for t in vocab if not any(c[t] for c in all_counts.values())])
    (out/'preflight_data.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    bindings[str(out/'preflight_data.json')]=file_sha(out/'preflight_data.json')
    p=dict(version='h0_p3_intermediate_v1',parent_protocol=a.parent_protocol,parent_checkpoint=a.parent_checkpoint,
        historical_experiment=a.historical_experiment,input_audit=parent['input_audit'],manifest=str(manifest),
        source=parent['backbones']['w2vbert']['path'],seed=42,accum=4,batch_samples=1600000,
        start_step=2000,stop_step=4000,save_steps=[3000,4000],schedule_steps=40000,lr=1e-5,
        aux_lambda=.3,new_parameter_lr_factor=100.,parent_exposure=snapshots['2000'],
        expected_final_exposure=snapshots['4000'],bindings=bindings,
        conditions={'B0':'final CTC; matched AdamW reset',
                    'B1':'final CTC + 0.3 * Layer12 CTC',
                    'B2':'final CTC on h24 + tanh(g) * LayerNorm(h12)',
                    'B3':'B2 + 0.3 * Layer12 CTC'},
        initialization={'gate':0.,'aux_head':'copy parent final CTC head',
                        'LayerNorm':'weight=1,bias=0, shared between auxiliary and fusion',
                        'final_prediction':'identical to parent for all four arms at initialization'},
        optimization={'AdamW':'reset for all arms; beta(.9,.98),eps1e-8,wd0,clip5',
            'rng_amp':'restore parent state after constructing all new modules',
            'scheduler':'analytic original 40k tri-stage at global update 2001..4000',
            'masking':False,'layerdrop':.1,'new_parameters':'100x base LR, same schedule; no search',
            'precision':'original FP16 AMP with FP32 master parameters/CTC',
            'data':'original full H0 manifest; exact continuation batches; no synthetic or phone-specific sampling'},
        storage={'snapshots':'FP32 model + RNG + AMP, model-only Adam retention to fit eight snapshots',
                 'estimated_model_bytes':8*sum(v.numel()*v.element_size() for v in ck['model'].values()),
                 'resumption':'new snapshots are not exact Adam-resumable; interruption requires replay from parent',
                 'existing_assets_deleted':False},
        evaluation={'primary_candidate':'B3 at 4000 against B0 at 4000; B1/B2 mechanism ablations',
            'checkpoint_3000':'trajectory diagnostic; not post-hoc primary replacement',
            'calibration':'same seen CAL normal sites; global 1% budget; freeze before selection',
            'development_gate':['seen and unseen measured calibrated normal FPR <= .01',
                'whole unseen exact diagnosis recall greater than B0',
                'unseen greedy FPR below/equal B0 at same seen FPR within observed adjacent-step range',
                'no >=30-error phone loses more than 5 percentage points exact recall versus B0',
                'no >=30-error deletion/insertion category loses more than 5 points exact recall',
                'false insertion gap rate increase <= 0.2 percentage points'],
            'guardrails':'exploratory tolerances, not product requirements or simultaneous significance guarantees',
            'whole_scope':'46 dictionary phones + deletion + full gap insertion; every support count reported',
            'uncertainty':'paired speaker and prompt bootstrap separately; fixed CAL thresholds',
            'suri':'diagnostic only, never selection or training weighting',
            'reference_labels':'unchanged; P2 allowed-pronunciation sensitivity kept separate',
            'acceptance':'development gate cannot authorize deployment; new independent text/speakers and full API validation remain required'},
        interpretation={'suri_only':'local recovery with other-phone regressions is not success',
            'fusion_only':'investigate information transfer rather than auxiliary supervision',
            'recognition_without_low_FPR':'calibration transfer/normal-tail separation remains unresolved',
            'all_fail':'reconsider data/context/label supervision; do not add an arbitrary series of auxiliary losses'},
        references=['https://arxiv.org/abs/2102.03216',
            'https://www.isca-archive.org/interspeech_2021/nozaki21_interspeech.html',
            'https://www.isca-archive.org/interspeech_2024/hojo24_interspeech.html'],
        novelty='simple baselines; signed residual fusion is not a reproduction of cited inter-layer attention')
    (out/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(protocol=str(out/'protocol.json'),snapshots=snapshots,
                          continuation_draws=dict(draw_kinds),free_bytes=shutil.disk_usage(out).free)),flush=True)


if __name__=='__main__': main()
