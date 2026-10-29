"""Audit frozen scale comparison and restore four arms from original PCM/checkpoints."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),
               str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import gc
import hashlib
import json
import numpy as np
import torch
import eval_sibilant_directional as service
from train_sibilant_frozen_probe import write
from train_sibilant_hubert_adaptation import RawDataset,novel_proxy,retained_subgroups
from train_sibilant_xlsr_scale import PrefixDataset,score
from mdd.sibilant_xlsr_scale import load_original,capture_prefix,restore_scale
from mdd.sibilant_hubert_adaptation import tensor_digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',required=True)
    ap.add_argument('--fit-root',required=True)
    ap.add_argument('--out',required=True)
    a = ap.parse_args()
    p = json.loads(Path(a.protocol).read_text())
    fit,out = Path(a.fit_root),Path(a.out)
    report = json.loads((fit/'report.json').read_text())
    receipt = json.loads((fit/'finished.json').read_text())
    if receipt['report_sha256'] != service.sha(fit/'report.json'):
        raise ValueError('completed comparison report changed/missing')
    if json.loads((fit/'protocol.json').read_text()) != p:
        raise ValueError('executed protocol differs')
    for path,digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen source/input changed: '+path)
    features = service.read_jsonl(p['features'])
    labels = service.read_jsonl(Path(p['train_cache'])/'data.labels.jsonl')
    indexed = {z['feature_index']:z for z in labels if z['feature_index'] is not None}
    native = np.load(p['data']);lengths,relative = native['lengths'],native['relative_vowel']
    y = np.load(p['targets'])['y']
    raw = RawDataset(p['train_cache'],features,relative,lengths)
    device = torch.device(p['device'])
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=False)
    audit = {'arms':{},'original_PCM_prefix_checks':{},'service_improvement_proven':False,
             'deployment':False,'goal_complete':False}
    for size,spec in p['models'].items():
        prefix_path = fit/(size+'_prefix')
        cache_report = json.loads((prefix_path/'report.json').read_text())
        for name,digest in cache_report['sha256'].items():
            if service.sha(prefix_path/name) != digest:
                raise ValueError('complete native prefix/projection changed')
        encoder,meta = load_original(spec['path'],device)
        data = PrefixDataset(prefix_path,relative,lengths)
        # Rebuild the first original-source 64 prefixes, not a read of cached
        # hidden features or a replay of live trained model parameters.
        reconstructed = np.zeros((64,34,meta['width']),np.float16)
        actual_lengths,actual_skips = [],[]
        with torch.no_grad():
            for i in range(64):
                x,n,skip,_,_ = capture_prefix(encoder,raw.source(features[i]['uid']),features[i])
                reconstructed[i,:len(x)] = x.cpu().numpy()
                actual_lengths.append(n);actual_skips.append(skip)
        np.testing.assert_array_equal(reconstructed,data.prefix[:64])
        np.testing.assert_array_equal(actual_lengths,data.source_lengths[:64])
        np.testing.assert_array_equal(actual_skips,data.skips[:64])
        audit['original_PCM_prefix_checks'][size] = {'original_sources':64,'cached_prefix_bitexact':True,
            'original_tail_sha256':tensor_digest(encoder.encoder.layers[-2:].state_dict())}
        for mode in ('frozen','tail2'):
            arm = size+'_'+mode
            result = report['arms'][arm]
            artifact_path = fit/(arm+'.pt')
            if service.sha(artifact_path) != result['artifact_sha256']:
                raise ValueError('trained coupled artifact changed')
            if service.sha(fit/(arm+'.scores.npz')) != result['score_sha256']:
                raise ValueError('saved complete TRAIN scores changed')
            artifact = torch.load(artifact_path,map_location='cpu')
            binding = dict(cache_report['binding'],original_tail_sha256=audit['original_PCM_prefix_checks'][size]['original_tail_sha256'])
            restored = restore_scale(artifact,binding,encoder.encoder.layers[-2:],encoder.encoder.layer_norm,device)
            scores = np.load(fit/(arm+'.scores.npz'))
            ps,av = scores['probabilities'],scores['available']
            expected = novel_proxy(ps,av,y,features,indexed)
            if expected != result['TRAIN_novel_proxy']:
                raise ValueError('complete TRAIN diagnostic does not reproduce')
            subgroups = retained_subgroups(ps,av,y,features,labels,expected)
            if subgroups != json.loads((fit/(arm+'.subgroups.json')).read_text()):
                raise ValueError('retained normal/unknown/no-input subgroups differ')
            # The comparison model receives the reconstructed PCM-derived
            # prefixes here, after serialized projection/tail/head restoration.
            args = (torch.as_tensor(reconstructed,device=device).float(),
                    torch.as_tensor(actual_lengths,device=device),torch.as_tensor(actual_skips,device=device),
                    torch.as_tensor(lengths[:64],device=device),torch.as_tensor(relative[:64],device=device))
            with torch.no_grad():
                logits,role = restored(*args)
            from mdd.sibilant_vowel_presence import observed_vowel_support
            mask = torch.arange(32,device=device)[None] < args[3][:,None]
            np.testing.assert_array_equal(logits.softmax(-1).cpu().numpy(),ps[:64])
            np.testing.assert_array_equal(observed_vowel_support(role,mask).cpu().numpy(),av[:64])
            if (result['actual_optimizer_updates'] != 300 or not result['prefix_unchanged']
                    or result['tail_changed'] != (mode == 'tail2') or result['skipped_draws'] != 76):
                raise ValueError('fixed replay/acoustic tuning evidence differs')
            audit['arms'][arm] = {'whole_TRAIN_metrics_reproduced':True,'raw_first64_predictions_bitexact':True,
                'retained_sites':len(labels),'unknown':int((y<0).sum()),'no_input':sum(z['feature_index'] is None for z in labels),
                'actual_optimizer_updates':300,'novel_TP':expected['novel_TP'],'wrong_actual':expected['wrong_actual']}
            write(out/'progress.json',audit)
            print(arm,json.dumps(audit['arms'][arm]),flush=True)
            del restored,artifact
        del encoder,data,reconstructed
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    candidate = report['arms']['2b_tail2']['TRAIN_novel_proxy']
    control_best = max([564]+[report['arms'][k]['TRAIN_novel_proxy']['novel_TP']
                             for k in ('300m_frozen','300m_tail2','2b_frozen')])
    gate = bool(candidate['novel_TP'] > control_best and candidate['wrong_actual'] <= 3
                and candidate['S_to_SS']['novel_TP'] >= 240 and candidate['SS_to_S']['novel_TP'] >= 324)
    if gate != report['proceed_CAL']:
        raise ValueError('frozen preselected candidate gate differs')
    audit.update(passed=True,proceed_CAL=gate,fit_report_sha256=receipt['report_sha256'],
                 audit_source_sha256=service.sha(Path(__file__)))
    write(out/'report.json',audit)
    write(out/'finished.json',{'report_sha256':service.sha(out/'report.json')})


if __name__ == '__main__':
    main()
