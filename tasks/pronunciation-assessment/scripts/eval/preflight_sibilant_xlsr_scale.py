"""Functional native-audio/full-forward and dense-tail-gradient checks for XLS-R."""
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
import json
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from train_sibilant_hubert_adaptation import RawDataset
from mdd.sibilant_xlsr_scale import load_original, capture_prefix, tail_hidden
from mdd.sibilant_hubert_adaptation import tensor_digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',required=True)
    ap.add_argument('--out',required=True)
    a = ap.parse_args()
    p,out = json.loads(Path(a.protocol).read_text()),Path(a.out)
    for name,digest in p['sha256'].items():
        if service.sha(name) != digest:
            raise ValueError('frozen XLS-R preflight input changed: '+name)
    features = service.read_jsonl(p['features'])
    data = np.load(p['native_data'])
    source = RawDataset(p['train_cache'],features,data['relative_vowel'],data['lengths'])
    device = torch.device(p.get('device','cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=False)
    (out/'protocol.json').write_text(Path(a.protocol).read_text())
    reports = {}
    for size,spec in p['models'].items():
        tick = time.monotonic()
        model,meta = load_original(spec['path'],device)
        cases,values = [],[]
        for i in range(16):
            wave = source.source(features[i]['uid'])
            before = wave.copy()
            with torch.no_grad():
                prefix,n,skip,h,info = capture_prefix(model,wave,features[i])
            np.testing.assert_array_equal(before,wave)
            cases.append(dict(index=i,source_length=n,prefix_length=len(prefix),skip=skip,
                selected_frames=len(h),**info))
            if i == 0:
                values.append((prefix.clone(),n))
        # Keep completed acoustic parity checks even if the later backward fails.
        (out/(size+'_forward.json')).write_text(json.dumps(
            {'metadata':meta,'cases':cases,'optimizer_updates':0},indent=2)+'\n')
        # A genuine gradient through original dense acoustic transformer layers;
        # no optimizer update, no phonetic performance claim from this check.
        tail = model.encoder.layers[-2:]
        prior = tensor_digest(tail.state_dict())
        tail.requires_grad_(True)
        prefix,n = values[0]
        output = tail_hidden(tail,model.encoder.layer_norm,prefix[None].float(),torch.tensor([n],device=device))
        objective = output[0,:n,::17].square().mean()
        objective.backward()
        gradient = [float(sum(z.grad.square().sum() for z in layer.parameters() if z.grad is not None).sqrt()) for layer in tail]
        forbidden = [name for name,value in model.named_parameters() if value.grad is not None
            and not name.startswith(tuple('encoder.layers.'+str(j)+'.' for j in (meta['depth']-2,meta['depth']-1)))]
        if forbidden or not all(np.isfinite(z) and z > 0 for z in gradient) or tensor_digest(tail.state_dict()) != prior:
            raise ValueError('dense acoustic tail gradient/isolation failed')
        report = dict(metadata=meta,cases=cases,gradient_norms=gradient,forbidden_gradients=forbidden,
            source_unchanged=True,optimizer_updates=0,seconds=time.monotonic()-tick,
            GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None)
        reports[size] = report
        (out/(size+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(size,json.dumps({k:v for k,v in report.items() if k != 'cases'},ensure_ascii=False),flush=True)
        del model,tail,prefix,output,objective,values
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    result = {'scope':p['scope'],'models':reports,'passed':True,'classification_or_error_performance_evaluated':False,
        'new_supervised_optimizer_updates':0,'deployment':False,'goal_complete':False}
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (out/'finished.json').write_text(json.dumps({'report_sha256':service.sha(out/'report.json')}))


if __name__ == '__main__':
    main()
