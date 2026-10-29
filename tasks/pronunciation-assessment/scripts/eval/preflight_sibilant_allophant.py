"""Functional-only frozen public attribute inference on original TRAIN ROIs."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'tasks/pronunciation-assessment/src'),
               str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import importlib.util
import json
import time
import types
import numpy as np
import torch
import eval_sibilant_directional as service
from mdd.sibilant_allophant import load_attributes, normalize_single
from mdd.sibilant_hubert_adaptation import tensor_digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for name, digest in p['sha256'].items():
        if service.sha(name) != digest:
            raise ValueError('frozen attribute preflight input changed: '+name)
    cache = Path(p['train_cache'])
    if json.loads((cache/'report.json').read_text())['partition'] != 'train' or p['indices'] != list(range(16)):
        raise ValueError('fixed first16 TRAIN functional check required')
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_bytes(Path(args.protocol).read_bytes())
    tick = time.monotonic()
    torch.set_num_threads(2)
    features = service.read_jsonl(p['features'])
    selected = [features[i] for i in p['indices']]
    wanted = {f['uid'] for f in selected}
    rows = {}
    for line in (cache/'data.responses.jsonl').open():
        row = json.loads(line)['row']
        if row['uid'] in wanted:
            rows[row['uid']] = {k:row[k] for k in ['uid','wav','frames']}
    if set(rows) != wanted:
        raise ValueError('original source rows missing')
    checkpoint = torch.load(p['checkpoint'],map_location='cpu',weights_only=True)
    config = json.loads(Path(p['backbone_config']).read_text())
    preprocessor = json.loads(Path(p['preprocessor']).read_text())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, metadata = load_attributes(checkpoint,config,preprocessor,device)
    before = tensor_digest(model.state_dict())
    spec = importlib.util.spec_from_file_location('official_reference_py38',p['official_reference'])
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    context = types.SimpleNamespace(_normalize=True,_use_attention_mask=True,_model=model.encoder)
    context._length_functions = [reference.conv_length(k,s,use_padding=False)
        for k,s in zip(config['conv_kernel'],config['conv_stride'])]
    context.downsampled_lengths = types.MethodType(reference.reference_lengths,context)
    reference_heads = {name:reference.HierarchicalClassifier(layer).eval() for name,layer in model.heads.items()}
    sources = {uid:service.load_audio(row)[:320000] for uid,row in rows.items()}
    records, probabilities, hiddens = [], [], []
    for feature in selected:
        source = sources[feature['uid']]
        unchanged = source.copy()
        start = feature['left']//320*320
        raw = torch.as_tensor(source[start:feature['right']],device=device)
        value = raw[None]
        batch = types.SimpleNamespace(audio_features=value,lengths=torch.as_tensor([len(raw)],device=device))
        with torch.no_grad():
            normalized = normalize_single(raw)
            original_normalized = reference.zero_mean_unit_var_norm(value,batch.lengths,reference.mask_sequence(batch.lengths))
            if not torch.equal(normalized,original_normalized):
                raise ValueError('official normalization parity failed')
            logits, hidden = model.observed(source,feature)
            direct_logits, direct_hidden = model.observed(source,feature)
            official_states, official_length = reference.reference_forward(context,batch)
            first,last = (feature['left']+319)//320,(feature['right']-400)//320
            index = torch.arange(first,last+1,device=device)-start//320
            if not torch.equal(hidden,direct_hidden) or not torch.equal(hidden,official_states[-1][index,0]):
                raise ValueError('repeat or original backbone forward parity failed')
            if int(official_length[0]) != len(official_states[-1]):
                raise ValueError('official convolution length disagrees')
            for name,z in logits.items():
                original = reference_heads[name](official_states[-1],official_length,
                    torch.zeros(1,device=device,dtype=torch.long))[index,0]
                if not torch.equal(z,direct_logits[name]) or not torch.equal(z,original):
                    raise ValueError('repeat or original attribute head parity failed: '+name)
            posterior = torch.cat([logits[z['name']].softmax(-1) for z in metadata['attributes']],-1)
            if len(hidden) != feature['input_frames'] or not bool(torch.isfinite(posterior).all()):
                raise ValueError('nonfinite observations or changed coordinates')
        np.testing.assert_array_equal(source,unchanged)
        probabilities.append(posterior.cpu().numpy())
        hiddens.append(hidden.cpu().numpy())
        records.append({'feature_index':feature['feature_index'],'uid':feature['uid'],
            'q_runtime':feature['q_runtime'],'frames':len(hidden),'clip_sha256':feature['clip_sha256'],
            'posterior_width':posterior.shape[1],'official_source_normalization_forward_heads_exact':True,
            'repeated_call_exact':True,'source_pcm_unchanged':True})
        print(json.dumps({'functional_inputs':len(records),'seconds':time.monotonic()-tick}),flush=True)
    if before != tensor_digest(model.state_dict()):
        raise ValueError('frozen public model changed')
    reloaded, _ = load_attributes(torch.load(p['checkpoint'],map_location='cpu',weights_only=True),config,preprocessor,device)
    with torch.no_grad():
        logits, hidden = reloaded.observed(sources[selected[0]['uid']],selected[0])
        restored = torch.cat([logits[z['name']].softmax(-1) for z in metadata['attributes']],-1).cpu().numpy()
    np.testing.assert_array_equal(restored,probabilities[0])
    np.testing.assert_array_equal(hidden.cpu().numpy(),hiddens[0])
    width = max(len(z) for z in probabilities)
    ps = np.zeros((16,width,metadata['posterior_width']),np.float32)
    hs = np.zeros((16,width,1024),np.float32)
    lengths = np.asarray([len(z) for z in probabilities])
    for i,(a,b) in enumerate(zip(probabilities,hiddens)):
        ps[i,:len(a)],hs[i,:len(b)] = a,b
    np.savez(out/'observations.npz',posterior=ps,hidden=hs,lengths=lengths)
    report = {'scope':p['scope'],'functional_inputs':16,'metadata':metadata,'records':records,
        'reference_scope':'Original selected function bodies and hierarchy source, same local HF/torch backend; not official package/decoder/environment parity',
        'all36_attributeheads_andnativecoordinates_checked':True,'checkpoint_and_source_pcm_unchanged':True,
        'exact_reload_first_input':True,'training_updates':0,'performance_or_error_detection_evaluated':False,
        'seconds':time.monotonic()-tick,'deployment':False,'goal_complete':False}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (out/'finished.json').write_text(json.dumps({'report_sha256':service.sha(out/'report.json')}))
    print(json.dumps({k:v for k,v in report.items() if k not in ['records','metadata']},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
