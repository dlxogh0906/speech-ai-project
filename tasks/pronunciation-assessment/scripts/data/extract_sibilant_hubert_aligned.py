"""Native fp32 HuBERT, aligned short input and fixed global observed-V grid."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src'),
               str(ROOT / 'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from transformers import HubertConfig, HubertModel, Wav2Vec2FeatureExtractor
from safetensors.torch import load_file
import eval_sibilant_directional as service


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen source changed: ' + path)
    cache = Path(p['cache'])
    if json.loads((cache / 'report.json').read_text())['partition'] != p['partition']:
        raise ValueError('source partition differs')
    features = service.read_jsonl(cache / 'data.features.jsonl')
    relative_source = np.load(cache / 'data.npz')['relative_vowel']
    selected = (np.load(p['source_indices']).tolist() if p.get('source_indices')
                else list(range(len(features))))
    if len(set(selected)) != len(selected) or any(not 0 <= i < len(features) for i in selected):
        raise ValueError('invalid source indices')
    wanted = {features[i]['uid'] for i in selected}
    rows = {}
    with (cache / 'data.responses.jsonl').open() as f:
        for line in f:
            z = json.loads(line)
            if z['row']['uid'] in wanted:
                rows[z['row']['uid']] = z['row']
    manifest = json.loads(Path(p['model_manifest']).read_text())
    folder = Path(manifest['model_dir'])
    for name, digest in manifest['sha256'].items():
        if service.sha(folder / name) != digest:
            raise ValueError('pinned model file changed')
    binding = {
        'backbone_key': 'hubert_ko:' + manifest['revision'] + ':' + manifest['sha256']['model.safetensors']
            + ':h6+h12:nativefp32:storedfp16:aligned_short',
        'feature_schema': 'frozen_hubert_ko_h6_h12_fp16_aligned_short_v1',
        'coordinate_schema': 'hubert_global_conv400_hop320_observedV_v1'}
    if binding != p['binding']:
        raise ValueError('native encoder binding differs')
    config = HubertConfig.from_pretrained(str(folder), local_files_only=True)
    model, info = HubertModel.from_pretrained(None, config=config,
        state_dict=load_file(str(folder / 'model.safetensors')), output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('native model load differs: ' + str(info))
    stride, receptive = 1, 1
    for kernel, step in zip(config.conv_kernel, config.conv_stride):
        receptive += (kernel - 1) * stride
        stride *= step
    if (receptive, stride) != (400, 320):
        raise ValueError('global convolution grid differs')
    fe = Wav2Vec2FeatureExtractor.from_pretrained(str(folder), local_files_only=True)
    if fe.sampling_rate != 16000 or fe.do_normalize:
        raise ValueError('native nonnormalizing 16k processor required')
    device = torch.device(p.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval().requires_grad_(False).to(device)
    torch.set_num_threads(2)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'protocol.json').write_text(Path(args.protocol).read_text())
    arrays, relatives, metadata = [], [], []
    last_uid, source = None, None
    tick = time.monotonic()
    with torch.no_grad():
        for source_i in selected:
            f = features[source_i]
            if f['uid'] != last_uid:
                source = service.load_audio(rows[f['uid']])[:320000]
                last_uid = f['uid']
            left, right = f['left'], f['right']
            if hashlib.sha256(source[left:right].tobytes()).hexdigest() != f['clip_sha256']:
                raise ValueError('original PCM differs')
            encoder_left = (left // 320) * 320
            first, last = (left + 319) // 320, (right - 400) // 320
            indices = np.arange(first, last + 1)
            if not len(indices) or first * 320 < left or last * 320 + 400 > right:
                raise ValueError('global frames outside original ROI')
            value = fe(source[encoder_left:right], sampling_rate=16000,
                       return_tensors='pt', padding=False).input_values.to(device)
            hidden = model(value, output_hidden_states=True).hidden_states
            local = torch.cat([hidden[6], hidden[12]], -1)[0, indices - encoder_left // 320]
            encoded = local.float().cpu().numpy().astype(np.float16)
            if not np.isfinite(encoded).all():
                raise ValueError('nonfinite latent features')
            oldn = f['input_frames']
            centers = left + (np.arange(oldn) + .5) * (right - left) / oldn
            anchors = centers / 16000. - relative_source[source_i, :oldn] * .31
            if np.ptp(anchors) > 1e-6:
                raise ValueError('observed V reference differs')
            reference = float(np.median(anchors))
            relative = np.clip(((indices * 320 + 199.5) / 16000. - reference) / .31, -2, 2).astype(np.float32)
            metadata.append(dict(f, feature_index=len(arrays), source_feature_index=source_i,
                source_input_frames=oldn, source_coordinate_schema=f['coordinate_schema'],
                input_frames=len(indices), encoder_left=encoder_left, global_frame_start=first,
                local_frame_start=first - encoder_left // 320,
                frame_first_source_sample=first * 320 + 199.5,
                observed_reference_seconds=reference, **{k: binding[k] for k in ('feature_schema', 'coordinate_schema')}))
            arrays.append(encoded)
            relatives.append(relative)
            if len(arrays) % 250 == 0:
                print(p['partition'], 'features', len(arrays), 'of', len(selected),
                      'seconds', round(time.monotonic() - tick, 1), flush=True)
    lengths = np.array([len(a) for a in arrays])
    width = int(lengths.max())
    x = np.zeros((len(arrays), width, 1536), np.float16)
    relative = np.zeros((len(arrays), width), np.float32)
    for i, (a, v) in enumerate(zip(arrays, relatives)):
        x[i, :len(a)] = a
        relative[i, :len(v)] = v
    np.savez(out / 'data.npz', x=x, relative_vowel=relative, lengths=lengths)
    with (out / 'features.jsonl').open('w') as f:
        for row in metadata:
            service.write_line(f, row)
    report = {'scope': p['scope'], 'partition': p['partition'], 'binding': binding,
              'shape': list(x.shape), 'dtype': str(x.dtype), 'native_inference_dtype': 'float32',
              'all_original_PCM_hashes_match': True, 'all_frames_inside_original_ROI': True,
              'TAG_or_expectedC_encoder_input': False, 'seconds': time.monotonic() - tick,
              'data_sha256': service.sha(out / 'data.npz'),
              'features_sha256': service.sha(out / 'features.jsonl'), 'deployment': False}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / 'finished.json').write_text(json.dumps({'sha256': service.sha(out / 'report.json')}) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
