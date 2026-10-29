"""Coupled native HuBERT tail/head artifacts and original-PCM global-grid encoding.

No TAG labels or annotation times enter this module. Callers supply the existing
runtime ROI and observed-vowel reference. Waveforms are encoded individually:
padding them would change the native encoder's group normalization.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from mdd.sibilant_frozen_probe import FrozenProbe, projection_digest
from mdd.sibilant_vowel_presence import CLASSES

READOUT = 'native_hubert_tail_cv_v1'
COORDINATES = 'hubert_global_conv400_hop320_observedV_v1'
PRECISION = 'native_fp32_unpadded_h6_h12_fp16_roundtrip_v1'


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(a.shape).encode())
        digest.update(str(a.dtype).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def binding_from_manifest(manifest):
    return {'repo_id': manifest['repo_id'], 'revision': manifest['revision'],
            'native_files_sha256': dict(manifest['sha256']),
            'encoder_depth': 12, 'tail_depth': 2, 'hidden_layers': [6, 12],
            'coordinate_schema': COORDINATES, 'precision': PRECISION,
            'classes': list(CLASSES)}


def load_native(manifest_path, device):
    # Keep heavy optional imports out of artifact-only callers and tests.
    from transformers import HubertConfig, HubertModel, Wav2Vec2FeatureExtractor
    from safetensors.torch import load_file
    manifest = json.loads(Path(manifest_path).read_text())
    folder = Path(manifest['model_dir'])
    for name, digest in manifest['sha256'].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
            raise ValueError('native model file changed: ' + name)
    config = HubertConfig.from_pretrained(str(folder), local_files_only=True)
    receptive, stride = 1, 1
    for kernel, step in zip(config.conv_kernel, config.conv_stride):
        receptive += (kernel - 1) * stride
        stride *= step
    if (config.num_hidden_layers, config.hidden_size, receptive, stride) != (12, 768, 400, 320):
        raise ValueError('native model architecture/global grid differs')
    encoder, info = HubertModel.from_pretrained(None, config=config,
        state_dict=load_file(str(folder / 'model.safetensors')), output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('native weights do not load exactly: ' + str(info))
    processor = Wav2Vec2FeatureExtractor.from_pretrained(str(folder), local_files_only=True)
    if processor.sampling_rate != 16000 or processor.do_normalize:
        raise ValueError('native 16k nonnormalizing processor required')
    encoder.eval().requires_grad_(False).to(device)
    binding = binding_from_manifest(manifest)
    encoder._sibilant_native_binding = binding
    return encoder, processor, binding


def encode_clip(encoder, processor, source, feature):
    left, right = int(feature['left']), int(feature['right'])
    if not 0 <= left < right <= len(source):
        raise ValueError('runtime ROI outside original waveform')
    if hashlib.sha256(source[left:right].tobytes()).hexdigest() != feature['clip_sha256']:
        raise ValueError('runtime original PCM changed')
    start = left // 320 * 320
    first, last = (left + 319) // 320, (right - 400) // 320
    if last < first:
        raise ValueError('no full receptive field in runtime ROI')
    device = next(encoder.parameters()).device
    value = processor(source[start:right], sampling_rate=16000,
                      return_tensors='pt', padding=False).input_values.to(device)
    hidden = encoder(value, output_hidden_states=True).hidden_states
    indices = torch.arange(first, last + 1, device=device) - start // 320
    # Explicit fp16 storage roundtrip is part of the fixed readout contract,
    # including during gradient propagation. The native encoder stays fp32.
    return torch.cat([hidden[6], hidden[12]], -1)[0, indices].to(torch.float16).float()


def coupled_artifact(encoder, probe, binding, mode):
    if mode not in ('frozen', 'tail2'):
        raise ValueError('unknown encoder update mode')
    projection = {k: getattr(probe, k).detach().cpu().clone()
                  for k in ('mean', 'scale', 'components')}
    tail = {k: v.detach().cpu().clone() for k, v in encoder.encoder.layers[-2:].state_dict().items()}
    head = {k: v.detach().cpu().clone() for k, v in probe.head.state_dict().items()}
    bound = dict(binding, mode=mode, projection_sha256=projection_digest(projection),
                 tail_sha256=tensor_digest(tail), head_sha256=tensor_digest(head))
    return {'readout': READOUT, 'binding': bound, 'projection': projection,
            'tail': tail, 'head': head}


def load_coupled(artifact, expected, encoder, device):
    """Validate all state and native identity before mutating the supplied encoder.

    The encoder must have been loaded with load_native using the manifest that
    produced expected. Head/tail state dictionaries cannot be used separately.
    """
    if artifact.get('readout') != READOUT:
        raise ValueError('coupled native-tail artifact required')
    bound = artifact['binding']
    native = getattr(encoder, '_sibilant_native_binding', None)
    if native is None or any(bound.get(k) != v for k, v in native.items()):
        raise ValueError('encoder was not loaded from the bound native manifest')
    for key, value in expected.items():
        if bound.get(key) != value:
            raise ValueError('native artifact binding mismatch: ' + key)
    if (bound.get('classes') != list(CLASSES) or bound.get('encoder_depth') != 12
            or bound.get('tail_depth') != 2 or bound.get('hidden_layers') != [6, 12]
            or bound.get('coordinate_schema') != COORDINATES
            or bound.get('precision') != PRECISION or bound.get('mode') not in ('frozen', 'tail2')
            or len(encoder.encoder.layers) != 12):
        raise ValueError('native depth/classes/coordinates/precision mismatch')
    projection = artifact['projection']
    mean, scale, components = [projection[k] for k in ('mean', 'scale', 'components')]
    if (mean.shape != (1536,) or scale.shape != (1536,) or components.shape != (1536, 256)
            or not all(bool(torch.isfinite(v).all()) for v in (mean, scale, components))
            or not bool((scale > 0).all())
            or bound.get('projection_sha256') != projection_digest(projection)):
        raise ValueError('fixed projection differs')
    for part in ('tail', 'head'):
        if bound.get(part + '_sha256') != tensor_digest(artifact[part]):
            raise ValueError('coupled ' + part + ' tensors changed')
    probe = FrozenProbe(projection).to(device).eval()
    for part, target in [('head', probe.head), ('tail', encoder.encoder.layers[-2:])]:
        actual, reference = artifact[part], target.state_dict()
        if set(actual) != set(reference) or any(
                actual[k].shape != reference[k].shape or actual[k].dtype != reference[k].dtype
                or not bool(torch.isfinite(actual[k]).all()) for k in reference):
            raise ValueError('coupled ' + part + ' keys/shapes/dtypes differ')
    probe.head.load_state_dict(artifact['head'], strict=True)
    encoder.encoder.layers[-2:].load_state_dict(artifact['tail'], strict=True)
    encoder.eval().requires_grad_(False)
    return probe
