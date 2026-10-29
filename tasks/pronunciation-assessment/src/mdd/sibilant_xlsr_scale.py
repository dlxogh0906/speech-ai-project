"""Original XLS-R prefix caching and dense last-two-layer acoustic adaptation.

The float16 prefix/output roundtrips are explicit runtime contracts shared by
frozen and tuned controls. Runtime inputs contain audio and observed-V position;
neither expected C, TAG identity nor TAG time enters this module.
"""
import hashlib
import numpy as np
import torch
from mdd.sibilant_frozen_probe import FrozenProbe, projection_digest
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_vowel_presence import CLASSES

READOUT = 'xlsr_scale_dense_tail2_cv_v1'
PRECISION = 'nativefp32_prefixfp16_tailfp32_outputfp16_v1'


def load_original(path, device):
    from shared.fairseq_guard import ensure_fairseq
    ensure_fairseq()
    from fairseq import checkpoint_utils
    models, cfg, _ = checkpoint_utils.load_model_ensemble_and_task([str(path)], strict=True)
    if len(models) != 1:
        raise ValueError('one original XLS-R checkpoint required')
    model = models[0].eval().requires_grad_(False).to(device)
    depth, width = len(model.encoder.layers), model.cfg.encoder_embed_dim
    if (depth, width) not in ((24,1024),(48,1920)) or not cfg.task.normalize or cfg.task.sample_rate != 16000:
        raise ValueError('declared original XLS-R family/native input required')
    if not model.encoder.layer_norm_first or model.crop_seq_to_multiple != 1:
        raise ValueError('unsupported final normalization or native frame cropping')
    return model, {'depth':depth, 'width':width, 'normalize':bool(cfg.task.normalize),
        'sample_rate':16000, 'layer_norm_first':True, 'precision':PRECISION,
        'live_parameters':sum(z.numel() for z in model.parameters()),
        'last2_parameters':sum(z.numel() for z in model.encoder.layers[-2:].parameters())}


def normalized_source(source, feature, device):
    left, right = int(feature['left']), int(feature['right'])
    if (source.ndim != 1 or source.dtype != np.float32 or not np.isfinite(source).all()
            or not 0 <= left < right <= len(source)):
        raise ValueError('finite original float32 mono PCM/ROI required')
    if hashlib.sha256(source[left:right].tobytes()).hexdigest() != feature['clip_sha256']:
        raise ValueError('runtime original PCM changed')
    start = left//320*320
    # Exact official RawAudioDataset normalization, before adding any padding.
    value = torch.as_tensor(source[start:right].copy(),device=device)
    value = torch.nn.functional.layer_norm(value,value.shape)
    first, last = (left+319)//320, (right-400)//320
    if last < first:
        raise ValueError('no native receptive field inside original ROI')
    return value[None], first-start//320, last-first+1


def tail_hidden(layers, final_norm, prefix, source_lengths):
    if prefix.ndim != 3 or len(source_lengths) != len(prefix):
        raise ValueError('B,T,C prefix and original native lengths required')
    if bool((source_lengths < 1).any()) or bool((source_lengths > prefix.shape[1]).any()):
        raise ValueError('invalid original native lengths')
    padding = torch.arange(prefix.shape[1],device=prefix.device)[None] >= source_lengths[:,None]
    x = prefix.float().masked_fill(padding[...,None],0.).transpose(0,1)
    for layer in layers:
        # Original 2B checkpoints wrap layers with fairseq activation checkpointing.
        # Its custom autograd Function sees only inputs, not layer parameters;
        # a detached cached prefix therefore suppresses the entire backward graph.
        # Replay the same original forward without that memory-saving wrapper.
        # Full-model extraction keeps the wrapper, and capture_prefix checks parity.
        forward = getattr(layer,'precheckpoint_forward',layer)
        x,_ = forward(x,self_attn_padding_mask=padding,need_weights=False)
    return final_norm(x.transpose(0,1))


def capture_prefix(model, source, feature, quantize=True):
    device = next(model.parameters()).device
    value, skip, count = normalized_source(source,feature,device)
    captured = {}
    def hook(module,args,kwargs):
        captured['x'] = args[0].detach().clone().transpose(0,1)
    handle = model.encoder.layers[-2].register_forward_pre_hook(hook,with_kwargs=True)
    try:
        output = model.extract_features(value,padding_mask=None,mask=False)['x']
    finally:
        handle.remove()
    prefix = captured['x']
    n = output.shape[1]
    if n != (value.shape[1]-400)//320+1 or skip+count > n or prefix.shape[1] > 34:
        raise ValueError('original native frames/grid differ')
    if count != feature['input_frames']:
        raise ValueError('source/native ROI frame count differs')
    lengths = torch.tensor([n],device=device)
    original_replay = tail_hidden(model.encoder.layers[-2:],model.encoder.layer_norm,prefix,lengths)
    torch.testing.assert_close(original_replay[:,:n],output,rtol=3e-5,atol=3e-5)
    stored = prefix.to(torch.float16) if quantize else prefix
    hidden = tail_hidden(model.encoder.layers[-2:],model.encoder.layer_norm,stored.float(),lengths)
    return stored[0], n, skip, hidden[0,skip:skip+count], {
        'original_replay_max_difference':float((original_replay[:,:n]-output).abs().max()),
        'prefix_roundtrip_hidden_max_difference':float((hidden[:,:n]-output).abs().max())}


class ScaleReadout(torch.nn.Module):
    def __init__(self, layers, final_norm, projection, mode):
        super().__init__()
        if mode not in ('frozen','tail2'):
            raise ValueError('declared scale mode required')
        self.mode = mode
        self.layers = layers
        self.final_norm = final_norm.requires_grad_(False)
        self.probe = FrozenProbe(projection)
        self.layers.requires_grad_(mode == 'tail2')

    def observed_hidden(self,prefix,source_lengths,skips,lengths):
        hidden = tail_hidden(self.layers,self.final_norm,prefix,source_lengths)
        index = skips[:,None]+torch.arange(32,device=hidden.device)[None]
        mask = torch.arange(32,device=hidden.device)[None] < lengths[:,None]
        safe = index.clamp(max=hidden.shape[1]-1)
        selected = hidden[torch.arange(len(hidden),device=hidden.device)[:,None],safe]
        selected = selected.to(torch.float16).float().masked_fill(~mask[...,None],0.)
        return selected,mask

    def forward(self,prefix,source_lengths,skips,lengths,relative):
        hidden,mask = self.observed_hidden(prefix,source_lengths,skips,lengths)
        return self.probe(hidden,mask,relative)


def scale_artifact(model, binding):
    projection = {k:getattr(model.probe,k).detach().cpu().clone() for k in ('mean','scale','components')}
    head = {k:v.detach().cpu().clone() for k,v in model.probe.head.state_dict().items()}
    tail = ({k:v.detach().cpu().clone() for k,v in model.layers.state_dict().items()}
        if model.mode == 'tail2' else None)
    return {'readout':READOUT,'binding':dict(binding,mode=model.mode,classes=list(CLASSES),
        precision=PRECISION,projection_sha256=projection_digest(projection),head_sha256=tensor_digest(head),
        tail_sha256=tensor_digest(tail) if tail is not None else None),
        'projection':projection,'head':head,'tail':tail}


def restore_scale(artifact, binding, layers, final_norm, device):
    bound = artifact['binding']
    if (artifact.get('readout') != READOUT or any(bound.get(k) != v for k,v in binding.items())
            or bound.get('classes') != list(CLASSES) or bound.get('precision') != PRECISION
            or projection_digest(artifact['projection']) != bound.get('projection_sha256')
            or tensor_digest(artifact['head']) != bound.get('head_sha256')):
        raise ValueError('coupled XLS-R artifact source/projection/head mismatch')
    if bound['mode'] == 'tail2':
        if tensor_digest(artifact['tail']) != bound['tail_sha256']:
            raise ValueError('coupled dense tail weights changed')
        layers.load_state_dict(artifact['tail'],strict=True)
    elif (bound['mode'] != 'frozen' or artifact['tail'] is not None
            or tensor_digest(layers.state_dict()) != bound.get('original_tail_sha256')):
        raise ValueError('frozen XLS-R artifact requires unchanged original tail weights')
    model = ScaleReadout(layers,final_norm,artifact['projection'],bound['mode']).to(device).eval()
    model.probe.head.load_state_dict(artifact['head'],strict=True)
    return model
