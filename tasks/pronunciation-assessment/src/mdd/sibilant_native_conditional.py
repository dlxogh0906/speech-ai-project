"""Original native HuBERT prefix replay with scoped conditional actual5 heads.

No annotation or actual-phone identity enters capture/replay. Prefixes stay
float32; only the original h6/h12 readout storage roundtrip uses float16.
Individual unpadded tails preserve the original attention computation.
"""
import copy
import hashlib
import numpy as np
import torch
from mdd.sibilant_hubert_adaptation import (encode_clip,tensor_digest,PRECISION,
                                          binding_from_manifest)
from mdd.sibilant_conditional_retained import (TOKENS,new_conditional,
    conditional_artifact,load_conditional)
from mdd.sibilant_vowel_presence import observed_vowel_support

READOUT = 'native_hubert_conditional_tail2_actual5_v1'


def prefix_digest(encoder):
    return tensor_digest({k:v for k,v in encoder.state_dict().items()
                          if not k.startswith(('encoder.layers.10.','encoder.layers.11.'))})


def native_tail(encoder,prefix):
    if prefix.ndim!=3 or prefix.shape[0]!=1 or prefix.shape[-1]!=768:
        raise ValueError('one unpadded native float32 prefix required')
    if prefix.dtype!=torch.float32:
        raise ValueError('prefix precision must remain original float32')
    x = prefix
    for layer in encoder.encoder.layers[-2:]:
        x = layer(x,attention_mask=None,output_attentions=False)[0]
    if encoder.config.do_stable_layer_norm:
        x = encoder.encoder.layer_norm(x)
    return x


def relative_grid(feature,width=32):
    n = int(feature['input_frames'])
    if not 1<=n<=width:
        raise ValueError('original native ROI length required')
    value = np.zeros(width,np.float32)
    centers = float(feature['frame_first_source_sample'])+np.arange(n)*320
    value[:n] = np.clip((centers/16000.-feature['observed_reference_seconds'])/.31,-2,2).astype(np.float32)
    return value


def capture_native(encoder,processor,source,feature):
    """Only original waveform and frozen runtime geometry; no labels."""
    left,right = int(feature['left']),int(feature['right'])
    if (source.ndim!=1 or source.dtype!=np.float32 or not np.isfinite(source).all()
            or not 0<=left<right<=len(source)):
        raise ValueError('finite original mono float32 source/ROI required')
    if hashlib.sha256(source[left:right].tobytes()).hexdigest()!=feature['clip_sha256']:
        raise ValueError('original source ROI changed')
    start = left//320*320;first,last = (left+319)//320,(right-400)//320
    n,skip = last-first+1,first-start//320
    if n!=feature['input_frames'] or not 1<=n<=32:
        raise ValueError('runtime native receptive-field grid differs')
    device = next(encoder.parameters()).device
    value = processor(source[start:right],sampling_rate=16000,return_tensors='pt',padding=False).input_values.to(device)
    hidden = encoder(value,output_hidden_states=True).hidden_states
    prefix = hidden[10];size = prefix.shape[1]
    if size!=(right-start-400)//320+1 or not skip+n<=size<=34:
        raise ValueError('native encoder input extent differs')
    replay = native_tail(encoder,prefix.detach())
    torch.testing.assert_close(replay,hidden[12],rtol=2e-5,atol=2e-5)
    early = hidden[6][0,skip:skip+n].to(torch.float16)
    observed = torch.cat([early.float(),hidden[12][0,skip:skip+n].to(torch.float16).float()],-1)
    return prefix[0].detach(),early.detach(),observed.detach(),skip,{
        'native_full_vs_replay_max_difference':float((replay-hidden[12]).abs().max()),
        'source_frames':size,'observed_frames':n}


def hidden_from_cache(encoder,prefix,early,skip,count):
    late = native_tail(encoder,prefix[None])[0,skip:skip+count].to(torch.float16).float()
    if early.shape!=(count,768) or late.shape!=early.shape:
        raise ValueError('cached early/late native grid differs')
    x = torch.cat([early.float(),late],-1)
    return torch.nn.functional.pad(x,(0,0,0,32-count))


class NativeConditional(torch.nn.Module):
    def __init__(self,encoder,heads,mode):
        super().__init__()
        if mode not in ('frozen','tail2'):
            raise ValueError('native frozen/tail2 treatment required')
        self.encoder = encoder.eval().requires_grad_(False)
        self.encoder.encoder.layers[-2:].requires_grad_(mode=='tail2')
        self.heads = heads
        self.mode = mode

    def summary_batch(self,cache,indices):
        summary,moments,phases = [],[],[]
        device = next(self.parameters()).device
        for i in indices:
            i = int(i);n,size,skip = [int(cache[k][i]) for k in ('lengths','source_lengths','skips')]
            prefix = torch.as_tensor(np.array(cache['prefix'][i,:size],copy=True),device=device)
            early = torch.as_tensor(np.array(cache['early'][i,:n],copy=True),device=device)
            x = hidden_from_cache(self.encoder,prefix,early,skip,n)[None]
            mask = torch.arange(32,device=device)[None]<n
            relative = torch.as_tensor(np.array(cache['relative'][i:i+1],copy=True),device=device)
            # Both scoped heads have the same frozen PCA and support network.
            s,m,phase = self.heads.heads[0].summarize(x,mask,relative)
            summary.append(s[0]);moments.append(m[0]);phases.append(phase[0])
        return torch.stack(summary),torch.stack(moments),torch.stack(phases)

    def from_cache(self,cache,indices,expected_index):
        summary,moments,phases = self.summary_batch(cache,indices)
        device = summary.device
        lengths = torch.as_tensor(np.asarray(cache['lengths'][indices]),device=device)
        mask = torch.arange(32,device=device)[None]<lengths[:,None]
        logits = self.heads.from_summary(summary,moments,expected_index)
        return logits,observed_vowel_support(phases,mask)


def export_native(model,base,binding):
    if prefix_digest(model.encoder)!=binding['prefix_sha256']:
        raise ValueError('frozen native prefix changed')
    current = tensor_digest(model.encoder.encoder.layers[-2:].state_dict())
    if model.mode=='frozen' and current!=binding['original_tail_sha256']:
        raise ValueError('frozen tail changed')
    tail = ({k:v.detach().cpu().clone() for k,v in model.encoder.encoder.layers[-2:].state_dict().items()}
            if model.mode=='tail2' else None)
    return {'readout':READOUT,'binding':dict(binding,mode=model.mode,precision=PRECISION,
        tail_sha256=current),'tail':tail,'conditional':conditional_artifact(model.heads,base)}


def restore_native(artifact,encoder,expected_binding,head_binding,device):
    bound = artifact.get('binding',{})
    native = getattr(encoder,'_sibilant_native_binding',None)
    if (artifact.get('readout')!=READOUT or bound.get('precision')!=PRECISION
            or bound.get('mode') not in ('frozen','tail2') or native is None
            or any(bound.get(k)!=v for k,v in expected_binding.items())
            or any(bound.get('native',{}).get(k)!=v for k,v in native.items())
            or prefix_digest(encoder)!=bound.get('prefix_sha256')):
        raise ValueError('coupled native source/prefix/mode binding differs')
    heads = load_conditional(artifact['conditional'],head_binding,device)
    if bound['mode']=='tail2':
        tail = artifact.get('tail')
        target = encoder.encoder.layers[-2:].state_dict()
        if (not isinstance(tail,dict) or set(tail)!=set(target)
                or any(tail[k].shape!=v.shape or tail[k].dtype!=v.dtype
                       or not bool(torch.isfinite(tail[k]).all()) for k,v in target.items())
                or tensor_digest(tail)!=bound.get('tail_sha256')):
            raise ValueError('coupled native tail tensors differ')
    else:
        if (artifact.get('tail') is not None
                or tensor_digest(encoder.encoder.layers[-2:].state_dict())!=bound.get('original_tail_sha256')
                or bound.get('tail_sha256')!=bound.get('original_tail_sha256')):
            raise ValueError('frozen native artifact requires original tail')
    # Validate everything before changing the caller's native encoder.
    if bound['mode']=='tail2':encoder.encoder.layers[-2:].load_state_dict(artifact['tail'],strict=True)
    return NativeConditional(encoder,heads,bound['mode']).to(device).eval()


def runtime_logits(model,processor,source,feature):
    """Direct original-PCM inference with automatic expected-C routing."""
    token = feature['canonical']
    if token not in TOKENS:raise ValueError('runtime canonical S/SS required')
    hidden = encode_clip(model.encoder,processor,source,feature)
    x = torch.nn.functional.pad(hidden,(0,0,0,32-len(hidden)))[None]
    device = x.device;mask = torch.arange(32,device=device)[None]<len(hidden)
    relative = torch.as_tensor(relative_grid(feature)[None],device=device)
    logits,phase = model.heads(x,mask,relative,torch.tensor([TOKENS.index(token)],device=device))
    return logits,observed_vowel_support(phase,mask)
