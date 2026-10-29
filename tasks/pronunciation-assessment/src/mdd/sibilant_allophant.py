"""Frozen attribute-only Allophant inference in the existing Py3.8 runtime.

This reads the 36 independent OUTPUT classifiers. It does not decode phones,
map IPA to Hangul, or interpret learned attributes as physical measurements.
"""
import csv
import hashlib
import io
import json
import numpy as np
import torch


def normalize_single(waveform):
    if waveform.ndim != 1 or len(waveform) < 400 or not bool(torch.isfinite(waveform).all()):
        raise ValueError('finite unpadded single original waveform required')
    value = waveform[None]
    length = torch.as_tensor([len(waveform)], device=value.device)
    mean = (value.sum(1)/length).unsqueeze(1)
    variance = ((value-mean)**2).sum(1)/length
    return (value-mean)/(variance.unsqueeze(1)+1e-7).sqrt()


def attribute_metadata(checkpoint):
    if (checkpoint['sample_rate'] != 16000 or checkpoint['config']['preprocessing']['feature_type'] != 'RAW'
            or checkpoint['config']['nn']['acoustic_model']['model_id'] != 'facebook/wav2vec2-xls-r-300m'):
        raise ValueError('unsupported public acoustic architecture')
    nodes = checkpoint['attribute_graph']['nodes']
    if (len(nodes) != 37 or any(z['dependencies'] != ['OUTPUT'] or z['time_layer_config'] is not None for z in nodes)
            or len({z['name'] for z in nodes}) != 37 or nodes[-1]['name'] != 'phoneme'):
        raise ValueError('independent final OUTPUT attribute heads required')
    # Match original keep-first phoneme rows and sorted contour vocabulary.
    unique = {}
    for row in csv.DictReader(io.StringIO(checkpoint['phonetic_indexer_state']['table_file'])):
        unique.setdefault(row['Phoneme'], row)
    result = []
    for node in nodes[:-1]:
        name = node['name']
        categories = sorted({v for row in unique.values() for v in row[name].split(',')})
        if len(categories) != node['size']:
            raise ValueError('attribute class vocabulary differs: '+name)
        result.append({'name':name, 'categories':['CTC_BLANK']+categories})
    return result


def migrate_weight_norm(state, expected_keys):
    state = dict(state)
    mappings = {}
    for old, new in [('encoder.pos_conv_embed.conv.weight_g',
                      'encoder.pos_conv_embed.conv.parametrizations.weight.original0'),
                     ('encoder.pos_conv_embed.conv.weight_v',
                      'encoder.pos_conv_embed.conv.parametrizations.weight.original1')]:
        if old in state and old not in expected_keys and new in expected_keys:
            if new in state:
                raise ValueError('ambiguous old/new weight norm parameters')
            state[new] = state.pop(old)
            mappings[old] = new
    if set(state) != set(expected_keys):
        raise ValueError('backbone state keys differ after the explicit weight-norm rename')
    return state, mappings


class FrozenAttributes(torch.nn.Module):
    def __init__(self, encoder, heads, metadata):
        super().__init__()
        self.encoder = encoder
        self.heads = torch.nn.ModuleDict(heads)
        self.metadata = metadata
        self.requires_grad_(False).eval()

    def forward(self, waveform):
        value = normalize_single(waveform)
        hidden = self.encoder(value, attention_mask=torch.ones_like(value,dtype=torch.long),
                              output_hidden_states=True).hidden_states[-1]
        return {z['name']:self.heads[z['name']](hidden)[0] for z in self.metadata}, hidden[0]

    def observed(self, source, feature):
        source = np.asarray(source)
        left, right = int(feature['left']), int(feature['right'])
        if source.dtype != np.float32 or not 0 <= left < right <= len(source):
            raise ValueError('original float32 source ROI required')
        if hashlib.sha256(source[left:right].tobytes()).hexdigest() != feature['clip_sha256']:
            raise ValueError('original source PCM differs')
        start = left//320*320
        first, last = (left+319)//320, (right-400)//320
        if (last < first or feature['input_frames'] != last-first+1
                or feature['frame_first_source_sample'] != first*320+199.5):
            raise ValueError('native convolution coordinate contract differs')
        device = next(self.parameters()).device
        wave = torch.as_tensor(source[start:right],device=device)
        logits, hidden = self(wave)
        index = torch.arange(first,last+1,device=device)-start//320
        return {name:z[index] for name,z in logits.items()}, hidden[index]


def load_attributes(checkpoint, backbone_config, preprocessor, device):
    from transformers import Wav2Vec2Config, Wav2Vec2Model
    metadata = attribute_metadata(checkpoint)
    if (preprocessor.get('sampling_rate') != 16000 or not preprocessor.get('do_normalize')
            or not preprocessor.get('return_attention_mask') or preprocessor.get('feature_size') != 1):
        raise ValueError('official normalized RAW waveform configuration required')
    config = Wav2Vec2Config.from_dict(backbone_config)
    config._attn_implementation = 'eager'
    if (config.hidden_size, config.num_hidden_layers, config.conv_kernel, config.conv_stride) != (
            1024,24,[10,3,3,3,3,2,2],[5,2,2,2,2,2,2]):
        raise ValueError('public XLSR geometry differs')
    encoder = Wav2Vec2Model(config)
    prefix = '_acoustic_model._model.'
    state = checkpoint['model_state']
    backbone = {k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
    backbone, renames = migrate_weight_norm(backbone, encoder.state_dict())
    encoder.load_state_dict(backbone, strict=True)
    heads = {}
    for entry in metadata:
        name = entry['name']
        layer = torch.nn.Linear(1024,len(entry['categories']))
        key = '_projection._layers.'+name+'._time_distributed_layer.'
        layer.load_state_dict({k:state[key+k] for k in ['weight','bias']},strict=True)
        heads[name] = layer
    model = FrozenAttributes(encoder,heads,metadata).to(device).eval()
    return model, {'attributes':metadata,'weight_norm_key_renames':renames,
        'loaded_backbone_tensors':len(backbone),'attribute_heads':len(heads),
        'posterior_width':sum(len(z['categories']) for z in metadata),
        'ignored_original_phone_composition_branch':True,
        'scope':'Attribute-only original linear heads, same local HF backbone; no packaged phoneme decoder parity claim'}
