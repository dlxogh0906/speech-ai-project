"""Precision, unpadded gradient flow, and coupled native-artifact integrity."""
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src')]
import numpy as np
import torch
from mdd.sibilant_frozen_probe import FrozenProbe,projection_digest,READOUT
from mdd.sibilant_vowel_presence import CLASSES
from mdd.sibilant_conditional_retained import new_conditional
from mdd.sibilant_native_conditional import (native_tail,NativeConditional,prefix_digest,
    export_native,restore_native)
from mdd.sibilant_hubert_adaptation import tensor_digest


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__();self.linear = torch.nn.Linear(768,768)
    def forward(self,x,attention_mask=None,output_attentions=False):
        return (x+self.linear(x).tanh(),)


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__();self.encoder = torch.nn.Module()
        self.encoder.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(10)]+[Layer(),Layer()])
        self.encoder.layer_norm = torch.nn.LayerNorm(768)
        self.config = SimpleNamespace(do_stable_layer_norm=False)
        self._sibilant_native_binding = {'unit':'native'}


class NativeTest(unittest.TestCase):
    def setup_model(self):
        torch.manual_seed(5);torch.set_num_threads(2)
        encoder = Encoder()
        projection = {'mean':torch.zeros(1536),'scale':torch.ones(1536),'components':torch.randn(1536,3)*.01}
        probe = FrozenProbe(projection).eval()
        base = {'readout':READOUT,'projection':projection,'head':copy.deepcopy(probe.head.state_dict()),
            'binding':{'backbone_key':'unit','classes':list(CLASSES),'projection_sha256':projection_digest(projection)}}
        binding = {'native':encoder._sibilant_native_binding,'prefix_sha256':prefix_digest(encoder),
            'original_tail_sha256':tensor_digest(encoder.encoder.layers[-2:].state_dict())}
        model = NativeConditional(encoder,new_conditional(base,base['binding'],'cpu'),'tail2').eval()
        return encoder,base,binding,model

    def test_float32_unpadded_prefix_and_gradient_matches_direct_layers(self):
        encoder,base,binding,model = self.setup_model()
        prefix = torch.randn(1,5,768)
        expected = encoder.encoder.layers[11](encoder.encoder.layers[10](prefix)[0])[0]
        self.assertTrue(torch.equal(native_tail(encoder,prefix),expected))
        with self.assertRaises(ValueError):native_tail(encoder,prefix.half())
        cache = {'prefix':np.full((2,7,768),np.nan,np.float32),
            'early':np.full((2,32,768),np.nan,np.float16),'lengths':np.asarray([4,3]),
            'source_lengths':np.asarray([5,4]),'skips':np.asarray([1,1]),
            'relative':np.zeros((2,32),np.float32)}
        cache['prefix'][0,:5] = prefix.detach().numpy()[0];cache['prefix'][1,:4] = prefix.detach().numpy()[0,:4]
        cache['early'][0,:4] = 0.;cache['early'][1,:3] = 0.
        logits,available = model.from_cache(cache,np.asarray([0,1]),torch.tensor([0,1]))
        self.assertTrue(bool(torch.isfinite(logits).all()));self.assertEqual(available.shape,(2,))
        loss = logits.square().mean();loss.backward()
        for layer in encoder.encoder.layers[-2:]:
            self.assertGreater(sum(float(v.grad.abs().sum()) for v in layer.parameters()),0.)
        self.assertTrue(all(v.grad is None for v in encoder.encoder.layer_norm.parameters()))
        self.assertEqual(prefix_digest(encoder),binding['prefix_sha256'])

    def test_native_and_head_scope_tampering_rejected_before_encoder_mutation(self):
        encoder,base,binding,model = self.setup_model()
        with torch.no_grad():model.encoder.encoder.layers[-1].linear.bias.add_(.01)
        artifact = export_native(model,base,binding)
        target = Encoder();target.encoder.layer_norm.load_state_dict(encoder.encoder.layer_norm.state_dict())
        restored = restore_native(artifact,target,binding,base['binding'],'cpu')
        self.assertEqual(tensor_digest(restored.encoder.encoder.layers[-2:].state_dict()),artifact['binding']['tail_sha256'])
        before = tensor_digest(target.state_dict())
        broken = copy.deepcopy(artifact);key = next(iter(broken['tail']));broken['tail'][key].add_(1.)
        with self.assertRaises(ValueError):restore_native(broken,target,binding,base['binding'],'cpu')
        self.assertEqual(tensor_digest(target.state_dict()),before)
        broken = copy.deepcopy(artifact);broken['conditional']['heads']['ㅅ']['runtime_expected_C'] = 'ㅆ'
        with self.assertRaises(ValueError):restore_native(broken,target,binding,base['binding'],'cpu')
        self.assertEqual(tensor_digest(target.state_dict()),before)


if __name__=='__main__':unittest.main()
