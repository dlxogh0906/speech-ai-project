"""Original normalization and forbidden padding in cached acoustic adaptation."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src')]
import hashlib
import unittest
import numpy as np
import torch
from mdd.sibilant_xlsr_scale import (normalized_source,tail_hidden,ScaleReadout,
                                    scale_artifact,restore_scale)
from mdd.sibilant_hubert_adaptation import tensor_digest


class AttentionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(4,2,dropout=0.)
    def forward(self,x,self_attn_padding_mask=None,need_weights=False):
        return self.attention(x,x,x,key_padding_mask=self_attn_padding_mask,need_weights=need_weights)


class ScaleTest(unittest.TestCase):
    def test_coupled_tail_restore_and_frozen_mismatch_rejection(self):
        import copy
        layers = torch.nn.ModuleList([AttentionLayer(),AttentionLayer()]).eval()
        original = copy.deepcopy(layers)
        norm = torch.nn.LayerNorm(4).eval()
        projection = {'mean':torch.zeros(4),'scale':torch.ones(4),'components':torch.eye(4)}
        binding = {'original_tail_sha256':tensor_digest(layers.state_dict())}
        frozen = ScaleReadout(layers,norm,projection,'frozen').eval()
        frozen_artifact = scale_artifact(frozen,binding)
        tuned = ScaleReadout(layers,norm,projection,'tail2').eval()
        with torch.no_grad():
            next(layers.parameters()).add_(.1)
        with self.assertRaises(ValueError):
            restore_scale(frozen_artifact,binding,layers,norm,torch.device('cpu'))
        artifact = scale_artifact(tuned,binding)
        restored = restore_scale(artifact,binding,original,copy.deepcopy(norm),torch.device('cpu'))
        self.assertEqual(tensor_digest(restored.layers.state_dict()),artifact['binding']['tail_sha256'])
        artifact['tail'][next(iter(artifact['tail']))].add_(.01)
        with self.assertRaises(ValueError):
            restore_scale(artifact,binding,original,norm,torch.device('cpu'))

    def test_original_checkpoint_wrapper_with_detached_cached_input(self):
        from shared.fairseq_guard import ensure_fairseq
        ensure_fairseq()
        from fairseq.modules.checkpoint_activations import checkpoint_wrapper
        import copy
        torch.manual_seed(3)
        original = torch.nn.ModuleList([AttentionLayer(),AttentionLayer()]).eval()
        wrapped = copy.deepcopy(original)
        for layer in wrapped:
            checkpoint_wrapper(layer)
        norm = torch.nn.LayerNorm(4).eval().requires_grad_(False)
        prefix = torch.randn(1,5,4)
        lengths = torch.tensor([5])
        # Reproduce the 2B failure: parameters alone do not activate its wrapper.
        broken,_ = wrapped[0](prefix.transpose(0,1),need_weights=False)
        self.assertFalse(broken.requires_grad)
        expected = tail_hidden(original,norm,prefix,lengths)
        actual = tail_hidden(wrapped,norm,prefix,lengths)
        self.assertTrue(torch.equal(expected,actual))
        expected[...,0].square().mean().backward()
        actual[...,0].square().mean().backward()
        for a,b in zip(original.parameters(),wrapped.parameters()):
            self.assertIsNotNone(b.grad)
            self.assertTrue(torch.equal(a.grad,b.grad))
        self.assertFalse(prefix.requires_grad)
        self.assertTrue(all(p.grad is None for p in norm.parameters()))

    def test_exact_native_norm_and_immutable_original_roi(self):
        wave = np.sin(np.arange(1600)/13.).astype(np.float32)
        before = wave.copy()
        feature = {'left':11,'right':1500,'clip_sha256':hashlib.sha256(wave[11:1500].tobytes()).hexdigest()}
        actual,skip,count = normalized_source(wave,feature,torch.device('cpu'))
        expected = torch.nn.functional.layer_norm(torch.from_numpy(wave[:1500]),(1500,))
        self.assertTrue(torch.equal(actual[0],expected))
        self.assertEqual((skip,count),(1,3))
        np.testing.assert_array_equal(before,wave)
        changed = wave.copy();changed[80] += 1
        with self.assertRaises(ValueError):
            normalized_source(changed,feature,torch.device('cpu'))

    def test_cached_padding_cannot_enter_valid_acoustic_frames(self):
        torch.manual_seed(2)
        layers = torch.nn.ModuleList([AttentionLayer(),AttentionLayer()]).eval()
        norm = torch.nn.LayerNorm(4).eval()
        x = torch.randn(1,5,4)
        a = tail_hidden(layers,norm,x,torch.tensor([3]))[:,:3]
        poisoned = x.clone();poisoned[:,3:] = float('nan')
        b = tail_hidden(layers,norm,poisoned,torch.tensor([3]))[:,:3]
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.isfinite(b).all())
        with self.assertRaises(ValueError):
            tail_hidden(layers,norm,x,torch.tensor([6]))


if __name__ == '__main__':
    unittest.main()
