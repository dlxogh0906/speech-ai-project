import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path.insert(0,str(ROOT/'tasks/pronunciation-assessment/src'))
import unittest
import torch
from mdd.sibilant_allophant import normalize_single, migrate_weight_norm, attribute_metadata


class AllophantTest(unittest.TestCase):
    def test_unpadded_normalization_retains_input_and_rejects_bad(self):
        wave = torch.linspace(-.4,.6,800)
        before = wave.clone()
        norm = normalize_single(wave)
        torch.testing.assert_close(wave,before,rtol=0,atol=0)
        torch.testing.assert_close(norm,normalize_single(3*wave+.7),rtol=2e-5,atol=2e-6)
        self.assertTrue(torch.equal(normalize_single(torch.ones(800)),torch.zeros(1,800)))
        for bad in [torch.ones(399),torch.ones(2,800),torch.full((800,),float('nan'))]:
            with self.assertRaises(ValueError):normalize_single(bad)

    def test_weight_norm_rename_preserves_values_not_silent_drop(self):
        old='encoder.pos_conv_embed.conv.weight_g';new='encoder.pos_conv_embed.conv.parametrizations.weight.original0'
        value=torch.tensor([3.]);state,mapping=migrate_weight_norm({old:value},[new])
        self.assertIs(state[new],value)
        self.assertEqual(mapping,{old:new})
        with self.assertRaises(ValueError):migrate_weight_norm({old:value,new:value},[new])
        with self.assertRaises(ValueError):migrate_weight_norm({old:value},[new,'missing'])

    def test_unsupported_architecture_rejected_before_model_instantiation(self):
        d={'sample_rate':8000,'config':{'preprocessing':{'feature_type':'RAW'},
            'nn':{'acoustic_model':{'model_id':'facebook/wav2vec2-xls-r-300m'}}}}
        with self.assertRaises(ValueError):attribute_metadata(d)
        d['sample_rate']=16000;d['attribute_graph']={'nodes':[]}
        with self.assertRaises(ValueError):attribute_metadata(d)


if __name__ == '__main__':
    unittest.main()
