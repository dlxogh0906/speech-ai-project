import copy
import io
import sys
import unittest
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tasks/pronunciation-assessment/src'))
from mdd.sibilant_frozen_raw_roles import FrozenRawRoles, role_artifact, load_roles
from mdd.sibilant_raw_cv import RawCVResidual, coarse_role_loss, cv_artifact, load_cv
from mdd.sibilant_hires_acoustics import NAMES
from mdd.sibilant_frozen_probe import FrozenProbe, READOUT, projection_digest
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_vowel_presence import CLASSES


class FrozenRolesTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(77)
        projection = {'mean': torch.zeros(6), 'scale': torch.ones(6), 'components': torch.randn(6, 3)}
        self.base = FrozenProbe(projection).eval()
        self.base_artifact = {'readout': READOUT, 'projection': projection,
            'head': copy.deepcopy(self.base.head.state_dict()), 'binding': {'backbone_key': 'unit',
                'classes': list(CLASSES), 'projection_sha256': projection_digest(projection)}}
        self.normalization = {'mean': [0.]*len(NAMES), 'scale': [1.]*len(NAMES)}
        self.raw = torch.randn(2, 25, 2*len(NAMES))
        self.mask = torch.tensor([[True]*25, [True]*17+[False]*8])
        self.relative = torch.linspace(-.5, .5, 25)[None].repeat(2, 1)
        teacher = RawCVResidual(copy.deepcopy(self.base), 'guided').eval()
        optimizer = torch.optim.AdamW([v for v in teacher.parameters() if v.requires_grad], lr=.01)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            _, role, _ = teacher.raw_terms(self.raw, self.mask, self.relative)
            target = torch.tensor([[0]*5+[1]*10+[2]*10, [0]*5+[1]*6+[2]*6+[-100]*8])
            coarse_role_loss(role, target).backward()
            optimizer.step()
        self.teacher_artifact = cv_artifact(teacher, self.base_artifact, self.normalization)
        self.permutation = torch.arange(25)[None].repeat(2, 1)
        self.permutation[0] = torch.arange(24, -1, -1)
        self.permutation[1, :17] = torch.arange(16, -1, -1)

    def model(self, mode):
        teacher = load_cv(self.teacher_artifact, {'backbone_key': 'unit'}, 'guided', 'cpu')
        torch.manual_seed(79)
        return FrozenRawRoles(copy.deepcopy(self.base), teacher, mode).eval()

    def test_teacher_assignment_only_control_preserves_distributions_and_padding(self):
        aligned, control = self.model('aligned'), self.model('permuted')
        a = aligned.teacher_roles(self.raw, self.mask, self.relative)
        b = control.teacher_roles(self.raw, self.mask, self.relative, self.permutation)
        self.assertFalse(a.requires_grad)
        self.assertTrue(torch.equal(b, a.gather(1, self.permutation[..., None].expand_as(a))))
        self.assertTrue(torch.equal(a.sort(1).values, b.sort(1).values))
        bad = self.permutation.clone()
        bad[1] = torch.arange(24, -1, -1)
        with self.assertRaises(ValueError):
            control.teacher_roles(self.raw, self.mask, self.relative, bad)

    def test_student_gradient_base_teacher_immutability_reload_and_teacher_coupling(self):
        model = self.model('aligned')
        x, mask, relative = torch.randn(2, 8, 6), torch.ones(2, 8, dtype=torch.bool), torch.zeros(2, 8)
        args = (x, mask, relative, self.raw, self.mask, self.relative)
        self.assertTrue(torch.equal(model(*args)[0], self.base(x, mask, relative)[0]))
        original_base, original_teacher = tensor_digest(model.base.state_dict()), tensor_digest(model.teacher.state_dict())
        optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=.01)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(model(*args)[0], torch.tensor([0, 1])).backward()
            optimizer.step()
        self.assertGreater(float(model.layers[0].weight.grad.abs().sum()), 0.)
        self.assertTrue(all(v.grad is None for v in model.teacher.parameters()))
        self.assertEqual(original_teacher, tensor_digest(model.teacher.state_dict()))
        self.assertEqual(original_base, tensor_digest(model.base.state_dict()))
        saved = role_artifact(model, self.base_artifact, self.teacher_artifact, self.normalization)
        buffer = io.BytesIO()
        torch.save(saved, buffer)
        buffer.seek(0)
        restored = load_roles(torch.load(buffer), {'backbone_key': 'unit'}, 'aligned', 'cpu')
        self.assertTrue(torch.equal(restored(*args)[0], model(*args)[0]))
        with self.assertRaises(ValueError):
            load_roles(saved, {}, 'permuted', 'cpu')
        with self.assertRaises(ValueError):
            load_cv(saved, {}, 'guided', 'cpu')
        other = load_cv(self.teacher_artifact, {'backbone_key': 'unit'}, 'guided', 'cpu')
        with torch.no_grad():
            other.role.bias.add_(.1)
        bad = copy.deepcopy(saved)
        bad['teacher'] = cv_artifact(other, self.base_artifact, self.normalization)
        with self.assertRaises(ValueError):
            load_roles(bad, {}, 'aligned', 'cpu')


if __name__ == '__main__':
    unittest.main()
