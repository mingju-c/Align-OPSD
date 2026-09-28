"""CPU numerical regression for C; load the pure module without Ray imports."""
import ast
import math
import unittest
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence
from unittest.mock import patch
import torch

path = Path(__file__).resolve().parents[1] / 'verl/trainer/ppo/correspondence_credit.py'
tree = ast.parse(path.read_text())
# Keep all definitions, omitting package imports that require the GPU training stack.
module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] +
                    [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))], type_ignores=[])
ns = dict(torch=torch, math=math, dataclass=dataclass, field=field, Sequence=Sequence)
exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), ns)

class FlatTurnTests(unittest.TestCase):
    def test_token_mode_has_token_varying_credit(self):
        identities = [SimpleNamespace(task_id='x', rollout_id=str(i // 3), turn_id=i % 3) for i in range(6)]
        gaps = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]] * 2)
        mask = torch.ones_like(gaps, dtype=torch.bool)
        adv = torch.tensor([1.] * 3 + [-1.] * 3)
        cfg = ns['CorrespondenceCreditConfig'](segmentation_mode='token', weighting_mode='bounded')
        out = ns['compute_correspondence_credit_from_tensors'](
            thinking_similarity=torch.zeros(6, 6), structural_mask=torch.zeros(6, 6, dtype=torch.bool),
            similarity_threshold=.7, corrected_token_gap=gaps, policy_loss_mask=mask,
            identities=identities, trajectory_advantage=adv, config=cfg)
        self.assertTrue((out.segment_index_per_turn == -1).all())
        self.assertFalse(torch.allclose(out.token_advantage[0], out.token_advantage[1]))
        self.assertTrue((out.turn_weight >= .9 - 1e-6).all())
        self.assertTrue((out.turn_weight <= 1.1 + 1e-6).all())

    def test_credit_and_no_segmentation(self):
        identities = [SimpleNamespace(task_id='x', rollout_id=str(i // 12), turn_id=i % 12) for i in range(36)]
        gaps = torch.arange(144, dtype=torch.float32).reshape(36, 4) / 100
        mask = torch.arange(4)[None, :] < (torch.arange(36) % 4 + 1)[:, None]
        adv = torch.tensor([2.] * 12 + [-1.] * 12 + [0.] * 12)
        kwargs = dict(thinking_similarity=torch.zeros(36, 36), structural_mask=torch.zeros(36, 36, dtype=torch.bool),
                      similarity_threshold=.7, corrected_token_gap=gaps, policy_loss_mask=mask,
                      identities=identities, trajectory_advantage=adv)
        for mode in ('legacy', 'bounded'):
            cfg = ns['CorrespondenceCreditConfig'](segmentation_mode='turn', weighting_mode=mode)
            with patch.dict(ns, {name: lambda *a, **k: self.fail('C invoked segmentation') for name in
                                ('build_dense_correspondence_profiles', 'compute_adjacent_jsd', 'segment_trajectories')}):
                out = ns['compute_correspondence_credit_from_tensors'](**kwargs, config=cfg)
                other = ns['compute_correspondence_credit_from_tensors'](**kwargs, config=replace(cfg, max_segment_turns=20))
            self.assertEqual(out.trajectory_segmentations, ())
            self.assertTrue((out.segment_index_per_turn == -1).all())
            torch.testing.assert_close(out.token_advantage, other.token_advantage)
            self.assertTrue((out.token_advantage[~mask] == 0).all())
            for start in (0, 12, 24):
                sl = slice(start, start + 12)
                counts = mask[sl].sum(-1).float()
                torch.testing.assert_close((counts * out.turn_weight[sl]).sum(), counts.sum())
                q = torch.softmax(counts.log() + adv[start].sign() * out.turn_gap[sl] / cfg.credit_temperature, 0)
                torch.testing.assert_close(out.within_segment_turn_budget[sl], q)
            self.assertGreater(float(out.turn_weight[11]), float(out.turn_weight[0]))
            self.assertLess(float(out.turn_weight[23]), float(out.turn_weight[12]))
            torch.testing.assert_close(out.turn_weight[24:], torch.ones(12))
            torch.testing.assert_close(out.token_advantage, mask * adv[:, None] * out.turn_weight[:, None])
            if mode == 'bounded':
                self.assertTrue((out.turn_weight >= .9 - 1e-6).all())
                self.assertTrue((out.turn_weight <= 1.1 + 1e-6).all())

if __name__ == '__main__':
    unittest.main()
