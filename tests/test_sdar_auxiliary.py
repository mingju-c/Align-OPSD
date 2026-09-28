"""Run with unittest; no GPU or rollout engine required."""
import unittest
import ast
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir
from pathlib import Path
from omegaconf import OmegaConf

from verl.trainer.main_beyond_timestamps import _validate_beyond_timestamps_config
from verl.trainer.ppo.sdar_utils import compute_sdar_loss


class SdarAuxiliaryTests(unittest.TestCase):
    def test_actor_selects_diagonal_without_removed_m2_diagnostics(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / 'verl/workers/actor/dp_actor.py').read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'update_policy')
        start = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'select_keys')
        end = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'batch')
        code = compile(ast.Module(body=method.body[start:end], type_ignores=[]), '<actor keys>', 'exec')
        for enabled in [False, True]:
            cfg = OmegaConf.create(dict(use_kl_loss=False, use_sdar_loss=enabled,
                sdar_use_diagonal_teacher=enabled, loss_agg_mode='token-mean'))
            scope = dict(self=SimpleNamespace(config=cfg), multi_turn=False,
                         data=SimpleNamespace(batch={'canonical_row_weight': None}))
            exec(code, scope)
            self.assertNotIn('teacher_log_probs', scope['select_keys'])
            self.assertEqual('sdar_diagonal_teacher_log_probs' in scope['select_keys'], enabled)

    def test_formula_gradient_and_teacher_detachment(self):
        student = torch.tensor([[-2., -1., -3.]], requires_grad=True)
        teacher = torch.tensor([[-1., -2., -1.]], requires_grad=True)
        mask = torch.tensor([[1., 1., 0.]])
        loss, _ = compute_sdar_loss(student, teacher, mask, gate_beta=5.)
        gate = torch.sigmoid(5 * (teacher.detach() - student.detach()))
        torch.testing.assert_close(loss, (gate * (teacher - student) * mask).sum() / 2)
        (loss * .01).backward()
        torch.testing.assert_close(student.grad, -.01 * gate * mask / 2)
        self.assertIsNone(teacher.grad)

    def test_duplicate_rows_have_no_extra_mass(self):
        student = torch.tensor([[-2., -1.], [-3., -2.]])
        teacher = torch.tensor([[-1., -2.], [-1., -1.]])
        base, metrics = compute_sdar_loss(student, teacher, torch.ones_like(student))
        idx = torch.tensor([0, 0, 1])
        duplicate, repeated = compute_sdar_loss(
            student[idx], teacher[idx], torch.ones_like(student[idx]),
            loss_weight=torch.tensor([[.5], [.5], [1.]]))
        torch.testing.assert_close(base, duplicate)
        self.assertAlmostEqual(metrics['sdar/gate_mean'], repeated['sdar/gate_mean'])

    def test_config_defaults_and_opt_in(self):
        root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(version_base=None, config_dir=str(root / 'verl/trainer/config')):
            cfg = compose(config_name='beyond_timestamps_trainer')
        _validate_beyond_timestamps_config(cfg)
        self.assertFalse(cfg.algorithm.sdar_aux.enabled)
        self.assertEqual(cfg.algorithm.sdar_aux.coef, .01)
        self.assertEqual(cfg.algorithm.sdar_aux.gate_beta, 5.)
        cfg.algorithm.sdar_aux.enabled = True
        _validate_beyond_timestamps_config(cfg)
        for key, value in [('coef', -1.), ('coef', float('nan')), ('gate_beta', 0.)]:
            previous = cfg.algorithm.sdar_aux[key]
            cfg.algorithm.sdar_aux[key] = value
            with self.assertRaises(ValueError):
                _validate_beyond_timestamps_config(cfg)
            cfg.algorithm.sdar_aux[key] = previous


if __name__ == '__main__':
    unittest.main()
