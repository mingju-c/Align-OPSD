"""CPU audit tests: python tests/test_retrospective_logging.py (no Ray required)."""

import ast
import copy
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

if __package__:
    from .test_rectification_probability_mixture import (
        ROOT, CharacterTokenizer, core, diagnostics, load_module, text_batch,
    )
else:
    from test_rectification_probability_mixture import (
        ROOT, CharacterTokenizer, core, diagnostics, load_module, text_batch,
    )

audit = load_module("_audit_logging", "verl/trainer/ppo/retrospective_logging.py")


def fixture(tag="action"):
    ids, sampled = text_batch(*([f"<think>Check size.</think><{tag}>inspect</{tag}>"] * 4))
    mask = sampled.clone()
    mask[0] = False  # legitimate zero-policy-token turn
    m, length = ids.shape
    identities = tuple(core.TurnIdentity("task", f"rollout-{i}", i) for i in range(m))
    diagonal = torch.full((m, length), .38).log().requires_grad_()
    student = torch.full((m, length), .40).log().requires_grad_()
    h = torch.zeros(m, m)
    weights = torch.tensor([.5, .3, .2])
    h[:3, 3] = .95 + .1 * weights.log()
    selected = ((), (), (), (0, 1, 2))
    cross = {(i, 3): torch.full((length,), p).log() for i, p in enumerate((.7, .1, .5))}
    combined = core.combine_cross_teacher_scores(diagonal, student, h, selected, cross)
    result = SimpleNamespace(
        **{k: combined[k] for k in ("diagonal_token_gap", "corrected_token_gap", "cross_teacher_log_probs",
                                   "corrected_teacher_log_probs", "correspondence_confidence",
                                   "correction_strength", "selected_source_weights")},
        diagonal_teacher_log_probs=diagonal, policy_loss_mask=mask, sampled_response_mask=sampled,
        turn_map=SimpleNamespace(identities=identities), selected_source_indices=selected,
        thinking_similarity=h, teacher_thinking=("Check size.",) * m, teacher_actions=("inspect",) * m,
    )
    regions, _ = diagnostics.response_region_masks(CharacterTokenizer(), ids, sampled)
    # EXACT production build_teacher_batch keys: no 'prompts' tensor.
    prompts = torch.tensor([[0, 101 + i, 201 + i, 301 + i] for i in range(m)])
    teacher = SimpleNamespace(batch={
        "input_ids": torch.cat([prompts, ids], -1), "responses": ids,
        "attention_mask": torch.cat([prompts.ne(0), sampled], -1),
    })
    batch = SimpleNamespace(batch={"responses": ids})
    return result, student, regions, combined, cross, teacher, batch


class AuditDataTests(unittest.TestCase):
    def test_three_benchmark_protocols_replay_exact_scores_and_preserve_inputs(self):
        for bench, tag in (("ALFWorld", "action"), ("WebShop", "action"),
                           ("Search-QA search", "search"), ("Search-QA answer", "answer")):
            with self.subTest(benchmark=bench):
                result, student, regions, combined, cross, teacher, batch = fixture(tag)
                before = {k: v.clone() for k, v in combined.items() if isinstance(v, torch.Tensor)}
                old_rng = torch.random.get_rng_state().clone()
                records = audit.turn_gap_records(result, student, regions, 1e-4)
                snapshot = audit.token_snapshot(
                    index=3, reason="test", result=result, batch=batch, teacher_batch=teacher,
                    student_log_probs=student, combined=combined, cross_scores=cross, region_masks=regions,
                    logger=SimpleNamespace(max_tokens=512, max_context_tokens=4096))
                snapshot = json.loads(json.dumps(audit.json_safe(snapshot), allow_nan=False))
                source_lps = torch.tensor([source["teacher_log_probs"] for source in snapshot["sources"]])
                w = torch.tensor([source["weight"] for source in snapshot["sources"]])
                replay = torch.logsumexp(w.log()[:, None] + source_lps, 0)
                torch.testing.assert_close(replay, torch.tensor(snapshot["cross_teacher_log_probs"]))
                diagonal = torch.tensor(snapshot["diagonal_teacher_log_probs"])
                a = snapshot["alpha"]
                torch.testing.assert_close((1-a)*diagonal + a*replay,
                                           torch.tensor(snapshot["corrected_teacher_log_probs"]))
                self.assertEqual(snapshot["diagonal_privileged_prompt"]["token_ids"], [104, 204, 304])
                self.assertEqual(snapshot["sources"][0]["privileged_prompt"]["token_ids"], [101, 201, 301])
                self.assertGreater(sum(snapshot["region_masks"]["action"]), 0)
                self.assertEqual(records[0]["regions"]["response"]["count"], 0)
                self.assertTrue(records[1]["cross_is_diagonal_fallback"])
                self.assertFalse(records[3]["cross_is_diagonal_fallback"])
                for key, original in before.items():
                    torch.testing.assert_close(combined[key], original)
                self.assertTrue(torch.equal(old_rng, torch.random.get_rng_state()))
                self.assertIsNone(student.grad)

    def test_per_turn_counts_pool_to_full_paired_statistics(self):
        result, student, regions, *_ = fixture()
        records = audit.turn_gap_records(result, student, regions, 1e-4)
        matched = torch.tensor([False, False, False, True])[:, None]
        for name, mask in {"response": result.policy_loss_mask, **regions}.items():
            stats = diagnostics.paired_gap_statistics(
                result.diagonal_token_gap, result.cross_teacher_log_probs-student,
                result.corrected_token_gap, mask & result.policy_loss_mask & matched, 1e-4)
            saved = records[3]["regions"][name]
            self.assertEqual(saved["count"], stats["token_count"])
            self.assertAlmostEqual(saved["cross_mean"], stats["cross_gap_mean"], places=6)
            for event in ("neg_to_pos", "pos_to_neg", "increase", "decrease"):
                self.assertEqual(saved[f"tol/cross_{event}_count"], stats[f"tol/cross_{event}_count"])

    def test_token_and_context_caps_keep_original_positions_and_flag_truncation(self):
        result, student, regions, combined, cross, teacher, batch = fixture()
        snapshot = audit.token_snapshot(
            index=3, reason="test", result=result, batch=batch, teacher_batch=teacher,
            student_log_probs=student, combined=combined, cross_scores=cross, region_masks=regions,
            logger=SimpleNamespace(max_tokens=7, max_context_tokens=2))
        self.assertEqual(snapshot["positions"], list(range(1, 8)))
        self.assertTrue(snapshot["tokens_truncated"])
        self.assertEqual(snapshot["diagonal_privileged_prompt"]["token_ids"], [204, 304])
        self.assertTrue(snapshot["diagonal_privileged_prompt"]["truncated"])
        for name in ("token_ids", "policy_mask", "student_log_probs", "cross_teacher_log_probs"):
            self.assertEqual(len(snapshot[name]), 7)

    def test_empty_source_snapshot_does_not_invent_cross_evidence(self):
        result, student, regions, combined, cross, teacher, batch = fixture()
        snapshot = audit.token_snapshot(
            index=1, reason="fallback", result=result, batch=batch, teacher_batch=teacher,
            student_log_probs=student, combined=combined, cross_scores=cross, region_masks=regions,
            logger=SimpleNamespace(max_tokens=512, max_context_tokens=4096))
        self.assertEqual(snapshot["sources"], [])
        self.assertEqual(snapshot["alpha"], 0.)
        self.assertEqual(snapshot["cross_teacher_log_probs"], snapshot["diagonal_teacher_log_probs"])

    def test_nonfinite_and_empty_region_have_explicit_denominators(self):
        result, student, regions, *_ = fixture()
        result.diagonal_token_gap[3, 2] = float("nan")
        result.corrected_token_gap[3, 3] = float("inf")
        records = audit.turn_gap_records(result, student, {"empty": torch.zeros_like(regions["action"])}, 1e-4)
        self.assertEqual(records[3]["regions"]["response"]["nonfinite_count"], 2)
        self.assertEqual(records[3]["regions"]["empty"]["count"], 0)
        json.dumps(audit.json_safe(records), allow_nan=False)

    def test_selection_is_bounded_deterministic_and_consumes_no_global_rng(self):
        result, student, regions, *_ = fixture()
        records = audit.turn_gap_records(result, student, regions, 1e-4)
        original = copy.deepcopy(records)
        py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
        for limit in (1, 2, 8):
            selected = audit.choose_sample_turns(records, limit, 10)
            self.assertEqual(selected, audit.choose_sample_turns(records, limit, 10))
            self.assertLessEqual(len(selected), limit)
            self.assertEqual(len({i for i, _ in selected}), len(selected))
        self.assertEqual(records, original)
        self.assertEqual(random.getstate(), py_state)
        self.assertTrue(np.array_equal(np_state[1], np.random.get_state()[1]))
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))

    def test_credit_records_preserve_sign_residual_and_boundary_identity(self):
        records = [{}, {}, {}]
        fields = ("profile_valid", "js_divergence", "boundary_valid", "turn_gap", "outcome_aligned_score",
                  "within_segment_turn_budget", "raw_credit_density", "projected_credit_density", "turn_weight")
        credit = SimpleNamespace(**{k: torch.tensor([.5, 1., 1.5]) for k in fields},
            segment_index_per_turn=torch.tensor([0, 0, 1]), segment_budget=torch.tensor([.6, .4]),
            segmentation_threshold=float("inf"), trajectory_segmentations=(SimpleNamespace(
                turn_indices=(0, 1, 2), semantic_boundaries=(1,),
                correspondence_guided_fallback_boundaries=(), forced_max_boundaries=(2,)),))
        original = credit.turn_weight.clone()
        audit.attach_credit(records, credit, torch.zeros(3), -torch.ones(3), torch.tensor([0., 0., -.2]))
        self.assertEqual(records[1]["credit"]["boundary_type"], "semantic")
        self.assertEqual(records[2]["credit"]["boundary_type"], "forced_max")
        self.assertEqual(records[0]["credit"]["trajectory_advantage"], -1.)
        self.assertLess(records[2]["credit"]["invalid_action_residual"], 0.)
        self.assertIsNone(audit.json_safe(records)[0]["credit"]["segmentation_threshold"])
        torch.testing.assert_close(credit.turn_weight, original)


class AuditWriterTests(unittest.TestCase):
    def logger(self, directory, **overrides):
        return audit.RetrospectiveLogger({"directory": directory, **overrides},
            resolved_config={"trainer": {"experiment_name": "../test/run"}, "api_key": "secret-value"},
            restored_step=37, code_root=ROOT)

    def test_resume_creates_unique_directories_and_records_config(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = self.logger(directory), self.logger(directory)
            self.assertNotEqual(first.directory, second.directory)
            self.assertTrue(first.directory.is_relative_to(directory))
            manifest = json.loads((first.directory / "manifest.jsonl").read_text())
            self.assertEqual(manifest["restored_step"], 37)
            self.assertEqual(manifest["config"]["api_key"], "[REDACTED]")
            self.assertIn("verl/trainer/ppo/retrospective_logging.py", manifest["code_sha256"])
            self.assertTrue(first.due(38))
            self.assertTrue(first.due(40))
            self.assertFalse(first.due(39))
            self.assertTrue(first.due(39, last=True))

    def test_disk_cap_drops_details_but_keeps_full_precision_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = self.logger(directory)
            logger.byte_cap = 100
            self.assertFalse(logger.write("samples", {"text": "x" * 200}, detail=True))
            self.assertEqual(logger.dropped_records, 1)
            tiny = 1.23456789012345e-8
            self.assertTrue(logger.write("metrics", {"value": tiny, "nan": float("nan")}))
            saved = json.loads((logger.directory / "metrics.jsonl").read_text())
            self.assertEqual(saved["value"], tiny)
            self.assertIsNone(saved["nan"])
            self.assertLessEqual(logger.detail_bytes, logger.byte_cap)

    def test_disabled_writer_creates_no_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = self.logger(directory, enabled=False)
            self.assertFalse(logger.due(38))
            self.assertFalse(logger.write("metrics", {"value": 1}))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_io_failure_is_reported_not_propagated(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = self.logger(directory)
            with patch.object(Path, "open", side_effect=OSError("simulated full disk")):
                with self.assertWarnsRegex(RuntimeWarning, "simulated full disk"):
                    self.assertFalse(logger.write("samples", {"value": 1}, detail=True))
            self.assertEqual(logger.metrics()["audit/errors"], 1.)
            self.assertEqual(logger.detail_bytes, 0)

    def test_logging_limits_are_rejected_before_distributed_startup(self):
        path = ROOT / "verl/trainer/main_beyond_timestamps.py"
        fn = next(node for node in ast.parse(path.read_text()).body
                  if isinstance(node, ast.FunctionDef) and node.name == "_validate_beyond_timestamps_config")
        # Exercise the actual diagnostic validation prefix without Ray/Hydra.
        stop = next(i for i, node in enumerate(fn.body)
                    if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "teacher_budget" for t in node.targets))
        fn.body = fn.body[:stop]
        scope = {"math": math}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), scope)
        validate = scope["_validate_beyond_timestamps_config"]
        for key in ("sample_interval", "sample_turns", "sample_tokens", "context_tokens", "max_detail_mb"):
            for value in (0, -1, True, 1.5, "10"):
                cfg = SimpleNamespace(algorithm={"thinking_correspondence": {
                    "diagnostics": {"retrospective": {key: value}}}})
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    validate(cfg)
            validate(SimpleNamespace(algorithm={"thinking_correspondence": {
                "diagnostics": {"retrospective": {key: 10}}}}))

    def test_region_cache_does_not_repeat_tokenization_or_change_statistics(self):
        result, student, _, combined, *_ = fixture()
        cache = {}
        tokenizer = CharacterTokenizer()
        args = dict(tokenizer=tokenizer,
                    responses=fixture()[-1].batch["responses"],
                    sampled_response_mask=result.sampled_response_mask,
                    policy_loss_mask=result.policy_loss_mask,
                    selected_source_indices=result.selected_source_indices,
                    combined=combined, student_log_probs=student)
        baseline = diagnostics.build_rectification_diagnostics(**args)
        cached = diagnostics.build_rectification_diagnostics(**args, region_cache=cache)
        self.assertEqual(baseline, cached)
        self.assertIn("action", cache["masks"])
        self.assertEqual(cache["counts"]["exact_roundtrip_rows"], 4)

    def test_shared_launchers_and_hook_order_are_unchanged(self):
        for bench in ("alfworld", "webshop", "search"):
            launcher = (ROOT / f"examples/beyond_timestamps_trainer/run_{bench}_3b.sh").read_text()
            self.assertIn("python3 -m verl.trainer.main_beyond_timestamps", launcher)
        loop = (ROOT / "verl/trainer/ppo/beyond_timestamps_ray_trainer.py").read_text()
        self.assertLess(loop.index('audit.write("turns"'), loop.index("self.last_mechanism1_result = None"))
        self.assertLess(loop.index("self.last_mechanism1_result = None"), loop.index("self.actor_rollout_wg.update_actor"))
        # All edited Python files parse without importing distributed dependencies.
        for name in ("retrospective_logging", "rectification_diagnostics", "thinking_correspondence_trainer",
                     "beyond_timestamps_ray_trainer"):
            ast.parse((ROOT / f"verl/trainer/ppo/{name}.py").read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
