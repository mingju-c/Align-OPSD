"""CPU-only unittest coverage; no Ray, CUDA, model download, or pytest required.

Run: python tests/test_rectification_probability_mixture.py
Load the dependency-light production modules directly, avoiding verl's optional
distributed package initialization in minimal analysis environments.
"""

import ast
import importlib.util
import math
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = load_module("_rectification_core", "verl/trainer/ppo/thinking_correspondence.py")
diagnostics = load_module("_rectification_diagnostics", "verl/trainer/ppo/rectification_diagnostics.py")


class CharacterTokenizer:
    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text],
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def text_batch(*texts):
    length = max(map(len, texts), default=0) + 2
    ids = torch.zeros(len(texts), length, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, text in enumerate(texts):
        ids[i, 1:1 + len(text)] = torch.tensor([ord(c) for c in text])
        mask[i, 1:1 + len(text)] = True
    return ids, mask


class AggregationTests(unittest.TestCase):
    def example(self, mode="probability_mixture", dtype=torch.float32, alpha_max=.2):
        p = torch.tensor([[.70, .80, .95], [.10, .30, .90], [.50, .70, .80]])
        diagonal = torch.tensor([[.38, .75, .92]] * 4).log().to(dtype)
        student = torch.tensor([[.40, .80, .90]] * 4).log().to(dtype)
        weights = torch.tensor([.5, .3, .2])
        h = torch.zeros(4, 4)
        h[:3, 3] = .95 + .1 * weights.log()
        cross = {(k, 3): p[k].log().to(dtype) for k in range(3)}
        out = core.combine_cross_teacher_scores(
            diagonal, student, h, ((), (), (), (0, 1, 2)), cross,
            aggregation_mode=mode, alpha_max=alpha_max,
        )
        return out, diagonal, student, p, weights

    def test_probability_mixture_matches_hand_calculation(self):
        out, diagonal, student, p, w = self.example()
        expected = (w[:, None] * p).sum(0).log()
        torch.testing.assert_close(out["cross_teacher_log_probs"][3], expected)
        self.assertAlmostEqual(float(expected[0].exp()), .48, places=6)
        cross_gap = expected - student[3]
        self.assertLess(float(out["diagonal_token_gap"][3, 0]), 0)
        self.assertGreater(float(cross_gap[0]), 0)
        alpha = out["correction_strength"][3]
        torch.testing.assert_close(out["corrected_teacher_log_probs"][3],
                                   (1 - alpha) * diagonal[3] + alpha * expected)

    def test_legacy_option_preserves_weighted_log_formula(self):
        out, _, _, p, w = self.example("log_mean")
        expected = (w[:, None] * p.log()).sum(0)
        torch.testing.assert_close(out["cross_teacher_log_probs"][3], expected)
        self.assertAlmostEqual(float(expected[0].exp()), .36504216, places=6)

    def test_both_modes_have_identical_retrieval_and_confidence(self):
        new, *_ = self.example()
        old, *_ = self.example("log_mean")
        torch.testing.assert_close(new["correspondence_confidence"], old["correspondence_confidence"])
        torch.testing.assert_close(new["correction_strength"], old["correction_strength"])
        for a, b in zip(new["selected_source_weights"], old["selected_source_weights"]):
            torch.testing.assert_close(a, b)

    def test_fp32_and_bfloat16_inputs_have_finite_detached_fp32_scores(self):
        for dtype in (torch.float32, torch.bfloat16):
            out, diagonal, *_ = self.example(dtype=dtype)
            for key in ("cross_teacher_log_probs", "corrected_teacher_log_probs", "corrected_token_gap"):
                self.assertEqual(out[key].dtype, torch.float32)
                self.assertFalse(out[key].requires_grad)
                self.assertTrue(bool(torch.isfinite(out[key]).all()))
            torch.testing.assert_close(out["corrected_teacher_log_probs"][:3], diagonal.float()[:3])

    def test_probability_score_is_bounded_by_sources_and_above_log_mean(self):
        torch.manual_seed(23)
        for _ in range(10):
            diagonal = -torch.rand(4, 20) * 10
            sources = -torch.rand(3, 20) * 40
            h = torch.rand(4, 4) * .3 + .7
            out = core.combine_cross_teacher_scores(
                diagonal, diagonal, h, ((), (), (), (0, 1, 2)),
                {(i, 3): sources[i] for i in range(3)},
            )
            c = out["cross_teacher_log_probs"][3]
            self.assertTrue(bool((c >= sources.min(0).values - 1e-5).all()))
            self.assertTrue(bool((c <= sources.max(0).values + 1e-5).all()))
            self.assertTrue(bool((c >= out["log_mean_cross_teacher_log_probs"][3] - 1e-5).all()))

    def test_equal_sources_do_not_multiply_scores_by_k(self):
        score = torch.tensor([-2., -10000.])
        for k in (1, 2, 3):
            out = core.combine_cross_teacher_scores(
                torch.zeros(4, 2), torch.zeros(4, 2), torch.ones(4, 4),
                ((), (), (), tuple(range(k))), {(i, 3): score for i in range(k)},
            )
            torch.testing.assert_close(out["cross_teacher_log_probs"][3], score)

    def test_extreme_probabilities_and_weight_temperature_are_stable(self):
        h = torch.ones(3, 3)
        h[0, 2] = .7
        out = core.combine_cross_teacher_scores(
            -torch.ones(3, 2), -torch.ones(3, 2), h, ((), (), (0, 1)),
            {(0, 2): torch.tensor([-10000., -20000.]),
             (1, 2): torch.tensor([-11000., -21000.])},
            aggregation_temperature=1e-5,
        )
        self.assertTrue(bool(torch.isfinite(out["cross_teacher_log_probs"]).all()))

    def test_probability_mixture_is_normalized_when_given_whole_vocab(self):
        # The helper treats columns independently; feeding all vocabulary entries
        # at one fixed prefix verifies the implied mixture's normalization.
        p = torch.tensor([[.7, .2, .1], [.1, .3, .6], [.5, .4, .1]])
        out = core.combine_cross_teacher_scores(
            torch.zeros(4, 3), torch.zeros(4, 3), torch.ones(4, 4),
            ((), (), (), (0, 1, 2)), {(i, 3): p[i].log() for i in range(3)},
        )
        self.assertAlmostEqual(float(out["cross_teacher_log_probs"][3].exp().sum()), 1., places=6)

    def test_zero_alpha_and_full_confidence_endpoints(self):
        out, diagonal, *_ = self.example(alpha_max=0)
        torch.testing.assert_close(out["corrected_teacher_log_probs"], diagonal)
        out = core.combine_cross_teacher_scores(
            -torch.ones(2, 2), -torch.ones(2, 2), torch.ones(2, 2),
            ((), (0,)), {(0, 1): torch.tensor([-2., -3.])}, alpha_max=1.,
        )
        torch.testing.assert_close(out["corrected_teacher_log_probs"][1], torch.tensor([-2., -3.]))

    def test_scoring_inputs_and_teacher_gradients_are_untouched(self):
        d = torch.tensor([[-1.], [-2.]], requires_grad=True)
        s = torch.tensor([[-2.], [-3.]], requires_grad=True)
        c = torch.tensor([-4.], requires_grad=True)
        out = core.combine_cross_teacher_scores(d, s, torch.ones(2, 2), ((), (0,)), {(0, 1): c})
        self.assertFalse(out["corrected_token_gap"].requires_grad)
        self.assertIsNone(d.grad)
        self.assertIsNone(s.grad)
        self.assertIsNone(c.grad)

    def test_invalid_mode_fails_instead_of_silently_reverting(self):
        with self.assertRaisesRegex(ValueError, "aggregation_mode"):
            self.example("typo")


class RegionTests(unittest.TestCase):
    def decode_mask(self, ids, mask):
        return CharacterTokenizer().decode(ids[0, mask[0]].tolist())

    def test_all_action_protocols_unicode_and_padding(self):
        for tag in ("action", "search", "answer"):
            text = f"<think>核验尺寸</think><{tag}>inspect dimensions</{tag}>EOS"
            ids, sampled = text_batch(text)
            masks, counts = diagnostics.response_region_masks(CharacterTokenizer(), ids, sampled)
            self.assertEqual(self.decode_mask(ids, masks["thinking"]), "核验尺寸")
            self.assertEqual(self.decode_mask(ids, masks["action"]), "inspect dimensions")
            self.assertEqual(counts["exact_roundtrip_rows"], 1)
            self.assertFalse(bool((masks["thinking"] & masks["action"]).any()))
            self.assertFalse(bool(((masks["thinking"] | masks["action"]) & ~sampled).any()))

    def test_case_insensitive_tags_and_action_without_thinking(self):
        ids, sampled = text_batch("<ACTION>buy</ACTION>")
        masks, _ = diagnostics.response_region_masks(CharacterTokenizer(), ids, sampled)
        self.assertEqual(self.decode_mask(ids, masks["action"]), "buy")
        self.assertEqual(int(masks["thinking"].sum()), 0)

    def test_incomplete_duplicate_and_nested_blocks_are_not_guessed(self):
        for text in ("<think>unfinished", "<action>a</action><action>b</action>",
                     "<think><action>a</action></think>"):
            ids, sampled = text_batch(text)
            masks, counts = diagnostics.response_region_masks(CharacterTokenizer(), ids, sampled)
            self.assertEqual(counts["malformed_region_rows"], 1)
            self.assertEqual(int(masks["thinking"].sum() + masks["action"].sum()), 0)

    def test_roundtrip_mismatch_is_reported(self):
        class WrongTokenizer(CharacterTokenizer):
            def __call__(self, text, **kwargs):
                result = super().__call__(text, **kwargs)
                result["input_ids"][0] += 1
                return result
        ids, sampled = text_batch("<action>buy</action>")
        masks, counts = diagnostics.response_region_masks(WrongTokenizer(), ids, sampled)
        self.assertEqual(counts["token_roundtrip_mismatch_rows"], 1)
        self.assertEqual(int(masks["action"].sum()), 0)

    def test_unsupported_offsets_do_not_crash_training(self):
        class SlowTokenizer(CharacterTokenizer):
            def __call__(self, text, **kwargs):
                raise NotImplementedError("no offsets")
        ids, sampled = text_batch("<action>buy</action>")
        _, counts = diagnostics.response_region_masks(SlowTokenizer(), ids, sampled)
        self.assertEqual(counts["offset_unavailable_rows"], 1)

    def test_boundary_straddling_tokens_are_excluded(self):
        class Pieces:
            pieces = ["<action>", "buy", " now</action>"]
            def decode(self, ids, **kwargs):
                return "".join(self.pieces[i] for i in ids)
            def __call__(self, text, **kwargs):
                return {"input_ids": [0, 1, 2], "offset_mapping": [(0, 8), (8, 11), (11, 24)]}
        masks, _ = diagnostics.response_region_masks(Pieces(), torch.tensor([[0, 1, 2]]), torch.ones(1, 3))
        self.assertEqual(masks["action"].tolist(), [[False, True, False]])


class PairedStatisticsTests(unittest.TestCase):
    def test_strict_tolerance_and_conditional_denominators(self):
        d = torch.tensor([[-1., 1., -1., 1., 0., 1e-6]])
        c = torch.tensor([[1., -1., -2., 2., 1., -1e-6]])
        r = .8 * d + .2 * c
        s = diagnostics.paired_gap_statistics(d, c, r, torch.ones_like(d), sign_epsilon=1e-4)
        self.assertEqual(s["token_count"], 6)
        self.assertEqual(s["strict/cross_neg_to_pos_count"], 1)
        self.assertEqual(s["strict/cross_pos_to_neg_count"], 2)
        self.assertEqual(s["strict/cross_neg_to_pos_given_negative"], .5)
        self.assertEqual(s["strict/cross_pos_to_neg_given_positive"], 2 / 3)
        self.assertEqual(s["tol/cross_pos_to_neg_count"], 1)
        self.assertEqual(s["tol/diagonal_neutral_count"], 2)
        self.assertEqual(s["strict/corrected_neg_to_pos_count"], 0)

    def test_empty_scope_is_finite_with_explicit_zero_denominators(self):
        x = torch.ones(1, 2)
        s = diagnostics.paired_gap_statistics(x, x, x, torch.zeros_like(x))
        self.assertEqual(s["token_count"], 0)
        self.assertTrue(all(math.isfinite(v) for v in s.values()))

    def test_nonfinite_pairs_are_counted_and_excluded(self):
        d = torch.tensor([[-1., float("nan"), 1.]])
        c = torch.tensor([[1., 1., -1.]])
        s = diagnostics.paired_gap_statistics(d, c, c, torch.ones_like(c))
        self.assertEqual(s["nonfinite_token_count"], 1)
        self.assertEqual(s["token_count"], 2)

    def test_invalid_epsilon_is_rejected(self):
        x = torch.ones(1, 2)
        for eps in (-1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                diagnostics.paired_gap_statistics(x, x, x, x, sign_epsilon=eps)

    def test_end_to_end_regions_matched_mask_and_no_side_effects(self):
        ids, sampled = text_batch("<think>abc</think><action>xy</action>", "<think>d</think><action>z</action>")
        policy = sampled.clone()
        # Deliberately exclude one action token from the actor loss.
        policy[0, 1 + len("<think>abc</think><action>")] = False
        student = torch.full(ids.shape, -2.)
        diagonal = torch.full(ids.shape, -2.1)
        sources = ((1,), ())
        combined = core.combine_cross_teacher_scores(
            diagonal, student, torch.ones(2, 2), sources,
            {(1, 0): torch.full((ids.shape[1],), -1.8)},
        )
        original_gap = combined["corrected_token_gap"].clone()
        stats = diagnostics.build_rectification_diagnostics(
            tokenizer=CharacterTokenizer(), responses=ids, sampled_response_mask=sampled,
            policy_loss_mask=policy, selected_source_indices=sources, combined=combined,
            student_log_probs=student,
        )
        n = int(policy[0].sum())
        self.assertEqual(stats["paired/response/token_count"], n)
        self.assertEqual(stats["paired/response/strict/cross_neg_to_pos_count"], n)
        self.assertEqual(stats["paired/response/strict/corrected_neg_to_pos_count"], 0)
        self.assertEqual(stats["paired/thinking/token_count"], 3)
        self.assertEqual(stats["paired/action/token_count"], 1)
        self.assertEqual(stats["paired/unmatched_policy_token_count"], int(policy[1].sum()))
        self.assertEqual(stats["regions/paired_other_token_count"] + 3 + 1, n)
        torch.testing.assert_close(combined["corrected_token_gap"], original_gap)

    def test_region_breakdown_can_be_disabled_without_tokenizer(self):
        x = torch.full((2, 3), -1.)
        out = core.combine_cross_teacher_scores(x, x, torch.ones(2, 2), ((), (0,)), {(0, 1): x[0]})
        stats = diagnostics.build_rectification_diagnostics(
            tokenizer=None, responses=torch.ones(2, 3, dtype=torch.long),
            sampled_response_mask=torch.ones(2, 3), policy_loss_mask=torch.ones(2, 3),
            selected_source_indices=((), (0,)), combined=out, student_log_probs=x,
            region_breakdown=False,
        )
        self.assertEqual(stats["paired/response/token_count"], 3)
        self.assertFalse(any(k.startswith("paired/action/") for k in stats))


class ConfigContractTests(unittest.TestCase):
    def test_real_entry_validation_accepts_both_modes_and_rejects_invalid_values(self):
        class Node(dict):
            def __getattr__(self, key):
                return self[key]
        source = ROOT / "verl/trainer/main_beyond_timestamps.py"
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "_validate_beyond_timestamps_config")
        scope = {"math": math}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
        validate = scope["_validate_beyond_timestamps_config"]
        m1 = Node(enabled=True, consumer="mechanism2", diagnostics=Node(sign_epsilon=1e-4))
        cfg = Node(
            algorithm=Node(thinking_correspondence=m1, mechanism2=Node(enabled=True),
                           adv_estimator="grpo", use_kl_in_reward=False),
            actor_rollout_ref=Node(actor=Node(strategy="fsdp", loss_agg_mode="token-mean")),
            env=Node(rollout=Node(n=8)),
        )
        for mode in ("probability_mixture", "log_mean"):
            m1["aggregation_mode"] = mode
            validate(cfg)
        m1["aggregation_mode"] = "typo"
        with self.assertRaisesRegex(ValueError, "aggregation_mode"):
            validate(cfg)
        m1["aggregation_mode"] = "probability_mixture"
        for eps in (-.1, float("nan"), float("inf")):
            m1["diagnostics"]["sign_epsilon"] = eps
            with self.assertRaisesRegex(ValueError, "sign_epsilon"):
                validate(cfg)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
