import math

import torch

from verl.trainer.ppo.correspondence_credit import (
    CorrespondenceCreditConfig,
    adaptive_segmentation_threshold,
    aggregate_policy_masked_turn_gap,
    build_dense_correspondence_profiles,
    compute_adjacent_jsd,
    compute_correspondence_credit,
    compute_correspondence_credit_from_tensors,
    compute_invalid_action_advantage_residual,
    compute_unique_trajectory_grpo_advantage,
    cosine_profile_distance,
    normalized_entropy_distance,
    normalized_js_divergence,
    profile_distance,
    project_weighted_box_budget,
    segment_trajectories,
)
from verl.trainer.ppo.thinking_correspondence import (
    Mechanism1Result,
    TurnIdentity,
    canonicalize_turns,
)


def _identities(turns_per_rollout=(3, 3)):
    identities = []
    for rollout_index, num_turns in enumerate(turns_per_rollout):
        identities.extend(
            TurnIdentity("task", f"rollout-{rollout_index}", turn)
            for turn in range(num_turns)
        )
    return tuple(identities)


def _cross_rollout_mask(identities):
    return torch.tensor(
        [
            [
                source.task_id == target.task_id and source.rollout_id != target.rollout_id
                for target in identities
            ]
            for source in identities
        ],
        dtype=torch.bool,
    )


def _base_h(identities, value=0.8):
    h = torch.zeros((len(identities), len(identities)), dtype=torch.float32)
    mask = _cross_rollout_mask(identities)
    h[mask] = value
    return h, mask


def test_dense_profiles_use_h_columns_and_all_eligible_sources():
    # Sources 0/1 are eligible for target 2.  Source 0 is below gamma but must
    # remain in the dense profile; gamma controls validity only.
    h = torch.tensor(
        [
            [0.0, 0.0, 0.60],
            [0.0, 0.0, 0.90],
            [0.0, 0.0, 0.00],
        ],
        dtype=torch.float32,
    )
    mask = torch.zeros_like(h, dtype=torch.bool)
    mask[0, 2] = True
    mask[1, 2] = True
    profiles, valid = build_dense_correspondence_profiles(h, mask, 0.70, 0.10)

    expected = torch.softmax(torch.tensor([0.60, 0.90]) / 0.10, dim=0)
    assert torch.allclose(profiles[2, :2], expected)
    assert profiles[2, 0] > 0  # no threshold/top-K truncation
    assert profiles[2].sum() == 1
    assert valid.tolist() == [False, False, True]
    assert torch.equal(profiles[:2], torch.zeros_like(profiles[:2]))


def test_profile_can_exist_but_be_invalid_on_absolute_similarity():
    h = torch.tensor([[0.0, 0.3], [0.4, 0.0]], dtype=torch.float32)
    mask = torch.tensor([[False, True], [True, False]])
    profiles, valid = build_dense_correspondence_profiles(h, mask, 0.70, 0.10)
    assert torch.allclose(profiles.sum(dim=-1), torch.ones(2))
    assert not valid.any()


def test_normalized_jsd_is_symmetric_zero_for_equal_and_one_for_disjoint():
    equal = torch.tensor([0.2, 0.8, 0.0])
    left = torch.tensor([1.0, 0.0])
    right = torch.tensor([0.0, 1.0])
    assert normalized_js_divergence(equal, equal).item() == 0.0
    assert torch.allclose(
        normalized_js_divergence(left, right),
        normalized_js_divergence(right, left),
    )
    assert torch.allclose(normalized_js_divergence(left, right), torch.tensor(1.0))


def test_optional_profile_distances_have_expected_ranges_and_dispatch():
    uniform = torch.tensor([0.25, 0.25, 0.25, 0.25])
    spike = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert normalized_entropy_distance(uniform, uniform).item() == 0.0
    assert torch.allclose(normalized_entropy_distance(uniform, spike), torch.tensor(1.0))
    assert cosine_profile_distance(uniform, uniform).item() < 1e-6
    assert torch.allclose(
        cosine_profile_distance(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])),
        torch.tensor(1.0),
    )
    for mode in ("jsd", "entropy", "cosine"):
        assert torch.allclose(profile_distance(uniform, spike, mode),
                              {"jsd": normalized_js_divergence,
                               "entropy": normalized_entropy_distance,
                               "cosine": cosine_profile_distance}[mode](uniform, spike))
    try:
        CorrespondenceCreditConfig(profile_distance="invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid profile distance accepted")


def test_mask_change_or_invalid_profile_cannot_create_semantic_jsd():
    identities = (
        TurnIdentity("task", "rollout", 0),
        TurnIdentity("task", "rollout", 1),
        TurnIdentity("task", "rollout", 2),
    )
    profiles = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    valid = torch.tensor([True, True, False])
    structural = torch.tensor(
        [[True, False, False], [False, True, False], [False, False, True]]
    )
    jsd, boundary_valid, _, stats = compute_adjacent_jsd(
        profiles, valid, structural, identities
    )
    assert not boundary_valid.any()
    assert torch.equal(jsd, torch.zeros(3))
    assert stats["structural_mask_change_rate"] == 1.0


def test_batch_adaptive_quantile_is_clipped_and_empty_disables_semantics():
    jsd = torch.tensor([0.0, 0.001, 0.5, 0.9])
    valid = torch.tensor([False, True, True, True])
    assert adaptive_segmentation_threshold(jsd, valid, 0.5, 0.01, 0.10) == 0.10
    assert math.isinf(
        adaptive_segmentation_threshold(jsd, torch.zeros(4, dtype=torch.bool), 0.8, 0.01, 0.10)
    )


def test_segmentation_classifies_semantic_guided_and_forced_boundaries():
    identities = tuple(TurnIdentity("task", "rollout", turn) for turn in range(12))
    groups = (tuple(range(12)),)
    jsd = torch.zeros(12)
    valid = torch.zeros(12, dtype=torch.bool)
    jsd[3], valid[3] = 0.9, True
    jsd[7], valid[7] = 0.4, True

    trajectories, raw_candidates, accepted_candidates = segment_trajectories(
        identities,
        groups,
        jsd,
        valid,
        segmentation_threshold=0.5,
        min_segment_turns=2,
        max_segment_turns=4,
        fallback_jsd_min=0.1,
    )
    segmentation = trajectories[0]
    assert raw_candidates == accepted_candidates == 1
    assert segmentation.semantic_boundaries == (3,)
    assert segmentation.correspondence_guided_fallback_boundaries == (7,)
    assert segmentation.forced_max_boundaries == (9,)
    assert [(segment.start, segment.end) for segment in segmentation.segments] == [
        (0, 3),
        (3, 7),
        (7, 9),
        (9, 12),
    ]
    covered = tuple(index for segment in segmentation.segments for index in segment.canonical_turn_indices)
    assert covered == tuple(range(12))
    assert all(segment.num_turns <= 4 for segment in segmentation.segments)


def test_semantic_candidates_are_accepted_strongest_first_under_lmin():
    identities = tuple(TurnIdentity("task", "rollout", turn) for turn in range(7))
    jsd = torch.zeros(7)
    valid = torch.zeros(7, dtype=torch.bool)
    jsd[2], valid[2] = 0.6, True
    jsd[3], valid[3] = 0.9, True
    trajectories, _, _ = segment_trajectories(
        identities,
        (tuple(range(7)),),
        jsd,
        valid,
        segmentation_threshold=0.5,
        min_segment_turns=2,
        max_segment_turns=7,
        fallback_jsd_min=0.1,
    )
    # Accepting t=2 first would suppress the stronger t=3 candidate.
    assert trajectories[0].semantic_boundaries == (3,)


def test_policy_mask_turn_gap_is_mean_not_sum_and_rejects_empty_turn():
    gap = torch.tensor([[1.0, 3.0, 100.0], [-2.0, 2.0, 9.0]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)
    turn_gap, lengths = aggregate_policy_masked_turn_gap(gap, mask)
    assert torch.equal(lengths, torch.tensor([2, 2]))
    assert torch.allclose(turn_gap, torch.tensor([2.0, 0.0]))
    try:
        aggregate_policy_masked_turn_gap(gap, torch.tensor([[1, 0, 0], [0, 0, 0]]))
    except ValueError as error:
        assert "policy-loss token" in str(error)
    else:
        raise AssertionError("an empty policy-loss turn must be rejected")


def test_weighted_box_projection_preserves_budget_instead_of_plain_clamp():
    counts = torch.tensor([1.0, 9.0])
    raw = torch.tensor([5.0, 5.0 / 9.0])  # weighted budget = 10
    projected = project_weighted_box_budget(raw, counts, density_upper_bound=2.0)
    assert projected.min() >= 0
    assert projected.max() <= 2.0 + 1e-6
    assert torch.allclose((counts * projected).sum(), counts.sum(), atol=1e-5)
    assert not torch.allclose(projected, raw.clamp(max=2.0))


def test_full_hierarchy_uniform_evidence_reduces_exactly_to_grpo_for_any_lengths():
    identities = _identities((3, 3))
    h, structural = _base_h(identities)
    gap = torch.full((6, 4), 2.0)
    mask = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    advantage = torch.tensor([1.5, 1.5, 1.5, -2.0, -2.0, -2.0])
    result = compute_correspondence_credit_from_tensors(
        thinking_similarity=h,
        structural_mask=structural,
        similarity_threshold=0.7,
        corrected_token_gap=gap,
        policy_loss_mask=mask,
        identities=identities,
        trajectory_advantage=advantage,
        config=CorrespondenceCreditConfig(
            min_segment_turns=1,
            max_segment_turns=2,  # force splits; result must still be uniform
            mixing_coefficient=0.75,
        ),
    )
    assert torch.allclose(result.turn_weight, torch.ones(6), atol=1e-6)
    assert torch.allclose(result.projected_credit_density, torch.ones(6), atol=1e-6)
    assert torch.allclose(result.token_advantage, mask * advantage.unsqueeze(-1), atol=1e-6)
    assert result.diagnostics["credit_conservation_error_max"] <= 1e-5


def test_outcome_alignment_reverses_density_preference_for_negative_trajectory():
    identities = _identities((2, 2))
    h, structural = _base_h(identities)
    gap = torch.tensor([[3.0], [0.0], [3.0], [0.0]])
    mask = torch.ones_like(gap, dtype=torch.bool)
    advantage = torch.tensor([1.0, 1.0, -1.0, -1.0])
    result = compute_correspondence_credit_from_tensors(
        thinking_similarity=h,
        structural_mask=structural,
        similarity_threshold=0.7,
        corrected_token_gap=gap,
        policy_loss_mask=mask,
        identities=identities,
        trajectory_advantage=advantage,
        config=CorrespondenceCreditConfig(
            min_segment_turns=1,
            max_segment_turns=2,
            credit_temperature=1.0,
            mixing_coefficient=1.0,
            density_upper_bound=4.0,
        ),
    )
    assert result.outcome_aligned_score.tolist() == [3.0, 0.0, -3.0, -0.0]
    assert result.turn_weight[0] > result.turn_weight[1]
    assert result.turn_weight[2] < result.turn_weight[3]
    for trajectory in result.trajectory_segmentations:
        indices = torch.tensor(trajectory.turn_indices)
        assert torch.allclose(result.turn_weight[indices].sum(), torch.tensor(2.0), atol=1e-5)


def test_zero_advantage_gives_uniform_weights_and_zero_m2_advantage():
    identities = _identities((2, 2))
    h, structural = _base_h(identities)
    gap = torch.tensor([[100.0], [-100.0], [4.0], [-3.0]])
    result = compute_correspondence_credit_from_tensors(
        thinking_similarity=h,
        structural_mask=structural,
        similarity_threshold=0.7,
        corrected_token_gap=gap,
        policy_loss_mask=torch.ones_like(gap, dtype=torch.bool),
        identities=identities,
        trajectory_advantage=torch.zeros(4),
        config=CorrespondenceCreditConfig(min_segment_turns=1, max_segment_turns=2),
    )
    assert torch.allclose(result.turn_weight, torch.ones(4))
    assert torch.equal(result.token_advantage, torch.zeros_like(gap))


def test_full_result_is_detached_bounded_and_budget_conserving_after_projection():
    identities = _identities((3, 3))
    h, structural = _base_h(identities)
    h.requires_grad_(True)
    gap = torch.tensor([[20.0], [-20.0], [-20.0], [20.0], [-20.0], [-20.0]], requires_grad=True)
    mask = torch.ones_like(gap, dtype=torch.bool)
    result = compute_correspondence_credit_from_tensors(
        thinking_similarity=h,
        structural_mask=structural,
        similarity_threshold=0.7,
        corrected_token_gap=gap,
        policy_loss_mask=mask,
        identities=identities,
        trajectory_advantage=torch.ones(6),
        config=CorrespondenceCreditConfig(
            min_segment_turns=1,
            max_segment_turns=3,
            mixing_coefficient=0.5,
            density_upper_bound=1.5,
        ),
    )
    assert result.diagnostics["projection_activation_rate"] == 1.0
    assert result.turn_weight.min() >= 0.5
    assert result.turn_weight.max() <= 1.25 + 1e-5
    assert result.diagnostics["credit_conservation_error_max"] <= 1e-5
    assert all(
        not value.requires_grad
        for value in result.__dict__.values()
        if isinstance(value, torch.Tensor)
    )


def test_wrapper_consumes_mechanism1_sidecar_and_keeps_matrix_direction():
    identities = _identities((1, 1))
    turn_map = canonicalize_turns(
        [identity.task_id for identity in identities],
        [identity.rollout_id for identity in identities],
        [identity.turn_id for identity in identities],
    )
    h, structural = _base_h(identities)
    token = torch.zeros((2, 1))
    mask = torch.ones((2, 1), dtype=torch.bool)
    m1 = Mechanism1Result(
        turn_map=turn_map,
        thinking_similarity=h,
        structural_mask=structural,
        selected_source_indices=((1,), (0,)),
        selected_source_weights=(torch.ones(1), torch.ones(1)),
        correspondence_confidence=torch.ones(2),
        correction_strength=torch.full((2,), 0.2),
        diagonal_teacher_log_probs=token,
        cross_teacher_log_probs=token,
        corrected_teacher_log_probs=token,
        diagonal_token_gap=token,
        corrected_token_gap=token,
        sampled_response_mask=mask,
        policy_loss_mask=mask,
        similarity_threshold=0.7,
        alpha_max=0.2,
        encoder_contract={},
    )
    result = compute_correspondence_credit(
        m1,
        torch.tensor([1.0, -1.0]),
        CorrespondenceCreditConfig(min_segment_turns=1, max_segment_turns=1),
    )
    assert result.matrix_layout == "target_by_source"
    assert torch.equal(result.dense_profiles, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))


def test_nonconstant_advantage_within_trajectory_is_rejected():
    identities = _identities((2, 2))
    h, structural = _base_h(identities)
    try:
        compute_correspondence_credit_from_tensors(
            thinking_similarity=h,
            structural_mask=structural,
            similarity_threshold=0.7,
            corrected_token_gap=torch.zeros((4, 1)),
            policy_loss_mask=torch.ones((4, 1), dtype=torch.bool),
            identities=identities,
            trajectory_advantage=torch.tensor([1.0, 2.0, -1.0, -1.0]),
            config=CorrespondenceCreditConfig(min_segment_turns=1, max_segment_turns=2),
        )
    except ValueError as error:
        assert "identical across turns" in str(error)
    else:
        raise AssertionError("turn-varying trajectory advantage must be rejected")


def test_unique_trajectory_grpo_is_not_weighted_by_turn_count():
    identities = _identities((3, 1))
    advantage = compute_unique_trajectory_grpo_advantage(
        torch.tensor([1.0, 1.0, 1.0, -1.0]), identities
    )
    expected = 1.0 / math.sqrt(2.0)
    assert torch.allclose(advantage[:3], torch.full((3,), expected), atol=1e-5)
    assert torch.allclose(advantage[3:], torch.full((1,), -expected), atol=1e-5)


def test_turn_normalization_matches_real_turn_statistics_and_preserves_legacy():
    identities = _identities((1, 3))
    rewards = torch.tensor([1., 0., 0., 0.])
    actual = compute_unique_trajectory_grpo_advantage(
        rewards, identities, normalization_scope="turn"
    )
    expected = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
    assert torch.allclose(actual, expected)
    legacy = compute_unique_trajectory_grpo_advantage(rewards, identities)
    explicit = compute_unique_trajectory_grpo_advantage(
        rewards, identities, normalization_scope="trajectory"
    )
    assert torch.equal(legacy, explicit)
    assert not torch.allclose(actual, legacy)
    centered = compute_unique_trajectory_grpo_advantage(
        rewards, identities, normalization_scope="turn", normalize_by_std=False
    )
    assert torch.allclose(centered, rewards - rewards.mean())
    assert torch.equal(rewards, torch.tensor([1., 0., 0., 0.]))


def test_turn_normalization_constant_rewards_are_finite_and_invalid_mode_fails():
    identities = _identities((1, 3))
    result = compute_unique_trajectory_grpo_advantage(
        torch.ones(4), identities, normalization_scope="turn"
    )
    assert torch.equal(result, torch.zeros(4))
    try:
        compute_unique_trajectory_grpo_advantage(
            torch.ones(4), identities, normalization_scope="typo"
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid normalization mode accepted")


def test_turn_normalized_advantage_composes_with_m2_and_conserves_budget():
    identities = _identities((2, 3))
    advantage = compute_unique_trajectory_grpo_advantage(
        torch.tensor([1., 1., 0., 0., 0.]), identities,
        normalization_scope="turn",
    )
    h, structural = _base_h(identities)
    result = compute_correspondence_credit_from_tensors(
        thinking_similarity=h,
        structural_mask=structural,
        similarity_threshold=0.7,
        corrected_token_gap=torch.tensor([[3.], [0.], [3.], [1.], [0.]]),
        policy_loss_mask=torch.ones((5, 1), dtype=torch.bool),
        identities=identities,
        trajectory_advantage=advantage,
        config=CorrespondenceCreditConfig(min_segment_turns=1, max_segment_turns=8),
    )
    assert not torch.allclose(result.turn_weight, torch.ones(5))
    assert torch.allclose(
        result.token_advantage[:, 0], advantage * result.turn_weight, atol=1e-6
    )
    for rows in (slice(0, 2), slice(2, 5)):
        assert torch.allclose(
            result.token_advantage[rows].sum(), advantage[rows].sum(), atol=1e-5
        )
    assert result.diagnostics["credit_conservation_error_max"] <= 1e-5


def test_bounded_m2_hierarchy_preserves_budget_sign_and_band():
    identities = _identities((3, 3))
    h, structural = _base_h(identities)
    mask = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]] * 2, dtype=torch.bool)
    advantage = torch.tensor([2., 2., 2., -2., -2., -2.])
    for band, mixing in [(0.2, 0.5), (0., 0.5), (0.2, 0.), (0.2, 1.)]:
        result = compute_correspondence_credit_from_tensors(
            thinking_similarity=h, structural_mask=structural, similarity_threshold=0.7,
            corrected_token_gap=torch.tensor([[10.], [-10.], [2.]] * 2).expand(-1, 3),
            policy_loss_mask=mask, identities=identities, trajectory_advantage=advantage,
            config=CorrespondenceCreditConfig(
                min_segment_turns=1, max_segment_turns=2, weighting_mode="bounded",
                multiplier_band=band, mixing_coefficient=mixing,
            ),
        )
        assert result.turn_weight.min() >= 1 - band * mixing - 1e-6
        assert result.turn_weight.max() <= 1 + band * mixing + 1e-6
        assert torch.equal(torch.sign(result.token_advantage), torch.sign(advantage[:, None] * mask))
        for rows in (slice(0, 3), slice(3, 6)):
            assert torch.allclose(result.token_advantage[rows].sum(),
                                  (advantage[rows, None] * mask[rows]).sum(), atol=1e-5)
        if band * mixing == 0:
            assert torch.allclose(result.turn_weight, torch.ones(6))
        else:
            assert not torch.allclose(result.turn_weight, torch.ones(6))


def test_bounded_projection_random_lengths_and_config_validation():
    generator = torch.Generator().manual_seed(42)
    for _ in range(40):
        counts = torch.randint(1, 513, (17,), generator=generator).float()
        raw = torch.rand(17, generator=generator) ** 4
        raw *= counts.sum() / (counts * raw).sum()
        projected = project_weighted_box_budget(raw, counts, 1.2, density_lower_bound=0.8)
        assert projected.min() >= 0.8 - 1e-6
        assert projected.max() <= 1.2 + 1e-6
        assert torch.allclose((counts * projected).sum(), counts.sum(), rtol=1e-6)
    for kwargs in ({"weighting_mode": "typo"}, {"multiplier_band": -0.1},
                   {"multiplier_band": float("nan")}, {"multiplier_band": 1.1}):
        try:
            CorrespondenceCreditConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid bounded config accepted")


def test_invalid_action_residual_is_separate_and_zero_without_invalid_actions():
    identities = _identities((2, 2))
    outcomes = torch.tensor([1.0, 1.0, -1.0, -1.0])
    mask = torch.ones((4, 3), dtype=torch.bool)
    zero_token, zero_scalar = compute_invalid_action_advantage_residual(
        outcomes,
        mask,
        identities,
        torch.ones(4, dtype=torch.bool),
        penalty_coefficient=0.1,
    )
    assert torch.equal(zero_token, torch.zeros_like(zero_token))
    assert torch.equal(zero_scalar, torch.zeros_like(zero_scalar))

    token_residual, scalar_residual = compute_invalid_action_advantage_residual(
        outcomes,
        mask,
        identities,
        torch.tensor([True, False, True, True]),
        penalty_coefficient=0.1,
    )
    assert scalar_residual[1] != 0
    assert torch.equal(token_residual, mask.float() * scalar_residual.unsqueeze(-1))


def test_outcome_and_gap_cannot_change_correspondence_segmentation():
    identities = _identities((4, 4))
    h, structural = _base_h(identities)
    mask = torch.ones((8, 2), dtype=torch.bool)
    config = CorrespondenceCreditConfig(min_segment_turns=1, max_segment_turns=4)
    first = compute_correspondence_credit_from_tensors(
        thinking_similarity=h,
        structural_mask=structural,
        similarity_threshold=0.7,
        corrected_token_gap=torch.zeros((8, 2)),
        policy_loss_mask=mask,
        identities=identities,
        trajectory_advantage=torch.tensor([1.0] * 4 + [-1.0] * 4),
        config=config,
    )
    second = compute_correspondence_credit_from_tensors(
        thinking_similarity=h,
        structural_mask=structural,
        similarity_threshold=0.7,
        corrected_token_gap=torch.randn((8, 2)) * 100,
        policy_loss_mask=mask,
        identities=identities,
        trajectory_advantage=torch.tensor([-3.0] * 4 + [7.0] * 4),
        config=config,
    )
    assert torch.equal(first.dense_profiles, second.dense_profiles)
    assert torch.equal(first.profile_valid, second.profile_valid)
    assert torch.equal(first.js_divergence, second.js_divergence)
    assert first.trajectory_segmentations == second.trajectory_segmentations
