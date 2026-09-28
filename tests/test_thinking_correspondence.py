from types import SimpleNamespace

import torch

from verl.trainer.ppo.core_algos import compute_policy_loss
from verl.trainer.ppo.thinking_correspondence import (
    HFThinkingEncoder,
    Mechanism1Result,
    TurnIdentity,
    build_structural_mask,
    canonical_multiplicity_row_weights,
    canonicalize_turns,
    combine_cross_teacher_scores,
    compose_cross_teacher_inputs,
    cosine_matrix_by_task,
    expand_canonical_tensor,
    last_token_pool,
    parse_tagged_response,
    select_policy_loss_mask,
    select_sparse_sources,
)


def test_hf_encoder_uses_transformers_compatible_torch_dtype(monkeypatch, tmp_path):
    import transformers

    captured_kwargs = {}

    class DummyTokenizer:
        truncation_side = "right"

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(hidden_size=1)

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )

    def fake_model_from_pretrained(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return DummyModel()

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        fake_model_from_pretrained,
    )

    encoder = HFThinkingEncoder(str(tmp_path), device="cpu", dtype="float32")

    assert captured_kwargs["torch_dtype"] is torch.float32
    assert captured_kwargs["attn_implementation"] == "sdpa"
    assert "dtype" not in captured_kwargs
    assert encoder.tokenizer.padding_side == "left"
    assert encoder.preprocess("  choose   the green shirt ") == "choose the green shirt"
    assert encoder.preprocess(" inspect   candidate ") == "inspect candidate"


def test_last_token_pool_supports_left_and_right_padding():
    hidden = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    left_padded = torch.tensor([[0, 1, 1], [1, 1, 1]])
    right_padded = torch.tensor([[1, 1, 0], [1, 1, 1]])

    assert torch.equal(last_token_pool(hidden, left_padded), hidden[:, -1])
    assert torch.equal(last_token_pool(hidden, right_padded), torch.stack((hidden[0, 1], hidden[1, 2])))


def test_parser_supports_all_benchmark_action_protocols():
    alf = parse_tagged_response("<think> inspect state </think><action>go north</action>")
    search = parse_tagged_response("<think>find evidence</think><search>query</search>")
    answer = parse_tagged_response("<think>done</think><answer>Paris</answer>")
    malformed = parse_tagged_response("<think>not closed")

    assert (alf.thinking, alf.action) == ("inspect state", "go north")
    assert search.action == "query"
    assert answer.action == "Paris"
    assert not malformed.thinking_valid


def test_canonicalization_only_deduplicates_exact_identity_and_expands():
    # Rows 0/3 are framework copies. Rows 1/2 deliberately represent different
    # real turns even if their response/action content could be identical.
    turn_map = canonicalize_turns(
        task_ids=["task", "task", "task", "task"],
        rollout_ids=["r1", "r1", "r2", "r1"],
        turn_ids=[0, 1, 0, 0],
    )
    assert turn_map.num_canonical_turns == 3
    assert turn_map.canonical_to_rows[0] == (0, 3)
    canonical = torch.tensor([[10.0], [20.0], [30.0]])
    assert expand_canonical_tensor(canonical, turn_map).tolist() == [[10.0], [20.0], [30.0], [10.0]]


def test_canonical_multiplicity_weights_conserve_loss_and_gradient_across_micro_batches():
    turn_map = canonicalize_turns(
        ["task", "task", "task"],
        ["rollout-1", "rollout-2", "rollout-1"],
        [0, 0, 0],
    )
    row_weights = canonical_multiplicity_row_weights(turn_map)
    assert torch.allclose(row_weights, torch.tensor([0.5, 1.0, 0.5]))

    canonical_log_prob = torch.tensor(
        [[0.10, -0.20, 0.30], [-0.10, 0.20, 0.00]],
        requires_grad=True,
    )
    canonical_old_log_prob = torch.zeros_like(canonical_log_prob)
    canonical_advantage = torch.tensor(
        [[1.0, 0.5, -0.5], [-1.0, 0.25, 0.0]],
    )
    canonical_mask = torch.tensor(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
    )
    canonical_loss, *_ = compute_policy_loss(
        old_log_prob=canonical_old_log_prob,
        log_prob=canonical_log_prob,
        advantages=canonical_advantage,
        response_mask=canonical_mask,
        cliprange=0.2,
        cliprange_low=0.2,
        cliprange_high=0.2,
        loss_agg_mode="token-mean",
    )
    canonical_gradient = torch.autograd.grad(
        canonical_loss,
        canonical_log_prob,
        retain_graph=True,
    )[0]

    row_index = torch.tensor(turn_map.row_to_canonical)
    expanded_log_prob = canonical_log_prob.index_select(0, row_index)
    expanded_old_log_prob = canonical_old_log_prob.index_select(0, row_index)
    expanded_advantage = canonical_advantage.index_select(0, row_index)
    expanded_mask = canonical_mask.index_select(0, row_index)
    weighted_token_total = (expanded_mask * row_weights.unsqueeze(-1)).sum()

    # Simulate two actor micro-batches. Each returns its own weighted mean and
    # is scaled by its share of the distributed mini-batch denominator.
    weighted_loss = canonical_loss.new_zeros(())
    for micro_indices in (torch.tensor([0, 1]), torch.tensor([2])):
        micro_mask = expanded_mask.index_select(0, micro_indices)
        micro_weight = row_weights.index_select(0, micro_indices).unsqueeze(-1)
        micro_loss, *_ = compute_policy_loss(
            old_log_prob=expanded_old_log_prob.index_select(0, micro_indices),
            log_prob=expanded_log_prob.index_select(0, micro_indices),
            advantages=expanded_advantage.index_select(0, micro_indices),
            response_mask=micro_mask,
            cliprange=0.2,
            cliprange_low=0.2,
            cliprange_high=0.2,
            loss_agg_mode="token-mean",
            loss_weight=micro_weight,
        )
        micro_weighted_tokens = (micro_mask * micro_weight).sum()
        weighted_loss = weighted_loss + micro_loss * (
            micro_weighted_tokens / weighted_token_total
        )

    weighted_gradient = torch.autograd.grad(weighted_loss, canonical_log_prob)[0]
    assert torch.allclose(weighted_loss, canonical_loss)
    assert torch.allclose(weighted_gradient, canonical_gradient)


def test_canonical_coordinates_are_stable_under_batch_reordering():
    first = canonicalize_turns(["b", "a", "a"], ["r2", "r2", "r1"], [0, 0, 1])
    second = canonicalize_turns(["a", "b", "a"], ["r1", "r2", "r2"], [1, 0, 0])
    assert first.identities == second.identities


def test_structural_mask_is_dense_pre_threshold_and_cross_rollout_only():
    identities = (
        TurnIdentity("task-a", "rollout-1", 0),
        TurnIdentity("task-a", "rollout-1", 1),
        TurnIdentity("task-a", "rollout-2", 0),
        TurnIdentity("task-b", "rollout-3", 0),
    )
    mask = build_structural_mask(
        identities,
        teacher_thinking_valid=torch.tensor([1, 1, 1, 1], dtype=torch.bool),
        student_thinking_valid=torch.tensor([1, 1, 1, 0], dtype=torch.bool),
        protocol_ids=["alf", "alf", "alf", "alf"],
    )
    assert mask.shape == (4, 4)
    assert mask[2, 0] and mask[2, 1]
    assert mask[0, 2] and mask[1, 2]
    assert not mask[0, 1]  # same rollout
    assert not mask[:, 3].any()  # invalid target thinking
    assert not mask[3].any()  # different task from every eligible target


def test_task_block_similarity_matches_dense_similarity_on_eligible_cells():
    teacher = torch.nn.functional.normalize(torch.randn(4, 5), dim=-1)
    student = torch.nn.functional.normalize(torch.randn(4, 5), dim=-1)
    task_ids = ["a", "a", "b", "b"]
    block = cosine_matrix_by_task(teacher, student, task_ids)
    dense = teacher @ student.T
    same_task = torch.tensor([[left == right for right in task_ids] for left in task_ids])
    assert block.shape == (4, 4)
    assert torch.allclose(block[same_task], dense[same_task])
    assert torch.equal(block[~same_task], torch.zeros_like(block[~same_task]))


def test_sparse_selection_applies_per_rollout_cap_then_top_k_with_tie_break():
    identities = (
        TurnIdentity("task", "a", 0),
        TurnIdentity("task", "a", 1),
        TurnIdentity("task", "b", 0),
        TurnIdentity("task", "c", 0),
    )
    h = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.80],
            [0.0, 0.0, 0.0, 0.95],
            [0.0, 0.0, 0.0, 0.90],
            [0.0, 0.0, 0.0, 0.90],
        ]
    )
    structural = torch.zeros((4, 4), dtype=torch.bool)
    structural[:, 3] = True
    selected = select_sparse_sources(
        h,
        structural,
        identities,
        similarity_threshold=0.70,
        source_rollout_cap=1,
        top_k=2,
    )
    assert selected[3] == (1, 2)  # best from rollout a, then identity tie-break b before c


def test_no_match_is_exact_diagonal_fallback():
    diagonal = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
    student = torch.tensor([[-2.0, -2.5], [-3.5, -5.0]])
    output = combine_cross_teacher_scores(
        diagonal,
        student,
        thinking_similarity=torch.eye(2),
        selected_source_indices=((), ()),
        cross_scores={},
    )
    assert torch.equal(output["corrected_teacher_log_probs"], diagonal)
    assert torch.equal(output["corrected_token_gap"], diagonal - student)
    assert torch.equal(output["correction_strength"], torch.zeros(2))


def test_cross_inputs_take_prompt_from_source_and_full_response_from_target():
    prompts = torch.tensor([[10, 11], [20, 21], [30, 31]])
    prompt_mask = torch.ones_like(prompts)
    responses = torch.tensor([[100, 101, 0], [200, 201, 202], [300, 0, 0]])
    response_mask = responses.ne(0).long()
    output = compose_cross_teacher_inputs(
        prompts,
        prompt_mask,
        responses,
        response_mask,
        pairs=((2, 0), (0, 1)),
    )
    assert output["input_ids"].tolist() == [
        [30, 31, 100, 101, 0],
        [10, 11, 200, 201, 202],
    ]
    assert output["responses"].tolist() == responses[[0, 1]].tolist()


def test_policy_loss_mask_matches_actor_semantics():
    responses = torch.ones((2, 3), dtype=torch.long)
    attention = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    loss_mask = torch.tensor([[0, 0, 0, 1, 0], [0, 0, 1, 1, 0]])
    assert torch.equal(
        select_policy_loss_mask(responses, attention, multi_turn=False),
        attention[:, -3:],
    )
    assert torch.equal(
        select_policy_loss_mask(responses, attention, multi_turn=True, loss_mask=loss_mask),
        loss_mask[:, -3:],
    )


def test_legacy_cross_teacher_log_prob_average_and_confidence_correction():
    diagonal = torch.tensor([[-4.0, -4.0], [-2.0, -2.0], [-3.0, -3.0]])
    student = torch.tensor([[-5.0, -5.0], [-3.0, -3.0], [-4.0, -4.0]])
    h = torch.zeros((3, 3))
    h[0, 2] = 0.8
    h[1, 2] = 0.9
    cross = {
        (0, 2): torch.tensor([-1.0, -3.0]),
        (1, 2): torch.tensor([-3.0, -1.0]),
    }
    output = combine_cross_teacher_scores(
        diagonal,
        student,
        h,
        selected_source_indices=((), (), (0, 1)),
        cross_scores=cross,
        similarity_threshold=0.7,
        aggregation_temperature=0.1,
        alpha_max=0.2,
        aggregation_mode="log_mean",
    )
    weights = torch.softmax(torch.tensor([0.8, 0.9]) / 0.1, dim=0)
    expected_cross = weights[0] * cross[(0, 2)] + weights[1] * cross[(1, 2)]
    expected_rho = (weights * torch.tensor([(0.8 - 0.7) / 0.3, (0.9 - 0.7) / 0.3])).sum()
    expected = (1 - 0.2 * expected_rho) * diagonal[2] + (0.2 * expected_rho) * expected_cross
    assert torch.allclose(output["cross_teacher_log_probs"][2], expected_cross)
    assert torch.allclose(output["correspondence_confidence"][2], expected_rho)
    assert torch.allclose(output["corrected_teacher_log_probs"][2], expected)
    assert 0 <= output["correction_strength"].min() <= output["correction_strength"].max() <= 0.2


def test_mechanism1_sidecar_contract_is_self_contained_for_mechanism2():
    turn_map = canonicalize_turns(["task", "task", "task"], ["r1", "r2", "r1"], [0, 0, 0])
    h = torch.tensor([[0.0, 0.8], [0.9, 0.0]], dtype=torch.float32)
    structural = torch.tensor([[False, True], [True, False]])
    token = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
    mask = torch.ones_like(token, dtype=torch.long)
    result = Mechanism1Result(
        turn_map=turn_map,
        thinking_similarity=h,
        structural_mask=structural,
        selected_source_indices=((1,), (0,)),
        selected_source_weights=(torch.ones(1), torch.ones(1)),
        correspondence_confidence=torch.tensor([0.5, 0.25]),
        correction_strength=torch.tensor([0.1, 0.05]),
        diagonal_teacher_log_probs=token,
        cross_teacher_log_probs=token,
        corrected_teacher_log_probs=token,
        diagonal_token_gap=token,
        corrected_token_gap=token,
        sampled_response_mask=mask,
        policy_loss_mask=mask,
        similarity_threshold=0.7,
        alpha_max=0.2,
        encoder_contract={"model": "qwen3-embedding", "pooling": "last_token"},
    )
    assert result.matrix_layout == "source_by_target"
    assert result.similarity_threshold == 0.7
    assert result.turn_map.row_to_canonical == (0, 1, 0)
