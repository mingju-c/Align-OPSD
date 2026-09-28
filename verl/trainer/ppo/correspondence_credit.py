"""Pure Mechanism-2 correspondence segmentation and credit allocation.

This module deliberately performs no model forward pass and does not mutate a
``DataProto``.  It consumes the detached, canonical output of Mechanism 1 plus
one GRPO outcome advantage per canonical turn.  Matrix orientation is explicit:
Mechanism 1 supplies ``H[source, target]`` and this module returns dense
profiles ``P[target, source]``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence

import torch

from verl.trainer.ppo.thinking_correspondence import Mechanism1Result, TurnIdentity


@dataclass(frozen=True)
class CorrespondenceCreditConfig:
    """Hyperparameters for hierarchical or flat turn credit."""

    segmentation_mode: str = "adaptive"

    profile_temperature: float = 0.10
    # Distance used to detect a boundary between adjacent profile
    # distributions.  ``jsd`` preserves the original behavior; ``entropy``
    # compares normalized Shannon entropies and ``cosine`` compares the
    # profile vectors directly.
    profile_distance: str = "jsd"
    segmentation_quantile: float = 0.80
    segmentation_threshold_min: float = 0.01
    segmentation_threshold_max: float = 0.10
    min_segment_turns: int = 2
    max_segment_turns: int = 8
    credit_temperature: float = 1.0
    mixing_coefficient: float = 0.5
    density_upper_bound: float = 4.0
    weighting_mode: str = "legacy"
    multiplier_band: float = 0.2
    loss_aggregation: str = "token-mean"
    projection_tolerance: float = 1e-7
    projection_max_iterations: int = 100

    def __post_init__(self) -> None:
        if self.segmentation_mode not in {"adaptive", "token", "turn", "random"}:
            raise ValueError("segmentation_mode must be adaptive, token, turn, or random")
        if self.weighting_mode not in {"legacy", "bounded"}:
            raise ValueError("weighting_mode must be legacy or bounded")
        if not math.isfinite(self.multiplier_band) or not 0 <= self.multiplier_band <= 1:
            raise ValueError("multiplier_band must be finite and in [0, 1]")
        if self.profile_temperature <= 0:
            raise ValueError("profile_temperature must be positive")
        if self.profile_distance not in {"jsd", "entropy", "cosine"}:
            raise ValueError("profile_distance must be jsd, entropy, or cosine")
        if not 0.0 <= self.segmentation_quantile <= 1.0:
            raise ValueError("segmentation_quantile must be in [0, 1]")
        if not 0.0 <= self.segmentation_threshold_min <= self.segmentation_threshold_max <= 1.0:
            raise ValueError("segmentation thresholds must satisfy 0 <= min <= max <= 1")
        if self.min_segment_turns < 1:
            raise ValueError("min_segment_turns must be positive")
        # This is the spec's recommended relation and is necessary to guarantee
        # that every overlong interval can be split without creating a short one.
        if self.max_segment_turns < 2 * self.min_segment_turns - 1:
            raise ValueError("max_segment_turns must be at least 2 * min_segment_turns - 1")
        if self.credit_temperature <= 0:
            raise ValueError("credit_temperature must be positive")
        if not 0.0 <= self.mixing_coefficient <= 1.0:
            raise ValueError("mixing_coefficient must be in [0, 1]")
        if self.density_upper_bound < 1.0:
            raise ValueError("density_upper_bound must be at least 1")
        if self.loss_aggregation != "token-mean":
            raise ValueError("Mechanism 2 budget conservation requires token-mean aggregation")
        if self.projection_tolerance <= 0 or self.projection_max_iterations < 1:
            raise ValueError("invalid projection solver settings")


@dataclass(frozen=True)
class DecisionSegment:
    """One contiguous segment in a canonical trajectory.

    ``start`` and ``end`` are local offsets into ``turn_indices`` and use the
    half-open convention.  ``canonical_turn_indices`` gives the corresponding
    global canonical coordinates.
    """

    task_id: str
    rollout_id: str
    start: int
    end: int
    canonical_turn_indices: tuple[int, ...]

    @property
    def num_turns(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class TrajectorySegmentation:
    """Partition and classified boundary positions for one trajectory."""

    task_id: str
    rollout_id: str
    turn_indices: tuple[int, ...]
    semantic_boundaries: tuple[int, ...]
    correspondence_guided_fallback_boundaries: tuple[int, ...]
    forced_max_boundaries: tuple[int, ...]
    segments: tuple[DecisionSegment, ...]


@dataclass
class CorrespondenceCreditResult:
    """Detached canonical result of Mechanism 2."""

    dense_profiles: torch.Tensor  # [target, source]
    profile_valid: torch.Tensor  # [target]
    js_divergence: torch.Tensor  # [target], selected profile distance before target
    boundary_valid: torch.Tensor  # [target]
    segmentation_threshold: float
    trajectory_segmentations: tuple[TrajectorySegmentation, ...]
    segment_index_per_turn: torch.Tensor  # [target], global segment index
    turn_gap: torch.Tensor
    outcome_aligned_score: torch.Tensor
    segment_budget: torch.Tensor  # indexed by global segment index
    within_segment_turn_budget: torch.Tensor
    raw_credit_density: torch.Tensor
    projected_credit_density: torch.Tensor
    turn_weight: torch.Tensor
    token_advantage: torch.Tensor
    diagnostics: dict[str, float] = field(default_factory=dict)
    matrix_layout: str = "target_by_source"

    def __post_init__(self) -> None:
        if self.matrix_layout != "target_by_source":
            raise ValueError("dense profile layout must be target_by_source")
        m = self.dense_profiles.shape[0]
        if self.dense_profiles.shape != (m, m):
            raise ValueError("dense_profiles must have shape [M_target, M_source]")
        for name in (
            "profile_valid",
            "js_divergence",
            "boundary_valid",
            "segment_index_per_turn",
            "turn_gap",
            "outcome_aligned_score",
            "within_segment_turn_budget",
            "raw_credit_density",
            "projected_credit_density",
            "turn_weight",
        ):
            if getattr(self, name).shape != (m,):
                raise ValueError(f"{name} must have shape [M]")
        if self.token_advantage.ndim != 2 or self.token_advantage.shape[0] != m:
            raise ValueError("token_advantage must have shape [M, response_length]")
        if self.profile_valid.dtype != torch.bool or self.boundary_valid.dtype != torch.bool:
            raise ValueError("profile_valid and boundary_valid must be boolean")
        for name in (
            "dense_profiles",
            "js_divergence",
            "turn_gap",
            "outcome_aligned_score",
            "segment_budget",
            "within_segment_turn_budget",
            "raw_credit_density",
            "projected_credit_density",
            "turn_weight",
            "token_advantage",
        ):
            if not torch.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} must be finite")
        if (self.dense_profiles < 0).any():
            raise ValueError("dense profiles must be nonnegative")
        row_sums = self.dense_profiles.sum(dim=-1)
        if not torch.all(torch.isclose(row_sums, torch.zeros_like(row_sums), atol=1e-5) |
                         torch.isclose(row_sums, torch.ones_like(row_sums), atol=1e-5)):
            raise ValueError("each dense profile must sum to zero or one")
        if ((self.js_divergence < 0) | (self.js_divergence > 1)).any():
            raise ValueError("profile distance must lie in [0, 1]")
        if not (self.diagnostics.get("flat_turn_allocation", 0.0) or
                self.diagnostics.get("flat_token_allocation", 0.0)) and (self.segment_index_per_turn < 0).any():
            raise ValueError("every turn must belong to a segment")
        if (self.raw_credit_density < 0).any() or (self.projected_credit_density < 0).any():
            raise ValueError("credit densities must be nonnegative")
        for value in self.__dict__.values():
            if isinstance(value, torch.Tensor) and value.requires_grad:
                raise ValueError("Mechanism 2 results must be detached")


def _as_fp32(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.detach().to(device=device, dtype=torch.float32)


def build_dense_correspondence_profiles(
    thinking_similarity: torch.Tensor,
    structural_mask: torch.Tensor,
    similarity_threshold: float,
    profile_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``P[target, source]`` from all structurally eligible sources.

    The similarity threshold affects only profile validity.  It never truncates
    or reweights the probability distribution itself.
    """

    if thinking_similarity.ndim != 2 or thinking_similarity.shape[0] != thinking_similarity.shape[1]:
        raise ValueError("thinking_similarity must be square H[source, target]")
    if structural_mask.shape != thinking_similarity.shape or structural_mask.dtype != torch.bool:
        raise ValueError("structural_mask must be boolean with the same shape as H")
    if not -1.0 <= float(similarity_threshold) < 1.0:
        raise ValueError("similarity_threshold must be in [-1, 1)")
    if profile_temperature <= 0:
        raise ValueError("profile_temperature must be positive")

    h = thinking_similarity.detach().float()
    eligible = structural_mask.detach().bool()
    if not torch.isfinite(h).all():
        raise ValueError("thinking_similarity must be finite")

    # Transpose both tensors together: rows are targets and columns are the
    # unchanged global Teacher-source coordinates.
    logits = h.T / float(profile_temperature)
    eligible_target_source = eligible.T
    has_source = eligible_target_source.any(dim=-1)
    profiles = torch.zeros_like(logits, dtype=torch.float32)
    if has_source.any():
        valid_logits = logits[has_source].masked_fill(~eligible_target_source[has_source], -torch.inf)
        profiles[has_source] = torch.softmax(valid_logits, dim=-1)

    max_similarity = torch.full(
        (h.shape[1],), -torch.inf, dtype=torch.float32, device=h.device
    )
    if has_source.any():
        masked_h = h.T.masked_fill(~eligible_target_source, -torch.inf)
        max_similarity[has_source] = masked_h[has_source].max(dim=-1).values
    profile_valid = has_source & max_similarity.ge(float(similarity_threshold))
    return profiles.detach(), profile_valid.detach()


def normalized_js_divergence(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Normalized Jensen--Shannon divergence using natural logarithms."""

    if left.shape != right.shape:
        raise ValueError("JSD inputs must have equal shapes")
    p = left.detach().float()
    q = right.detach().float()
    if (p < 0).any() or (q < 0).any() or not torch.isfinite(p).all() or not torch.isfinite(q).all():
        raise ValueError("JSD inputs must be finite nonnegative distributions")
    if not torch.allclose(p.sum(), torch.ones((), device=p.device), atol=1e-5, rtol=1e-5):
        raise ValueError("left JSD input must sum to one")
    if not torch.allclose(q.sum(), torch.ones((), device=q.device), atol=1e-5, rtol=1e-5):
        raise ValueError("right JSD input must sum to one")
    midpoint = 0.5 * (p + q)
    p_positive = p > 0
    q_positive = q > 0
    kl_p = (p[p_positive] * (p[p_positive].log() - midpoint[p_positive].log())).sum()
    kl_q = (q[q_positive] * (q[q_positive].log() - midpoint[q_positive].log())).sum()
    jsd = 0.5 * (kl_p + kl_q) / math.log(2.0)
    return jsd.clamp(0.0, 1.0).detach()


def normalized_entropy_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Absolute difference of normalized Shannon entropy values."""

    if left.shape != right.shape:
        raise ValueError("entropy-distance inputs must have equal shapes")
    p = left.detach().float()
    q = right.detach().float()
    for name, value in (("left", p), ("right", q)):
        if (value < 0).any() or not torch.isfinite(value).all():
            raise ValueError(f"{name} entropy-distance input must be finite nonnegative")
        if not torch.allclose(value.sum(), torch.ones((), device=value.device), atol=1e-5, rtol=1e-5):
            raise ValueError(f"{name} entropy-distance input must sum to one")
    support = max(int(p.numel()), 1)
    normalizer = math.log(support) if support > 1 else 1.0
    hp = _entropy(p) / normalizer
    hq = _entropy(q) / normalizer
    return (hp - hq).abs().clamp(0.0, 1.0).detach()


def cosine_profile_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Cosine distance (1 - cosine similarity) for two probability profiles."""

    if left.shape != right.shape:
        raise ValueError("cosine-distance inputs must have equal shapes")
    p = left.detach().float()
    q = right.detach().float()
    if (p < 0).any() or (q < 0).any() or not torch.isfinite(p).all() or not torch.isfinite(q).all():
        raise ValueError("cosine-distance inputs must be finite nonnegative")
    if not torch.allclose(p.sum(), torch.ones((), device=p.device), atol=1e-5, rtol=1e-5):
        raise ValueError("left cosine-distance input must sum to one")
    if not torch.allclose(q.sum(), torch.ones((), device=q.device), atol=1e-5, rtol=1e-5):
        raise ValueError("right cosine-distance input must sum to one")
    similarity = torch.nn.functional.cosine_similarity(p.unsqueeze(0), q.unsqueeze(0), dim=-1)[0]
    return (1.0 - similarity).clamp(0.0, 1.0).detach()


def profile_distance(left: torch.Tensor, right: torch.Tensor, mode: str) -> torch.Tensor:
    """Dispatch the selectable adjacent-profile distance."""

    if mode == "jsd":
        return normalized_js_divergence(left, right)
    if mode == "entropy":
        return normalized_entropy_distance(left, right)
    if mode == "cosine":
        return cosine_profile_distance(left, right)
    raise ValueError("profile distance mode must be jsd, entropy, or cosine")


def _trajectory_groups(identities: Sequence[TurnIdentity]) -> tuple[tuple[int, ...], ...]:
    groups: dict[tuple[str, str], list[int]] = {}
    for index, identity in enumerate(identities):
        groups.setdefault((identity.task_id, identity.rollout_id), []).append(index)
    output = []
    for key in sorted(groups):
        indices = sorted(groups[key], key=lambda index: (identities[index].turn_id, index))
        turn_ids = [identities[index].turn_id for index in indices]
        if len(set(turn_ids)) != len(turn_ids):
            raise ValueError(f"duplicate canonical turn_id in trajectory {key}")
        output.append(tuple(indices))
    return tuple(output)


def compute_unique_trajectory_grpo_advantage(
    episode_outcome: torch.Tensor,
    identities: Sequence[TurnIdentity],
    *,
    normalize_by_std: bool = True,
    epsilon: float = 1e-6,
    normalization_scope: str = "trajectory",
) -> torch.Tensor:
    """Compute one pure GRPO outcome advantage per unique trajectory.

    Inputs must be canonical real turns, without framework-created copies.
    ``trajectory`` counts each rollout once; ``turn`` weights the normalization
    statistics by its real turn count. Both broadcast one scalar per trajectory.
    """

    if normalization_scope not in {"trajectory", "turn"}:
        raise ValueError("normalization_scope must be 'trajectory' or 'turn'")
    outcome = torch.as_tensor(episode_outcome).detach().float()
    if outcome.shape != (len(identities),) or not torch.isfinite(outcome).all():
        raise ValueError("episode_outcome must be finite with shape [M]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    trajectories = _trajectory_groups(identities)
    task_to_trajectories: dict[str, list[tuple[int, ...]]] = {}
    for rows in trajectories:
        task_to_trajectories.setdefault(identities[rows[0]].task_id, []).append(rows)

    advantage = torch.empty_like(outcome)
    for task_id in sorted(task_to_trajectories):
        task_trajectories = task_to_trajectories[task_id]
        values = []
        for rows in task_trajectories:
            indices = torch.tensor(rows, dtype=torch.long, device=outcome.device)
            trajectory_values = outcome.index_select(0, indices)
            if not torch.allclose(
                trajectory_values,
                trajectory_values[:1].expand_as(trajectory_values),
                atol=1e-6,
                rtol=1e-6,
            ):
                raise ValueError("episode outcome must be identical across turns of a trajectory")
            values.append(trajectory_values[0])
        group_values = torch.stack(values)
        if group_values.numel() < 2:
            raise ValueError("GRPO requires at least two unique sibling trajectories per task")
        statistics_values = group_values
        if normalization_scope == "turn":
            counts = torch.tensor(
                [len(rows) for rows in task_trajectories], device=outcome.device
            )
            statistics_values = group_values.repeat_interleave(counts)
        mean = statistics_values.mean()
        std = statistics_values.std()
        normalized = (
            (group_values - mean) / (std + epsilon)
            if normalize_by_std
            else group_values - mean
        )
        for rows, value in zip(task_trajectories, normalized, strict=True):
            advantage[list(rows)] = value
    return advantage.detach()


def compute_invalid_action_advantage_residual(
    episode_outcome: torch.Tensor,
    policy_loss_mask: torch.Tensor,
    identities: Sequence[TurnIdentity],
    is_action_valid: torch.Tensor,
    *,
    penalty_coefficient: float,
    normalize_by_std: bool = True,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Isolate the legacy per-turn invalid-action GRPO effect.

    The two GRPO normalizations use canonical real turns only. Their difference
    is added after M2 allocation, so invalidity cannot influence profiles,
    boundaries, outcome alignment, or the hierarchical credit budget.
    """

    outcome = torch.as_tensor(episode_outcome).detach().float()
    valid = torch.as_tensor(is_action_valid).detach().to(device=outcome.device, dtype=torch.bool)
    mask = policy_loss_mask.detach().to(device=outcome.device, dtype=torch.bool)
    m = len(identities)
    if outcome.shape != (m,) or valid.shape != (m,):
        raise ValueError("outcome and action validity must have canonical shape [M]")
    if mask.ndim != 2 or mask.shape[0] != m:
        raise ValueError("policy_loss_mask must have shape [M, response_length]")
    if penalty_coefficient < 0 or epsilon <= 0:
        raise ValueError("invalid penalty settings")

    def normalize_turn_rows(scores: torch.Tensor) -> torch.Tensor:
        normalized = torch.empty_like(scores)
        task_to_rows: dict[str, list[int]] = {}
        for row, identity in enumerate(identities):
            task_to_rows.setdefault(identity.task_id, []).append(row)
        for task_id in sorted(task_to_rows):
            rows = task_to_rows[task_id]
            indices = torch.tensor(rows, dtype=torch.long, device=scores.device)
            values = scores.index_select(0, indices)
            if values.numel() == 1:
                mean = values.new_zeros(())
                std = values.new_ones(())
            else:
                mean = values.mean()
                std = values.std()
            normalized[indices] = (
                (values - mean) / (std + epsilon)
                if normalize_by_std
                else values - mean
            )
        return normalized

    pure = normalize_turn_rows(outcome)
    penalized = normalize_turn_rows(
        outcome - float(penalty_coefficient) * (~valid).to(outcome.dtype)
    )
    scalar_residual = (penalized - pure).detach()
    token_residual = mask.to(torch.float32) * scalar_residual.unsqueeze(-1)
    return token_residual.detach(), scalar_residual


def compute_adjacent_profile_distance(
    profiles: torch.Tensor,
    profile_valid: torch.Tensor,
    structural_mask: torch.Tensor,
    identities: Sequence[TurnIdentity],
    distance_mode: str = "jsd",
) -> tuple[torch.Tensor, torch.Tensor, tuple[tuple[int, ...], ...], dict[str, float]]:
    """Compute adjacent profile distances without treating mask changes as evidence."""

    m = len(identities)
    if profiles.shape != (m, m) or profile_valid.shape != (m,) or structural_mask.shape != (m, m):
        raise ValueError("adjacent-profile-distance input shapes are inconsistent")
    if distance_mode not in {"jsd", "entropy", "cosine"}:
        raise ValueError("distance_mode must be jsd, entropy, or cosine")
    groups = _trajectory_groups(identities)
    distances = torch.zeros(m, dtype=torch.float32, device=profiles.device)
    boundary_valid = torch.zeros(m, dtype=torch.bool, device=profiles.device)
    adjacent_count = 0
    mask_change_count = 0
    both_profile_valid_count = 0
    for indices in groups:
        for previous, current in zip(indices[:-1], indices[1:], strict=True):
            adjacent_count += 1
            same_mask = torch.equal(structural_mask[:, previous], structural_mask[:, current])
            if not same_mask:
                mask_change_count += 1
            profiles_valid = bool(profile_valid[previous] and profile_valid[current])
            both_profile_valid_count += int(profiles_valid)
            if profiles_valid and same_mask:
                boundary_valid[current] = True
                distances[current] = profile_distance(
                    profiles[previous], profiles[current], distance_mode
                )
    stats = {
        "adjacent_pair_count": float(adjacent_count),
        "adjacent_profiles_valid_rate": both_profile_valid_count / max(adjacent_count, 1),
        "structural_mask_change_rate": mask_change_count / max(adjacent_count, 1),
        "valid_jsd_pair_rate": float(boundary_valid.sum().item()) / max(adjacent_count, 1),
        "valid_profile_distance_pair_rate": float(boundary_valid.sum().item()) / max(adjacent_count, 1),
        "profile_distance_mode_jsd": float(distance_mode == "jsd"),
        "profile_distance_mode_entropy": float(distance_mode == "entropy"),
        "profile_distance_mode_cosine": float(distance_mode == "cosine"),
    }
    return distances.detach(), boundary_valid.detach(), groups, stats


def compute_adjacent_jsd(
    profiles: torch.Tensor,
    profile_valid: torch.Tensor,
    structural_mask: torch.Tensor,
    identities: Sequence[TurnIdentity],
) -> tuple[torch.Tensor, torch.Tensor, tuple[tuple[int, ...], ...], dict[str, float]]:
    """Backward-compatible wrapper for the original JSD path."""

    return compute_adjacent_profile_distance(
        profiles, profile_valid, structural_mask, identities, distance_mode="jsd"
    )


def adaptive_segmentation_threshold(
    js_divergence: torch.Tensor,
    boundary_valid: torch.Tensor,
    quantile: float,
    threshold_min: float,
    threshold_max: float,
) -> float:
    """Return the clipped batch quantile, or infinity when no JSD is valid."""

    values = js_divergence.detach().float()[boundary_valid.detach().bool()]
    if values.numel() == 0:
        return float("inf")
    threshold = torch.quantile(values, float(quantile)).item()
    return float(min(max(threshold, float(threshold_min)), float(threshold_max)))


def _nearest_boundaries(boundaries: set[int], position: int) -> tuple[int, int]:
    return max(value for value in boundaries if value < position), min(
        value for value in boundaries if value > position
    )


def _balanced_split(left: int, right: int, min_turns: int) -> int:
    feasible = range(left + min_turns, right - min_turns + 1)
    midpoint = 0.5 * (left + right)
    try:
        return min(feasible, key=lambda position: (abs(position - midpoint), position))
    except ValueError as error:
        raise ValueError("overlong interval cannot be split without violating min_segment_turns") from error


def segment_trajectories(
    identities: Sequence[TurnIdentity],
    groups: Sequence[Sequence[int]],
    js_divergence: torch.Tensor,
    boundary_valid: torch.Tensor,
    segmentation_threshold: float,
    min_segment_turns: int,
    max_segment_turns: int,
    fallback_jsd_min: float,
    segmentation_mode: str = "adaptive",
    random_seed: int = 0,
) -> tuple[tuple[TrajectorySegmentation, ...], int, int]:
    """Apply semantic candidates, then JSD-guided/forced maximum-length splits."""

    if min_segment_turns < 1 or max_segment_turns < 2 * min_segment_turns - 1:
        raise ValueError("invalid min/max segment length relation")
    trajectories: list[TrajectorySegmentation] = []
    raw_candidate_count = 0
    accepted_candidate_count = 0
    for indices_sequence in groups:
        indices = tuple(indices_sequence)
        num_turns = len(indices)
        first_identity = identities[indices[0]]
        boundaries = {0, num_turns}
        semantic: set[int] = set()
        guided: set[int] = set()
        forced: set[int] = set()

        if segmentation_mode == "random":
            rng = random.Random(random_seed + len(trajectories))
            candidates = list(range(min_segment_turns, num_turns - min_segment_turns + 1))
            rng.shuffle(candidates)
            for position in candidates:
                left, right = _nearest_boundaries(boundaries, position)
                if position - left >= min_segment_turns and right - position >= min_segment_turns:
                    boundaries.add(position)
            # The maximum-length repair below remains active.
            raw_candidate_count += len(candidates)
            accepted_candidate_count += max(0, len(boundaries) - 2)
            candidates = []

        else:
            candidates = [
                local_position
                for local_position in range(1, num_turns)
                if bool(boundary_valid[indices[local_position]])
                and float(js_divergence[indices[local_position]]) > segmentation_threshold
            ]
        raw_candidate_count += len(candidates)
        candidates.sort(
            key=lambda position: (-float(js_divergence[indices[position]]), position)
        )
        for position in candidates:
            left, right = _nearest_boundaries(boundaries, position)
            if position - left >= min_segment_turns and right - position >= min_segment_turns:
                boundaries.add(position)
                semantic.add(position)
                accepted_candidate_count += 1

        # Repeatedly split the leftmost overlong interval.  Each inserted split
        # replaces it with two intervals, so termination is guaranteed.
        while True:
            ordered = sorted(boundaries)
            overlong = next(
                ((left, right) for left, right in zip(ordered[:-1], ordered[1:], strict=True)
                 if right - left > max_segment_turns),
                None,
            )
            if overlong is None:
                break
            left, right = overlong
            feasible_valid = [
                position
                for position in range(left + min_segment_turns, right - min_segment_turns + 1)
                if bool(boundary_valid[indices[position]])
                and float(js_divergence[indices[position]]) >= fallback_jsd_min
            ]
            if feasible_valid:
                position = min(
                    feasible_valid,
                    key=lambda item: (-float(js_divergence[indices[item]]), item),
                )
                guided.add(position)
            else:
                position = _balanced_split(left, right, min_segment_turns)
                forced.add(position)
            boundaries.add(position)

        ordered = sorted(boundaries)
        segments = tuple(
            DecisionSegment(
                task_id=first_identity.task_id,
                rollout_id=first_identity.rollout_id,
                start=left,
                end=right,
                canonical_turn_indices=indices[left:right],
            )
            for left, right in zip(ordered[:-1], ordered[1:], strict=True)
        )
        if tuple(index for segment in segments for index in segment.canonical_turn_indices) != indices:
            raise RuntimeError("segments must be a complete, ordered, non-overlapping partition")
        if any(segment.num_turns > max_segment_turns for segment in segments):
            raise RuntimeError("maximum segment length was not enforced")
        if num_turns >= min_segment_turns and any(
            segment.num_turns < min_segment_turns for segment in segments
        ):
            raise RuntimeError("minimum segment length was not enforced")
        trajectories.append(
            TrajectorySegmentation(
                task_id=first_identity.task_id,
                rollout_id=first_identity.rollout_id,
                turn_indices=indices,
                semantic_boundaries=tuple(sorted(semantic)),
                correspondence_guided_fallback_boundaries=tuple(sorted(guided)),
                forced_max_boundaries=tuple(sorted(forced)),
                segments=segments,
            )
        )
    return tuple(trajectories), raw_candidate_count, accepted_candidate_count


def aggregate_policy_masked_turn_gap(
    corrected_token_gap: torch.Tensor, policy_loss_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean corrected gap per actual training token and valid-token counts."""

    if corrected_token_gap.ndim != 2 or policy_loss_mask.shape != corrected_token_gap.shape:
        raise ValueError("corrected_token_gap and policy_loss_mask must have shape [M, L]")
    mask = policy_loss_mask.detach().bool()
    lengths = mask.sum(dim=-1)
    if (lengths <= 0).any():
        raise ValueError("every turn participating in Mechanism 2 must have a policy-loss token")
    gap = corrected_token_gap.detach().float()
    if not torch.isfinite(gap).all():
        raise ValueError("corrected_token_gap must be finite")
    turn_gap = (gap * mask.to(gap.dtype)).sum(dim=-1) / lengths.to(gap.dtype)
    return turn_gap.detach(), lengths.detach()


def project_weighted_box_budget(
    raw_density: torch.Tensor,
    token_counts: torch.Tensor,
    density_upper_bound: float,
    tolerance: float = 1e-7,
    max_iterations: int = 100,
    density_lower_bound: float = 0.0,
) -> torch.Tensor:
    """Project densities onto ``[h_min, h_max]`` with fixed token-weighted budget.

    Solves the weighted Euclidean projection from spec-2.  The KKT solution is
    ``clamp(raw - lambda, h_min, h_max)``; ``lambda`` is found by bisection.
    """

    raw = raw_density.detach().float()
    counts = token_counts.detach().to(device=raw.device, dtype=torch.float32)
    if raw.ndim != 1 or counts.shape != raw.shape:
        raise ValueError("raw_density and token_counts must be one-dimensional and aligned")
    if (raw < 0).any() or not torch.isfinite(raw).all():
        raise ValueError("raw_density must be finite and nonnegative")
    if (counts <= 0).any() or not torch.isfinite(counts).all():
        raise ValueError("token_counts must be finite and positive")
    if not math.isfinite(density_lower_bound) or not 0 <= density_lower_bound <= 1:
        raise ValueError("density_lower_bound must be finite and in [0, 1]")
    if not math.isfinite(density_upper_bound) or density_upper_bound < 1.0:
        raise ValueError("density_upper_bound must be at least one")

    target = counts.sum()
    raw_budget_error = ((counts * raw).sum() - target).abs()
    float32_roundoff = 32.0 * torch.finfo(torch.float32).eps * float(target) * max(raw.numel(), 1)
    if raw_budget_error > max(tolerance * float(target), float32_roundoff, 1e-5):
        raise ValueError("raw density does not conserve the token budget")
    if density_lower_bound == 1.0 or density_upper_bound == 1.0:
        return torch.ones_like(raw).detach()
    if bool(((raw <= density_upper_bound) & (raw >= density_lower_bound)).all()):
        return raw.detach()

    lower = float((raw - density_upper_bound).min().item())
    upper = float((raw - density_lower_bound).max().item())
    for _ in range(max_iterations):
        midpoint = 0.5 * (lower + upper)
        proposal = (raw - midpoint).clamp(float(density_lower_bound), float(density_upper_bound))
        budget = (counts * proposal).sum()
        if abs(float(budget - target)) <= tolerance * max(float(target), 1.0):
            lower = midpoint
            upper = midpoint
            break
        if budget > target:
            lower = midpoint
        else:
            upper = midpoint
    projected = (raw - 0.5 * (lower + upper)).clamp(float(density_lower_bound), float(density_upper_bound))

    # Remove the final bisection residual without elementwise clamping.  A tiny
    # common shift over non-bound coordinates preserves the KKT form.
    for _ in range(3):
        residual = target - (counts * projected).sum()
        if abs(float(residual)) <= tolerance * max(float(target), 1.0):
            break
        free = (projected > density_lower_bound) & (projected < density_upper_bound)
        denominator = counts[free].sum()
        if denominator <= 0:
            raise RuntimeError("weighted box projection has no free coordinate for residual correction")
        projected[free] += residual / denominator
    if (projected < density_lower_bound - tolerance).any() or (projected > density_upper_bound + tolerance).any():
        raise RuntimeError("weighted box projection violated its bounds")
    if abs(float((counts * projected).sum() - target)) > 5 * tolerance * max(float(target), 1.0):
        raise RuntimeError("weighted box projection failed to conserve the token budget")
    return projected.clamp(float(density_lower_bound), float(density_upper_bound)).detach()


def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
    positive = probabilities > 0
    return -(probabilities[positive] * probabilities[positive].log()).sum()


def _safe_stats(values: torch.Tensor, prefix: str) -> dict[str, float]:
    value = values.detach().float()
    if value.numel() == 0:
        return {f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0, f"{prefix}_min": 0.0, f"{prefix}_max": 0.0}
    return {
        f"{prefix}_mean": float(value.mean().item()),
        f"{prefix}_std": float(value.std(unbiased=False).item()),
        f"{prefix}_min": float(value.min().item()),
        f"{prefix}_max": float(value.max().item()),
    }


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.detach().float()
    y = right.detach().float()
    if x.numel() <= 1 or x.std(unbiased=False) == 0 or y.std(unbiased=False) == 0:
        return 0.0
    return float((((x - x.mean()) * (y - y.mean())).mean() / (x.std(unbiased=False) * y.std(unbiased=False))).item())


@torch.no_grad()
def compute_flat_turn_credit(gaps, policy_mask, identities, advantage, cfg):
    """Allocate directly over trajectory turns; no profiles, boundaries or spans.

    Neutral mass is the optimized token count N_k. The turn budget is
    softmax(log N_k + sign(A) * mean_token_gap_k / temperature).
    """
    turn_gap, counts = aggregate_policy_masked_turn_gap(gaps, policy_mask)
    counts = counts.float()
    score = advantage.sign() * turn_gap
    raw = torch.zeros_like(turn_gap)
    projected = torch.zeros_like(turn_gap)
    budgets = torch.zeros_like(turn_gap)
    lo = 1 - cfg.multiplier_band if cfg.weighting_mode == "bounded" else 0.0
    hi = 1 + cfg.multiplier_band if cfg.weighting_mode == "bounded" else cfg.density_upper_bound
    max_error = 0.0
    for rows in _trajectory_groups(identities):
        idx = torch.tensor(rows, device=gaps.device)
        a = advantage[idx]
        if not torch.allclose(a, a[:1].expand_as(a), atol=1e-6, rtol=1e-6):
            raise ValueError("trajectory_advantage must be identical across turns of a trajectory")
        n = counts[idx]
        q = torch.softmax(n.log() + score[idx] / cfg.credit_temperature, dim=0)
        budgets[idx] = q
        raw[idx] = n.sum() * q / n
        projected[idx] = project_weighted_box_budget(
            raw[idx], n, hi, tolerance=cfg.projection_tolerance,
            max_iterations=cfg.projection_max_iterations, density_lower_bound=lo)
        w = 1 - cfg.mixing_coefficient + cfg.mixing_coefficient * projected[idx]
        error = abs(float((n * w).sum() - n.sum()))
        max_error = max(max_error, error)
        if error > 1e-5 * max(float(n.sum()), 1.0):
            raise RuntimeError("flat turn weights failed to conserve trajectory credit")
    weight = 1 - cfg.mixing_coefficient + cfg.mixing_coefficient * projected
    m = len(identities)
    zeros = torch.zeros_like(turn_gap)
    flags = torch.zeros(m, dtype=torch.bool, device=gaps.device)
    diagnostics = {"flat_turn_allocation": 1.0, "segment_count": 0.0,
                   "mixed_credit_conservation_error_max": max_error}
    diagnostics.update(_safe_stats(weight, "turn_weight"))
    return CorrespondenceCreditResult(
        dense_profiles=torch.zeros((m, m), device=gaps.device),
        profile_valid=flags, js_divergence=zeros, boundary_valid=flags.clone(),
        segmentation_threshold=float("inf"), trajectory_segmentations=(),
        segment_index_per_turn=torch.full((m,), -1, dtype=torch.long, device=gaps.device),
        turn_gap=turn_gap, outcome_aligned_score=score, segment_budget=zeros[:0],
        within_segment_turn_budget=budgets, raw_credit_density=raw,
        projected_credit_density=projected, turn_weight=weight,
        token_advantage=policy_mask.float() * advantage[:, None] * weight[:, None],
        diagnostics=diagnostics)


def compute_flat_token_credit(gaps, policy_mask, identities, advantage, cfg):
    """Allocate outcome credit directly over optimized tokens, without spans."""
    mask = policy_mask.bool()
    density = torch.zeros_like(gaps, dtype=torch.float32)
    token_weights = torch.ones_like(gaps, dtype=torch.float32)
    max_error = 0.0
    lo = 1 - cfg.multiplier_band if cfg.weighting_mode == "bounded" else 0.0
    hi = 1 + cfg.multiplier_band if cfg.weighting_mode == "bounded" else cfg.density_upper_bound
    for rows in _trajectory_groups(identities):
        idx = torch.tensor(rows, dtype=torch.long, device=gaps.device)
        valid = mask.index_select(0, idx)
        scores = gaps.index_select(0, idx)[valid]
        signs = advantage.index_select(0, idx)[:, None].expand_as(gaps.index_select(0, idx))[valid].sign()
        total = int(valid.sum().item())
        q = torch.softmax(signs * scores / cfg.credit_temperature, dim=0)
        raw = total * q
        counts = torch.ones_like(raw)
        projected = project_weighted_box_budget(
            raw, counts, hi, tolerance=cfg.projection_tolerance,
            max_iterations=cfg.projection_max_iterations, density_lower_bound=lo)
        local_weights = torch.ones_like(gaps.index_select(0, idx))
        local_weights[valid] = projected
        token_weights.index_copy_(0, idx, local_weights)
        error = abs(float(projected.sum() - total))
        max_error = max(max_error, error)
        if error > 1e-5 * max(float(total), 1.0):
            raise RuntimeError("flat token weights failed to conserve trajectory credit")
    counts = mask.sum(dim=-1).float().clamp_min(1)
    turn_weight = (token_weights * mask).sum(dim=-1) / counts
    turn_gap, _ = aggregate_policy_masked_turn_gap(gaps, mask)
    m = len(identities)
    zeros = torch.zeros(m, dtype=torch.float32, device=gaps.device)
    flags = torch.zeros(m, dtype=torch.bool, device=gaps.device)
    diagnostics = {"flat_token_allocation": 1.0, "segment_count": 0.0,
                   "mixed_credit_conservation_error_max": max_error}
    diagnostics.update(_safe_stats(turn_weight, "turn_weight"))
    return CorrespondenceCreditResult(
        dense_profiles=torch.zeros((m, m), device=gaps.device), profile_valid=flags,
        js_divergence=zeros, boundary_valid=flags.clone(), segmentation_threshold=float("inf"),
        trajectory_segmentations=(), segment_index_per_turn=torch.full((m,), -1, dtype=torch.long, device=gaps.device),
        turn_gap=turn_gap, outcome_aligned_score=advantage.sign() * turn_gap,
        segment_budget=zeros[:0], within_segment_turn_budget=turn_weight,
        raw_credit_density=turn_weight, projected_credit_density=turn_weight,
        turn_weight=turn_weight, token_advantage=mask.float() * advantage[:, None] * token_weights,
        diagnostics=diagnostics)


def compute_correspondence_credit_from_tensors(
    *,
    thinking_similarity: torch.Tensor,
    structural_mask: torch.Tensor,
    similarity_threshold: float,
    corrected_token_gap: torch.Tensor,
    policy_loss_mask: torch.Tensor,
    identities: Sequence[TurnIdentity],
    trajectory_advantage: torch.Tensor,
    config: CorrespondenceCreditConfig | None = None,
) -> CorrespondenceCreditResult:
    """Run the complete spec-2 path on canonical tensors."""

    cfg = config or CorrespondenceCreditConfig()
    m = len(identities)
    if m == 0:
        raise ValueError("Mechanism 2 requires at least one canonical turn")
    if thinking_similarity.shape != (m, m) or structural_mask.shape != (m, m):
        raise ValueError("H/B shapes must match the canonical identity count")
    if corrected_token_gap.ndim != 2 or corrected_token_gap.shape[0] != m:
        raise ValueError("corrected_token_gap must have shape [M, response_length]")
    if policy_loss_mask.shape != corrected_token_gap.shape:
        raise ValueError("policy_loss_mask must align with corrected_token_gap")

    device = thinking_similarity.device
    h = _as_fp32(thinking_similarity, device)
    structural = structural_mask.detach().to(device=device, dtype=torch.bool)
    gaps = _as_fp32(corrected_token_gap, device)
    policy_mask = policy_loss_mask.detach().to(device=device, dtype=torch.bool)
    advantage = torch.as_tensor(trajectory_advantage).detach().to(device=device, dtype=torch.float32)
    if advantage.shape != (m,) or not torch.isfinite(advantage).all():
        raise ValueError("trajectory_advantage must be finite with shape [M]")

    if cfg.segmentation_mode == "token":
        return compute_flat_token_credit(gaps, policy_mask, identities, advantage, cfg)
    if cfg.segmentation_mode == "turn":
        return compute_flat_turn_credit(gaps, policy_mask, identities, advantage, cfg)

    profiles, profile_valid = build_dense_correspondence_profiles(
        h, structural, similarity_threshold, cfg.profile_temperature
    )
    jsd, boundary_valid, groups, adjacency_stats = compute_adjacent_profile_distance(
        profiles, profile_valid, structural, identities, distance_mode=cfg.profile_distance
    )
    threshold = adaptive_segmentation_threshold(
        jsd,
        boundary_valid,
        cfg.segmentation_quantile,
        cfg.segmentation_threshold_min,
        cfg.segmentation_threshold_max,
    )
    trajectory_segmentations, raw_candidate_count, accepted_candidate_count = segment_trajectories(
        identities=identities,
        groups=groups,
        js_divergence=jsd,
        boundary_valid=boundary_valid,
        segmentation_threshold=threshold,
        min_segment_turns=cfg.min_segment_turns,
        max_segment_turns=cfg.max_segment_turns,
        fallback_jsd_min=cfg.segmentation_threshold_min,
        segmentation_mode=cfg.segmentation_mode,
    )

    turn_gap, token_counts_long = aggregate_policy_masked_turn_gap(gaps, policy_mask)
    token_counts = token_counts_long.to(device=device, dtype=torch.float32)
    aligned_score = torch.sign(advantage) * turn_gap

    segment_index_per_turn = torch.full((m,), -1, dtype=torch.long, device=device)
    all_segments: list[DecisionSegment] = []
    for trajectory in trajectory_segmentations:
        for segment in trajectory.segments:
            segment_index = len(all_segments)
            all_segments.append(segment)
            segment_index_per_turn[list(segment.canonical_turn_indices)] = segment_index
    if (segment_index_per_turn < 0).any():
        raise RuntimeError("segmentation did not cover every canonical turn")

    segment_budget = torch.zeros(len(all_segments), dtype=torch.float32, device=device)
    turn_budget = torch.zeros(m, dtype=torch.float32, device=device)
    raw_density = torch.zeros(m, dtype=torch.float32, device=device)
    projected_density = torch.zeros(m, dtype=torch.float32, device=device)
    segment_entropies: list[float] = []
    within_segment_entropies: list[float] = []
    projection_trajectory_count = 0
    density_min = 1.0 - cfg.multiplier_band if cfg.weighting_mode == "bounded" else 0.0
    density_max = 1.0 + cfg.multiplier_band if cfg.weighting_mode == "bounded" else cfg.density_upper_bound
    conservation_errors: list[float] = []

    for trajectory in trajectory_segmentations:
        trajectory_indices = torch.tensor(trajectory.turn_indices, dtype=torch.long, device=device)
        trajectory_advantages = advantage.index_select(0, trajectory_indices)
        if not torch.allclose(
            trajectory_advantages,
            trajectory_advantages[:1].expand_as(trajectory_advantages),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError("trajectory_advantage must be identical across turns of a trajectory")

        local_segments = trajectory.segments
        global_segment_indices = torch.tensor(
            [int(segment_index_per_turn[segment.canonical_turn_indices[0]]) for segment in local_segments],
            dtype=torch.long,
            device=device,
        )
        segment_scores = []
        segment_tokens = []
        for segment in local_segments:
            indices = torch.tensor(segment.canonical_turn_indices, dtype=torch.long, device=device)
            segment_scores.append(aligned_score.index_select(0, indices).mean())
            segment_tokens.append(token_counts.index_select(0, indices).sum())
        segment_scores_tensor = torch.stack(segment_scores)
        segment_tokens_tensor = torch.stack(segment_tokens)
        b = torch.softmax(
            segment_tokens_tensor.log() + segment_scores_tensor / cfg.credit_temperature,
            dim=0,
        )
        segment_budget[global_segment_indices] = b
        segment_entropies.append(float(_entropy(b).item()))

        total_tokens = token_counts.index_select(0, trajectory_indices).sum()
        for local_segment_index, segment in enumerate(local_segments):
            indices = torch.tensor(segment.canonical_turn_indices, dtype=torch.long, device=device)
            q = torch.softmax(
                token_counts.index_select(0, indices).log()
                + aligned_score.index_select(0, indices) / cfg.credit_temperature,
                dim=0,
            )
            turn_budget[indices] = q
            within_segment_entropies.append(float(_entropy(q).item()))
            raw_density[indices] = (
                total_tokens * b[local_segment_index] * q / token_counts.index_select(0, indices)
            )

        raw_for_trajectory = raw_density.index_select(0, trajectory_indices)
        projected = project_weighted_box_budget(
            raw_for_trajectory,
            token_counts.index_select(0, trajectory_indices),
            density_max,
            tolerance=cfg.projection_tolerance,
            max_iterations=cfg.projection_max_iterations,
            density_lower_bound=density_min,
        )
        projected_density[trajectory_indices] = projected
        projection_trajectory_count += int(bool(
            ((raw_for_trajectory > density_max) | (raw_for_trajectory < density_min)).any()
        ))
        conservation_errors.append(
            abs(float((token_counts.index_select(0, trajectory_indices) * projected).sum() - total_tokens))
        )

    turn_weight = (1.0 - cfg.mixing_coefficient) + cfg.mixing_coefficient * projected_density
    token_advantage = (
        policy_mask.to(torch.float32)
        * advantage.unsqueeze(-1)
        * turn_weight.detach().unsqueeze(-1)
    )

    final_conservation_errors = []
    for trajectory in trajectory_segmentations:
        indices = torch.tensor(trajectory.turn_indices, dtype=torch.long, device=device)
        counts = token_counts.index_select(0, indices)
        total = counts.sum()
        error = abs(float((counts * turn_weight.index_select(0, indices)).sum() - total))
        final_conservation_errors.append(error)
        if error > 1e-5 * max(float(total), 1.0):
            raise RuntimeError("mixed turn weights failed to conserve trajectory credit")

    valid_profile_probabilities = profiles[profile_valid]
    if valid_profile_probabilities.numel():
        profile_entropy = torch.stack([_entropy(row) for row in valid_profile_probabilities])
        profile_max = valid_profile_probabilities.max(dim=-1).values
        top_count = min(3, profiles.shape[-1])
        profile_top_mass = valid_profile_probabilities.topk(top_count, dim=-1).values.sum(dim=-1)
        effective_support = profile_entropy.exp()
    else:
        profile_entropy = profile_max = profile_top_mass = effective_support = torch.empty(0, device=device)
    valid_jsd = jsd[boundary_valid]
    segment_lengths = torch.tensor([segment.num_turns for segment in all_segments], dtype=torch.float32, device=device)
    diagnostic_values: dict[str, float] = {
        "profile_valid_rate": float(profile_valid.float().mean().item()) if m else 0.0,
        "valid_jsd_sample_count": float(valid_jsd.numel()),
        "valid_profile_distance_sample_count": float(valid_jsd.numel()),
        "profile_distance_mode_jsd": float(cfg.profile_distance == "jsd"),
        "profile_distance_mode_entropy": float(cfg.profile_distance == "entropy"),
        "profile_distance_mode_cosine": float(cfg.profile_distance == "cosine"),
        # Keep infinity in the result to mean "no semantic threshold", but
        # expose only finite values to W&B/JSON metrics.
        "segmentation_threshold": float(threshold) if math.isfinite(threshold) else -1.0,
        "segmentation_threshold_available": float(math.isfinite(threshold)),
        "near_zero_jsd_rate": float((valid_jsd <= 1e-8).float().mean().item()) if valid_jsd.numel() else 0.0,
        "near_zero_profile_distance_rate": float((valid_jsd <= 1e-8).float().mean().item()) if valid_jsd.numel() else 0.0,
        "raw_semantic_candidate_count": float(raw_candidate_count),
        "accepted_semantic_boundary_count": float(accepted_candidate_count),
        "semantic_boundary_count": float(sum(len(item.semantic_boundaries) for item in trajectory_segmentations)),
        "correspondence_guided_fallback_boundary_count": float(
            sum(len(item.correspondence_guided_fallback_boundaries) for item in trajectory_segmentations)
        ),
        "forced_max_boundary_count": float(sum(len(item.forced_max_boundaries) for item in trajectory_segmentations)),
        "segment_count": float(len(all_segments)),
        "segment_budget_entropy_mean": sum(segment_entropies) / max(len(segment_entropies), 1),
        "within_segment_turn_entropy_mean": sum(within_segment_entropies) / max(len(within_segment_entropies), 1),
        "raw_density_over_cap_rate": float((raw_density > density_max).float().mean().item()) if m else 0.0,
        "raw_density_under_floor_rate": float((raw_density < density_min).float().mean().item()) if m else 0.0,
        "bounded_weighting_enabled": float(cfg.weighting_mode == "bounded"),
        "effective_density_lower_bound": density_min,
        "effective_density_upper_bound": density_max,
        "configured_turn_weight_min": 1.0 - cfg.mixing_coefficient + cfg.mixing_coefficient * density_min,
        "configured_turn_weight_max": 1.0 - cfg.mixing_coefficient + cfg.mixing_coefficient * density_max,
        "projection_activation_rate": projection_trajectory_count / max(len(trajectory_segmentations), 1),
        "projection_mean_absolute_change": float((projected_density - raw_density).abs().mean().item()) if m else 0.0,
        "credit_conservation_error_max": max(conservation_errors, default=0.0),
        "mixed_credit_conservation_error_max": max(final_conservation_errors, default=0.0),
        "weight_turn_length_correlation": _pearson(turn_weight, token_counts),
        "weight_turn_gap_correlation": _pearson(turn_weight, turn_gap),
        "weight_abs_turn_gap_correlation": _pearson(turn_weight, turn_gap.abs()),
        "weight_outcome_aligned_score_correlation": _pearson(turn_weight, aligned_score),
        "positive_outcome_alignment_rate": float((aligned_score > 0).float().mean().item()) if m else 0.0,
    }
    diagnostic_values.update(adjacency_stats)
    diagnostic_values.update(_safe_stats(profile_entropy, "profile_entropy"))
    diagnostic_values.update(_safe_stats(profile_max, "profile_max_probability"))
    diagnostic_values.update(_safe_stats(profile_top_mass, "profile_top3_mass"))
    diagnostic_values.update(_safe_stats(effective_support, "profile_effective_support"))
    diagnostic_values.update(_safe_stats(valid_jsd, "normalized_jsd"))
    diagnostic_values.update(_safe_stats(valid_jsd, "profile_distance"))
    if valid_jsd.numel():
        diagnostic_values.update(
            {
                "normalized_jsd_p50": float(torch.quantile(valid_jsd, 0.50).item()),
                "normalized_jsd_p80": float(torch.quantile(valid_jsd, 0.80).item()),
                "normalized_jsd_p90": float(torch.quantile(valid_jsd, 0.90).item()),
                "profile_distance_p50": float(torch.quantile(valid_jsd, 0.50).item()),
                "profile_distance_p80": float(torch.quantile(valid_jsd, 0.80).item()),
                "profile_distance_p90": float(torch.quantile(valid_jsd, 0.90).item()),
            }
        )
    diagnostic_values.update(_safe_stats(segment_lengths, "segment_length"))
    diagnostic_values.update(_safe_stats(turn_gap, "turn_gap"))
    diagnostic_values.update(_safe_stats(aligned_score, "outcome_aligned_score"))
    diagnostic_values.update(_safe_stats(raw_density, "raw_credit_density"))
    diagnostic_values.update(_safe_stats(projected_density, "projected_credit_density"))
    diagnostic_values.update(_safe_stats(turn_weight, "turn_weight"))
    if turn_weight.numel():
        diagnostic_values["turn_weight_p95"] = float(torch.quantile(turn_weight, 0.95).item())

    result = CorrespondenceCreditResult(
        dense_profiles=profiles.detach(),
        profile_valid=profile_valid.detach(),
        js_divergence=jsd.detach(),
        boundary_valid=boundary_valid.detach(),
        segmentation_threshold=threshold,
        trajectory_segmentations=trajectory_segmentations,
        segment_index_per_turn=segment_index_per_turn.detach(),
        turn_gap=turn_gap.detach(),
        outcome_aligned_score=aligned_score.detach(),
        segment_budget=segment_budget.detach(),
        within_segment_turn_budget=turn_budget.detach(),
        raw_credit_density=raw_density.detach(),
        projected_credit_density=projected_density.detach(),
        turn_weight=turn_weight.detach(),
        token_advantage=token_advantage.detach(),
        diagnostics=diagnostic_values,
    )

    lower_bound = 1.0 - cfg.mixing_coefficient + cfg.mixing_coefficient * density_min
    upper_bound = 1.0 - cfg.mixing_coefficient + cfg.mixing_coefficient * density_max
    if (result.turn_weight < lower_bound - 1e-5).any() or (result.turn_weight > upper_bound + 1e-5).any():
        raise RuntimeError("final turn weights violated their configured bounds")
    return result


def compute_correspondence_credit(
    mechanism1_result: Mechanism1Result,
    trajectory_advantage: torch.Tensor,
    config: CorrespondenceCreditConfig | None = None,
) -> CorrespondenceCreditResult:
    """Run Mechanism 2 directly from the canonical Mechanism-1 sidecar."""

    if mechanism1_result.matrix_layout != "source_by_target":
        raise ValueError("Mechanism 2 requires H[source, target]")
    return compute_correspondence_credit_from_tensors(
        thinking_similarity=mechanism1_result.thinking_similarity,
        structural_mask=mechanism1_result.structural_mask,
        similarity_threshold=mechanism1_result.similarity_threshold,
        corrected_token_gap=mechanism1_result.corrected_token_gap,
        policy_loss_mask=mechanism1_result.policy_loss_mask,
        identities=mechanism1_result.turn_map.identities,
        trajectory_advantage=trajectory_advantage,
        config=config,
    )


__all__ = [
    "CorrespondenceCreditConfig",
    "CorrespondenceCreditResult",
    "DecisionSegment",
    "TrajectorySegmentation",
    "adaptive_segmentation_threshold",
    "aggregate_policy_masked_turn_gap",
    "build_dense_correspondence_profiles",
    "compute_adjacent_jsd",
    "compute_adjacent_profile_distance",
    "compute_correspondence_credit",
    "compute_correspondence_credit_from_tensors",
    "compute_invalid_action_advantage_residual",
    "compute_unique_trajectory_grpo_advantage",
    "normalized_js_divergence",
    "normalized_entropy_distance",
    "cosine_profile_distance",
    "profile_distance",
    "project_weighted_box_budget",
    "segment_trajectories",
]
