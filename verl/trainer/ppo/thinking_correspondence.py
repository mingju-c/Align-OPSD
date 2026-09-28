"""Thinking-correspondence primitives for Mechanism 1.

The code in this module is deliberately independent of rewards and advantages.
It operates on canonical real turns, builds a dense source-by-target similarity
matrix, selects sparse cross-teacher pairs, and combines diagonal and cross
teacher-forced scores.  Trainer-specific model calls live in
``thinking_correspondence_trainer.py``.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True, order=True)
class TurnIdentity:
    """Identity of one real decision turn.

    Only rows with the exact same triple may share canonical computation.
    """

    task_id: str
    rollout_id: str
    turn_id: int


@dataclass(frozen=True)
class CanonicalTurnMap:
    """Bidirectional mapping between adjusted batch rows and real turns."""

    identities: tuple[TurnIdentity, ...]
    canonical_row_indices: tuple[int, ...]
    row_to_canonical: tuple[int, ...]
    canonical_to_rows: tuple[tuple[int, ...], ...]

    @property
    def num_canonical_turns(self) -> int:
        return len(self.identities)


@dataclass(frozen=True)
class ParsedResponse:
    thinking: str
    action: str
    thinking_valid: bool
    action_valid: bool


@dataclass
class EncoderDiagnostics:
    num_texts: int = 0
    invalid_texts: int = 0
    truncated_texts: int = 0
    encoded_tokens: int = 0
    max_encoded_tokens: int = 0
    num_batches: int = 0
    tokenization_seconds: float = 0.0
    transfer_seconds: float = 0.0
    forward_seconds: float = 0.0

    @property
    def truncation_rate(self) -> float:
        return self.truncated_texts / max(self.num_texts, 1)


@dataclass
class Mechanism1Result:
    """Detached canonical sidecar consumed by standalone M1 and future M2.

    Matrix direction is fixed: ``thinking_similarity[source, target]``.
    Dense tensors are never expanded to framework-created duplicate rows.
    """

    turn_map: CanonicalTurnMap
    thinking_similarity: torch.Tensor
    structural_mask: torch.Tensor
    selected_source_indices: tuple[tuple[int, ...], ...]
    selected_source_weights: tuple[torch.Tensor, ...]
    correspondence_confidence: torch.Tensor
    correction_strength: torch.Tensor
    diagonal_teacher_log_probs: torch.Tensor
    cross_teacher_log_probs: torch.Tensor
    corrected_teacher_log_probs: torch.Tensor
    diagonal_token_gap: torch.Tensor
    corrected_token_gap: torch.Tensor
    sampled_response_mask: torch.Tensor
    policy_loss_mask: torch.Tensor
    similarity_threshold: float
    alpha_max: float
    encoder_contract: dict[str, Any]
    matrix_layout: str = "source_by_target"
    student_thinking: tuple[str, ...] = ()
    teacher_thinking: tuple[str, ...] = ()
    student_actions: tuple[str, ...] = ()
    teacher_actions: tuple[str, ...] = ()
    action_similarity: torch.Tensor | None = None
    diagnostics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        m = self.turn_map.num_canonical_turns
        if self.matrix_layout != "source_by_target":
            raise ValueError("Mechanism 1 matrix layout must be source_by_target")
        if not -1.0 <= self.similarity_threshold < 1.0:
            raise ValueError("similarity_threshold must be in [-1, 1)")
        if not 0.0 <= self.alpha_max <= 1.0:
            raise ValueError("alpha_max must be in [0, 1]")
        if self.thinking_similarity.shape != (m, m):
            raise ValueError("thinking_similarity must be source-by-target [M, M]")
        if self.structural_mask.shape != (m, m):
            raise ValueError("structural_mask must have the same [M, M] shape")
        if self.structural_mask.dtype != torch.bool:
            raise ValueError("structural_mask must be boolean")
        if self.thinking_similarity.dtype != torch.float32:
            raise ValueError("thinking_similarity must be FP32")
        if not torch.isfinite(self.thinking_similarity).all():
            raise ValueError("thinking_similarity must be finite")
        if ((self.thinking_similarity < -1.000001) | (self.thinking_similarity > 1.000001)).any():
            raise ValueError("thinking_similarity must lie in [-1, 1]")
        if self.action_similarity is not None:
            if self.action_similarity.shape != (m, m):
                raise ValueError("action_similarity must be turn-by-turn [M, M]")
            if self.action_similarity.dtype != torch.float32 or not torch.isfinite(self.action_similarity).all():
                raise ValueError("action_similarity must be finite FP32")

        token_shape = self.diagonal_teacher_log_probs.shape
        if len(token_shape) != 2 or token_shape[0] != m:
            raise ValueError("canonical token tensors must have shape [M, response_length]")
        for name in (
            "cross_teacher_log_probs",
            "corrected_teacher_log_probs",
            "diagonal_token_gap",
            "corrected_token_gap",
            "sampled_response_mask",
            "policy_loss_mask",
        ):
            if getattr(self, name).shape != token_shape:
                raise ValueError(f"{name} must have shape {tuple(token_shape)}")
        for name in (
            "diagonal_teacher_log_probs",
            "cross_teacher_log_probs",
            "corrected_teacher_log_probs",
            "diagonal_token_gap",
            "corrected_token_gap",
        ):
            if not torch.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} must be finite")
        if self.correspondence_confidence.shape != (m,) or self.correction_strength.shape != (m,):
            raise ValueError("confidence and correction strength must have shape [M]")
        if len(self.selected_source_indices) != m or len(self.selected_source_weights) != m:
            raise ValueError("sparse selections must have one entry per target")
        if (self.policy_loss_mask.bool() & ~self.sampled_response_mask.bool()).any():
            raise ValueError("policy_loss_mask must be a subset of sampled_response_mask")
        if (self.policy_loss_mask.sum(dim=-1) <= 0).any():
            raise ValueError("every canonical turn must contain a policy-loss token")
        if not torch.isfinite(self.correspondence_confidence).all():
            raise ValueError("correspondence_confidence must be finite")
        if not torch.isfinite(self.correction_strength).all():
            raise ValueError("correction_strength must be finite")
        if ((self.correspondence_confidence < 0) | (self.correspondence_confidence > 1)).any():
            raise ValueError("correspondence_confidence must lie in [0, 1]")
        if ((self.correction_strength < 0) | (self.correction_strength > self.alpha_max + 1e-7)).any():
            raise ValueError("correction_strength must lie in [0, alpha_max]")

        for target, (sources, weights) in enumerate(
            zip(self.selected_source_indices, self.selected_source_weights, strict=True)
        ):
            if weights.shape != (len(sources),):
                raise ValueError("each sparse weight vector must align with its selected sources")
            if sources and not torch.allclose(weights.float().sum(), weights.new_tensor(1.0).float(), atol=1e-6):
                raise ValueError("non-empty sparse source weights must sum to one")
            for source in sources:
                if not self.structural_mask[source, target]:
                    raise ValueError("selected sources must be structurally eligible")

        if len(self.turn_map.row_to_canonical) != sum(len(rows) for rows in self.turn_map.canonical_to_rows):
            raise ValueError("canonical mapping row counts are inconsistent")
        row_count = len(self.turn_map.row_to_canonical)
        for canonical_index, rows in enumerate(self.turn_map.canonical_to_rows):
            if not rows or self.turn_map.canonical_row_indices[canonical_index] != rows[0]:
                raise ValueError("canonical mapping must retain the first row for each identity")
            if any(row < 0 or row >= row_count for row in rows):
                raise ValueError("canonical mapping contains an out-of-range row")
            if any(self.turn_map.row_to_canonical[row] != canonical_index for row in rows):
                raise ValueError("canonical mapping is not invertible")
        for value in self.__dict__.values():
            if isinstance(value, torch.Tensor) and value.requires_grad:
                raise ValueError("Mechanism1Result tensors must be detached")
        for weights in self.selected_source_weights:
            if weights.requires_grad:
                raise ValueError("Mechanism1Result sparse weights must be detached")


_THINK_RE = re.compile(r"<think\b[^>]*>(.*?)</think\s*>", re.IGNORECASE | re.DOTALL)
_ACTION_PATTERNS = (
    re.compile(r"<action\b[^>]*>(.*?)</action\s*>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<search\b[^>]*>(.*?)</search\s*>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<answer\b[^>]*>(.*?)</answer\s*>", re.IGNORECASE | re.DOTALL),
)


def parse_tagged_response(text: str | None) -> ParsedResponse:
    """Extract inner thinking/action text, stripping surrounding whitespace only."""

    value = "" if text is None else str(text)
    think_match = _THINK_RE.search(value)
    thinking = think_match.group(1).strip() if think_match else ""

    action = ""
    for pattern in _ACTION_PATTERNS:
        match = pattern.search(value)
        if match:
            action = match.group(1).strip()
            break

    return ParsedResponse(
        thinking=thinking,
        action=action,
        thinking_valid=bool(thinking),
        action_valid=bool(action),
    )


def canonicalize_turns(
    task_ids: Sequence[Any], rollout_ids: Sequence[Any], turn_ids: Sequence[Any]
) -> CanonicalTurnMap:
    """Canonicalize exact framework copies and nothing else.

    Canonical identities are sorted to keep source coordinates stable under
    batch reordering.  The first row for an identity supplies its tensors.
    """

    if not (len(task_ids) == len(rollout_ids) == len(turn_ids)):
        raise ValueError("identity fields must have equal lengths")

    row_keys = tuple(
        TurnIdentity(str(task), str(rollout), int(turn))
        for task, rollout, turn in zip(task_ids, rollout_ids, turn_ids, strict=True)
    )
    identities = tuple(sorted(set(row_keys)))
    identity_to_index = {identity: index for index, identity in enumerate(identities)}
    rows: list[list[int]] = [[] for _ in identities]
    row_to_canonical: list[int] = []
    for row_index, identity in enumerate(row_keys):
        canonical_index = identity_to_index[identity]
        row_to_canonical.append(canonical_index)
        rows[canonical_index].append(row_index)

    canonical_row_indices = tuple(group[0] for group in rows)
    return CanonicalTurnMap(
        identities=identities,
        canonical_row_indices=canonical_row_indices,
        row_to_canonical=tuple(row_to_canonical),
        canonical_to_rows=tuple(tuple(group) for group in rows),
    )


def expand_canonical_tensor(value: torch.Tensor, turn_map: CanonicalTurnMap) -> torch.Tensor:
    """Expand a canonical leading dimension back to adjusted batch rows."""

    if value.shape[0] != turn_map.num_canonical_turns:
        raise ValueError("tensor leading dimension does not match canonical turns")
    index = torch.tensor(turn_map.row_to_canonical, dtype=torch.long, device=value.device)
    return value.index_select(0, index)


def canonical_multiplicity_row_weights(
    turn_map: CanonicalTurnMap,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Give all adjusted copies a total optimization mass of one per real turn.

    A canonical turn represented by ``k`` framework rows receives weight ``1/k``
    on every row.  The weights stay separate from the policy mask so actor-side
    loss normalization cannot cancel them inside individual micro-batches.
    """

    canonical_weights = torch.tensor(
        [1.0 / len(rows) for rows in turn_map.canonical_to_rows],
        dtype=torch.float32,
        device=device,
    )
    row_weights = expand_canonical_tensor(canonical_weights, turn_map)
    for canonical_index, rows in enumerate(turn_map.canonical_to_rows):
        indices = torch.tensor(rows, dtype=torch.long, device=row_weights.device)
        total = row_weights.index_select(0, indices).sum()
        if not torch.allclose(total, total.new_tensor(1.0), atol=1e-6, rtol=0.0):
            raise RuntimeError(
                f"canonical row weights do not sum to one for turn {canonical_index}"
            )
    return row_weights.detach()


def select_policy_loss_mask(
    responses: torch.Tensor,
    attention_mask: torch.Tensor,
    multi_turn: bool,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single tensor-level source of truth for the actor's response mask."""

    response_length = responses.shape[-1]
    if multi_turn:
        if loss_mask is None:
            raise KeyError("multi-turn policy loss requires loss_mask")
        return loss_mask[:, -response_length:]
    return attention_mask[:, -response_length:]


def compute_policy_loss_mask(batch, multi_turn: bool) -> torch.Tensor:
    """DataProto wrapper around :func:`select_policy_loss_mask`."""

    return select_policy_loss_mask(
        responses=batch.batch["responses"],
        attention_mask=batch.batch["attention_mask"],
        multi_turn=multi_turn,
        loss_mask=batch.batch["loss_mask"] if "loss_mask" in batch.batch else None,
    )


def compose_cross_teacher_inputs(
    teacher_prompt_ids: torch.Tensor,
    teacher_prompt_mask: torch.Tensor,
    student_responses: torch.Tensor,
    student_response_mask: torch.Tensor,
    pairs: Sequence[tuple[int, int]],
) -> dict[str, torch.Tensor]:
    """Compose source Teacher contexts with target Student sampled responses."""

    if not pairs:
        raise ValueError("at least one source-target pair is required")
    source_indices = torch.tensor(
        [source for source, _ in pairs], dtype=torch.long, device=teacher_prompt_ids.device
    )
    target_indices = torch.tensor(
        [target for _, target in pairs], dtype=torch.long, device=student_responses.device
    )
    source_prompts = teacher_prompt_ids.index_select(0, source_indices)
    source_masks = teacher_prompt_mask.index_select(0, source_indices)
    target_responses = student_responses.index_select(0, target_indices)
    target_masks = student_response_mask.index_select(0, target_indices)
    return {
        "input_ids": torch.cat((source_prompts, target_responses), dim=-1),
        "attention_mask": torch.cat((source_masks, target_masks), dim=-1),
        "responses": target_responses,
    }


def build_structural_mask(
    identities: Sequence[TurnIdentity],
    teacher_thinking_valid: torch.Tensor,
    student_thinking_valid: torch.Tensor,
    protocol_ids: Sequence[Any] | None = None,
) -> torch.Tensor:
    """Build dense pre-threshold structural eligibility ``B[source, target]``."""

    m = len(identities)
    if teacher_thinking_valid.shape != (m,) or student_thinking_valid.shape != (m,):
        raise ValueError("thinking validity vectors must have shape [M]")
    if protocol_ids is not None and len(protocol_ids) != m:
        raise ValueError("protocol_ids must have length M")

    device = teacher_thinking_valid.device

    def encode_categories(values: Sequence[Any]) -> torch.Tensor:
        category = {value: index for index, value in enumerate(sorted({str(item) for item in values}))}
        return torch.tensor([category[str(item)] for item in values], dtype=torch.long, device=device)

    task = encode_categories([identity.task_id for identity in identities])
    rollout = encode_categories([identity.rollout_id for identity in identities])
    mask = task[:, None].eq(task[None, :]) & rollout[:, None].ne(rollout[None, :])
    mask &= teacher_thinking_valid.bool()[:, None] & student_thinking_valid.bool()[None, :]
    if protocol_ids is not None:
        protocol = encode_categories(protocol_ids)
        mask &= protocol[:, None].eq(protocol[None, :])
    return mask


def select_sparse_sources(
    thinking_similarity: torch.Tensor,
    structural_mask: torch.Tensor,
    identities: Sequence[TurnIdentity],
    similarity_threshold: float = 0.70,
    source_rollout_cap: int = 1,
    top_k: int = 3,
) -> tuple[tuple[int, ...], ...]:
    """Apply threshold, per-rollout cap, and global top-K deterministically."""

    if thinking_similarity.ndim != 2 or thinking_similarity.shape[0] != thinking_similarity.shape[1]:
        raise ValueError("thinking_similarity must be square")
    if structural_mask.shape != thinking_similarity.shape:
        raise ValueError("structural_mask shape mismatch")
    if len(identities) != thinking_similarity.shape[0]:
        raise ValueError("identity count mismatch")
    if not -1.0 <= similarity_threshold < 1.0:
        raise ValueError("similarity_threshold must be in [-1, 1)")
    if source_rollout_cap != 1:
        raise ValueError("Mechanism 1 requires source_rollout_cap=1")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    h = thinking_similarity.detach().float().cpu()
    eligible = structural_mask.detach().bool().cpu()
    selected_by_target: list[tuple[int, ...]] = []
    for target in range(h.shape[1]):
        by_rollout: dict[str, list[int]] = {}
        eligible_sources = torch.nonzero(
            eligible[:, target] & h[:, target].ge(similarity_threshold), as_tuple=True
        )[0].tolist()
        for source in eligible_sources:
            if float(h[source, target]) >= similarity_threshold:
                by_rollout.setdefault(identities[source].rollout_id, []).append(source)

        after_cap: list[int] = []
        for rollout_id in sorted(by_rollout):
            ranked = sorted(
                by_rollout[rollout_id],
                key=lambda source: (-float(h[source, target]), identities[source]),
            )
            after_cap.extend(ranked[:source_rollout_cap])
        ranked = sorted(
            after_cap,
            key=lambda source: (-float(h[source, target]), identities[source]),
        )
        selected_by_target.append(tuple(ranked[:top_k]))
    return tuple(selected_by_target)


def combine_cross_teacher_scores(
    diagonal_teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    thinking_similarity: torch.Tensor,
    selected_source_indices: Sequence[Sequence[int]],
    cross_scores: Mapping[tuple[int, int], torch.Tensor],
    similarity_threshold: float = 0.70,
    aggregation_temperature: float = 0.10,
    alpha_max: float = 0.20,
    aggregation_mode: str = "probability_mixture",
) -> dict[str, torch.Tensor]:
    """Mix source probabilities, then apply the unchanged diagonal log-score blend.

    log_mean retains the historical cross-source geometric score. Both
    alternatives use the same sampled-token scores for paired diagnostics;
    neither requires another teacher forward or full-vocabulary tensors.
    Score arithmetic is FP32 and detached.
    """

    if diagonal_teacher_log_probs.shape != student_log_probs.shape:
        raise ValueError("teacher and student log-prob shapes must match")
    m, response_length = diagonal_teacher_log_probs.shape
    if thinking_similarity.shape != (m, m):
        raise ValueError("thinking_similarity must be [M, M]")
    if len(selected_source_indices) != m:
        raise ValueError("selected_source_indices must have one entry per target")
    if not -1.0 <= similarity_threshold < 1.0:
        raise ValueError("similarity_threshold must be in [-1, 1)")
    if aggregation_temperature <= 0:
        raise ValueError("aggregation_temperature must be positive")
    if not 0.0 <= alpha_max <= 1.0:
        raise ValueError("alpha_max must be in [0, 1]")

    if aggregation_mode not in {"probability_mixture", "log_mean"}:
        raise ValueError("aggregation_mode must be probability_mixture or log_mean")
    device = diagonal_teacher_log_probs.device
    diagonal = diagonal_teacher_log_probs.detach().float()
    student = student_log_probs.detach().to(device=device, dtype=torch.float32)
    probability_cross = diagonal.clone()
    log_mean_cross = diagonal.clone()
    selected_weights: list[torch.Tensor] = []
    rho = torch.zeros(m, dtype=torch.float32, device=device)

    for target, sources in enumerate(selected_source_indices):
        if not sources:
            selected_weights.append(torch.empty(0, dtype=torch.float32, device=device))
            continue
        source_index = torch.tensor(tuple(sources), dtype=torch.long, device=thinking_similarity.device)
        similarities = thinking_similarity[source_index, target].detach().float()
        log_weights = torch.log_softmax(similarities / float(aggregation_temperature), dim=0)
        weights = log_weights.exp()
        stacked_scores = []
        for source in sources:
            key = (int(source), int(target))
            if key not in cross_scores:
                raise KeyError(f"missing cross-teacher score for pair {key}")
            score = cross_scores[key]
            if score.shape != (response_length,):
                raise ValueError(f"cross score for {key} must have shape [{response_length}]")
            stacked_scores.append(score.detach().to(device=device, dtype=torch.float32))
        stacked = torch.stack(stacked_scores, dim=0)
        log_mean_cross[target] = (weights.to(device).unsqueeze(-1) * stacked).sum(dim=0)
        # log(sum_k omega_k * p_k(y_r)), not sum_k omega_k * log p_k(y_r).
        probability_cross[target] = torch.logsumexp(
            log_weights.to(device).unsqueeze(-1) + stacked, dim=0
        )
        selected_weights.append(weights.to(device).detach())
        normalized_similarity = ((similarities - similarity_threshold) / (1.0 - similarity_threshold)).clamp(0, 1)
        rho[target] = (weights * normalized_similarity).sum().to(device)

    alpha = (float(alpha_max) * rho).clamp(0.0, float(alpha_max))
    cross_teacher = probability_cross if aggregation_mode == "probability_mixture" else log_mean_cross
    corrected_teacher = (
        (1.0 - alpha.unsqueeze(-1)) * diagonal
        + alpha.unsqueeze(-1) * cross_teacher
    )
    diagonal_gap = diagonal - student
    corrected_gap = corrected_teacher - student
    return {
        "selected_source_weights": tuple(selected_weights),
        "correspondence_confidence": rho.detach(),
        "correction_strength": alpha.detach(),
        "cross_teacher_log_probs": cross_teacher.detach(),
        "log_mean_cross_teacher_log_probs": log_mean_cross.detach(),
        "probability_mixture_cross_teacher_log_probs": probability_cross.detach(),
        "corrected_teacher_log_probs": corrected_teacher.detach(),
        "diagonal_token_gap": diagonal_gap.detach(),
        "corrected_token_gap": corrected_gap.detach(),
    }


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the final non-padding token, as required by Qwen3-Embedding."""

    if last_hidden_states.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("last hidden states and attention mask must have shapes [B, L, D] and [B, L]")
    if last_hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError("last hidden states and attention mask sequence dimensions must match")
    if (attention_mask.sum(dim=1) <= 0).any():
        raise ValueError("cannot pool an all-padding sequence")
    if bool(attention_mask[:, -1].all()):
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1).long() - 1
    batch_indices = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
    return last_hidden_states[batch_indices, sequence_lengths]


class HFThinkingEncoder:
    """Frozen offline Qwen3 model for symmetric thinking-turn embeddings."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        max_length: int = 4096,
        batch_size: int = 4,
        pooling: str = "last_token",
        padding_side: str = "left",
        dtype: str = "auto",
        revision: str | None = None,
        expected_model_sha256: str | None = None,
        truncation_side: str = "right",
        attention_implementation: str = "sdpa",
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        if max_length < 1 or batch_size < 1:
            raise ValueError("max_length and batch_size must be positive")
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.pooling = str(pooling)
        self.padding_side = str(padding_side)
        self.attention_implementation = str(attention_implementation)
        self.revision = revision
        if self.pooling not in {"last_token", "mean"}:
            raise ValueError("encoder pooling must be last_token or mean")
        if self.padding_side not in {"left", "right"}:
            raise ValueError("encoder padding_side must be left or right")
        if truncation_side not in {"left", "right"}:
            raise ValueError("encoder truncation_side must be left or right")

        if expected_model_sha256:
            model_file = Path(model_path) / "model.safetensors"
            if not model_file.is_file():
                raise FileNotFoundError(f"encoder weight file not found: {model_file}")
            digest = hashlib.sha256()
            with model_file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_sha256.lower() != str(expected_model_sha256).lower():
                raise ValueError(
                    f"encoder model.safetensors SHA-256 mismatch: {actual_sha256} "
                    f"!= {expected_model_sha256}"
                )

        if dtype == "auto":
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
                model_dtype = torch.bfloat16
            elif self.device.type == "cuda":
                model_dtype = torch.float16
            else:
                model_dtype = torch.float32
        else:
            dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
            if dtype not in dtype_map:
                raise ValueError(f"unsupported encoder dtype: {dtype}")
            model_dtype = dtype_map[dtype]

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.tokenizer.padding_side = self.padding_side
        self.tokenizer.truncation_side = truncation_side
        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "torch_dtype": model_dtype,
        }
        if self.attention_implementation not in {"", "auto"}:
            model_kwargs["attn_implementation"] = self.attention_implementation
        self.model = AutoModel.from_pretrained(model_path, **model_kwargs).to(self.device)
        position_limit = getattr(self.model.config, "max_position_embeddings", None)
        if position_limit is not None and self.max_length > int(position_limit):
            raise ValueError(
                f"encoder max_length={self.max_length} exceeds the model position limit "
                f"{int(position_limit)}"
            )
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def preprocess(self, text: str) -> str:
        """Normalize thinking text without adding role-dependent content."""

        return " ".join(str(text).split())

    def _pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "last_token":
            return last_token_pool(last_hidden_states, attention_mask)
        attention = attention_mask.unsqueeze(-1).to(last_hidden_states.dtype)
        return (last_hidden_states * attention).sum(dim=1) / attention.sum(dim=1).clamp_min(1)

    @torch.inference_mode()
    def encode(
        self,
        texts: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor, EncoderDiagnostics]:
        diagnostics = EncoderDiagnostics(num_texts=len(texts))
        valid = torch.tensor([bool(str(text).strip()) for text in texts], dtype=torch.bool)
        diagnostics.invalid_texts = int((~valid).sum().item())
        embeddings = torch.zeros((len(texts), int(self.model.config.hidden_size)), dtype=torch.float32)
        valid_indices = valid.nonzero(as_tuple=True)[0].tolist()
        if not valid_indices:
            return embeddings, valid, diagnostics

        prepared = [self.preprocess(texts[index]) for index in valid_indices]
        tokenization_start = time.perf_counter()
        lengths = self.tokenizer(
            prepared,
            add_special_tokens=True,
            truncation=False,
            return_length=True,
            verbose=False,
        )["length"]
        diagnostics.truncated_texts = sum(int(length > self.max_length) for length in lengths)
        clipped_lengths = [min(int(length), self.max_length) for length in lengths]
        diagnostics.encoded_tokens = sum(clipped_lengths)
        diagnostics.max_encoded_tokens = max(clipped_lengths, default=0)
        diagnostics.tokenization_seconds += time.perf_counter() - tokenization_start

        for offset in range(0, len(prepared), self.batch_size):
            diagnostics.num_batches += 1
            text_batch = prepared[offset : offset + self.batch_size]
            tokenization_start = time.perf_counter()
            encoded = self.tokenizer(
                text_batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            diagnostics.tokenization_seconds += time.perf_counter() - tokenization_start
            transfer_start = time.perf_counter()
            encoded = {key: value.to(self.device, non_blocking=self.device.type == "cuda") for key, value in encoded.items()}
            diagnostics.transfer_seconds += time.perf_counter() - transfer_start
            forward_start = time.perf_counter()
            output = self.model(**encoded).last_hidden_state
            pooled = self._pool(output, encoded["attention_mask"])
            pooled = F.normalize(pooled.float(), p=2, dim=-1).cpu()
            diagnostics.forward_seconds += time.perf_counter() - forward_start
            batch_indices = valid_indices[offset : offset + len(text_batch)]
            embeddings[torch.tensor(batch_indices, dtype=torch.long)] = pooled
        return embeddings, valid, diagnostics


# This cache is process-local.  When the function below is executed through
# Worker.execute_func_rank_zero, it lives in actor rank 0's process and can use
# that worker's Ray-assigned GPU without requesting an additional GPU resource.
_ACTOR_RANK_ZERO_ENCODER: HFThinkingEncoder | None = None
_ACTOR_RANK_ZERO_ENCODER_CONTRACT: tuple[Any, ...] | None = None


def encode_thinking_on_actor_rank_zero(
    texts: Sequence[str],
    encoder_config: Mapping[str, Any],
    offload_after_encode: bool = True,
) -> dict[str, Any]:
    """Encode on actor rank 0's assigned GPU, then optionally release VRAM.

    The frozen model remains cached in CPU memory after the first call.  Later
    iterations only stage its weights CPU -> GPU -> CPU around the encoder
    forward, so it does not occupy VRAM during Teacher forcing or actor update.
    """

    if not torch.cuda.is_available():
        raise RuntimeError(
            "actor_rank0 encoder execution requires a CUDA-enabled actor worker; "
            "use encoder.execution=trainer for the CPU fallback"
        )

    model_path = str(encoder_config.get("model_path", ""))
    if not model_path:
        raise ValueError("encoder.model_path is required for actor_rank0 execution")
    device = torch.device("cuda", torch.cuda.current_device())
    contract = (
        model_path,
        int(encoder_config.get("max_length", 4096)),
        int(encoder_config.get("batch_size", 4)),
        str(encoder_config.get("pooling", "last_token")),
        str(encoder_config.get("padding_side", "left")),
        str(encoder_config.get("dtype", "auto")),
        encoder_config.get("revision"),
        encoder_config.get("expected_model_sha256"),
        str(encoder_config.get("truncation_side", "right")),
        str(encoder_config.get("attention_implementation", "sdpa")),
    )

    global _ACTOR_RANK_ZERO_ENCODER, _ACTOR_RANK_ZERO_ENCODER_CONTRACT
    load_seconds = 0.0
    stage_in_seconds = 0.0
    stage_out_seconds = 0.0
    if _ACTOR_RANK_ZERO_ENCODER is None or _ACTOR_RANK_ZERO_ENCODER_CONTRACT != contract:
        _ACTOR_RANK_ZERO_ENCODER = None
        torch.cuda.empty_cache()
        load_start = time.perf_counter()
        _ACTOR_RANK_ZERO_ENCODER = HFThinkingEncoder(
            model_path=model_path,
            device=str(device),
            max_length=contract[1],
            batch_size=contract[2],
            pooling=contract[3],
            padding_side=contract[4],
            dtype=contract[5],
            revision=contract[6],
            expected_model_sha256=contract[7],
            truncation_side=contract[8],
            attention_implementation=contract[9],
        )
        torch.cuda.synchronize(device)
        load_seconds = time.perf_counter() - load_start
        _ACTOR_RANK_ZERO_ENCODER_CONTRACT = contract
    elif _ACTOR_RANK_ZERO_ENCODER.device != device:
        stage_in_start = time.perf_counter()
        _ACTOR_RANK_ZERO_ENCODER.model.to(device)
        _ACTOR_RANK_ZERO_ENCODER.device = device
        torch.cuda.synchronize(device)
        stage_in_seconds = time.perf_counter() - stage_in_start

    encode_start = time.perf_counter()
    try:
        embeddings, valid, diagnostics = _ACTOR_RANK_ZERO_ENCODER.encode(texts)
        torch.cuda.synchronize(device)
        encode_seconds = time.perf_counter() - encode_start
    finally:
        if offload_after_encode and _ACTOR_RANK_ZERO_ENCODER is not None:
            stage_out_start = time.perf_counter()
            _ACTOR_RANK_ZERO_ENCODER.model.to("cpu")
            _ACTOR_RANK_ZERO_ENCODER.device = torch.device("cpu")
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            stage_out_seconds = time.perf_counter() - stage_out_start

    return {
        "embeddings": embeddings,
        "valid": valid,
        "diagnostics": asdict(diagnostics),
        "load_seconds": load_seconds,
        "stage_in_seconds": stage_in_seconds,
        "encode_seconds": encode_seconds,
        "stage_out_seconds": stage_out_seconds,
        "device_index": int(device.index or 0),
    }


def cosine_matrix(teacher_embeddings: torch.Tensor, student_embeddings: torch.Tensor) -> torch.Tensor:
    """Return the FP32 source-by-target matrix H = E_teacher E_student^T."""

    if teacher_embeddings.ndim != 2 or student_embeddings.ndim != 2:
        raise ValueError("embeddings must be rank-2")
    if teacher_embeddings.shape != student_embeddings.shape:
        raise ValueError("teacher and student embeddings must have equal [M, d] shapes")
    teacher = F.normalize(teacher_embeddings.detach().float(), p=2, dim=-1)
    student = F.normalize(student_embeddings.detach().float(), p=2, dim=-1)
    matrix = teacher @ student.T
    return matrix.clamp(-1.0, 1.0).detach()


def cosine_matrix_by_task(
    teacher_embeddings: torch.Tensor,
    student_embeddings: torch.Tensor,
    task_ids: Sequence[Any],
) -> torch.Tensor:
    """Compute task blocks while preserving the logical global ``[M, M]`` shape.

    Cross-task cells are zero because the structural mask always excludes them.
    This is exactly equivalent to dense multiplication after applying that mask,
    while avoiding quadratic work across unrelated benchmark tasks.
    """

    if teacher_embeddings.shape != student_embeddings.shape:
        raise ValueError("teacher and student embeddings must have equal [M, d] shapes")
    if len(task_ids) != teacher_embeddings.shape[0]:
        raise ValueError("task_ids must have length M")
    teacher = F.normalize(teacher_embeddings.detach().float(), p=2, dim=-1)
    student = F.normalize(student_embeddings.detach().float(), p=2, dim=-1)
    output = torch.zeros((len(task_ids), len(task_ids)), dtype=torch.float32)
    groups: dict[str, list[int]] = {}
    for index, task_id in enumerate(task_ids):
        groups.setdefault(str(task_id), []).append(index)
    for task_id in sorted(groups):
        indices = torch.tensor(groups[task_id], dtype=torch.long)
        block = teacher.index_select(0, indices) @ student.index_select(0, indices).T
        output[indices[:, None], indices[None, :]] = block.clamp(-1.0, 1.0)
    return output.detach()


def action_cosine_matrix(
    encoder: HFThinkingEncoder, teacher_actions: Sequence[str], student_actions: Sequence[str]
) -> torch.Tensor:
    """Optional analysis-only action matrix; never consumed by source retrieval."""

    teacher_embeddings, _, _ = encoder.encode(teacher_actions, ("query",) * len(teacher_actions))
    student_embeddings, _, _ = encoder.encode(student_actions, ("document",) * len(student_actions))
    return cosine_matrix(teacher_embeddings, student_embeddings)
