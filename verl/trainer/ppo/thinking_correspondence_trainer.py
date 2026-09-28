"""Mechanism-1 computation and M1-to-M2 adapter.

This extends the repository's shared RL self-distillation infrastructure. It
canonicalizes real turns, freely generates one skill-conditioned Teacher
response per turn, performs diagonal and sparse cross Teacher forcing, then
hands corrected scores and the rollout-time gap to the BeyondTimestamps loop.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence

import numpy as np
import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.action_consistency import action_consistency_mask, validate_action_filter_config
from verl.trainer.ppo.correspondence_credit import (
    CorrespondenceCreditConfig,
    compute_correspondence_credit,
    compute_invalid_action_advantage_residual,
    compute_unique_trajectory_grpo_advantage,
)
from verl.trainer.ppo.rectification_diagnostics import build_rectification_diagnostics, response_region_masks
from verl.trainer.ppo.retrospective_logging import attach_credit, choose_sample_turns, token_snapshot, turn_gap_records
from verl.trainer.ppo.rlsd_ray_trainer import RLSDRayTrainer, build_teacher_batch
from verl.trainer.ppo.thinking_correspondence import (
    EncoderDiagnostics,
    HFThinkingEncoder,
    Mechanism1Result,
    build_structural_mask,
    canonical_multiplicity_row_weights,
    canonicalize_turns,
    combine_cross_teacher_scores,
    compose_cross_teacher_inputs,
    compute_policy_loss_mask,
    cosine_matrix,
    cosine_matrix_by_task,
    encode_thinking_on_actor_rank_zero,
    expand_canonical_tensor,
    parse_tagged_response,
    select_sparse_sources,
)
from verl.utils.model import compute_position_id_with_mask


class ThinkingCorrespondenceRayTrainer(RLSDRayTrainer):
    """Skill-conditioned standalone alignment with corrected M1 gaps."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config.algorithm.get("thinking_correspondence", {})
        self.m1_cfg = cfg
        self.ablation_mode = str(self.config.algorithm.get("ablation_mode", "E")).upper()
        if self.ablation_mode not in {"A", "B", "C", "D", "E"}:
            raise ValueError("algorithm.ablation_mode must be one of A, B, C, D, E")
        action_cfg = cfg.get("action_filter", {})
        self.m1_action_filter_enabled = bool(action_cfg.get("enabled", False))
        self.m1_action_threshold = validate_action_filter_config(action_cfg)
        self.m1_consumer = str(cfg.get("consumer", "standalone_alignment"))
        self.m1_similarity_threshold = float(cfg.get("similarity_threshold", 0.70))
        self.m1_source_rollout_cap = int(cfg.get("source_rollout_cap", 1))
        self.m1_top_k = int(cfg.get("top_k", 3))
        self.m1_aggregation_temperature = float(cfg.get("aggregation_temperature", 0.10))
        self.m1_aggregation_mode = str(cfg.get("aggregation_mode", "probability_mixture"))
        if self.m1_aggregation_mode not in {"probability_mixture", "log_mean"}:
            raise ValueError("thinking_correspondence.aggregation_mode must be probability_mixture or log_mean")
        self.m1_diagnostics_cfg = cfg.get("diagnostics", {})
        self.m1_alpha_max = float(cfg.get("alpha_max", 0.20))
        # A disables rectification while retaining the correspondence sidecar
        # needed by M2. B allocates directly over tokens; C allocates directly
        # over turns without spans; D uses random spans; E is full.
        if self.ablation_mode == "A":
            self.m1_alpha_max = 0.0
        rollout_cfg = self.config.actor_rollout_ref.rollout
        self.m1_teacher_max_response_length = cfg.get("teacher_max_response_length")
        self.m1_teacher_do_sample = bool(rollout_cfg.get("do_sample", True))
        legacy_teacher_do_sample = cfg.get("teacher_do_sample")
        if legacy_teacher_do_sample is not None and bool(legacy_teacher_do_sample) != self.m1_teacher_do_sample:
            raise ValueError(
                "Teacher generation must inherit actor_rollout_ref.rollout.do_sample; "
                "configure the shared Student/Teacher rollout instead of a Teacher-only override"
            )
        # Reuse action embeddings for filtering and optional matrix diagnostics.
        self.m1_compute_action_matrix = bool(
            cfg.get("compute_action_matrix", cfg.get("persist_action_matrix", False)) or self.m1_action_filter_enabled
        )
        encoder_cfg = cfg.get("encoder", {})
        self.m1_encoder_execution = str(encoder_cfg.get("execution", "trainer"))
        self.m1_encoder_offload_after_encode = bool(
            encoder_cfg.get("offload_after_encode", True)
        )
        self.m1_fuse_teacher_forcing = bool(cfg.get("fuse_teacher_forcing", False))
        self._thinking_encoder = None
        self._thinking_encoder_load_seconds = 0.0
        self.last_mechanism1_result: Mechanism1Result | None = None

        if not -1.0 <= self.m1_similarity_threshold < 1.0:
            raise ValueError("thinking_correspondence.similarity_threshold must be in [-1, 1)")
        if not 0.0 <= self.m1_alpha_max <= 1.0:
            raise ValueError("thinking_correspondence.alpha_max must be in [0, 1]")
        if self.m1_aggregation_temperature <= 0:
            raise ValueError("thinking_correspondence.aggregation_temperature must be positive")
        if self.m1_source_rollout_cap != 1:
            raise ValueError("Mechanism 1 requires thinking_correspondence.source_rollout_cap=1")
        if self.m1_consumer not in {"standalone_alignment", "mechanism2"}:
            raise ValueError(f"unsupported Mechanism-1 consumer: {self.m1_consumer}")
        if self.m1_encoder_execution not in {"trainer", "actor_rank0", "actor_all"}:
            raise ValueError(
                "thinking_correspondence.encoder.execution must be trainer, actor_rank0 or actor_all"
            )
        if self.m1_encoder_execution == "actor_all" and not self.m1_encoder_offload_after_encode:
            raise ValueError("actor_all requires encoder.offload_after_encode=true")
        m2_cfg = self.config.algorithm.get("mechanism2", {})
        self.m2_credit_config = None
        if self.m1_consumer == "mechanism2":
            self.m2_credit_config = CorrespondenceCreditConfig(
                profile_temperature=float(m2_cfg.get("profile_temperature", 0.10)),
                profile_distance=str(m2_cfg.get("profile_distance", "jsd")),
                segmentation_quantile=float(m2_cfg.get("segmentation_quantile", 0.80)),
                segmentation_threshold_min=float(m2_cfg.get("segmentation_threshold_min", 0.01)),
                segmentation_threshold_max=float(m2_cfg.get("segmentation_threshold_max", 0.10)),
                min_segment_turns=int(m2_cfg.get("min_segment_turns", 2)),
                max_segment_turns=int(m2_cfg.get("max_segment_turns", 8)),
                credit_temperature=float(m2_cfg.get("credit_temperature", 1.0)),
                mixing_coefficient=float(m2_cfg.get("mixing_coefficient", 0.5)),
                density_upper_bound=float(m2_cfg.get("density_upper_bound", 4.0)),
                weighting_mode=str(m2_cfg.get("weighting_mode", "legacy")),
                multiplier_band=float(m2_cfg.get("multiplier_band", 0.2)),
                segmentation_mode=("token" if self.ablation_mode == "B" else
                                   "turn" if self.ablation_mode == "C" else
                                   "random" if self.ablation_mode == "D" else "adaptive"),
                loss_aggregation=str(self.config.actor_rollout_ref.actor.loss_agg_mode),
            )

    def consume_mechanism1_result(self, batch: DataProto) -> tuple[DataProto, dict[str, float]]:
        """Consume M1 either as standalone alignment or as M2 credit input."""

        if self.m1_consumer == "standalone_alignment":
            return batch, {"m1/consumer_standalone_alignment": 1.0}

        mechanism2_start = time.perf_counter()
        result = self.last_mechanism1_result
        if result is None:
            raise RuntimeError("Mechanism 2 requires the current iteration's Mechanism1Result")
        turn_map = result.turn_map
        canonical_rows = np.asarray(turn_map.canonical_row_indices, dtype=np.int64)
        canonical_batch = batch.select_idxs(canonical_rows)
        current_policy_mask = compute_policy_loss_mask(
            canonical_batch,
            multi_turn=bool(self.config.actor_rollout_ref.rollout.multi_turn.enable),
        ).detach()
        sidecar_policy_mask = result.policy_loss_mask.to(
            device=current_policy_mask.device, dtype=torch.bool
        )
        if current_policy_mask.shape != sidecar_policy_mask.shape or not torch.equal(
            current_policy_mask.bool(), sidecar_policy_mask
        ):
            raise RuntimeError("Mechanism-1 sidecar policy mask drifted before Mechanism-2 consumption")

        if "episode_rewards" not in batch.non_tensor_batch:
            raise KeyError("Mechanism 2 requires raw episode_rewards")
        episode_outcome = torch.tensor(
            [float(batch.non_tensor_batch["episode_rewards"][row]) for row in canonical_rows],
            dtype=torch.float32,
            device=result.thinking_similarity.device,
        )
        normalize_by_std = bool(self.config.algorithm.get("norm_adv_by_std_in_grpo", True))
        normalization_scope = self.config.algorithm.get("grpo_normalization_scope", "trajectory")
        trajectory_advantage = compute_unique_trajectory_grpo_advantage(
            episode_outcome,
            turn_map.identities,
            normalize_by_std=normalize_by_std,
            normalization_scope=normalization_scope,
        )
        m2_result = compute_correspondence_credit(
            mechanism1_result=result,
            trajectory_advantage=trajectory_advantage,
            config=self.m2_credit_config,
        )

        use_invalid_penalty = bool(
            self.config.actor_rollout_ref.actor.get("use_invalid_action_penalty", True)
        )
        if use_invalid_penalty:
            if "is_action_valid" not in batch.non_tensor_batch:
                raise KeyError("invalid-action penalty requires is_action_valid")
            action_valid = torch.tensor(
                [bool(batch.non_tensor_batch["is_action_valid"][row]) for row in canonical_rows],
                dtype=torch.bool,
                device=episode_outcome.device,
            )
            invalid_token_residual, invalid_scalar_residual = compute_invalid_action_advantage_residual(
                episode_outcome,
                result.policy_loss_mask,
                turn_map.identities,
                action_valid,
                penalty_coefficient=float(
                    self.config.actor_rollout_ref.actor.invalid_action_penalty_coef
                ),
                normalize_by_std=normalize_by_std,
            )
            valid_action_ratio = float(action_valid.float().mean().item())
        else:
            invalid_token_residual = torch.zeros_like(m2_result.token_advantage)
            invalid_scalar_residual = torch.zeros_like(trajectory_advantage)
            valid_action_ratio = 1.0

        audit = getattr(self, "retrospective_logger", None)
        if audit is not None and audit.pending is not None:
            try:
                attach_credit(audit.pending["records"], m2_result, episode_outcome,
                              trajectory_advantage, invalid_scalar_residual)
            except Exception as exc:
                audit.failure("attach_credit", exc)

        canonical_advantage = (m2_result.token_advantage + invalid_token_residual).detach()
        if not torch.isfinite(canonical_advantage).all():
            raise RuntimeError("Mechanism-2 training advantage must be finite")
        policy_mask = result.policy_loss_mask.to(
            device=canonical_advantage.device, dtype=torch.bool
        )
        if canonical_advantage[policy_mask.logical_not()].count_nonzero():
            raise RuntimeError("Mechanism-2 advantage must be zero outside the policy-loss mask")
        batch.batch["advantages"] = expand_canonical_tensor(canonical_advantage, turn_map)
        batch.batch["returns"] = batch.batch["advantages"].clone()

        metrics = {f"m2/{key}": value for key, value in m2_result.diagnostics.items()}
        trajectory_outcomes: dict[tuple[str, str], float] = {}
        trajectory_lengths: dict[tuple[str, str], int] = {}
        for index, identity in enumerate(turn_map.identities):
            key = (identity.task_id, identity.rollout_id)
            trajectory_outcomes.setdefault(key, float(episode_outcome[index].item()))
            trajectory_lengths[key] = trajectory_lengths.get(key, 0) + 1
        unique_outcomes = torch.tensor(
            list(trajectory_outcomes.values()), dtype=torch.float32
        )
        unique_lengths = torch.tensor(
            list(trajectory_lengths.values()), dtype=torch.float32
        )
        metrics.update(
            {
                "m1/consumer_mechanism2": 1.0,
                "m2/grpo_normalization_turn": float(normalization_scope == "turn"),
                "m2/compute_seconds": time.perf_counter() - mechanism2_start,
                "m2/trajectory_advantage_mean": float(trajectory_advantage.mean().item()),
                "m2/trajectory_advantage_std": float(
                    trajectory_advantage.std(unbiased=False).item()
                ),
                "m2/trajectory_advantage_abs_mean": float(
                    trajectory_advantage.abs().mean().item()
                ),
                "m2/invalid_residual_abs_mean": float(
                    invalid_scalar_residual.abs().mean().item()
                ),
                "m2/framework_copy_fraction": 1.0
                - turn_map.num_canonical_turns / max(len(turn_map.row_to_canonical), 1),
                "episode/unique_trajectory_count": float(len(trajectory_outcomes)),
                "episode/outcome_mean": float(unique_outcomes.mean().item()),
                "episode/outcome_std": float(unique_outcomes.std(unbiased=False).item()),
                "episode/outcome_positive_rate": float((unique_outcomes > 0).float().mean().item()),
                "episode/outcome_perfect_rate": float((unique_outcomes >= 1).float().mean().item()),
                "episode/trajectory_length_mean": float(unique_lengths.mean().item()),
                "episode/trajectory_length_max": float(unique_lengths.max().item()),
                "episode/valid_action_ratio": valid_action_ratio,
            }
        )
        return batch, metrics

    def _resolved_encoder_config(self) -> dict[str, object]:
        encoder_cfg = self.m1_cfg.get("encoder", {})
        model_path = encoder_cfg.get("model_path") or os.environ.get("THINKING_ENCODER_PATH")
        if not model_path:
            raise ValueError(
                "Set algorithm.thinking_correspondence.encoder.model_path or "
                "THINKING_ENCODER_PATH to the offline Qwen3-Embedding-0.6B directory"
            )
        return {
            "model_path": str(model_path),
            "device": str(encoder_cfg.get("device", "auto")),
            "max_length": int(encoder_cfg.get("max_length", 4096)),
            "batch_size": int(encoder_cfg.get("batch_size", 4)),
            "pooling": str(encoder_cfg.get("pooling", "last_token")),
            "padding_side": str(encoder_cfg.get("padding_side", "left")),
            "dtype": str(encoder_cfg.get("dtype", "auto")),
            "revision": encoder_cfg.get("revision"),
            "expected_model_sha256": encoder_cfg.get("expected_model_sha256"),
            "truncation_side": str(encoder_cfg.get("truncation_side", "right")),
            "attention_implementation": str(
                encoder_cfg.get("attention_implementation", "sdpa")
            ),
        }

    def _get_thinking_encoder(self) -> HFThinkingEncoder:
        if self._thinking_encoder is None:
            load_start = time.perf_counter()
            encoder_cfg = self._resolved_encoder_config()
            self._thinking_encoder = HFThinkingEncoder(
                model_path=encoder_cfg["model_path"],
                device=encoder_cfg["device"],
                max_length=encoder_cfg["max_length"],
                batch_size=encoder_cfg["batch_size"],
                pooling=encoder_cfg["pooling"],
                padding_side=encoder_cfg["padding_side"],
                dtype=encoder_cfg["dtype"],
                revision=encoder_cfg.get("revision"),
                expected_model_sha256=encoder_cfg.get("expected_model_sha256"),
                truncation_side=encoder_cfg["truncation_side"],
                attention_implementation=encoder_cfg["attention_implementation"],
            )
            self._thinking_encoder_load_seconds = time.perf_counter() - load_start
        return self._thinking_encoder

    def _encode_thinking_texts(
        self,
        texts: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor, EncoderDiagnostics, dict[str, float]]:
        """Encode either in the CPU trainer or on actor rank 0's staged GPU."""

        if self.m1_encoder_execution == "actor_all":
            encoder_cfg = self._resolved_encoder_config()
            count = self.actor_rollout_wg.world_size
            shards = [tuple(texts[rank::count]) for rank in range(count)]
            started = time.perf_counter()
            # Explicit all-worker RPC bypasses the rank-zero dispatch wrapper.
            # Every worker offloads in finally before returning CPU results.
            results = self.actor_rollout_wg.execute_all_sync(
                "execute_func_rank_zero",
                [encode_thinking_on_actor_rank_zero] * count,
                shards, [encoder_cfg] * count, [True] * count,
            )
            wall = time.perf_counter() - started
            if len(results) != count:
                raise RuntimeError("Encoder worker result count mismatch")
            embeddings = torch.empty((len(texts), results[0]["embeddings"].shape[1]), dtype=torch.float32)
            valid = torch.empty(len(texts), dtype=torch.bool)
            for rank, result in enumerate(results):
                e, v = result["embeddings"], result["valid"]
                if e.device.type != "cpu" or v.device.type != "cpu":
                    raise RuntimeError("Encoder shards must return CPU tensors")
                if e.shape != embeddings[rank::count].shape or v.shape != valid[rank::count].shape:
                    raise RuntimeError("Encoder shard shape mismatch")
                embeddings[rank::count] = e
                valid[rank::count] = v
            summed = {"num_texts", "invalid_texts", "truncated_texts", "encoded_tokens", "num_batches"}
            diagnostics = EncoderDiagnostics(**{
                key: (sum if key in summed else max)(r["diagnostics"][key] for r in results)
                for key in EncoderDiagnostics.__dataclass_fields__
            })
            timings = {
                key: max(float(r[key]) for r in results)
                for key in ("load_seconds", "stage_in_seconds", "encode_seconds", "stage_out_seconds")
            }
            timings.update(device_index=-1.0, all_worker_wall_seconds=wall)
            return embeddings, valid, diagnostics, timings

        if self.m1_encoder_execution == "actor_rank0":
            encoder_cfg = self._resolved_encoder_config()
            # The worker uses its own Ray-assigned logical cuda device.  The
            # trainer-side device field is intentionally ignored in this mode.
            result = self.actor_rollout_wg.execute_func_rank_zero(
                encode_thinking_on_actor_rank_zero,
                tuple(texts),
                encoder_cfg,
                self.m1_encoder_offload_after_encode,
            )
            diagnostics = EncoderDiagnostics(**result["diagnostics"])
            timings = {
                "load_seconds": float(result["load_seconds"]),
                "stage_in_seconds": float(result["stage_in_seconds"]),
                "encode_seconds": float(result["encode_seconds"]),
                "stage_out_seconds": float(result["stage_out_seconds"]),
                "device_index": float(result["device_index"]),
            }
            return result["embeddings"], result["valid"], diagnostics, timings

        encoder_was_cached = self._thinking_encoder is not None
        encoder = self._get_thinking_encoder()
        encode_start = time.perf_counter()
        embeddings, valid, diagnostics = encoder.encode(texts)
        timings = {
            "load_seconds": 0.0 if encoder_was_cached else self._thinking_encoder_load_seconds,
            "stage_in_seconds": 0.0,
            "encode_seconds": time.perf_counter() - encode_start,
            "stage_out_seconds": 0.0,
            "device_index": -1.0,
        }
        return embeddings, valid, diagnostics, timings

    def _padded_compute_log_probs(self, batch: DataProto) -> torch.Tensor:
        padded, pad_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)
        output = self.actor_rollout_wg.compute_log_prob(padded)
        if pad_size:
            output = unpad_dataproto(output, pad_size=pad_size)
        return output.batch["old_log_probs"]

    @staticmethod
    def _validate_canonical_copies(batch: DataProto, turn_map) -> None:
        """Reject identity collisions instead of silently selecting one row."""

        tensor_keys = ("input_ids", "responses", "attention_mask", "loss_mask")
        metadata_keys = (
            "uid",
            "traj_uid",
            "turn_step",
            "data_source",
            "gamefile",
            "episode_rewards",
            "episode_lengths",
            "is_action_valid",
        )
        for rows in turn_map.canonical_to_rows:
            if len(rows) < 2:
                continue
            reference = rows[0]
            for row in rows[1:]:
                for key in tensor_keys:
                    if key in batch.batch and not torch.equal(batch.batch[key][reference], batch.batch[key][row]):
                        raise ValueError(f"rows with the same turn identity differ in tensor field {key}")
                for key in metadata_keys:
                    if key in batch.non_tensor_batch:
                        left = batch.non_tensor_batch[key][reference]
                        right = batch.non_tensor_batch[key][row]
                        if str(left) != str(right):
                            raise ValueError(f"rows with the same turn identity differ in metadata field {key}")

    def _generate_teacher_responses(self, teacher_batch: DataProto) -> torch.Tensor:
        response_length = teacher_batch.batch["responses"].shape[-1]
        prompt_length = teacher_batch.batch["input_ids"].shape[-1] - response_length
        prompt_batch = DataProto.from_dict(
            tensors={
                "input_ids": teacher_batch.batch["input_ids"][:, :prompt_length],
                "attention_mask": teacher_batch.batch["attention_mask"][:, :prompt_length],
                "position_ids": teacher_batch.batch["position_ids"][:, :prompt_length],
            },
            meta_info={
                **teacher_batch.meta_info,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                # Teacher and training-time Student use the exact same rollout
                # sampling policy. Only their prompts differ (privileged skill).
                "do_sample": self.m1_teacher_do_sample,
                "diagnostic_role": "teacher",
                "diagnostic_step": int(self.global_steps),
            },
        )
        if self.m1_teacher_max_response_length is not None:
            prompt_batch.meta_info["generation_max_tokens"] = int(self.m1_teacher_max_response_length)
        padded, pad_size = pad_dataproto_to_divisor(prompt_batch, self.actor_rollout_wg.world_size)
        output = self.actor_rollout_wg.generate_sequences(padded)
        if pad_size:
            output = unpad_dataproto(output, pad_size=pad_size)
        return output.batch["responses"]

    @staticmethod
    def _build_cross_teacher_batch(
        teacher_batch: DataProto,
        student_batch: DataProto,
        pairs: Sequence[tuple[int, int]],
    ) -> DataProto:
        """Use source teacher prompts and target Student sampled responses."""

        response_length = student_batch.batch["responses"].shape[-1]
        prompt_length = teacher_batch.batch["input_ids"].shape[-1] - response_length
        teacher_prompt_mask = teacher_batch.batch["attention_mask"][:, :prompt_length]
        target_response_mask = student_batch.batch["attention_mask"][:, -response_length:]
        tensors = compose_cross_teacher_inputs(
            teacher_prompt_ids=teacher_batch.batch["input_ids"][:, :prompt_length],
            teacher_prompt_mask=teacher_prompt_mask,
            student_responses=student_batch.batch["responses"],
            student_response_mask=target_response_mask,
            pairs=pairs,
        )
        tensors["position_ids"] = compute_position_id_with_mask(tensors["attention_mask"])
        return DataProto.from_dict(tensors=tensors)

    def _compute_teacher_log_probs(self, batch: DataProto) -> torch.Tensor:
        """Compute and attach the full outcome-free Mechanism-1 sidecar."""

        mechanism1_start = time.perf_counter()
        # Do not retain the previous iteration's dense matrices while building
        # the current sidecar.
        self.last_mechanism1_result = None
        required_identity_fields = ("uid", "traj_uid", "turn_step")
        missing = [key for key in required_identity_fields if key not in batch.non_tensor_batch]
        if missing:
            raise KeyError(f"Mechanism 1 requires turn identity fields: {missing}")

        turn_map = canonicalize_turns(
            batch.non_tensor_batch["uid"],
            batch.non_tensor_batch["traj_uid"],
            batch.non_tensor_batch["turn_step"],
        )
        self._validate_canonical_copies(batch, turn_map)
        # ``adjust_batch`` may append exact copies for distributed divisibility.
        # Preserve their compute rows, but make their combined actor objective
        # equal one canonical turn rather than counting every copy again.
        batch.batch["canonical_row_weight"] = canonical_multiplicity_row_weights(
            turn_map,
            device=batch.batch["responses"].device,
        )
        canonical_indices = np.asarray(turn_map.canonical_row_indices, dtype=np.int64)
        canonical_batch = batch.select_idxs(canonical_indices)
        m = turn_map.num_canonical_turns

        trace_dir = self.config.actor_rollout_ref.rollout.get("generation_trace_dir")
        construction_records = [] if trace_dir else None
        teacher_batch = build_teacher_batch(
            batch=canonical_batch,
            skill_provider=self.skill_provider,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.data.max_prompt_length,
            diagnostic_records=construction_records,
        )

        diagonal_teacher_log_probs = None
        diagonal_seconds = 0.0
        if not self.m1_fuse_teacher_forcing:
            diagonal_start = time.perf_counter()
            diagonal_teacher_log_probs = self._padded_compute_log_probs(teacher_batch).detach()
            diagonal_seconds = time.perf_counter() - diagonal_start

        teacher_generation_start = time.perf_counter()
        teacher_response_ids = self._generate_teacher_responses(teacher_batch)
        teacher_generation_seconds = time.perf_counter() - teacher_generation_start

        rollout_runtime_stats = self.actor_rollout_wg.get_rollout_runtime_stats()

        parsing_start = time.perf_counter()
        student_response_text = self.tokenizer.batch_decode(
            canonical_batch.batch["responses"], skip_special_tokens=True
        )
        teacher_response_text = self.tokenizer.batch_decode(teacher_response_ids, skip_special_tokens=True)
        parsed_student = tuple(parse_tagged_response(text) for text in student_response_text)
        parsed_teacher = tuple(parse_tagged_response(text) for text in teacher_response_text)
        if trace_dir:
            from verl.trainer.ppo.generation_trace import capture_pairs
            capture_pairs(trace_dir, self.global_steps, turn_map.identities,
                          canonical_batch, teacher_batch, teacher_response_ids,
                          parsed_student, parsed_teacher, construction_records, self.tokenizer)
        student_thinking = tuple(item.thinking for item in parsed_student)
        teacher_thinking = tuple(item.thinking for item in parsed_teacher)
        student_actions = tuple(item.action for item in parsed_student)
        teacher_actions = tuple(item.action for item in parsed_teacher)
        parsing_seconds = time.perf_counter() - parsing_start

        encoder_texts = student_thinking + teacher_thinking
        if self.m1_compute_action_matrix:
            encoder_texts += teacher_actions + student_actions
        all_embeddings, all_valid, encoder_diagnostics, encoder_timings = (
            self._encode_thinking_texts(encoder_texts)
        )
        encoder_load_seconds = encoder_timings["load_seconds"]
        encoder_seconds = encoder_timings["encode_seconds"]
        student_embeddings = all_embeddings[:m]
        teacher_embeddings = all_embeddings[m : 2 * m]
        student_valid = all_valid[:m]
        teacher_valid = all_valid[m : 2 * m]
        matrix_start = time.perf_counter()
        thinking_similarity = cosine_matrix_by_task(
            teacher_embeddings,
            student_embeddings,
            [identity.task_id for identity in turn_map.identities],
        )
        matrix_seconds = time.perf_counter() - matrix_start

        protocol_ids = None
        if "data_source" in canonical_batch.non_tensor_batch:
            protocol_ids = canonical_batch.non_tensor_batch["data_source"]
        structural_mask = build_structural_mask(
            identities=turn_map.identities,
            teacher_thinking_valid=teacher_valid,
            student_thinking_valid=student_valid,
            protocol_ids=protocol_ids,
        )
        action_similarity = None
        if self.m1_compute_action_matrix:
            teacher_action_embeddings = all_embeddings[2 * m : 3 * m]
            student_action_embeddings = all_embeddings[3 * m : 4 * m]
            action_similarity = cosine_matrix(
                teacher_action_embeddings, student_action_embeddings
            )

        selection_start = time.perf_counter()
        selected_sources = select_sparse_sources(
            thinking_similarity=thinking_similarity,
            structural_mask=structural_mask,
            identities=turn_map.identities,
            similarity_threshold=self.m1_similarity_threshold,
            source_rollout_cap=self.m1_source_rollout_cap,
            top_k=self.m1_top_k,
        )
        selection_seconds = time.perf_counter() - selection_start
        action_filter_start = time.perf_counter()
        action_filter_metrics = {"action_filter/enabled": float(self.m1_action_filter_enabled)}
        if self.m1_action_filter_enabled:
            candidates = torch.zeros_like(structural_mask, dtype=torch.bool)
            for target, sources in enumerate(selected_sources):
                for source in sources:
                    candidates[source, target] = True
            teacher_action_valid = all_valid[2 * m : 3 * m] & torch.tensor(
                [p.action_valid for p in parsed_teacher], device=all_valid.device)
            student_action_valid = all_valid[3 * m : 4 * m] & torch.tensor(
                [p.action_valid for p in parsed_student], device=all_valid.device)
            selection_mask = action_consistency_mask(
                candidates, action_similarity, teacher_actions, student_actions,
                teacher_action_valid, student_action_valid, self.m1_action_threshold,
                webshop=str(self.config.env.env_name).lower() == "webshop",
            )
            accepted = selection_mask.detach().cpu().tolist()
            selected_sources = tuple(
                tuple(source for source in sources if accepted[source][target])
                for target, sources in enumerate(selected_sources)
            )
            before, after = int(candidates.sum()), int(selection_mask.sum())
            action_filter_metrics.update({
                "action_filter/similarity_threshold": self.m1_action_threshold,
                "action_filter/candidate_pairs": float(before),
                "action_filter/accepted_pairs": float(after),
                "action_filter/rejected_pairs": float(before - after),
                "action_filter/retained_fraction": after / before if before else 0.0,
                "action_filter/new_fallback_turn_fraction": float(
                    (candidates.any(dim=0) & ~selection_mask.any(dim=0)).float().mean()),
            })
        action_filter_metrics["action_filter/seconds"] = time.perf_counter() - action_filter_start
        pairs = tuple(
            (source, target)
            for target, sources in enumerate(selected_sources)
            for source in sources
        )

        cross_batch_start = time.perf_counter()
        cross_batch = None
        if pairs:
            cross_batch = self._build_cross_teacher_batch(teacher_batch, canonical_batch, pairs)
        cross_batch_seconds = time.perf_counter() - cross_batch_start

        cross_scores: dict[tuple[int, int], torch.Tensor] = {}
        cross_seconds = 0.0
        teacher_forcing_seconds = diagonal_seconds
        if self.m1_fuse_teacher_forcing:
            teacher_forcing_start = time.perf_counter()
            scoring_batch = (
                DataProto.concat([teacher_batch, cross_batch])
                if cross_batch is not None
                else teacher_batch
            )
            all_teacher_log_probs = self._padded_compute_log_probs(scoring_batch).detach()
            teacher_forcing_seconds = time.perf_counter() - teacher_forcing_start
            diagonal_teacher_log_probs = all_teacher_log_probs[:m]
            if pairs:
                pair_log_probs = all_teacher_log_probs[m:]
                cross_scores = {
                    pair: pair_log_probs[index] for index, pair in enumerate(pairs)
                }
        elif cross_batch is not None:
            cross_start = time.perf_counter()
            pair_log_probs = self._padded_compute_log_probs(cross_batch).detach()
            cross_seconds = time.perf_counter() - cross_start
            teacher_forcing_seconds += cross_seconds
            cross_scores = {pair: pair_log_probs[index] for index, pair in enumerate(pairs)}

        if diagonal_teacher_log_probs is None:
            raise RuntimeError("diagonal Teacher scores were not produced")

        canonical_rows = torch.tensor(
            turn_map.canonical_row_indices,
            dtype=torch.long,
            device=batch.batch["old_log_probs"].device,
        )
        student_log_probs = batch.batch["old_log_probs"].index_select(0, canonical_rows).detach()
        combined = combine_cross_teacher_scores(
            diagonal_teacher_log_probs=diagonal_teacher_log_probs,
            student_log_probs=student_log_probs,
            thinking_similarity=thinking_similarity,
            selected_source_indices=selected_sources,
            cross_scores=cross_scores,
            similarity_threshold=self.m1_similarity_threshold,
            aggregation_temperature=self.m1_aggregation_temperature,
            alpha_max=self.m1_alpha_max,
            aggregation_mode=self.m1_aggregation_mode,
        )

        sampled_response_mask = canonical_batch.batch["attention_mask"][:, -student_log_probs.shape[-1] :].detach()
        policy_loss_mask = compute_policy_loss_mask(
            canonical_batch,
            multi_turn=bool(self.config.actor_rollout_ref.rollout.multi_turn.enable),
        ).detach()

        matched_counts = torch.tensor(
            [len(sources) for sources in selected_sources],
            dtype=torch.float32,
            device=sampled_response_mask.device,
        )
        matched_token_mask = sampled_response_mask.bool() & matched_counts.bool().unsqueeze(-1)
        if matched_token_mask.any():
            cross_diagonal_disagreement = (
                combined["cross_teacher_log_probs"] - diagonal_teacher_log_probs
            ).abs()[matched_token_mask].float().mean().item()
        else:
            cross_diagonal_disagreement = 0.0

        def masked_distribution(name: str, values: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
            selected = values.detach().float()[mask.bool()]
            if not selected.numel():
                return {
                    f"{name}_{stat}": 0.0
                    for stat in ("mean", "std", "min", "max", "p10", "p50", "p90")
                }
            return {
                f"{name}_mean": float(selected.mean().item()),
                f"{name}_std": float(selected.std(unbiased=False).item()),
                f"{name}_min": float(selected.min().item()),
                f"{name}_max": float(selected.max().item()),
                f"{name}_p10": float(torch.quantile(selected, 0.10).item()),
                f"{name}_p50": float(torch.quantile(selected, 0.50).item()),
                f"{name}_p90": float(torch.quantile(selected, 0.90).item()),
            }

        cross_gap = combined["cross_teacher_log_probs"] - student_log_probs
        cross_diagonal_delta = combined["cross_teacher_log_probs"] - diagonal_teacher_log_probs
        corrected_diagonal_delta = combined["corrected_teacher_log_probs"] - diagonal_teacher_log_probs
        gap_diagnostics = dict(action_filter_metrics)
        gap_diagnostics.update(masked_distribution("diagonal_gap", combined["diagonal_token_gap"], policy_loss_mask))
        gap_diagnostics.update(masked_distribution("cross_gap", cross_gap, policy_loss_mask.bool() & matched_token_mask))
        gap_diagnostics.update(masked_distribution("corrected_gap", combined["corrected_token_gap"], policy_loss_mask))
        gap_diagnostics.update(
            masked_distribution(
                "cross_diagonal_delta",
                cross_diagonal_delta,
                policy_loss_mask.bool() & matched_token_mask,
            )
        )
        gap_diagnostics.update(
            masked_distribution(
                "corrected_diagonal_delta", corrected_diagonal_delta, policy_loss_mask
            )
        )
        region_cache = {}
        paired_diagnostics_start = time.perf_counter()
        if bool(self.m1_diagnostics_cfg.get("enabled", True)):
            gap_diagnostics.update(build_rectification_diagnostics(
                tokenizer=self.tokenizer,
                responses=canonical_batch.batch["responses"],
                sampled_response_mask=sampled_response_mask,
                policy_loss_mask=policy_loss_mask,
                selected_source_indices=selected_sources,
                combined=combined,
                student_log_probs=student_log_probs,
                sign_epsilon=float(self.m1_diagnostics_cfg.get("sign_epsilon", 1e-4)),
                region_breakdown=bool(self.m1_diagnostics_cfg.get("region_breakdown", True)),
                region_cache=region_cache,
            ))
        gap_diagnostics["diagnostics/compute_seconds"] = time.perf_counter() - paired_diagnostics_start
        gap_diagnostics["diagnostics/enabled"] = float(self.m1_diagnostics_cfg.get("enabled", True))
        gap_diagnostics["aggregation/probability_mixture"] = float(self.m1_aggregation_mode == "probability_mixture")
        gap_diagnostics["aggregation/diagonal_blend_log_mean"] = 1.0
        turn_mask = torch.ones_like(combined["correspondence_confidence"], dtype=torch.bool)
        gap_diagnostics.update(
            masked_distribution("correspondence_confidence", combined["correspondence_confidence"], turn_mask)
        )
        gap_diagnostics.update(masked_distribution("correction_strength", combined["correction_strength"], turn_mask))

        structurally_eligible = structural_mask.bool()
        threshold_eligible = structurally_eligible & thinking_similarity.ge(
            self.m1_similarity_threshold
        )
        eligible_counts = structurally_eligible.sum(dim=0).float()
        candidate_counts = threshold_eligible.sum(dim=0).float()
        valid_diagonal = student_valid.bool() & teacher_valid.bool()
        selected_similarity_values = torch.tensor(
            [
                float(thinking_similarity[source, target].item())
                for target, sources in enumerate(selected_sources)
                for source in sources
            ],
            dtype=torch.float32,
        )
        similarity_diagnostics = {}
        similarity_diagnostics.update(
            masked_distribution(
                "structural_similarity", thinking_similarity, structurally_eligible
            )
        )
        similarity_diagnostics.update(
            masked_distribution(
                "threshold_similarity", thinking_similarity, threshold_eligible
            )
        )
        similarity_diagnostics.update(
            masked_distribution(
                "diagonal_similarity", thinking_similarity.diagonal(), valid_diagonal
            )
        )
        similarity_diagnostics.update(
            masked_distribution(
                "selected_similarity",
                selected_similarity_values,
                torch.ones_like(selected_similarity_values, dtype=torch.bool),
            )
        )
        similarity_diagnostics.update(
            masked_distribution(
                "eligible_sources_per_target",
                eligible_counts,
                torch.ones_like(eligible_counts, dtype=torch.bool),
            )
        )
        similarity_diagnostics.update(
            masked_distribution(
                "threshold_candidates_per_target",
                candidate_counts,
                torch.ones_like(candidate_counts, dtype=torch.bool),
            )
        )

        response_length = canonical_batch.batch["responses"].shape[-1]
        student_response_lengths = canonical_batch.batch["attention_mask"][:, -response_length:].sum(dim=-1).float()
        teacher_pad_token_id = self.tokenizer.pad_token_id
        if teacher_pad_token_id is None:
            teacher_pad_token_id = self.tokenizer.eos_token_id
        teacher_response_lengths = teacher_response_ids.ne(teacher_pad_token_id).sum(dim=-1).float()
        response_diagnostics = {
            "teacher_output_budget": float(teacher_response_ids.shape[-1]),
            # Token-count diagnostic, not a replacement for the engine finish reason.
            "teacher_response_at_budget_fraction": float(
                (teacher_response_lengths >= teacher_response_ids.shape[-1]).float().mean().item()
            ),
        }
        response_diagnostics.update(
            masked_distribution(
                "student_response_tokens",
                student_response_lengths,
                torch.ones_like(student_response_lengths, dtype=torch.bool),
            )
        )
        response_diagnostics.update(
            masked_distribution(
                "teacher_response_tokens",
                teacher_response_lengths,
                torch.ones_like(teacher_response_lengths, dtype=torch.bool),
            )
        )

        mechanism1_seconds = time.perf_counter() - mechanism1_start
        encoder_total_seconds = float(
            encoder_load_seconds
            + encoder_timings["stage_in_seconds"]
            + encoder_seconds
            + encoder_timings["stage_out_seconds"]
        )
        diagnostics = {
            "canonical_turn_count": float(m),
            "adjusted_row_count": float(len(batch)),
            "canonical_row_weight_sum": float(
                batch.batch["canonical_row_weight"].sum().item()
            ),
            "cross_pair_count": float(len(pairs)),
            "valid_match_rate": float((matched_counts > 0).float().mean().item()) if m else 0.0,
            "fallback_turn_fraction": float((matched_counts == 0).float().mean().item()) if m else 0.0,
            "mean_matched_teachers": float(matched_counts.mean().item()) if m else 0.0,
            "top_k_saturation_rate": float((matched_counts >= self.m1_top_k).float().mean().item()) if m else 0.0,
            "structural_eligibility_density": float(structurally_eligible.float().mean().item()) if m else 0.0,
            "threshold_eligibility_density": float(threshold_eligible.float().mean().item()) if m else 0.0,
            "threshold_candidate_zero_rate": float((candidate_counts == 0).float().mean().item()) if m else 0.0,
            "correction_active_rate": float((combined["correction_strength"] > 0).float().mean().item()) if m else 0.0,
            "invalid_student_thinking_fraction": float((~student_valid).float().mean().item()) if m else 0.0,
            "invalid_teacher_thinking_fraction": float((~teacher_valid).float().mean().item()) if m else 0.0,
            "invalid_student_action_fraction": float(np.mean([not item.action_valid for item in parsed_student])) if m else 0.0,
            "invalid_teacher_action_fraction": float(np.mean([not item.action_valid for item in parsed_teacher])) if m else 0.0,
            "thinking_truncation_rate": float(encoder_diagnostics.truncation_rate),
            "parsing_seconds": parsing_seconds,
            "teacher_generation_seconds": teacher_generation_seconds,
            "teacher_generation_turns_per_second": m / max(teacher_generation_seconds, 1e-12),
            "diagonal_scoring_seconds": diagonal_seconds,
            "teacher_forcing_seconds": teacher_forcing_seconds,
            "teacher_forcing_rows_per_second": (m + len(pairs)) / max(teacher_forcing_seconds, 1e-12),
            "teacher_forcing_fused": float(self.m1_fuse_teacher_forcing),
            "teacher_forcing_row_count": float(m + len(pairs)),
            "encoder_seconds": encoder_seconds,
            "encoder_total_seconds": encoder_total_seconds,
            "encoder_load_seconds": encoder_load_seconds,
            "encoder_stage_in_seconds": encoder_timings["stage_in_seconds"],
            "encoder_stage_out_seconds": encoder_timings["stage_out_seconds"],
            "encoder_actor_rank0": float(self.m1_encoder_execution == "actor_rank0"),
            "encoder_actor_all": float(self.m1_encoder_execution == "actor_all"),
            "encoder_all_worker_wall_seconds": encoder_timings.get("all_worker_wall_seconds", 0.0),
            "encoder_actor_device_index": encoder_timings["device_index"],
            "encoder_tokenization_seconds": encoder_diagnostics.tokenization_seconds,
            "encoder_transfer_seconds": encoder_diagnostics.transfer_seconds,
            "encoder_forward_seconds": encoder_diagnostics.forward_seconds,
            "encoder_text_count": float(encoder_diagnostics.num_texts),
            "encoder_invalid_text_count": float(encoder_diagnostics.invalid_texts),
            "encoder_encoded_tokens": float(encoder_diagnostics.encoded_tokens),
            "encoder_max_encoded_tokens": float(encoder_diagnostics.max_encoded_tokens),
            "encoder_batch_count": float(encoder_diagnostics.num_batches),
            "encoder_tokens_per_second": encoder_diagnostics.encoded_tokens / max(encoder_total_seconds, 1e-12),
            "encoder_forward_tokens_per_second": encoder_diagnostics.encoded_tokens / max(encoder_diagnostics.forward_seconds, 1e-12),
            "matrix_construction_seconds": matrix_seconds,
            "source_selection_seconds": selection_seconds,
            "cross_batch_construction_seconds": cross_batch_seconds,
            "cross_scoring_seconds": cross_seconds,
            "cross_diagonal_abs_disagreement": float(cross_diagonal_disagreement),
            "mechanism1_total_seconds": mechanism1_seconds,
            "teacher_generation_time_fraction": teacher_generation_seconds / max(mechanism1_seconds, 1e-12),
            "teacher_forcing_time_fraction": teacher_forcing_seconds / max(mechanism1_seconds, 1e-12),
            "encoder_time_fraction": encoder_total_seconds / max(mechanism1_seconds, 1e-12),
            "teacher_do_sample": float(self.m1_teacher_do_sample),
            "student_teacher_sampling_shared": 1.0,
            "teacher_sampling_temperature": float(self.config.actor_rollout_ref.rollout.temperature),
            "teacher_sampling_top_p": float(self.config.actor_rollout_ref.rollout.top_p),
            "teacher_sampling_top_k": float(self.config.actor_rollout_ref.rollout.top_k),
            **{f"vllm_{key}": value for key, value in rollout_runtime_stats.items()},
            **gap_diagnostics,
            **similarity_diagnostics,
            **response_diagnostics,
        }
        encoder_contract = self._resolved_encoder_config()
        self.last_mechanism1_result = Mechanism1Result(
            turn_map=turn_map,
            thinking_similarity=thinking_similarity.detach(),
            structural_mask=structural_mask.detach(),
            selected_source_indices=selected_sources,
            selected_source_weights=combined["selected_source_weights"],
            correspondence_confidence=combined["correspondence_confidence"],
            correction_strength=combined["correction_strength"],
            diagonal_teacher_log_probs=diagonal_teacher_log_probs,
            cross_teacher_log_probs=combined["cross_teacher_log_probs"],
            corrected_teacher_log_probs=combined["corrected_teacher_log_probs"],
            diagonal_token_gap=combined["diagonal_token_gap"],
            corrected_token_gap=combined["corrected_token_gap"],
            sampled_response_mask=sampled_response_mask,
            policy_loss_mask=policy_loss_mask,
            similarity_threshold=self.m1_similarity_threshold,
            alpha_max=self.m1_alpha_max,
            encoder_contract={
                "model_path": str(encoder_contract["model_path"]),
                "revision": str(encoder_contract.get("revision") or ""),
                "input_scope": "thinking_text_only",
                "role_conditioning": "symmetric_none",
                "max_length": int(encoder_contract["max_length"]),
                "padding_side": str(encoder_contract["padding_side"]),
                "truncation_side": str(encoder_contract["truncation_side"]),
                "pooling": str(encoder_contract["pooling"]),
                "normalization": "fp32_l2",
                "attention_implementation": str(
                    encoder_contract["attention_implementation"]
                ),
                "execution": self.m1_encoder_execution,
                "offload_after_encode": self.m1_encoder_offload_after_encode,
            },
            student_thinking=student_thinking,
            teacher_thinking=teacher_thinking,
            student_actions=student_actions,
            teacher_actions=teacher_actions,
            action_similarity=action_similarity,
            diagnostics=diagnostics,
        )

        audit = getattr(self, "retrospective_logger", None)
        if audit is not None and audit.enabled and audit.detail_bytes < audit.byte_cap:
            audit_start = time.perf_counter()
            try:
                regions = region_cache.get("masks")
                if regions is None:
                    regions, counts = response_region_masks(
                        self.tokenizer, canonical_batch.batch["responses"], sampled_response_mask)
                    region_cache.update(masks=regions, counts=counts)
                records = turn_gap_records(self.last_mechanism1_result, student_log_probs, regions,
                                           float(self.m1_diagnostics_cfg.get("sign_epsilon", 1e-4)))
                for i, record in enumerate(records):
                    for field in ("data_source", "episode_rewards", "episode_lengths", "is_action_valid"):
                        if field in canonical_batch.non_tensor_batch:
                            record[field] = canonical_batch.non_tensor_batch[field][i]
                    record["selected_sources"] = [
                        {"canonical_index": source,
                         "similarity": float(thinking_similarity[source, i]),
                         "weight": float(combined["selected_source_weights"][i][k])}
                        for k, source in enumerate(selected_sources[i])]
                snapshots = []
                if audit.due(self.global_steps, self.global_steps >= self.total_training_steps):
                    for i, reason in choose_sample_turns(records, audit.max_turns, self.global_steps):
                        snapshots.append(token_snapshot(
                            index=i, reason=reason, result=self.last_mechanism1_result,
                            batch=canonical_batch, teacher_batch=teacher_batch,
                            student_log_probs=student_log_probs, combined=combined, cross_scores=cross_scores,
                            region_masks=regions, logger=audit))
                audit.pending = {"records": records, "snapshots": snapshots,
                                 "region_counts": region_cache.get("counts", {}),
                                 "encoder_contract": self.last_mechanism1_result.encoder_contract}
            except Exception as exc:
                audit.pending = None
                audit.failure("capture", exc)
            diagnostics["audit_capture_seconds"] = time.perf_counter() - audit_start

        expanded_teacher = expand_canonical_tensor(combined["corrected_teacher_log_probs"], turn_map)
        if self.config.algorithm.get("sdar_aux", {}).get("enabled", False):
            batch.batch["sdar_diagonal_teacher_log_probs"] = expand_canonical_tensor(
                diagonal_teacher_log_probs.detach(), turn_map
            )
        batch.batch["m1_corrected_gap"] = expand_canonical_tensor(combined["corrected_token_gap"], turn_map)
        batch.batch["m1_correspondence_confidence"] = expand_canonical_tensor(
            combined["correspondence_confidence"], turn_map
        )
        batch.batch["m1_correction_strength"] = expand_canonical_tensor(
            combined["correction_strength"], turn_map
        )
        return expanded_teacher
