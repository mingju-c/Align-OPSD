"""Local, read-only audit trail. No model calls and no training RNG consumption."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time
import uuid
import warnings

import torch


def json_safe(value):
    """Strict JSON; redact credential fields, represent nonfinite numbers as null."""
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return {str(k): ("[REDACTED]" if re.search(
            r"(^|_)(password|secret|api_key|access_token|auth_token|credential)(_|$)", str(k), re.I
        ) else json_safe(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


class RetrospectiveLogger:
    """One unique directory per process invocation, including resumed runs.

    Detail has a cumulative byte cap; scalar metrics continue after the cap.
    Files are closed after each append. Failures warn and are exposed as metrics,
    never silently change the objective or restart a run.
    """

    def __init__(self, config, *, resolved_config, restored_step, code_root):
        self.config = dict(config)
        self.enabled = bool(config.get("enabled", True))
        self.interval = int(config.get("sample_interval", 10))
        self.max_turns = int(config.get("sample_turns", 8))
        self.max_tokens = int(config.get("sample_tokens", 512))
        self.max_context_tokens = int(config.get("context_tokens", 4096))
        self.byte_cap = int(config.get("max_detail_mb", 1024)) * 1024 * 1024
        if min(self.interval, self.max_turns, self.max_tokens, self.max_context_tokens, self.byte_cap) <= 0:
            raise ValueError("retrospective logging limits and interval must be positive")
        self.first_step = int(restored_step) + 1
        self.errors = self.dropped_records = self.detail_bytes = self.bytes_written = 0
        self.pending = None
        self.directory = None
        self.run_id = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "_" + uuid.uuid4().hex[:10]
        if not self.enabled:
            return
        experiment = str(resolved_config.get("trainer", {}).get("experiment_name", "run"))
        experiment = re.sub(r"[^a-zA-Z0-9_.-]+", "_", experiment)[:120].strip(".") or "run"
        try:
            self.directory = Path(config.get("directory", "outputs/retrospective")).expanduser().resolve() / experiment / self.run_id
            self.directory.mkdir(parents=True, exist_ok=False)
            code_root = Path(code_root)
            def git(*args):
                try:
                    return subprocess.check_output(["git", "-C", str(code_root), *args], timeout=5,
                                                   stderr=subprocess.DEVNULL).decode().strip()
                except (OSError, subprocess.SubprocessError):
                    return None
            hashes = {str(p.relative_to(code_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for folder in ("verl/trainer/ppo", "verl/trainer/config")
                      for p in sorted((code_root / folder).glob("*")) if p.is_file()}
            self.write("manifest", {"restored_step": restored_step, "config": resolved_config,
                       "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
                       "code_sha256": hashes, "torch_version": torch.__version__,
                       "semantics": "sampled-token log probabilities; scores are not utility labels"})
            print(f"[retrospective] local audit directory: {self.directory}", flush=True)
        except Exception as exc:
            self.failure("initialize", exc)
            self.enabled = False

    def failure(self, stage, exc):
        self.errors += 1
        warnings.warn(f"retrospective logging {stage} failed: {type(exc).__name__}: {exc}", RuntimeWarning)

    def due(self, step, last=False):
        return self.enabled and self.detail_bytes < self.byte_cap and (
            step == self.first_step or step % self.interval == 0 or last)

    def write(self, stream, record, *, detail=False):
        if not self.enabled or self.directory is None:
            return False
        try:
            payload = dict(schema_version=1, run_id=self.run_id, **record)
            line = (json.dumps(json_safe(payload), ensure_ascii=False, allow_nan=False,
                               separators=(",", ":")) + "\n").encode("utf-8")
            if detail and self.detail_bytes + len(line) > self.byte_cap:
                self.dropped_records += 1
                return False
            with (self.directory / f"{stream}.jsonl").open("ab") as handle:
                handle.write(line)
            self.bytes_written += len(line)
            if detail:
                self.detail_bytes += len(line)
            return True
        except Exception as exc:
            self.failure(stream, exc)
            return False

    def metrics(self):
        return {f"audit/{key}": float(value) for key, value in {
            "enabled": self.enabled, "errors": self.errors, "bytes_written": self.bytes_written,
            "detail_bytes": self.detail_bytes, "dropped_records": self.dropped_records,
            "detail_cap_reached": self.dropped_records > 0,
        }.items()}


def turn_gap_records(result, student_log_probs, region_masks, sign_epsilon):
    """All canonical turns, common policy mask, both token and turn denominators.

    Counts/sums allow exact pooling over steps, unlike averages of averages.
    Cross values on unmatched turns are explicitly labelled diagonal fallback.
    """
    policy = result.policy_loss_mask.detach().bool().cpu()
    d = result.diagonal_token_gap.detach().float().cpu()
    c = result.cross_teacher_log_probs.detach().float().cpu() - student_log_probs.detach().float().cpu()
    r = result.corrected_token_gap.detach().float().cpu()
    records = [{"canonical_index": i, "identity": dataclasses.asdict(identity),
                "matched_sources": len(result.selected_source_indices[i]),
                "cross_is_diagonal_fallback": not bool(result.selected_source_indices[i]),
                "confidence": float(result.correspondence_confidence[i]),
                "alpha": float(result.correction_strength[i]), "regions": {}}
               for i, identity in enumerate(result.turn_map.identities)]
    finite = torch.isfinite(d) & torch.isfinite(c) & torch.isfinite(r)
    for name, mask in {"response": policy, **{k: policy & v for k, v in region_masks.items()}}.items():
        valid = mask & finite
        counts = valid.sum(-1)
        fields = {"eligible_count": mask.sum(-1), "count": counts,
                  "nonfinite_count": (mask & ~finite).sum(-1)}
        for field, values in (("diagonal", d), ("cross", c), ("corrected", r),
                              ("cross_delta", c - d), ("corrected_delta", r - d)):
            total = torch.where(valid, values, 0).sum(-1)
            fields[field + "_sum"] = total
            fields[field + "_mean"] = total / counts.clamp_min(1)
        for band, eps in (("strict", 0.), ("tol", sign_epsilon)):
            neg, pos = d < -eps, d > eps
            fields[band + "/diagonal_negative_count"] = (valid & neg).sum(-1)
            fields[band + "/diagonal_positive_count"] = (valid & pos).sum(-1)
            for label, values in (("cross", c), ("corrected", r)):
                for event, condition in (("neg_to_pos", neg & (values > eps)),
                                         ("pos_to_neg", pos & (values < -eps)),
                                         ("increase", values - d > eps), ("decrease", values - d < -eps)):
                    fields[f"{band}/{label}_{event}_count"] = (valid & condition).sum(-1)
        columns = {k: v.tolist() for k, v in fields.items()}
        for i, record in enumerate(records):
            record["regions"][name] = {k: v[i] for k, v in columns.items()}
    return records


def choose_sample_turns(records, limit, step):
    """Half deterministic hash sample, half targeted examples; never use training RNG."""
    if not records:
        return []
    order = sorted(range(len(records)), key=lambda i: hashlib.sha256(
        f"{step}:{records[i]['identity']}".encode()).digest())
    chosen = {i: "hash_sample" for i in order[:max(1, limit // 2)]}
    candidates = []
    for event in ("neg_to_pos", "pos_to_neg"):
        for i in order:
            if records[i]["regions"]["response"][f"tol/cross_{event}_count"]:
                candidates.append((i, "cross_" + event))
                break
    for i in order:
        if records[i]["cross_is_diagonal_fallback"]:
            candidates.append((i, "no_match_fallback"))
            break
    extreme = sorted(order, key=lambda i: abs(records[i]["regions"]["response"]["cross_delta_mean"]), reverse=True)
    candidates += [(i, "large_absolute_cross_shift") for i in extreme]
    for i, reason in candidates:
        if len(chosen) >= limit:
            break
        chosen.setdefault(i, reason)
    return list(chosen.items())


def token_snapshot(*, index, reason, result, batch, teacher_batch, student_log_probs,
                   combined, cross_scores, region_masks, logger):
    """Bounded original-position arrays, including each selected source's score.

    Save heads when capped, with original lengths and truncation flags. Prompt
    IDs are the ACTUAL privileged scoring contexts (tails if capped), not a
    fabricated natural-language explanation. Local only; no extra forwards.
    """
    sampled = result.sampled_response_mask[index].detach().bool().cpu()
    all_positions = sampled.nonzero(as_tuple=True)[0].tolist()
    positions = all_positions[:logger.max_tokens]
    def values(tensor, row=None):
        if row is not None:
            tensor = tensor[row]
        return tensor.detach().cpu()[positions].tolist()
    def context(row):
        # build_teacher_batch stores input_ids + responses, NOT a prompts key.
        prompt_length = (teacher_batch.batch["input_ids"].shape[-1]
                         - teacher_batch.batch["responses"].shape[-1])
        prompts = teacher_batch.batch["input_ids"][row, :prompt_length].detach().cpu()
        mask = teacher_batch.batch["attention_mask"][row, :prompt_length].detach().bool().cpu()
        ids = prompts[mask].tolist()
        return {"token_ids": ids[-logger.max_context_tokens:], "original_length": len(ids),
                "truncated": len(ids) > logger.max_context_tokens, "retained": "tail"}
    sources = []
    for k, source in enumerate(result.selected_source_indices[index]):
        sources.append({"canonical_index": source, "identity": dataclasses.asdict(result.turn_map.identities[source]),
                        "similarity": float(result.thinking_similarity[source, index]),
                        "weight": float(result.selected_source_weights[index][k]),
                        "teacher_log_probs": values(cross_scores[(source, index)]),
                        "privileged_prompt": context(source),
                        "retrieval_thinking": result.teacher_thinking[source][:4096],
                        "retrieval_thinking_truncated": len(result.teacher_thinking[source]) > 4096,
                        "generated_action": result.teacher_actions[source][:1024],
                        "generated_action_truncated": len(result.teacher_actions[source]) > 1024})
    return {"canonical_index": index, "identity": dataclasses.asdict(result.turn_map.identities[index]),
            "selection_reason": reason, "representative_rate_denominator": False,
            "sampled_token_count": len(all_positions), "tokens_truncated": len(all_positions) > len(positions),
            "positions": positions, "token_ids": values(batch.batch["responses"], index),
            "policy_mask": values(result.policy_loss_mask, index),
            "region_masks": {name: values(mask, index) for name, mask in region_masks.items()},
            "student_log_probs": values(student_log_probs, index),
            "diagonal_teacher_log_probs": values(result.diagonal_teacher_log_probs, index),
            "cross_teacher_log_probs": values(result.cross_teacher_log_probs, index),
            "corrected_teacher_log_probs": values(result.corrected_teacher_log_probs, index),
            "log_mean_cross_teacher_log_probs": values(combined["log_mean_cross_teacher_log_probs"], index),
            "probability_mixture_cross_teacher_log_probs": values(combined["probability_mixture_cross_teacher_log_probs"], index),
            "alpha": float(result.correction_strength[index]),
            "diagonal_privileged_prompt": context(index), "sources": sources}


def attach_credit(records, result, episode_outcome, trajectory_advantage, invalid_residual):
    """Attach scalars only; never retain the dense correspondence matrix."""
    fields = ("profile_valid", "js_divergence", "boundary_valid", "segment_index_per_turn",
              "turn_gap", "outcome_aligned_score", "within_segment_turn_budget", "raw_credit_density",
              "projected_credit_density", "turn_weight")
    columns = {key: getattr(result, key).detach().cpu().tolist() for key in fields}
    columns.update(episode_reward=episode_outcome.detach().cpu().tolist(),
                   trajectory_advantage=trajectory_advantage.detach().cpu().tolist(),
                   invalid_action_residual=invalid_residual.detach().cpu().tolist())
    for i, record in enumerate(records):
        credit = {key: values[i] for key, values in columns.items()}
        segment = int(credit["segment_index_per_turn"])
        credit["segment_budget"] = float(result.segment_budget[segment]) if segment >= 0 else None
        credit["segmentation_threshold"] = result.segmentation_threshold
        record["credit"] = credit
    for trajectory in result.trajectory_segmentations:
        for kind, offsets in (("semantic", trajectory.semantic_boundaries),
                              ("correspondence_fallback", trajectory.correspondence_guided_fallback_boundaries),
                              ("forced_max", trajectory.forced_max_boundaries)):
            for offset in offsets:
                records[trajectory.turn_indices[offset]]["credit"]["boundary_type"] = kind
