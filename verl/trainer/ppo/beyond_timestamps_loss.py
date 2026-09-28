"""Actor-side alignment objective for standalone BeyondTimestamps M1."""

from __future__ import annotations

import torch

from verl.trainer.ppo.core_algos import agg_loss


def compute_beyond_timestamps_alignment_loss(
    student_log_probs: torch.Tensor,
    corrected_teacher_log_probs: torch.Tensor,
    rollout_corrected_gap: torch.Tensor,
    response_mask: torch.Tensor,
    gate_beta: float = 5.0,
    loss_agg_mode: str = "token-mean",
    loss_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Distill from M1's corrected Teacher with a rollout-frozen gate.

    The corrected gap is produced once during rollout scoring. Detaching it here
    prevents PPO epochs from changing correspondence confidence as the current
    policy moves. Gradients therefore flow only through ``student_log_probs``.
    ``loss_weight`` removes any extra objective mass from framework-created
    copies without changing the policy token mask.
    """

    if corrected_teacher_log_probs.shape != student_log_probs.shape:
        raise ValueError("corrected_teacher_log_probs must match student_log_probs")
    if rollout_corrected_gap.shape != student_log_probs.shape:
        raise ValueError("rollout_corrected_gap must match student_log_probs")
    if response_mask.shape != student_log_probs.shape:
        raise ValueError("response_mask must match student_log_probs")

    teacher = corrected_teacher_log_probs.detach()
    corrected_gap = rollout_corrected_gap.detach().to(
        device=student_log_probs.device,
        dtype=student_log_probs.dtype,
    )
    gate = torch.sigmoid(gate_beta * corrected_gap).detach()
    loss = agg_loss(
        loss_mat=gate * (teacher - student_log_probs),
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        loss_weight=loss_weight,
    )

    with torch.no_grad():
        metric_mask = response_mask
        if loss_weight is not None:
            metric_mask = response_mask.to(dtype=loss_weight.dtype) * loss_weight.to(
                device=response_mask.device,
            )
        mask_sum = metric_mask.sum().clamp(min=1)
        gate_mean = (gate * metric_mask).sum() / mask_sum
        gate_active = ((gate > 0.5).to(metric_mask.dtype) * metric_mask).sum() / mask_sum
        gap_mean = (corrected_gap * metric_mask).sum() / mask_sum

    return loss, {
        "beyond_timestamps/alignment_loss": loss.detach().item(),
        "beyond_timestamps/gate_mean": gate_mean.item(),
        "beyond_timestamps/gate_active_ratio": gate_active.item(),
        "beyond_timestamps/corrected_gap_mean": gap_mean.item(),
    }
