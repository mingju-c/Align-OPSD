"""Read-only, paired sampled-token diagnostics for cross-context rectification.

These statistics describe score movement, not action correctness or calibrated
utility. Region masks NEVER replace the actor's policy-loss mask.
"""

from __future__ import annotations

import math
import re

import torch


_BLOCK = re.compile(
    r"<(?P<tag>think|action|search|answer)\b[^>]*>(?P<body>.*?)</(?P=tag)\s*>",
    re.IGNORECASE | re.DOTALL,
)
_MARKER = re.compile(r"</?(?:think|action|search|answer)\b[^>]*>", re.IGNORECASE)


def response_region_masks(tokenizer, responses, sampled_response_mask):
    """Return conservative CPU masks on ORIGINAL sampled-token positions.

    Decode without dropping special tokens or cleaning spaces, then require
    exact token-ID roundtrip through an offset-capable tokenizer. Content-only
    spans exclude tags, EOS, and boundary-straddling tokens. Unsupported offsets,
    non-roundtripping IDs, malformed/nested/duplicate protocol blocks are
    reported, never guessed. They still participate in full-response statistics.
    """
    if responses.ndim != 2 or responses.shape != sampled_response_mask.shape:
        raise ValueError("response IDs and sampled response mask must have shape [M, L]")
    ids = responses.detach().cpu()
    sampled = sampled_response_mask.detach().bool().cpu()
    thinking = torch.zeros_like(sampled)
    action = torch.zeros_like(sampled)
    counts = dict(
        rows=float(ids.shape[0]), offset_unavailable_rows=0.0,
        token_roundtrip_mismatch_rows=0.0, malformed_region_rows=0.0,
        exact_roundtrip_rows=0.0, thinking_region_rows=0.0, action_region_rows=0.0,
        thinking_boundary_token_count=0.0, action_boundary_token_count=0.0,
    )
    for row in range(ids.shape[0]):
        positions = sampled[row].nonzero(as_tuple=True)[0].tolist()
        original = ids[row, positions].tolist()
        if not original:
            continue
        try:
            text = tokenizer.decode(
                original, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = encoded["offset_mapping"]
        except (NotImplementedError, TypeError, ValueError, KeyError):
            counts["offset_unavailable_rows"] += 1
            continue
        if list(encoded["input_ids"]) != original or len(offsets) != len(original):
            counts["token_roundtrip_mismatch_rows"] += 1
            continue
        counts["exact_roundtrip_rows"] += 1
        blocks = list(_BLOCK.finditer(text))
        thinks = [b for b in blocks if b.group("tag").lower() == "think"]
        actions = [b for b in blocks if b.group("tag").lower() != "think"]
        if len(list(_MARKER.finditer(text))) != 2 * len(blocks) or len(thinks) > 1 or len(actions) > 1:
            counts["malformed_region_rows"] += 1
            continue
        for name, matches, target in (("thinking", thinks, thinking), ("action", actions, action)):
            if not matches or not matches[0].group("body").strip():
                continue
            counts[f"{name}_region_rows"] += 1
            lo, hi = matches[0].span("body")
            for position, (start, end) in zip(positions, offsets, strict=True):
                if lo <= start < end <= hi:
                    target[row, position] = True
                elif start < hi and end > lo:
                    counts[f"{name}_boundary_token_count"] += 1
    return {"thinking": thinking, "action": action}, counts


def _distribution(name, values):
    if not values.numel():
        return {f"{name}_{key}": 0.0 for key in ("mean", "std", "p10", "p50", "p90")}
    q = torch.quantile(values, values.new_tensor([0.1, 0.5, 0.9])).tolist()
    return {
        f"{name}_mean": float(values.mean()),
        f"{name}_std": float(values.std(unbiased=False)),
        **{f"{name}_{key}": value for key, value in zip(("p10", "p50", "p90"), q, strict=True)},
    }


def paired_gap_statistics(diagonal_gap, cross_gap, corrected_gap, mask, sign_epsilon=1e-4):
    """Same-token means, signed changes and BOTH flip denominators.

    Unconditional rates use token_count; conditional rates use the original
    diagonal-negative/positive count. Empty denominators give 0 plus count=0.
    Strict and tolerance-band signs are both retained, so rounded near-zero
    values cannot silently be interpreted as robust sign reversals.
    """
    if not math.isfinite(sign_epsilon) or sign_epsilon < 0:
        raise ValueError("sign_epsilon must be finite and nonnegative")
    if any(x.shape != mask.shape for x in (diagonal_gap, cross_gap, corrected_gap)):
        raise ValueError("paired gap tensors and mask must have the same shape")
    mask = mask.detach().bool().cpu()
    values = [x.detach().float().cpu() for x in (diagonal_gap, cross_gap, corrected_gap)]
    finite = torch.isfinite(values[0]) & torch.isfinite(values[1]) & torch.isfinite(values[2])
    selected = mask & finite
    d, c, r = [x[selected] for x in values]
    n = d.numel()
    out = {
        "eligible_token_count": float(mask.sum()),
        "nonfinite_token_count": float((mask & ~finite).sum()),
        "token_count": float(n),
    }
    for name, x in (("diagonal_gap", d), ("cross_gap", c), ("corrected_gap", r),
                    ("cross_delta", c - d), ("corrected_delta", r - d)):
        out.update(_distribution(name, x))
    for band, eps in (("strict", 0.0), ("tol", sign_epsilon)):
        neg, pos = d < -eps, d > eps
        nneg, npos = int(neg.sum()), int(pos.sum())
        out[f"{band}/diagonal_negative_count"] = float(nneg)
        out[f"{band}/diagonal_positive_count"] = float(npos)
        out[f"{band}/diagonal_neutral_count"] = float(n - nneg - npos)
        for name, x in (("cross", c), ("corrected", r)):
            up = int((neg & (x > eps)).sum())
            down = int((pos & (x < -eps)).sum())
            increased, decreased = int(((x - d) > eps).sum()), int(((x - d) < -eps).sum())
            out.update({
                f"{band}/{name}_positive_count": float((x > eps).sum()),
                f"{band}/{name}_negative_count": float((x < -eps).sum()),
                f"{band}/{name}_neg_to_pos_count": float(up),
                f"{band}/{name}_pos_to_neg_count": float(down),
                f"{band}/{name}_neg_to_pos_rate": up / n if n else 0.0,
                f"{band}/{name}_pos_to_neg_rate": down / n if n else 0.0,
                f"{band}/{name}_neg_to_pos_given_negative": up / nneg if nneg else 0.0,
                f"{band}/{name}_pos_to_neg_given_positive": down / npos if npos else 0.0,
                f"{band}/{name}_increase_count": float(increased),
                f"{band}/{name}_decrease_count": float(decreased),
            })
    return out


def build_rectification_diagnostics(
    *, tokenizer, responses, sampled_response_mask, policy_loss_mask,
    selected_source_indices, combined, student_log_probs, sign_epsilon=1e-4, region_breakdown=True,
    region_cache=None,
):
    """Build canonical paired metrics; leave scores, masks and gradients intact."""
    policy = policy_loss_mask.detach().bool().cpu()
    sampled = sampled_response_mask.detach().bool().cpu()
    if policy.shape != responses.shape or sampled.shape != policy.shape:
        raise ValueError("responses and masks must align")
    if len(selected_source_indices) != policy.shape[0]:
        raise ValueError("source selections must align with canonical turns")
    if (policy & ~sampled).any():
        raise ValueError("policy mask must be a subset of sampled response mask")
    matched = torch.tensor([bool(s) for s in selected_source_indices]).unsqueeze(-1)
    paired = policy & matched
    d = combined["diagonal_token_gap"].detach().float().cpu()
    student = student_log_probs.detach().float().cpu()
    if student.shape != policy.shape:
        raise ValueError("student log probabilities must align with response tokens")
    r = combined["corrected_token_gap"].detach().float().cpu()
    c = combined["cross_teacher_log_probs"].detach().float().cpu() - student
    log_mean = combined["log_mean_cross_teacher_log_probs"].detach().float().cpu() - student
    probability = combined["probability_mixture_cross_teacher_log_probs"].detach().float().cpu() - student
    regions = {}
    region_counts = {}
    if region_breakdown:
        regions, region_counts = response_region_masks(tokenizer, responses, sampled_response_mask)
    if region_cache is not None:
        region_cache.update(masks=regions, counts=region_counts)
    out = {
        "diagnostics/sign_epsilon": float(sign_epsilon),
        "diagnostics/region_breakdown_enabled": float(region_breakdown),
        "paired/policy_token_count": float(policy.sum()),
        "paired/unmatched_policy_token_count": float((policy & ~matched).sum()),
        **{f"regions/{key}": value for key, value in region_counts.items()},
    }
    scopes = {"response": paired}
    scopes.update({name: paired & mask for name, mask in regions.items()})
    if regions:
        other = paired & ~(regions["thinking"] | regions["action"])
        out["regions/paired_other_token_count"] = float(other.sum())
    for name, mask in scopes.items():
        stats = paired_gap_statistics(d, c, r, mask, sign_epsilon)
        # Formula-only uplift is NOT evidence of semantic calibration.
        formula_mask = mask & torch.isfinite(log_mean) & torch.isfinite(probability)
        stats["formula_token_count"] = float(formula_mask.sum())
        stats.update(_distribution("probability_minus_log_mean", (probability - log_mean)[formula_mask]))
        stats.update(_distribution("log_mean_cross_gap", log_mean[formula_mask]))
        stats.update(_distribution("probability_cross_gap", probability[formula_mask]))
        stats["formula_neg_to_pos_count"] = float((formula_mask & (log_mean < 0) & (probability > 0)).sum())
        stats["formula_tol_neg_to_pos_count"] = float(
            (formula_mask & (log_mean < -sign_epsilon) & (probability > sign_epsilon)).sum()
        )
        out.update({f"paired/{name}/{key}": value for key, value in stats.items()})
    return out
