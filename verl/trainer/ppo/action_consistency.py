"""Filter cross sources by action agreement, without consulting token gaps."""
import math
import re

import torch


def validate_action_filter_config(cfg):
    threshold = cfg.get("similarity_threshold", 0.8)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("action_filter.similarity_threshold must be a number in [-1, 1]")
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("action_filter.similarity_threshold must be a number in [-1, 1]")
    return float(threshold)


def action_consistency_mask(candidates, similarities, teacher_actions, student_actions,
                            teacher_valid, student_valid, threshold, *, webshop=False):
    """Return accepted source-by-target pairs. Click targets require exact identity.

    Filter the already selected thinking top-K pairs without replacement.
    No score-sign selection.
    """
    threshold = validate_action_filter_config({"similarity_threshold": threshold})
    if candidates.shape != similarities.shape or candidates.shape != (len(teacher_actions), len(student_actions)):
        raise ValueError("action filter matrices must have source-by-target shape")
    device = candidates.device
    pool = candidates.detach().bool().cpu()
    sims = similarities.detach().float().cpu()
    tv = torch.as_tensor(teacher_valid).bool().cpu()
    sv = torch.as_tensor(student_valid).bool().cpu()
    accepted = pool & tv[:, None] & sv[None, :] & torch.isfinite(sims) & sims.ge(threshold)
    if webshop:
        def parse(action):
            match = re.fullmatch(r"\s*(search|click)\[(.*)\]\s*", action, flags=re.DOTALL)
            if not match or not match[2].strip():
                return None
            return match[1], match[2].strip()
        teachers = [parse(a) for a in teacher_actions]
        students = [parse(a) for a in student_actions]
        for source, target in accepted.nonzero(as_tuple=False).tolist():
            a, b = teachers[source], students[target]
            if a is None or b is None or a[0] != b[0] or (a[0] == "click" and a[1] != b[1]):
                accepted[source, target] = False
    return accepted.to(device)
