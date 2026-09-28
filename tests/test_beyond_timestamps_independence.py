from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.trainer.main_beyond_timestamps import _validate_beyond_timestamps_config
from verl.trainer.ppo.beyond_timestamps_loss import (
    compute_beyond_timestamps_alignment_loss,
)
from verl.trainer.ppo.thinking_correspondence import (
    canonical_multiplicity_row_weights,
    canonicalize_turns,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _compose_beyond_config(*overrides: str):
    with initialize_config_dir(
        version_base=None,
        config_dir=str(REPO_ROOT / "verl/trainer/config"),
    ):
        return compose(
            config_name="beyond_timestamps_trainer",
            overrides=list(overrides),
        )


def test_dedicated_config_declares_complete_method_tree():
    config = OmegaConf.load(
        REPO_ROOT / "verl/trainer/config/beyond_timestamps_trainer.yaml"
    )

    assert config.defaults == ["ppo_trainer", "_self_"]
    assert config.algorithm.thinking_correspondence.enabled is True
    assert config.algorithm.thinking_correspondence.consumer == "mechanism2"
    assert config.algorithm.mechanism2.enabled is True
    assert config.algorithm.thinking_correspondence.alignment.coef == 0.01
    assert config.algorithm.thinking_correspondence.encoder.pooling == "last_token"
    assert config.algorithm.thinking_correspondence.encoder.padding_side == "left"
    assert config.algorithm.thinking_correspondence.encoder.max_length == 4096


def test_default_and_m1_only_compositions_pass_mode_validation():
    full = _compose_beyond_config()
    m1_only = _compose_beyond_config(
        "algorithm.thinking_correspondence.consumer=standalone_alignment",
        "algorithm.mechanism2.enabled=false",
    )

    _validate_beyond_timestamps_config(full)
    _validate_beyond_timestamps_config(m1_only)


def test_mode_validation_rejects_consumer_flag_drift():
    invalid = _compose_beyond_config("algorithm.mechanism2.enabled=false")

    with pytest.raises(ValueError, match="enabled must be true exactly"):
        _validate_beyond_timestamps_config(invalid)


def test_mode_validation_requires_token_mean_for_copy_weighting():
    invalid = _compose_beyond_config(
        "algorithm.thinking_correspondence.consumer=standalone_alignment",
        "algorithm.mechanism2.enabled=false",
        "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean",
    )

    with pytest.raises(ValueError, match="canonical multiplicity weighting"):
        _validate_beyond_timestamps_config(invalid)


def test_public_entry_owns_runner_instead_of_delegating_to_sdar():
    source = (
        REPO_ROOT / "verl/trainer/main_beyond_timestamps.py"
    ).read_text(encoding="utf-8")

    assert 'config_name="beyond_timestamps_trainer"' in source
    assert "class BeyondTimestampsTaskRunner" in source
    assert "BeyondTimestampsRayTrainer" in source
    assert "main_sdar" not in source
    assert "run_sdar" not in source


def test_baseline_sdar_stack_has_no_beyond_mode_switches():
    baseline_files = [
        REPO_ROOT / "verl/trainer/main_sdar.py",
        REPO_ROOT / "verl/trainer/ppo/skillsd_ray_trainer.py",
        REPO_ROOT / "verl/trainer/ppo/sdar_utils.py",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in baseline_files)

    assert "thinking_correspondence" not in source
    assert "mechanism2" not in source
    assert "m1_corrected_gap" not in source


def test_all_twelve_method_launchers_use_beyond_entry_and_declared_keys():
    full = sorted((REPO_ROOT / "examples/beyond_timestamps_trainer").glob("run_*.sh"))
    m1_only = sorted(
        (REPO_ROOT / "examples/thinking_correspondence_trainer").glob("run_*.sh")
    )

    assert len(full) == 6
    assert len(m1_only) == 6
    for path in full + m1_only:
        source = path.read_text(encoding="utf-8")
        assert "python3 -m verl.trainer.main_beyond_timestamps" in source
        assert "main_sdar" not in source
        assert "+algorithm.thinking_correspondence" not in source
        assert "+algorithm.mechanism2" not in source
        assert 'default_thinking_encoder_path="$REPO_ROOT/models/' in source
        assert "THINKING_ENCODER_PATH" in source
        assert "Align-OPD" not in source

    for path in full:
        source = path.read_text(encoding="utf-8")
        assert "algorithm.thinking_correspondence.consumer=mechanism2" in source
        assert "algorithm.mechanism2.enabled=true" in source

    for path in m1_only:
        source = path.read_text(encoding="utf-8")
        assert "algorithm.thinking_correspondence.consumer=standalone_alignment" in source
        assert "algorithm.mechanism2.enabled=false" in source


def test_alignment_uses_rollout_frozen_corrected_gap():
    student = torch.zeros((1, 2), dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([[1.0, -1.0]], requires_grad=True)
    corrected_gap = torch.tensor([[0.2, -0.2]], requires_grad=True)
    response_mask = torch.ones((1, 2), dtype=torch.float32)

    loss, metrics = compute_beyond_timestamps_alignment_loss(
        student_log_probs=student,
        corrected_teacher_log_probs=teacher,
        rollout_corrected_gap=corrected_gap,
        response_mask=response_mask,
        gate_beta=5.0,
    )
    expected_gate = torch.sigmoid(torch.tensor([[1.0, -1.0]]))
    expected_loss = (expected_gate * teacher.detach()).mean()

    assert torch.allclose(loss.detach(), expected_loss)
    loss.backward()
    assert torch.allclose(student.grad, -expected_gate / 2)
    assert teacher.grad is None
    assert corrected_gap.grad is None
    assert metrics["beyond_timestamps/gate_active_ratio"] == 0.5


def test_alignment_is_invariant_to_framework_copies():
    turn_map = canonicalize_turns(
        ["task", "task", "task"],
        ["rollout-1", "rollout-2", "rollout-1"],
        [0, 0, 0],
    )
    row_index = torch.tensor(turn_map.row_to_canonical)
    row_weight = canonical_multiplicity_row_weights(turn_map).unsqueeze(-1)
    canonical_student = torch.tensor(
        [[-0.2, 0.1], [0.3, -0.4]],
        requires_grad=True,
    )
    canonical_teacher = torch.tensor([[0.4, -0.1], [0.2, 0.5]])
    canonical_gap = canonical_teacher - canonical_student.detach()
    canonical_mask = torch.ones_like(canonical_student, dtype=torch.long)

    canonical_loss, _ = compute_beyond_timestamps_alignment_loss(
        student_log_probs=canonical_student,
        corrected_teacher_log_probs=canonical_teacher,
        rollout_corrected_gap=canonical_gap,
        response_mask=canonical_mask,
    )
    canonical_gradient = torch.autograd.grad(
        canonical_loss,
        canonical_student,
        retain_graph=True,
    )[0]

    weighted_loss, _ = compute_beyond_timestamps_alignment_loss(
        student_log_probs=canonical_student.index_select(0, row_index),
        corrected_teacher_log_probs=canonical_teacher.index_select(0, row_index),
        rollout_corrected_gap=canonical_gap.index_select(0, row_index),
        response_mask=canonical_mask.index_select(0, row_index),
        loss_weight=row_weight,
    )
    weighted_gradient = torch.autograd.grad(weighted_loss, canonical_student)[0]

    assert torch.allclose(weighted_loss, canonical_loss)
    assert torch.allclose(weighted_gradient, canonical_gradient)
