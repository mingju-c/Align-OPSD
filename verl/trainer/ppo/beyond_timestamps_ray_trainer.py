"""Training loop owned by BeyondTimestamps.

The loop consumes Mechanism 1's correspondence sidecar and either applies the
standalone corrected-gap alignment loss or delegates credit construction to
Mechanism 2. Baseline SkillSD/SDAR trainers remain untouched.
"""

import json
import os
from pprint import pprint
from pathlib import Path
import time

from verl.trainer.ppo.retrospective_logging import RetrospectiveLogger

import numpy as np
import ray
import torch
from tqdm import tqdm

from agent_system.multi_turn_rollout import adjust_batch
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    _timer,
    apply_invalid_action_penalty,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.thinking_correspondence_trainer import ThinkingCorrespondenceRayTrainer
from verl.utils.memory_monitor import monitor_memory
from verl.utils.metric import reduce_metrics
from verl.utils.torch_functional import masked_mean


class BeyondTimestampsRayTrainer(ThinkingCorrespondenceRayTrainer):
    """Train either standalone M1 or the coupled M1+M2 method."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        alignment_cfg = self.m1_cfg.get("alignment", {})
        self.m1_alignment_coef = float(alignment_cfg.get("coef", 0.01))

    def fit(self):
        """Run the BeyondTimestamps training lifecycle."""

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self.retrospective_logger = RetrospectiveLogger(
            self.m1_diagnostics_cfg.get("retrospective", {}),
            resolved_config=OmegaConf.to_container(self.config, resolve=True),
            restored_step=self.global_steps, code_root=Path(__file__).resolve().parents[3],
        )

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            with monitor_memory("initial_validation", {}, self.global_steps):
                val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            self.retrospective_logger.write("metrics", {"step": self.global_steps,
                                                       "phase": "initial_validation", "metrics": val_metrics})
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training")
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                self.retrospective_logger.pending = None
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                gen_batch.meta_info["diagnostic_step"] = int(self.global_steps)
                gen_batch.meta_info["diagnostic_role"] = "student"

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw), monitor_memory("gen", metrics, self.global_steps):
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
                        )

                    del batch
                    batch = gen_batch_output

                    batch = adjust_batch(self.config, batch)
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    # Mechanism 1: free Teacher generation and correspondence correction.
                    with _timer("teacher_forward", timing_raw), monitor_memory("teacher_forward", metrics, self.global_steps):
                        teacher_log_probs = self._compute_teacher_log_probs(batch)
                        batch.batch["teacher_log_probs"] = teacher_log_probs
                        m1_result = getattr(self, "last_mechanism1_result", None)
                        if m1_result is not None:
                            metrics.update({f"m1/{key}": value for key, value in m1_result.diagnostics.items()})
                            metrics["m1/correspondence_confidence_mean"] = (
                                m1_result.correspondence_confidence.float().mean().item()
                            )
                            metrics["m1/correction_strength_mean"] = (
                                m1_result.correction_strength.float().mean().item()
                            )

                    if self.use_reference_policy:
                        with _timer("ref", timing_raw), monitor_memory("ref", metrics, self.global_steps):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        m1_consumer = getattr(self, "consume_mechanism1_result", None)
                        mechanism2_mode = getattr(self, "m1_consumer", None) == "mechanism2"

                        # M2 must see the untouched episode outcome. Its consumer
                        # computes invalid-action credit as a separate residual
                        # after hierarchical allocation. All legacy paths retain
                        # the published penalty-before-GRPO behavior.
                        if (
                            not mechanism2_mode
                            and self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True)
                        ):
                            batch, invalid_metrics = apply_invalid_action_penalty(
                                batch,
                                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                            )
                            metrics.update(invalid_metrics)

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if mechanism2_mode:
                            if m1_consumer is None:
                                raise RuntimeError("Mechanism 2 requires a Mechanism-1 consumer")
                            batch, m1_consumer_metrics = m1_consumer(batch)
                            metrics.update(m1_consumer_metrics)
                        else:
                            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                                use_pf_ppo=self.config.algorithm.use_pf_ppo,
                                pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                                pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                                step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                                gigpo_mode=self.config.algorithm.gigpo.mode,
                                gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                                gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                            )
                            if m1_consumer is not None:
                                batch, m1_consumer_metrics = m1_consumer(batch)
                                metrics.update(m1_consumer_metrics)

                        audit = self.retrospective_logger
                        if audit.pending is not None:
                            audit_start = time.perf_counter()
                            pending = audit.pending
                            audit.pending = None
                            audit.write("turns", {"step": self.global_steps,
                                "region_counts": pending["region_counts"],
                                "encoder_contract": pending["encoder_contract"],
                                "turns": pending["records"]}, detail=True)
                            for snapshot in pending["snapshots"]:
                                i = snapshot["canonical_index"]
                                snapshot["turn_summary"] = pending["records"][i]
                                audit.write("samples", {"step": self.global_steps, **snapshot}, detail=True)
                            metrics["audit/detail_write_seconds"] = time.perf_counter() - audit_start

                        if m1_consumer is not None:
                            # The consumer must materialize everything needed by the
                            # actor into the batch. Release canonical MxM sidecar
                            # tensors before the policy update.
                            self.last_mechanism1_result = None
                            m1_result = None

                        # Standalone M1 keeps the standard GRPO advantages and adds
                        # corrected-gap alignment. M2 materializes its token credit
                        # above and deliberately disables direct alignment.

                        # Log teacher-student gap metrics
                        response_mask = batch.batch["response_mask"]
                        student_log_probs = batch.batch["old_log_probs"]
                        teacher_lp = batch.batch["teacher_log_probs"]
                        delta_t = (teacher_lp - student_log_probs) * response_mask
                        if mechanism2_mode:
                            metrics["m2/direct_alignment_enabled"] = 0.0
                            metrics["sdar/aux_enabled"] = float(
                                self.config.algorithm.get("sdar_aux", {}).get("enabled", False)
                            )
                        else:
                            metrics["m1/teacher_student_gap_mean"] = masked_mean(delta_t, response_mask).item()
                            metrics["m1/teacher_student_gap_std"] = masked_mean(delta_t ** 2, response_mask).sqrt().item()
                            metrics["m1/alignment_coef"] = self.m1_alignment_coef

                        # Save per-token gap data if SAVE_BEYOND_TIMESTAMPS_DEBUG=1, at test_freq interval
                        if os.environ.get("SAVE_BEYOND_TIMESTAMPS_DEBUG", "0") == "1" and \
                                self.config.trainer.test_freq > 0 and \
                                self.global_steps % self.config.trainer.test_freq == 0:
                            save_dir = os.environ.get(
                                "SAVE_BEYOND_TIMESTAMPS_DEBUG_DIR",
                                "outputs/beyond_timestamps_debug"
                            )
                            os.makedirs(save_dir, exist_ok=True)
                            save_path = os.path.join(save_dir, f"step_{self.global_steps}.jsonl")

                            bs = response_mask.shape[0]
                            turn_steps = batch.non_tensor_batch.get("turn_step", np.zeros(bs, dtype=object))
                            traj_uids = batch.non_tensor_batch.get("traj_uid", np.array([""] * bs, dtype=object))
                            episode_rewards_arr = batch.non_tensor_batch.get("episode_rewards", np.zeros(bs, dtype=object))
                            episode_lengths_arr = batch.non_tensor_batch.get("episode_lengths", np.zeros(bs, dtype=object))
                            response_ids = batch.batch["responses"]

                            with open(save_path, "w") as f:
                                for i in range(bs):
                                    mask_i = response_mask[i].bool()
                                    valid_count = mask_i.sum().item()
                                    if valid_count == 0:
                                        continue
                                    token_ids_i = response_ids[i][mask_i].cpu().tolist()
                                    tokens_i = [self.tokenizer.decode([tid]) for tid in token_ids_i]
                                    gaps_i = delta_t[i][mask_i].cpu().tolist()
                                    teacher_lps_i = teacher_lp[i][mask_i].cpu().tolist()
                                    student_lps_i = student_log_probs[i][mask_i].cpu().tolist()

                                    record = {
                                        "global_step": self.global_steps,
                                        "sample_idx": i,
                                        "turn_step": int(turn_steps[i]),
                                        "traj_uid": str(traj_uids[i]),
                                        "episode_reward": float(episode_rewards_arr[i]),
                                        "episode_length": float(episode_lengths_arr[i]),
                                        "tokens": tokens_i,
                                        "token_ids": token_ids_i,
                                        "gaps": gaps_i,
                                        "teacher_log_probs": teacher_lps_i,
                                        "student_log_probs": student_lps_i,
                                    }
                                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            print(f"[BeyondTimestamps Debug] Saved per-token gap data to {save_path} ({bs} samples)")

                        if mechanism2_mode:
                            # These diagnostics have already been consumed. Avoid
                            # transferring unused Teacher/M1 tensors to the actor.
                            for key in (
                                "teacher_log_probs",
                                "m1_corrected_gap",
                                "m1_correspondence_confidence",
                                "m1_correction_strength",
                            ):
                                batch.batch.pop(key, None)

                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw), monitor_memory("update_actor", metrics, self.global_steps):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    test_start_step = self.config.trainer.get("test_start_step", 0)
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or (self.global_steps >= test_start_step and self.global_steps % self.config.trainer.test_freq == 0)):
                        with _timer("testing", timing_raw), monitor_memory("testing", metrics, self.global_steps):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    save_start_step = self.config.trainer.get("save_start_step", 0)
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or (
                            self.global_steps >= save_start_step
                            and self.global_steps % self.config.trainer.save_freq == 0
                        )
                    ):
                        with _timer("save_checkpoint", timing_raw), monitor_memory("save_checkpoint", metrics, self.global_steps):
                            self._save_checkpoint()

                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                metrics.update(self.retrospective_logger.metrics())
                self.retrospective_logger.write("metrics", {
                    "step": self.global_steps, "phase": "training", "metrics": metrics})
                # Expose an I/O failure from this very append to the normal logger.
                metrics.update(self.retrospective_logger.metrics())
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
