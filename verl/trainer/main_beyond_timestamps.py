"""Independent entry point for BeyondTimestamps M1 and coupled M1+M2 training."""

import math

import hydra
import ray
from omegaconf import OmegaConf


def _validate_beyond_timestamps_config(config) -> None:
    """Reject incompatible M1/M2 combinations before starting Ray."""
    ablation_mode = str(config.algorithm.get("ablation_mode", "E")).upper()
    if ablation_mode not in {"A", "B", "C", "D", "E"}:
        raise ValueError("algorithm.ablation_mode must be one of A, B, C, D, E")

    m1_cfg = config.algorithm.get("thinking_correspondence", {})
    encoder_cfg = m1_cfg.get("encoder", {})
    if encoder_cfg.get("execution", "trainer") not in {"trainer", "actor_rank0", "actor_all"}:
        raise ValueError("encoder.execution must be trainer, actor_rank0 or actor_all")
    if encoder_cfg.get("execution") == "actor_all" and not encoder_cfg.get("offload_after_encode", True):
        raise ValueError("actor_all requires encoder.offload_after_encode=true")
    aux = config.algorithm.get("sdar_aux", {})
    if not isinstance(aux.get("enabled", False), bool):
        raise ValueError("sdar_aux.enabled must be boolean")
    for key, default in (("coef", 0.01), ("gate_beta", 5.0)):
        value = aux.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"sdar_aux.{key} must be finite numeric")
        if value < 0 or (key == "gate_beta" and value == 0):
            raise ValueError(f"Invalid sdar_aux.{key}")
    from verl.trainer.ppo.action_consistency import validate_action_filter_config
    validate_action_filter_config(m1_cfg.get("action_filter", {}))
    if str(m1_cfg.get("aggregation_mode", "probability_mixture")) not in {"probability_mixture", "log_mean"}:
        raise ValueError("aggregation_mode must be probability_mixture or log_mean")
    sign_epsilon = float(m1_cfg.get("diagnostics", {}).get("sign_epsilon", 1e-4))
    if not math.isfinite(sign_epsilon) or sign_epsilon < 0:
        raise ValueError("diagnostics.sign_epsilon must be finite and nonnegative")
    audit_cfg = m1_cfg.get("diagnostics", {}).get("retrospective", {})
    for key, default in (("sample_interval", 10), ("sample_turns", 8), ("sample_tokens", 512),
                         ("context_tokens", 4096), ("max_detail_mb", 1024)):
        value = audit_cfg.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"diagnostics.retrospective.{key} must be a positive integer")
    teacher_budget = m1_cfg.get("teacher_max_response_length")
    if teacher_budget is not None:
        if isinstance(teacher_budget, bool) or not isinstance(teacher_budget, int) or teacher_budget <= 0:
            raise ValueError("teacher_max_response_length must be a positive integer or null")
        rollout = config.actor_rollout_ref.rollout
        if rollout.name != "vllm":
            raise ValueError("Independent Teacher output budgets currently require vllm")
        context_limit = int(rollout.max_model_len or (rollout.prompt_length + rollout.response_length))
        if int(rollout.prompt_length) + teacher_budget > context_limit:
            raise ValueError("Teacher prompt + output exceeds rollout.max_model_len; increase the context limit")
    m2_cfg = config.algorithm.get("mechanism2", {})
    profile_distance = str(m2_cfg.get("profile_distance", "jsd"))
    if profile_distance not in {"jsd", "entropy", "cosine"}:
        raise ValueError(
            "algorithm.mechanism2.profile_distance must be jsd, entropy, or cosine"
        )
    if config.algorithm.get("grpo_normalization_scope", "trajectory") not in {"trajectory", "turn"}:
        raise ValueError("algorithm.grpo_normalization_scope must be 'trajectory' or 'turn'")
    m1_enabled = bool(m1_cfg.get("enabled", False))
    consumer = str(m1_cfg.get("consumer", "standalone_alignment"))
    m2_enabled = bool(m2_cfg.get("enabled", False))
    if not m1_enabled:
        raise ValueError("BeyondTimestamps requires algorithm.thinking_correspondence.enabled=true")
    if consumer not in {"standalone_alignment", "mechanism2"}:
        raise ValueError(f"Unsupported thinking_correspondence consumer: {consumer}")
    mechanism2 = m1_enabled and consumer == "mechanism2"
    if mechanism2 != m2_enabled:
        raise ValueError(
            "algorithm.mechanism2.enabled must be true exactly when "
            "thinking_correspondence.consumer=mechanism2"
        )
    if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
        raise NotImplementedError("Mechanism 1/2 currently require the FSDP data-parallel actor")
    actor_cfg = config.actor_rollout_ref.actor
    if actor_cfg.loss_agg_mode != "token-mean":
        raise ValueError(
            "BeyondTimestamps canonical multiplicity weighting requires "
            "actor.loss_agg_mode=token-mean"
        )
    if str(actor_cfg.get("policy_loss", {}).get("loss_mode", "vanilla")) != "vanilla":
        raise ValueError(
            "BeyondTimestamps canonical multiplicity weighting currently requires "
            "actor.policy_loss.loss_mode=vanilla"
        )
    if mechanism2:
        if str(config.algorithm.adv_estimator).lower() != "grpo":
            raise ValueError("Mechanism 2 requires algorithm.adv_estimator=grpo")
        if config.algorithm.use_kl_in_reward:
            raise ValueError("Mechanism 2 requires pure outcome GRPO; set algorithm.use_kl_in_reward=false")
        if bool(config.algorithm.get("use_pf_ppo", False)):
            raise ValueError("Mechanism 2 does not support PF-PPO reweighting")
        if int(config.env.rollout.n) < 2:
            raise ValueError("Mechanism 2 requires env.rollout.n>=2 sibling trajectories per task")
    if bool(config.actor_rollout_ref.actor.get("use_sdl_loss", False)) or bool(
        config.actor_rollout_ref.actor.get("use_sdar_loss", False)
    ):
        raise ValueError("Disable direct SDL/SDAR actor flags; use algorithm.sdar_aux.enabled")


@hydra.main(config_path="config", config_name="beyond_timestamps_trainer", version_base=None)
def main(config):
    run_beyond_timestamps(config)


def run_beyond_timestamps(config) -> None:
    _validate_beyond_timestamps_config(config)
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = BeyondTimestampsTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class BeyondTimestampsTaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf, open_dict

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        m1_cfg = config.algorithm.thinking_correspondence
        m1_consumer = str(m1_cfg.consumer)
        standalone_alignment = m1_consumer == "standalone_alignment"
        mechanism2 = m1_consumer == "mechanism2"
        alignment_cfg = m1_cfg.alignment
        sdar_aux = config.algorithm.get("sdar_aux", {})
        with open_dict(config):
            actor_cfg = config.actor_rollout_ref.actor
            actor_cfg.use_sdl_loss = False
            actor_cfg.use_sdar_loss = bool(sdar_aux.get("enabled", False))
            actor_cfg.sdar_use_diagonal_teacher = actor_cfg.use_sdar_loss
            actor_cfg.sdar_loss_coef = float(sdar_aux.get("coef", 0.01))
            actor_cfg.sdar_gate_beta = float(sdar_aux.get("gate_beta", 5.0))
            actor_cfg.use_beyond_timestamps_alignment = standalone_alignment
            actor_cfg.beyond_timestamps_alignment_coef = float(alignment_cfg.coef)
            actor_cfg.beyond_timestamps_gate_beta = float(alignment_cfg.beta)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from agent_system.environments import make_envs

        envs, val_envs = make_envs(config)

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == "episode":
            from agent_system.reward_manager import EpisodeRewardManager

            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError

        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        assert config.actor_rollout_ref.rollout.n == 1, (
            "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. "
            "In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"
        )

        from agent_system.multi_turn_rollout import TrajectoryCollector

        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        from verl.trainer.ppo.rlsd_utils import SkillProvider

        skills_dir = str(m1_cfg.skills_dir)
        skill_all = bool(m1_cfg.skill_all)
        skill_provider = SkillProvider(skills_dir=skills_dir, skill_all=skill_all)
        method_label = "BeyondTimestamps" if mechanism2 else "BeyondTimestamps-M1"
        print(f"[{method_label}] Loaded skills from {skills_dir}")
        print(f"[{method_label}] Available skills: {list(skill_provider.skill_contents.keys())}")
        print(f"[{method_label}] Task-to-skill mapping: {skill_provider.task_to_skill}")
        print(f"[{method_label}] direct_alignment_enabled: {standalone_alignment}")
        print(f"[{method_label}] diagonal_sdar_aux: {dict(sdar_aux)}")
        if standalone_alignment:
            print(f"[{method_label}] alignment_coef: {alignment_cfg.coef}")
            print(f"[{method_label}] gate_beta: {alignment_cfg.beta}")

        from verl.trainer.ppo.beyond_timestamps_ray_trainer import BeyondTimestampsRayTrainer

        trainer = BeyondTimestampsRayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
            skill_provider=skill_provider,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
