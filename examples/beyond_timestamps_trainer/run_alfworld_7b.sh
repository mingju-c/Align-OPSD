#!/usr/bin/env bash
set -euo pipefail
# BeyondTimestamps M1 + M2 training on ALFWorld with a 7B policy.
# Defaults use 8 GPUs. Activate the task environment before launching.

ENGINE=vllm
if [[ $# -gt 0 && "$1" != *=* && "$1" != -* ]]; then
    ENGINE=$1
    shift
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/common.sh" "$@"
default_thinking_encoder_path="$REPO_ROOT/models/Qwen3-Embedding-0.6B"
thinking_encoder_path="${THINKING_ENCODER_PATH:-$default_thinking_encoder_path}"
if [[ "$ALIGNOPSD_CONFIG_ONLY" != true ]]; then
for required_encoder_file in config.json tokenizer.json model.safetensors; do
    if [[ ! -f "$thinking_encoder_path/$required_encoder_file" ]]; then
        echo "Missing offline Qwen3 embedding file: $thinking_encoder_path/$required_encoder_file" >&2
        echo "Download Qwen/Qwen3-Embedding-0.6B first or set THINKING_ENCODER_PATH." >&2
        exit 1
    fi
done
fi

num_cpus_per_env_worker=0.1

# Mechanism 1 hyperparameters
skill_all=false

train_data_size=16
val_data_size=128
group_size=8
experiment_name="${BEYOND_TIMESTAMPS_EXPERIMENT_NAME:-Align_7B_alfworld_probmix_turn_th080_ct05}"
data_dir="$ALFWORLD_PREPARED_DATA_DIR"


if [[ "$ALIGNOPSD_CONFIG_ONLY" != true ]]; then
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --local_dir "$data_dir" \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size
fi

python3 -m verl.trainer.main_beyond_timestamps \
    "data.train_files=$data_dir/text/train.parquet" \
    "data.val_files=$data_dir/text/test.parquet" \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    "actor_rollout_ref.model.path=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE:-1.0} \
    actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P:-1.0} \
    actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K:--1} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.rollout.max_model_len=6144 \
    "algorithm.thinking_correspondence.encoder.model_path=$thinking_encoder_path" \
    algorithm.adv_estimator=grpo \
    algorithm.grpo_normalization_scope=${GRPO_NORMALIZATION_SCOPE:-turn} \
    algorithm.mechanism2.credit_temperature=${M2_CREDIT_TEMPERATURE:-0.5} \
    algorithm.mechanism2.density_upper_bound=4.0 \
    algorithm.mechanism2.enabled=true \
    algorithm.mechanism2.max_segment_turns=8 \
    algorithm.mechanism2.min_segment_turns=2 \
    algorithm.mechanism2.mixing_coefficient=0.50 \
    algorithm.mechanism2.profile_distance=jsd \
    algorithm.mechanism2.profile_temperature=0.10 \
    algorithm.mechanism2.segmentation_quantile=0.80 \
    algorithm.mechanism2.segmentation_threshold_max=0.10 \
    algorithm.mechanism2.segmentation_threshold_min=0.01 \
    algorithm.mechanism2.weighting_mode=legacy \
    algorithm.sdar_aux.enabled=false \
    algorithm.thinking_correspondence.action_filter.enabled=false \
    algorithm.thinking_correspondence.aggregation_mode=probability_mixture \
    algorithm.thinking_correspondence.aggregation_temperature=0.10 \
    algorithm.thinking_correspondence.alpha_max=0.80 \
    algorithm.thinking_correspondence.consumer=mechanism2 \
    algorithm.thinking_correspondence.enabled=true \
    algorithm.thinking_correspondence.encoder.attention_implementation=${THINKING_ENCODER_ATTN_IMPLEMENTATION:-sdpa} \
    algorithm.thinking_correspondence.encoder.batch_size=${THINKING_ENCODER_BATCH_SIZE:-16} \
    algorithm.thinking_correspondence.encoder.device=${THINKING_ENCODER_DEVICE:-auto} \
    algorithm.thinking_correspondence.encoder.execution=${THINKING_ENCODER_EXECUTION:-actor_rank0} \
    algorithm.thinking_correspondence.encoder.expected_model_sha256=0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd \
    algorithm.thinking_correspondence.encoder.max_length=4096 \
    algorithm.thinking_correspondence.encoder.offload_after_encode=${THINKING_ENCODER_OFFLOAD_AFTER_ENCODE:-true} \
    algorithm.thinking_correspondence.encoder.padding_side=left \
    algorithm.thinking_correspondence.encoder.pooling=last_token \
    algorithm.thinking_correspondence.encoder.revision=c54f2e6e80b2d7b7de06f51cec4959f6b3e03418 \
    algorithm.thinking_correspondence.encoder.truncation_side=right \
    algorithm.thinking_correspondence.similarity_threshold=0.80 \
    algorithm.thinking_correspondence.skill_all=$skill_all \
    algorithm.thinking_correspondence.skills_dir=skills/alfworld \
    algorithm.thinking_correspondence.source_rollout_cap=1 \
    algorithm.thinking_correspondence.teacher_max_response_length=2048 \
    algorithm.thinking_correspondence.top_k=3 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    "trainer.experiment_name=$experiment_name" \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.n_gpus_per_node=${NUM_GPUS:-8} \
    trainer.nnodes=1 \
    trainer.project_name='beyond_timestamps_alfworld' \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.resume_mode=auto \
    trainer.save_freq=${SAVE_FREQ:--1} \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-150} \
    trainer.val_before_train=True "$@"
