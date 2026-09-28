#!/usr/bin/env bash
set -euo pipefail
# Thinking-correspondence Mechanism 1 standalone training.


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

data_dir="$WEBSHOP_PREPARED_DATA_DIR"

num_cpus_per_env_worker=0.1

# Mechanism 1 hyperparameters
skill_all=false

train_data_size=16
val_data_size=128
group_size=8
experiment_name="${M1_METHOD_NAME:-ThinkingCorrespondence}_webshop7b_gamma0.70_alpha0.20_skill${skill_all}"

if [[ "$ALIGNOPSD_CONFIG_ONLY" != true ]]; then
python3 -m examples.data_preprocess.prepare \
    --local_dir "$data_dir" \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size
fi

python3 -m verl.trainer.main_beyond_timestamps \
    algorithm.adv_estimator=grpo \
    "data.train_files=$data_dir/text/train.parquet" \
    "data.val_files=$data_dir/text/test.parquet" \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    "actor_rollout_ref.model.path=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.max_model_len=6144 \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.thinking_correspondence.enabled=true \
    algorithm.thinking_correspondence.consumer=standalone_alignment \
    algorithm.mechanism2.enabled=false \
    algorithm.thinking_correspondence.similarity_threshold=0.70 \
    algorithm.thinking_correspondence.source_rollout_cap=1 \
    algorithm.thinking_correspondence.top_k=3 \
    algorithm.thinking_correspondence.aggregation_temperature=0.10 \
    algorithm.thinking_correspondence.alpha_max=0.20 \
    "algorithm.thinking_correspondence.encoder.model_path=$thinking_encoder_path" \
    algorithm.thinking_correspondence.encoder.revision=c54f2e6e80b2d7b7de06f51cec4959f6b3e03418 \
    algorithm.thinking_correspondence.encoder.expected_model_sha256=0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd \
    algorithm.thinking_correspondence.encoder.pooling=last_token \
    algorithm.thinking_correspondence.encoder.padding_side=left \
    algorithm.thinking_correspondence.encoder.truncation_side=right \
    algorithm.thinking_correspondence.encoder.max_length=4096 \
    algorithm.thinking_correspondence.encoder.attention_implementation=${THINKING_ENCODER_ATTN_IMPLEMENTATION:-sdpa} \
    algorithm.thinking_correspondence.encoder.device=${THINKING_ENCODER_DEVICE:-auto} \
    algorithm.thinking_correspondence.encoder.batch_size=${THINKING_ENCODER_BATCH_SIZE:-4} \
    algorithm.thinking_correspondence.alignment.beta=5.0 \
    algorithm.thinking_correspondence.alignment.coef=0.01 \
    algorithm.thinking_correspondence.skills_dir=skills/webshop \
    algorithm.thinking_correspondence.skill_all=$skill_all \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_webshopv1' \
    "trainer.experiment_name=$experiment_name" \
    trainer.n_gpus_per_node=${NUM_GPUS:-2} \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.resume_mode=disable \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True "$@"
