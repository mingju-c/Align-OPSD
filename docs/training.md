# Training Guide

Run commands from the repository root. See the [README](../README.md) for an overview.

## Installation

Requires Linux and NVIDIA GPUs. Run installation commands from the repository root.

For **ALFWorld/Search**:

```bash
conda create -n alignopsd python=3.12.14 pip ninja packaging -y
conda activate alignopsd
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```

For **WebShop**, use a separate environment:

```bash
conda create -n alignopsd-webshop python=3.10 pip ninja packaging -y
conda activate alignopsd-webshop
(
  cd agent_system/environments/env_package/webshop/webshop
  bash setup.sh -d small
)
python -m pip install -r requirements-webshop.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```

WebShop setup installs its dependencies and 1,000-product assets/index. Keep this environment active so Java is on `PATH`. Set `WEBSHOP_ASSET_ROOT` to reuse an existing installation.

## Assets

Download the encoder at the required revision and the policy model:

```bash
hf download Qwen/Qwen3-Embedding-0.6B \
  --revision c54f2e6e80b2d7b7de06f51cec4959f6b3e03418 \
  --local-dir ./models/Qwen3-Embedding-0.6B
hf download Qwen/Qwen2.5-3B-Instruct --local-dir ./models/Qwen2.5-3B-Instruct
export MODEL_PATH=./models/Qwen2.5-3B-Instruct
```

For 7B, use `Qwen/Qwen2.5-7B-Instruct` instead. Without `MODEL_PATH`, each launcher uses its corresponding Hugging Face model ID.

**ALFWorld**:

```bash
export ALFWORLD_DATA="$PWD/data/alfworld"
alfworld-download -f
```

ALFWorld/WebShop placeholder Parquet files are generated automatically in separate task directories.

**Search** (Search-R1 NQ/HotpotQA):

```bash
python -m examples.data_preprocess.preprocess_search_r1_dataset \
  --local_dir ./data/searchR1_processed_direct
python examples/search/searchr1_download.py --local_dir ./data/searchR1
cat ./data/searchR1/part_aa ./data/searchR1/part_ab > ./data/searchR1/e5_Flat.index
gzip -dk ./data/searchR1/wiki-18.jsonl.gz
```

Start the retrieval service in an environment containing PyTorch, Transformers, Datasets, FastAPI, Uvicorn and a CUDA-compatible FAISS build:

```bash
bash examples/search/retriever/retrieval_launch.sh
```

The training data directory must contain both `train.parquet` and `test.parquet`. `SEARCH_INDEX_DIR`, `RETRIEVER_MODEL_PATH` and `SEARCH_PORT` configure the retrieval server. `FAISS_GPU=false` places the index on CPU; the dense encoder still requires CUDA.

## Training

```bash
bash examples/beyond_timestamps_trainer/run_alfworld_3b.sh
bash examples/beyond_timestamps_trainer/run_webshop_3b.sh
bash examples/beyond_timestamps_trainer/run_search_3b.sh
```

Replace `3b` with `7b` for larger policies. M1-only launchers are in `examples/thinking_correspondence_trainer/`.

See [configuration defaults and training schedules](../examples/beyond_timestamps_trainer/CONFIGS.md).

M1-only launchers start fresh by default. To resume, select a checkpoint directory and pass `trainer.resume_mode=auto`.

Standard launchers accept an optional engine (`vllm`), followed by Hydra overrides. For a short training check:

```bash
NUM_GPUS=4 bash examples/beyond_timestamps_trainer/run_search_3b.sh vllm \
  data.train_batch_size=4 env.rollout.n=2 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  trainer.total_training_steps=2 trainer.test_freq=-1 \
  trainer.save_freq=-1 trainer.val_before_train=false trainer.resume_mode=disable \
  trainer.logger='[console]' trainer.experiment_name=smoke
```

Batch sizes, tensor parallelism and memory settings must fit the hardware. Search 3B defaults to four GPUs; several other profiles use eight. WebShop 3B targets 80 GB GPUs.

See [`.env.example`](../.env.example) for optional paths and settings. Relative paths resolve from the repository root; `.env` is not loaded automatically. W&B defaults to offline mode; use interactive login for online logging.

## Validation

```bash
python scripts/check_training.py --task search --model-size 3b --config-only
# --task also accepts alfworld/webshop; --model-size accepts 3b/7b.
python scripts/check_release.py
python -m pip install pytest
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests
```

Omit `--config-only` to also check GPUs, dependencies, model/benchmark assets and the retrieval service. Preflight and regression tests do not verify full training; run the short training command above on the target GPU machine to check model loading and optimizer execution.

