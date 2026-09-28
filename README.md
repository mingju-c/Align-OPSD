# AlignOPSD

Code for correspondence-driven credit assignment in agentic reinforcement learning. The implementation keeps the internal `BeyondTimestamps` names for configuration/checkpoint compatibility.

This release focuses on **ALFWorld, WebShop and Search**, with 3B/7B policies and M1-only ablations. Model weights, benchmark data, checkpoints, logs, unrelated baseline launchers, and upstream integration tests are excluded.

| Directory | Contents |
| --- | --- |
| `examples/` | Main/M1 launchers, data preparation and retrieval |
| `verl/` | Method implementation and shared distributed training runtime |
| `agent_system/` | The three benchmark integrations |
| `skills/` | Skill mappings and only the texts they reference |
| `gigpo/` | Shared advantage helper imported by the PPO runtime |
| `scripts/` | Path setup, preflight/release checks and checkpoint merging |
| `tests/` | Method and launcher regression tests |

Shared PPO, RLSD and SDAR code remains where imported by the method. Unused launch recipes, visual card-game assets, THOR scene generators, generated skill memories and old documentation are omitted.

## Installation

Run installation commands from the checkout root. Install this checkout in editable mode. Training requires Linux and NVIDIA GPUs.

For **ALFWorld/Search**:

```bash
conda create -n alignopsd python=3.12.14 pip ninja packaging -y
conda activate alignopsd
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```

This profile pins PyTorch 2.8.0, vLLM 0.11.0, Transformers 4.57.3, Ray 2.50.0 and TensorDict 0.10.0. Shared dependencies live in `setup.py`.

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

WebShop setup installs Java 11, Pyserini, spaCy models and the 1,000-product assets/index. Activate this environment so its Java executable is on `PATH`; an existing asset installation can be selected with `WEBSHOP_ASSET_ROOT`.

## Assets

Download the frozen encoder before training; the method checks its revision and weight checksum:

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

Each launcher defines its model size, method parameters and training schedule directly. See [configuration defaults and the ALFWorld step-145 schedule](examples/beyond_timestamps_trainer/CONFIGS.md).

M1-only experiment names include the model size, and these launchers start fresh by default. To resume a saved run, select its checkpoint directory and pass `trainer.resume_mode=auto` explicitly.

The optional first argument of a standard launcher selects the engine (`vllm`). Remaining arguments are Hydra overrides and take precedence. For example:

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

| Variable | Default |
| --- | --- |
| `DATA_ROOT` | `<checkout>/data` |
| `MODEL_PATH` | Launcher-specific Qwen2.5-3B/7B-Instruct model ID |
| `THINKING_ENCODER_PATH` | `<checkout>/models/Qwen3-Embedding-0.6B` |
| `SEARCH_DATA_DIR` | `<DATA_ROOT>/searchR1_processed_direct` |
| `SEARCH_URL` | `http://127.0.0.1:8000/retrieve` |
| `ALFWORLD_DATA` | `<DATA_ROOT>/alfworld` |
| `WEBSHOP_ASSET_ROOT` | Bundled WebShop directory |
| `WANDB_MODE` | `offline` |

Launchers resolve relative paths from the checkout root and also work when invoked elsewhere. `.env.example` lists optional settings; `.env` is not loaded automatically. Use interactive W&B login for online logging.

## Validation

```bash
python scripts/check_training.py --task search --model-size 3b --config-only
# --task also accepts alfworld/webshop; --model-size accepts 3b/7b.
python scripts/check_release.py
python -m pip install pytest
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests
```

Omit `--config-only` to check GPU visibility, installed dependencies, encoder files, cached policy configuration/tokenizer, benchmark assets and the retrieval service. Preflight does not load full model weights or run an optimizer step.

The regression suite covers the method and launcher configuration. Prepare the assets and run the short training command above on the target GPU machine to check model loading and optimizer execution.

## License

Built on SDAR, verl-agent/GiGPO, veRL, ALFWorld, WebShop and Search-R1. Original copyright headers and bundled licenses are retained; see `LICENSE` and `Notice.txt`.
