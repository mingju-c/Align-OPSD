# Configuration defaults

These tables describe the launchers in this directory with no environment or command-line overrides. Run commands from the extracted `AlignOPSD` directory after following the README installation and asset instructions.

## Main method

All six profiles use Qwen2.5-Instruct of the indicated size, enable M1 and M2, use eight sibling rollouts per task, and set a 150-step training limit.

| Profile | GPUs | GRPO normalization | M1 threshold | M1 top-k | M2 credit temperature | SDAR auxiliary | Action filter |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ALFWorld 3b | 8 | turn | 0.65 | 2 | 0.5 | true | true |
| ALFWorld 7b | 8 | turn | 0.8 | 3 | 0.5 | false | false |
| Search 3b | 4 | trajectory | 0.7 | 3 | 1.0 | false | false |
| Search 7b | 8 | turn | 0.7 | 3 | 0.5 | false | false |
| WebShop 3b | 4 | turn | 0.8 | 3 | 0.2 | false | true |
| WebShop 7b | 8 | turn | 0.8 | 2 | 0.5 | false | false |

Search 3B uses `log_mean` aggregation; the other profiles use `probability_mixture`. The action-filter threshold is 0.80 when enabled. ALFWorld 3B uses SDAR coefficient 0.01 and gate beta 5.0.

| Profile | Train batch | Validation batch | PPO micro-batch per GPU | Evaluate every | Save every | Validate before training |
| --- | --- | --- | --- | --- | --- | --- |
| ALFWorld 3b | 16 | 128 | 16 | 5 | 5 | true |
| ALFWorld 7b | 16 | 128 | 32 | 5 | -1 | true |
| Search 3b | 128 | 512 | 4 | 5 | 5 | false |
| Search 7b | 128 | 512 | 16 | 150 | -1 | false |
| WebShop 3b | 16 | 128 | 8 | 10 | 50 | true |
| WebShop 7b | 16 | 128 | 8 | 5 | -1 | true |

Evaluation and saving intervals are measured in training steps; `-1` disables periodic saving. GPU counts and batch sizes should be adjusted together for the available hardware.

## ALFWorld 3B step-145 schedule

The default ALFWorld 3B launcher evaluates and saves every five steps. To use the final-only evaluation and saving schedule for the step-145 run:

```bash
bash examples/beyond_timestamps_trainer/run_alfworld_3b.sh \
  trainer.total_training_steps=145 \
  trainer.total_epochs=145 \
  trainer.test_freq=145 \
  trainer.save_freq=145 \
  trainer.val_before_train=false \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.experiment_name=AlignOPSD_alfworld3b_final145
```

The launcher supplies M1 threshold 0.65, top-k 2, PPO micro-batch 16, turn-level GRPO normalization, M2 credit temperature 0.5, and SDAR coefficient 0.01. Evaluation samples responses at temperature 0.4, so an identical score is not guaranteed on every rerun.

## M1-only ablations

The six launchers in `examples/thinking_correspondence_trainer/` disable M2 and use the standalone M1 alignment objective. They use separate experiment names for 3B and 7B, honor `ALFWORLD_DATA`, and start with `trainer.resume_mode=disable`.

To resume an existing checkpoint, set `trainer.default_local_dir` to the appropriate directory and explicitly pass `trainer.resume_mode=auto`. Command-line Hydra overrides take precedence over launcher defaults.
