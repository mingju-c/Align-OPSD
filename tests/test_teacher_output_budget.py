"""CPU contract tests for the real rollout methods with a stub inference engine."""
import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length

ROOT = Path(__file__).resolve().parents[1]


def rollout_methods():
    # Avoid importing/initializing vLLM CUDA libraries in a CPU contract test.
    tree = ast.parse((ROOT / 'verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py').read_text())
    methods = {}
    scope = dict(contextmanager=contextmanager, torch=torch, np=np, DataProto=DataProto,
                 TensorDict=TensorDict, get_response_mask=get_response_mask,
                 pad_2d_list_to_length=pad_2d_list_to_length, vllm_version='0.8.5')
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in {'update_sampling_params', 'generate_sequences'}:
            node.decorator_list = [ast.Name(id='contextmanager', ctx=ast.Load())] if node.name == 'update_sampling_params' else []
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<rollout methods>', 'exec'), scope)
            methods[node.name] = scope[node.name]
    return type('RolloutHarness', (), methods)


@pytest.mark.parametrize('do_sample', [True, False])
def test_teacher_budget_does_not_leak_into_next_student_call(do_sample):
    worker = rollout_methods()()
    worker.config = NS(free_cache_engine=False, response_length=512, prompt_length=2048, max_model_len=4096)
    worker.sampling_params = NS(max_tokens=512, n=1, temperature=1., top_p=1., top_k=-1)
    worker.pad_token_id = 0
    worker.lora_kwargs = {}
    seen = []
    def generate(**kwargs):
        budget = kwargs['sampling_params'].max_tokens
        seen.append(budget)
        ids = [5] * min(budget, 700) + ([2] if budget > 700 else [])
        sample = NS(token_ids=ids, logprobs=[{token: NS(logprob=-.1)} for token in ids])
        return [NS(outputs=[sample])]
    worker.inference_engine = NS(generate=generate)
    def prompts(budget=None):
        meta = dict(eos_token_id=2, do_sample=do_sample)
        if budget is not None:
            meta['generation_max_tokens'] = budget
        return DataProto.from_dict(tensors={
            'input_ids': torch.tensor([[3, 4]]), 'attention_mask': torch.ones(1, 2, dtype=torch.long),
            'position_ids': torch.tensor([[0, 1]])},
            non_tensors={'raw_prompt_ids': np.array([[3, 4]], dtype=object)}, meta_info=meta)
    teacher = worker.generate_sequences(prompts(2048))
    assert teacher.batch['responses'].shape == (1, 2048)
    assert teacher.batch['responses'][0, 699] == 5
    assert teacher.batch['input_ids'].shape == (1, 2050)
    assert teacher.batch['rollout_log_probs'].shape == (1, 2048)
    assert teacher.batch['attention_mask'].shape == teacher.batch['position_ids'].shape == (1, 2050)
    student = worker.generate_sequences(prompts())
    assert student.batch['responses'].shape == (1, 512)
    assert seen == [2048, 512]
    assert worker.sampling_params.max_tokens == 512
    with pytest.raises(ValueError, match='exceeds'):
        worker.generate_sequences(prompts(4096))
    assert seen == [2048, 512]
    def fail(**kwargs):
        raise RuntimeError('engine failed')
    worker.inference_engine.generate = fail
    with pytest.raises(RuntimeError, match='engine failed'):
        worker.generate_sequences(prompts(2048))
    assert worker.sampling_params.max_tokens == 512


def test_b_config_keeps_student_and_mechanisms_unchanged():
    from hydra import compose, initialize_config_dir
    from verl.trainer.main_beyond_timestamps import _validate_beyond_timestamps_config
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'verl/trainer/config')):
        cfg = compose(config_name='beyond_timestamps_trainer', overrides=[
            'data.max_prompt_length=2048', 'data.max_response_length=512',
            'algorithm.thinking_correspondence.teacher_max_response_length=2048',
            'actor_rollout_ref.rollout.max_model_len=4096', 'trainer.resume_mode=disable'])
    _validate_beyond_timestamps_config(cfg)
    assert cfg.actor_rollout_ref.rollout.response_length == 512
    assert cfg.algorithm.thinking_correspondence.similarity_threshold == .70
    assert cfg.algorithm.mechanism2.credit_temperature == 1.
    cfg.actor_rollout_ref.rollout.max_model_len = 2560
    with pytest.raises(ValueError, match='exceeds'):
        _validate_beyond_timestamps_config(cfg)
    cfg.algorithm.thinking_correspondence.teacher_max_response_length = 0
    with pytest.raises(ValueError, match='positive integer'):
        _validate_beyond_timestamps_config(cfg)
