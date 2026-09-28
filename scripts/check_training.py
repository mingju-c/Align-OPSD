#!/usr/bin/env python3
"""Validate the actual launch configuration and, by default, local runtime assets."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['alfworld', 'webshop', 'search'], required=True)
    parser.add_argument('--model-size', choices=['3b', '7b'], default='3b')
    parser.add_argument('--method', choices=['full', 'm1'], default='full')
    parser.add_argument('--config-only', action='store_true', help='Validate imports and Hydra overrides without requiring GPUs or assets.')
    args, overrides = parser.parse_known_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    env = os.environ.copy()
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    folder = 'beyond_timestamps_trainer' if args.method == 'full' else 'thinking_correspondence_trainer'
    launcher = ROOT / 'examples' / folder / f'run_{args.task}_{args.model_size}.sh'
    result = subprocess.run(['bash', str(launcher), '--cfg', 'job', '--resolve', *overrides], env=env, text=True, capture_output=True)
    if result.returncode:
        print(result.stderr or result.stdout, file=sys.stderr)
        return result.returncode
    from omegaconf import OmegaConf
    from verl.trainer.main_beyond_timestamps import _validate_beyond_timestamps_config
    config = OmegaConf.create(result.stdout)
    _validate_beyond_timestamps_config(config)
    import verl
    if Path(verl.__file__).resolve().parent != ROOT / 'verl':
        raise RuntimeError('The active interpreter imported a different checkout of verl.')
    print(f'PASS: {args.task}/{args.model_size}/{args.method} launch configuration and method validation')
    if args.config_only:
        print('GPU execution, model loading, benchmark assets and services were not checked.')
        return 0

    errors = []
    def check(label, fn):
        try:
            fn()
            print(f'PASS: {label}')
        except Exception as exc:
            errors.append(f'{label}: {exc}')
            print(f'FAIL: {label}: {exc}')

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    for module in ['torch', 'vllm', 'flash_attn', 'ray', 'datasets', 'agent_system.environments']:
        check(f'import {module}', lambda name=module: importlib.import_module(name))
    import torch
    check('CUDA devices', lambda: require(torch.cuda.is_available() and torch.cuda.device_count() >= config.trainer.n_gpus_per_node,
                                         f'need {config.trainer.n_gpus_per_node} visible GPUs; found {torch.cuda.device_count()}'))
    encoder = Path(config.algorithm.thinking_correspondence.encoder.model_path).expanduser()
    for name in ['config.json', 'tokenizer.json', 'model.safetensors']:
        check(f'encoder {name}', lambda name=name: require((encoder / name).is_file(), f'missing {encoder / name}'))
    policy = config.actor_rollout_ref.model.path
    def check_policy():
        from transformers import AutoConfig, AutoTokenizer
        AutoConfig.from_pretrained(policy, local_files_only=True)
        AutoTokenizer.from_pretrained(policy, local_files_only=True)
    check('cached policy config and tokenizer', check_policy)
    skills = Path(config.algorithm.thinking_correspondence.skills_dir)
    check('skill mapping', lambda: require((skills / 'skill_mapping.json').is_file(), f'missing {skills / "skill_mapping.json"}'))
    data_root = Path(os.environ.get('DATA_ROOT', ROOT / 'data')).expanduser()
    if args.task == 'search':
        check('Search environment imports', lambda: importlib.import_module('agent_system.environments.env_package.search.envs'))
        for split in ['train_files', 'val_files']:
            check(f'Search {split}', lambda split=split: require(Path(config.data[split]).expanduser().is_file(), f'missing {config.data[split]}'))
        def check_service():
            request = urllib.request.Request(config.env.search.search_url, data=json.dumps({'query': 'test', 'topk': 1, 'return_scores': True}).encode(), headers={'Content-Type': 'application/json'})
            # Loopback retrieval should never go through a machine's HTTP proxy.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=30) as response:
                payload = json.load(response)
            require(bool(payload.get('result')), 'retrieval service returned no results')
        check('retrieval service', check_service)
    elif args.task == 'alfworld':
        check('ALFWorld imports', lambda: importlib.import_module('agent_system.environments.env_package.alfworld.envs'))
        assets = Path(os.environ.get('ALFWORLD_DATA', data_root / 'alfworld')).expanduser()
        for item in ['json_2.1.1/train', 'json_2.1.1/valid_seen', 'json_2.1.1/valid_unseen', 'logic/alfred.pddl', 'logic/alfred.twl2']:
            check(f'ALFWorld {item}', lambda item=item: require((assets / item).exists(), f'missing {assets / item}'))
    else:
        check('WebShop imports', lambda: importlib.import_module('agent_system.environments.env_package.webshop.envs'))
        assets = Path(os.environ.get('WEBSHOP_ASSET_ROOT', ROOT / 'agent_system/environments/env_package/webshop/webshop')).expanduser()
        for item in ['data/items_shuffle_1000.json', 'data/items_ins_v2_1000.json', 'data/items_human_ins.json', 'search_engine/indexes/segments_1']:
            check(f'WebShop {item}', lambda item=item: require((assets / item).is_file(), f'missing {assets / item}'))
        check('Java', lambda: subprocess.run(['java', '-version'], check=True, capture_output=True))
        check('Pyserini', lambda: importlib.import_module('pyserini.search'))
    if errors:
        print(f'{len(errors)} preflight checks failed. Prepare the dependencies/assets in README.md before training.')
        return 1
    print('Preflight passed. Weight loading and actual GPU training still require a training run.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
