"""Launch real shell entrypoints from a relocated directory with spaces, without GPUs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from hydra import compose, initialize_config_dir
import pytest

from verl.trainer.main_beyond_timestamps import _validate_beyond_timestamps_config

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = sorted((ROOT / 'examples/beyond_timestamps_trainer').glob('run_*.sh')) + sorted(
    (ROOT / 'examples/thinking_correspondence_trainer').glob('run_*.sh')
)


def _launch_relocated(launcher, tmp_path, custom_assets=False):
    checkout = tmp_path / 'checkout with spaces'
    target = checkout / launcher.relative_to(ROOT)
    target.parent.mkdir(parents=True)
    shutil.copy2(launcher, target)
    (checkout / 'scripts').mkdir()
    shutil.copy2(ROOT / 'scripts/common.sh', checkout / 'scripts/common.sh')
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    python = bindir / 'python3'
    python.write_text(f'#!{sys.executable}\nimport json, os, sys\nprint(json.dumps({{"cwd": os.getcwd(), "argv": sys.argv[1:], "alfworld_data": os.environ.get("ALFWORLD_DATA")}}))\n')
    python.chmod(0o755)
    env = os.environ.copy()
    for name in ['DATA_ROOT', 'SEARCH_DATA_DIR', 'ALFWORLD_PREPARED_DATA_DIR', 'WEBSHOP_PREPARED_DATA_DIR', 'TRAIN_DATA', 'VAL_DATA', 'ALFWORLD_DATA']:
        env.pop(name, None)
    env.update(PATH=str(bindir) + os.pathsep + env.get('PATH', ''), MODEL_PATH='./models/policy with spaces',
               THINKING_ENCODER_PATH='./models/encoder with spaces')
    if custom_assets:
        env['ALFWORLD_DATA'] = str(tmp_path / 'custom assets' / 'alfworld')
    result = subprocess.run(['bash', str(target), '--cfg', 'job', 'trainer.total_training_steps=2'],
                            cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    call = json.loads(result.stdout)
    assert Path(call['cwd']) == checkout
    expected_assets = tmp_path / 'custom assets' / 'alfworld' if custom_assets else checkout / 'data' / 'alfworld'
    assert Path(call['alfworld_data']) == expected_assets
    assert call['argv'][:2] == ['-m', 'verl.trainer.main_beyond_timestamps']
    overrides = call['argv'][2:]
    cfg = overrides.index('--cfg')
    del overrides[cfg:cfg + 2]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'verl/trainer/config')):
        config = compose(config_name='beyond_timestamps_trainer', overrides=overrides)
    _validate_beyond_timestamps_config(config)
    assert config.actor_rollout_ref.model.path == './models/policy with spaces'
    assert config.algorithm.thinking_correspondence.encoder.model_path == './models/encoder with spaces'
    assert Path(config.data.train_files).is_relative_to(checkout / 'data')
    assert config.trainer.total_training_steps == 2
    return config


@pytest.mark.parametrize('launcher', LAUNCHERS, ids=lambda p: f'{p.parent.name}/{p.name}')
@pytest.mark.parametrize('custom_assets', [False, True], ids=['default-assets', 'custom-assets'])
def test_launchers_survive_relocation_and_compose(launcher, tmp_path, custom_assets):
    _launch_relocated(launcher, tmp_path, custom_assets)


@pytest.mark.parametrize('task', ['alfworld', 'webshop', 'search'])
def test_m1_model_sizes_have_separate_checkpoint_directories(task, tmp_path):
    configs = [
        _launch_relocated(
            ROOT / 'examples/thinking_correspondence_trainer' / f'run_{task}_{size}.sh',
            tmp_path / size,
        )
        for size in ['3b', '7b']
    ]
    assert configs[0].trainer.default_local_dir != configs[1].trainer.default_local_dir
    assert all(config.trainer.resume_mode == 'disable' for config in configs)
