"""Exercise unpacked WebShop setup without installations, network, or GPUs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEBSHOP = ROOT / 'agent_system/environments/env_package/webshop/webshop'


@pytest.fixture
def setup_environment(tmp_path):
    checkout = tmp_path / 'unpacked webshop'
    for relative in ['setup.sh', 'search_engine/run_indexing.sh']:
        script = checkout / relative
        script.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WEBSHOP / relative, script)
        script.chmod(0o644)
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    log = tmp_path / 'commands.jsonl'
    stub = (
        f'#!{sys.executable}\n'
        'import json, os, pathlib, sys\n'
        'name = pathlib.Path(sys.argv[0]).name\n'
        'args = sys.argv[1:]\n'
        'with open(os.environ["SETUP_TEST_LOG"], "a") as stream:\n'
        '    stream.write(json.dumps({"name": name, "args": args, "cwd": os.getcwd()}) + "\\n")\n'
        'stage = os.environ.get("SETUP_TEST_FAIL", "")\n'
        'if stage == "dependencies" and name == "pip": sys.exit(17)\n'
        'if stage == "index" and args[:2] == ["-m", "pyserini.index.lucene"]: sys.exit(23)\n'
    )
    for name in ['pip', 'conda', 'gdown', 'python']:
        executable = bindir / name
        executable.write_text(stub)
        executable.chmod(0o755)
    env = {
        **os.environ,
        'PATH': str(bindir) + os.pathsep + os.environ.get('PATH', ''),
        'SETUP_TEST_LOG': str(log),
    }
    env.pop('SETUP_TEST_FAIL', None)
    return checkout, env, log


@pytest.mark.parametrize('failure,expected_code', [('', 0), ('dependencies', 17), ('index', 23)])
def test_setup_without_executable_bits_and_propagates_failures(setup_environment, tmp_path, failure, expected_code):
    checkout, env, log = setup_environment
    env['SETUP_TEST_FAIL'] = failure
    result = subprocess.run(['bash', str(checkout / 'setup.sh'), '-d', 'small'],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == expected_code, result.stderr
    assert 'Permission denied' not in result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    indexing = [call for call in calls if call['args'][:2] == ['-m', 'pyserini.index.lucene']]
    if failure == 'dependencies':
        assert len(calls) == 1
    elif failure == 'index':
        assert len(indexing) == 1
        assert calls[-1] == indexing[0]
    else:
        assert len(indexing) == 4
        assert all(Path(call['cwd']) == checkout / 'search_engine' for call in indexing)


def test_setup_rejects_invalid_dataset_before_installing(setup_environment, tmp_path):
    checkout, env, log = setup_environment
    result = subprocess.run(['bash', str(checkout / 'setup.sh'), '-d', 'invalid'],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert not log.exists()
