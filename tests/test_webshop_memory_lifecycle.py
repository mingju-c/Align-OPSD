from types import SimpleNamespace

import numpy as np
import pytest

from agent_system.environments.env_package.webshop import envs as webshop
from verl.utils import memory_monitor


class RemoteMethod:
    def __init__(self, fn):
        self.remote = fn


class FakeWorker:
    def __init__(self, seed):
        self.seed = seed
        self.ready = RemoteMethod(lambda: True)
        self.get_goals = RemoteMethod(lambda: list(range(700)))
        self.reset = RemoteMethod(lambda idx: (idx, {}))


@pytest.fixture
def fake_ray(monkeypatch):
    created, killed = [], []

    def create(seed, kwargs, heap):
        worker = FakeWorker(seed)
        created.append(worker)
        return worker

    monkeypatch.setattr(webshop.ray, 'is_initialized', lambda: True)
    monkeypatch.setattr(webshop.ray, 'remote', lambda **resources: lambda cls: SimpleNamespace(remote=create))
    monkeypatch.setattr(webshop.ray, 'get', lambda refs: refs)
    monkeypatch.setattr(webshop.ray, 'kill', lambda actor, no_restart=True: killed.append(actor))
    return created, killed


def test_lazy_validation_restarts_without_rewinding_task_rng(fake_ray):
    created, killed = fake_ray
    env = webshop.WebshopMultiProcessEnv(1000, 4, 1, {}, is_train=False, lazy_start=True)
    expected = np.random.RandomState(1000)
    assert not created
    first, _ = env.reset()
    assert first == expected.choice(range(500), size=4, replace=False).tolist()
    assert len(created) == 4
    # Multiple validation batches reuse the pool until the validation pass ends.
    env.reset()
    expected.choice(range(500), size=4, replace=False)
    assert len(created) == 4
    env.release_workers()
    assert len(killed) == 4 and not env._workers
    third, _ = env.reset()
    assert third == expected.choice(range(500), size=4, replace=False).tolist()
    assert len(created) == 8
    assert [w.seed for w in created[:4]] == [w.seed for w in created[4:]]
    env.close()
    env.close()
    assert len(killed) == 8
    with pytest.raises(RuntimeError, match='closed'):
        env.reset()


def test_training_pool_remains_eager_and_grouped(fake_ray):
    created, _ = fake_ray
    env = webshop.WebshopMultiProcessEnv(0, 2, 3, {}, is_train=True)
    assert len(created) == 6
    tasks, _ = env.reset()
    assert tasks[0] == tasks[1] == tasks[2]
    assert tasks[3] == tasks[4] == tasks[5]
    assert all(500 <= task < 700 for task in tasks)
    env.close()


def test_partial_startup_failure_kills_created_actors(fake_ray, monkeypatch):
    created, killed = fake_ray
    env = webshop.WebshopMultiProcessEnv(0, 2, 1, {}, is_train=False, lazy_start=True)
    def fail(refs):
        raise RuntimeError('worker startup failed')
    monkeypatch.setattr(webshop.ray, 'get', fail)
    with pytest.raises(RuntimeError, match='worker startup failed'):
        env.reset()
    assert killed == created and not env._workers
    env.close()


def test_memory_monitor_records_failure_without_hiding_it(monkeypatch):
    values = iter([{'ray_used_gib': 10.0}, {'ray_used_gib': 12.0}])
    monkeypatch.setattr(memory_monitor, 'memory_snapshot', lambda: next(values))
    metrics = {}
    with pytest.raises(ValueError, match='training failure'):
        with memory_monitor.monitor_memory('save', metrics, 15, interval=60):
            raise ValueError('training failure')
    assert metrics['memory/save/ray_used_gib_before'] == 10
    assert metrics['memory/save/ray_used_gib_after'] == 12
    assert metrics['memory/save/ray_used_gib_sampled_peak'] == 12


@pytest.mark.parametrize('fail', [False, True])
def test_validation_releases_resources_on_success_and_failure(fail):
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    released = []
    def validate():
        if fail:
            raise RuntimeError('validation failed')
        return {'score': 1.0}
    trainer = SimpleNamespace(
        _validate_impl=validate,
        val_envs=SimpleNamespace(release_validation_resources=lambda: released.append(True)),
    )
    if fail:
        with pytest.raises(RuntimeError, match='validation failed'):
            RayPPOTrainer._validate(trainer)
    else:
        assert RayPPOTrainer._validate(trainer) == {'score': 1.0}
    assert released == [True]
