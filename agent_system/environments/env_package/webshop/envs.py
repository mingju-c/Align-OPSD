# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ray
import gym
import numpy as np

# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *WebAgentTextEnv* instance.
    """
    
    def __init__(self, seed, env_kwargs, jvm_max_heap_mb=512):
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webshop'))
        sys.path.append(project_root)
        import jnius_config
        if not jnius_config.vm_running:
            if jvm_max_heap_mb < 64:
                raise ValueError("WebShop JVM heap must be at least 64 MiB")
            jnius_config.add_options("-Xms64m", f"-Xmx{int(jvm_max_heap_mb)}m")
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)
        
        env_kwargs = dict(env_kwargs)
        env_kwargs['seed'] = seed
        self.env = gym.make('WebAgentTextEnv-v0', **env_kwargs)
    
    def ready(self):
        """Acknowledge that environment and search engine initialization finished."""
        return True

    def step(self, action):
        """Execute a step in the environment"""
        obs, reward, done, info = self.env.step(action)
        info = dict(info or {})  # make a *copy* so we can mutate safely
        info['available_actions'] = self.env.get_available_actions()
        info['task_score'] = reward

        # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
        if done and reward == 1.0:
            info['won'] = True
            reward = 10.0
        else:
            info['won'] = False
            reward = 0

        return obs, reward, done, info
    
    def reset(self, idx):
        """Reset the environment with given session index"""
        obs, info = self.env.reset(session=idx)
        info = dict(info or {})
        info['available_actions'] = self.env.get_available_actions()
        info['won'] = False
        return obs, info
    
    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render)
        return rendered
    
    def get_available_actions(self):
        """Get available actions"""
        return self.env.get_available_actions()
    
    def get_goals(self):
        """Get environment goals"""
        return self.env.server.goals

    def release_rollout_memory(self):
        """Collect Python and Lucene JVM garbage without resetting the episode."""
        import gc
        import time
        import psutil
        from jnius import autoclass

        process = psutil.Process()
        runtime = autoclass("java.lang.Runtime").getRuntime()
        java_before = runtime.totalMemory() - runtime.freeMemory()
        before = process.memory_info().rss
        started = time.monotonic()
        # Release JNI wrappers before collecting the Java-owned search heap.
        # System.gc is a request; report actual RSS instead of assuming all
        # collectors immediately return unused pages to the operating system.
        gc.collect()
        autoclass("java.lang.System").gc()
        return {
            "java_used_before": java_before,
            "java_used_after": runtime.totalMemory() - runtime.freeMemory(),
            "java_committed_after": runtime.totalMemory(),
            "rss_before": before,
            "rss_after": process.memory_info().rss,
            "seconds": time.monotonic() - started,
        }
    
    def close(self):
        """Close the environment"""
        self.env.close()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
        lazy_start: bool = False,
        jvm_max_heap_mb: int = 512,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train: assert group_n == 1

        self._rng = np.random.RandomState(seed)

        self._env_kwargs = env_kwargs if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}

        self._seed = seed
        self._jvm_max_heap_mb = jvm_max_heap_mb
        self._resources_per_worker = resources_per_worker
        self._workers = []
        self._closed = False
        self.goal_idxs = range(500) if not is_train else None
        if not lazy_start:
            self._start_workers()

    def _start_workers(self):
        if self._closed:
            raise RuntimeError("Cannot restart a closed WebShop environment")
        if self._workers:
            return
        env_worker = ray.remote(**self._resources_per_worker)(WebshopWorker)
        try:
            for i in range(self.num_processes):
                self._workers.append(env_worker.remote(
                    self._seed + (i // self.group_n), dict(self._env_kwargs), self._jvm_max_heap_mb
                ))
            # Explicit readiness replaces the previous arbitrary startup sleep.
            ray.get([worker.ready.remote() for worker in self._workers])
            if self.is_train:
                goals = ray.get(self._workers[0].get_goals.remote())
                self.goal_idxs = range(500, len(goals))
        except BaseException:
            self.release_workers()
            raise

    def release_workers(self):
        """Destroy actor processes and their JVMs; retain task-sampling RNG."""
        workers, self._workers = self._workers, []
        for worker in workers:
            ray.kill(worker, no_restart=True)


    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self._workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        self._start_workers()
        idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        idx = np.repeat(idx, self.group_n).tolist()

        # Send reset commands to all workers
        futures = []
        for worker, i in zip(self._workers, idx):
            future = worker.reset.remote(i)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            future = self._workers[env_idx].render.remote(mode)
            return ray.get(future)

        futures = []
        for worker in self._workers:
            future = worker.render.remote(mode)
            futures.append(future)
        
        return ray.get(futures)

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def release_rollout_memory(self):
        """Collect in bounded groups before training or checkpointing resumes."""
        import time

        started = time.monotonic()
        results = []
        # Avoid hundreds of simultaneous JVM full collections.
        for offset in range(0, len(self._workers), 16):
            results.extend(ray.get([
                worker.release_rollout_memory.remote()
                for worker in self._workers[offset:offset + 16]
            ]))
        before = sum(result["rss_before"] for result in results) / 1024**3
        after = sum(result["rss_after"] for result in results) / 1024**3
        java_before = sum(result["java_used_before"] for result in results) / 1024**3
        java_after = sum(result["java_used_after"] for result in results) / 1024**3
        phase = "train" if self.is_train else "validation"
        print(
            f"[WebShop GC] {phase}: workers={len(results)}, "
            f"summed worker RSS={before:.2f}->{after:.2f} GiB, "
            f"Java used={java_before:.2f}->{java_after:.2f} GiB, "
            f"elapsed={time.monotonic() - started:.2f}s",
            flush=True,
        )

    def close(self):
        if getattr(self, '_closed', False):
            return

        self.release_workers()
        self._closed = True

    def __del__(self):  # noqa: D401
        if getattr(self, "_workers", None) and ray.is_initialized():
            self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
    lazy_start: bool = False,
    jvm_max_heap_mb: int = 512,
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
        lazy_start=lazy_start,
        jvm_max_heap_mb=jvm_max_heap_mb,
    )
