"""Sample host/container memory separately around training stages."""

import threading
from contextlib import contextmanager
from pathlib import Path

import psutil
from ray._private.utils import get_system_memory, get_used_memory


def memory_snapshot():
    gib = 1024**3
    values = {
        "ray_used_gib": get_used_memory() / gib,
        "ray_limit_gib": get_system_memory() / gib,
        "host_used_gib": psutil.virtual_memory().used / gib,
        "driver_rss_gib": psutil.Process().memory_info().rss / gib,
    }
    # cgroup v2 breakdown supplements Ray's working-set calculation. File
    # memory includes shmem; these fields must not be summed as disjoint usage.
    try:
        root = Path("/sys/fs/cgroup")
        values["cgroup_current_gib"] = int((root / "memory.current").read_text()) / gib
        stats = dict(line.split() for line in (root / "memory.stat").read_text().splitlines())
        for key in ("anon", "file", "shmem"):
            values[f"cgroup_{key}_gib"] = int(stats[key]) / gib
    except (OSError, ValueError, KeyError):
        pass
    return values


@contextmanager
def monitor_memory(stage, metrics, step, interval=0.5):
    """Record sampled peaks, including on exceptions; not an exact peak meter."""
    before = memory_snapshot()
    peaks = dict(before)
    stop = threading.Event()

    def sample():
        while not stop.wait(interval):
            for key, value in memory_snapshot().items():
                peaks[key] = max(peaks.get(key, value), value)

    print(f"[Memory] step={step} stage={stage} begin={before}", flush=True)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        after = memory_snapshot()
        for key, value in after.items():
            peaks[key] = max(peaks.get(key, value), value)
            metrics[f"memory/{stage}/{key}_before"] = before.get(key, value)
            metrics[f"memory/{stage}/{key}_after"] = value
            metrics[f"memory/{stage}/{key}_sampled_peak"] = peaks[key]
        print(f"[Memory] step={step} stage={stage} end={after} sampled_peak={peaks}", flush=True)
