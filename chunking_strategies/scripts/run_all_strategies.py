
# run_all_strategies.py
import os
import sys
import json
import time
import threading
import subprocess
import shutil
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import median
from typing import Dict, Any, List, Optional, Tuple

# Security/Environment fix for potential SSL certificate issues on Windows/Conda
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    # We print to stderr so it doesn't interfere with potential JSON output if any, 
    # though run_all_strategies is mostly a CLI tool.
    print(f"Warning: SSL_CERT_FILE is set to a non-existent path: {os.environ['SSL_CERT_FILE']}. Unsetting.", file=sys.stderr)
    del os.environ["SSL_CERT_FILE"]

# Ensure imports work relative to the chunking_strategies package root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunkbench_config import (
    ACTIVE_DOMAIN,
    CHUNKING_GPUS,
    CHUNKING_WORKERS,
    DOCUMENTS_FILE,
    MONITOR_GPU_INTERVAL,
    MONITOR_RSS_INTERVAL,
    chunking_stats_dir,
    chunks_dir,
    require,
)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
CHILD_SCRIPT = os.path.join(SCRIPT_DIR, "run_single_strategy.py")
CHUNKS_DIR = chunks_dir()
STATS_DIR = chunking_stats_dir()

# -------------------------------
# Parent-side monitors (per child PID)
# -------------------------------

def _safe_import_psutil():
    try:
        import psutil
        return psutil
    except Exception as e:
        print(
            f"[WARN] psutil unavailable ({e}); RAM metrics will be empty. "
            "Install it with: pip install psutil",
            file=sys.stderr,
        )
        return None

PSUTIL = _safe_import_psutil()

# Try NVML; if not available, we'll fall back to nvidia-smi
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

def get_child_rss_mb(pid: int) -> Optional[float]:
    """Returns RSS MB for a given PID, or None if unavailable."""
    if PSUTIL is None:
        return None
    try:
        proc = PSUTIL.Process(pid)
        return proc.memory_info().rss / (1024 ** 2)
    except Exception:
        return None

class PerProcessGPUMonitor:
    """
    Monitors per-PID GPU memory using NVML if available; otherwise best-effort via nvidia-smi.
    Utilization (percent) is device-level (approximate), but memory is per-PID when possible.
    """
    def __init__(self, pid: int, interval: float = MONITOR_GPU_INTERVAL):
        self.pid = pid
        self.interval = interval
        self._stop = threading.Event()
        self.peak_mem_mb: Optional[float] = None
        self.peak_util_percent: Optional[float] = None
        self.thread: Optional[threading.Thread] = None
        self._has_nvidia_smi = shutil.which("nvidia-smi") is not None

    def _sample_nvml(self):
        try:
            dev_count = pynvml.nvmlDeviceGetCount()
            device_peak_mem = 0.0
            device_peak_util = 0.0
            for i in range(dev_count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)

                # Device-level util for context (approximate)
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                    device_peak_util = max(device_peak_util, float(util))
                except Exception:
                    pass

                # Per-process memory: try graphics or compute process lists
                pid_mem_mb = 0.0
                lists = []
                for getter in (
                    getattr(pynvml, "nvmlDeviceGetGraphicsRunningProcesses", None),
                    getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses", None),
                ):
                    if getter:
                        try:
                            infos = getter(h)
                            lists.append(infos)
                        except Exception:
                            pass
                for infos in lists:
                    for info in infos or []:
                        try:
                            if info.pid == self.pid:
                                used_mb = info.usedGpuMemory / (1024 ** 2)  # bytes to MB
                                pid_mem_mb = max(pid_mem_mb, used_mb)
                        except Exception:
                            pass

                device_peak_mem = max(device_peak_mem, pid_mem_mb)

            if device_peak_mem > 0.0:
                self.peak_mem_mb = max(self.peak_mem_mb or 0.0, device_peak_mem)
            if device_peak_util > 0.0:
                self.peak_util_percent = max(self.peak_util_percent or 0.0, device_peak_util)
        except Exception:
            pass

    def _sample_nvidia_smi(self):
        # Fallback: query per-process memory via nvidia-smi compute apps (availability varies)
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
            ).decode("utf-8").strip()
            if out:
                peak_mb = 0.0
                for line in out.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2:
                        p = int(parts[0])
                        mem_mb = float(parts[1])
                        if p == self.pid:
                            peak_mb = max(peak_mb, mem_mb)
                if peak_mb > 0.0:
                    self.peak_mem_mb = max(self.peak_mem_mb or 0.0, peak_mb)

            # Device util for rough context
            out2 = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
            ).decode("utf-8").strip()
            if out2:
                util_peaks = [float(u.strip()) for u in out2.splitlines() if u.strip()]
                if util_peaks:
                    self.peak_util_percent = max(self.peak_util_percent or 0.0, max(util_peaks))
        except Exception:
            pass

    def _run(self):
        while not self._stop.is_set():
            if NVML_AVAILABLE:
                self._sample_nvml()
            elif self._has_nvidia_smi:
                self._sample_nvidia_smi()
            else:
                # Nothing available
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._stop.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=2.0)

def monitor_child_process(
    pid: int,
    stop_event: threading.Event,
    interval: float = MONITOR_RSS_INTERVAL,
) -> Dict[str, Any]:
    """
    Tracks baseline and peak RSS MB of the child PID and GPU memory/util during its lifetime.

    Termination is driven by `stop_event`, which the parent sets once the child
    has actually exited. Relying on psutil to detect the exit is not safe: it is
    an optional dependency, and a reaped-but-not-yet-cleaned child still reports
    as running, so the loop could never guarantee an exit on its own.
    """
    rss_baseline_mb = get_child_rss_mb(pid)
    peak_rss_mb = rss_baseline_mb or 0.0

    gpu_mon = PerProcessGPUMonitor(pid=pid, interval=MONITOR_GPU_INTERVAL)
    gpu_mon.start()

    try:
        while not stop_event.is_set():
            if PSUTIL:
                # Exit early if the process is already gone, but never depend on
                # this alone -- stop_event is the authoritative signal.
                try:
                    if not PSUTIL.Process(pid).is_running():
                        break
                except Exception:
                    break

            rss_now_mb = get_child_rss_mb(pid)
            if rss_now_mb is not None:
                peak_rss_mb = max(peak_rss_mb, rss_now_mb)

            stop_event.wait(interval)
    finally:
        gpu_mon.stop()

    ram_delta_mb = None
    if rss_baseline_mb is not None:
        ram_delta_mb = max(0.0, (peak_rss_mb - rss_baseline_mb))

    return {
        "ram_baseline_mb": rss_baseline_mb,
        "ram_peak_mb": peak_rss_mb,
        "ram_delta_mb": ram_delta_mb,
        "gpu_peak_mem_mb": gpu_mon.peak_mem_mb,
        "gpu_peak_util_percent": gpu_mon.peak_util_percent,
    }

# -------------------------------
# Strategy specs loading
# -------------------------------

CONFIG_FILE = os.path.join(PACKAGE_ROOT, "chunking", "chunker_config.json")

def load_strategies_from_config(config_path: str) -> List[Tuple[str, Dict[str, Any]]]:
    if not os.path.exists(config_path):
        print(f"[WARN] Config file {config_path} not found. Returning empty strategies.")
        return []
        
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    strategies = []
    for s in data.get("strategies", []):
        if s.get("enabled", True):
            strategies.append((s["name"], s["spec"]))
    return strategies

# -------------------------------
# Parent orchestration
# -------------------------------

def save_json(obj: Any, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

_gpu_counter = itertools.count()


def run_strategy(strategy_name: str, spec: Dict[str, Any]) -> None:
    """Runs one strategy in a monitored subprocess and writes its metrics."""
    print(f"\n=== Running strategy in subprocess: {strategy_name} ===")

    env = os.environ.copy()

    # Spread concurrent children across GPUs round-robin. Each child then sees a
    # single device, so the embedding chunkers never contend for one card and no
    # shared model pool is involved.
    if CHUNKING_GPUS:
        gpu = CHUNKING_GPUS[next(_gpu_counter) % len(CHUNKING_GPUS)]
        env["CUDA_VISIBLE_DEVICES"] = gpu
        print(f"    pinned to GPU {gpu}")

    # Spawn child process
    proc = subprocess.Popen(
        [sys.executable, CHILD_SCRIPT, strategy_name],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    # Start parent-side monitoring for child PID
    monitor_thread_result: Dict[str, Any] = {}
    stop_event = threading.Event()

    def _monitor():
        res = monitor_child_process(proc.pid, stop_event)
        monitor_thread_result.update(res)

    t0 = time.perf_counter()
    mon_thread = threading.Thread(target=_monitor, daemon=True)
    mon_thread.start()

    # Send spec JSON to child via stdin
    spec_json = json.dumps(spec)
    try:
        out, err = proc.communicate(input=spec_json.encode("utf-8"))
    finally:
        # Always release the monitor, even if the child was killed or the parent
        # was interrupted -- otherwise the join below would hang forever.
        stop_event.set()

    mon_thread.join(timeout=MONITOR_GPU_INTERVAL + 5.0)
    if mon_thread.is_alive():
        print(
            f"[WARN] Resource monitor for {strategy_name} did not stop; "
            "continuing without its metrics.",
            file=sys.stderr,
        )
    t1 = time.perf_counter()

    elapsed_sec = round(t1 - t0, 6)

    if proc.returncode != 0:
        print(f"[ERROR] Child failed for {strategy_name}:\n{err.decode()}", file=sys.stderr)
        # Still write monitor data for debugging
        metrics_file = os.path.join(STATS_DIR, f"{strategy_name}_metrics_parent.json")
        save_json({
            "strategy": strategy_name,
            "elapsed_sec": elapsed_sec,
            "monitor": monitor_thread_result,
            "child_stdout": out.decode(),
            "child_stderr": err.decode(),
        }, metrics_file)
        return

    # Child should have written core metrics to STATS_DIR/{strategy}_metrics_core.json
    core_metrics_file = os.path.join(STATS_DIR, f"{strategy_name}_metrics_core.json")
    if not os.path.exists(core_metrics_file):
        print(f"[WARN] Core metrics not found for {strategy_name}: {core_metrics_file}", file=sys.stderr)
        core_metrics = {}
    else:
        with open(core_metrics_file, "r", encoding="utf-8") as f:
            core_metrics = json.load(f)

    # Merge
    consolidated = {
        "strategy": strategy_name,
        "elapsed_sec_parent": elapsed_sec,  # whole child run
        "resource_monitor": monitor_thread_result,  # RAM/GPU (per-PID)
        "core_metrics": core_metrics,  # per-doc + aggregated from child
    }

    metrics_file = os.path.join(STATS_DIR, f"{strategy_name}_metrics.json")
    save_json(consolidated, metrics_file)
    print(f"[OK] Saved consolidated metrics to {metrics_file}")


def main():
    require("DOCUMENTS_FILE", "DOMAINS", "EMBEDDING_MODEL", "LLM_CHUNKER_MODEL")

    if not os.path.exists(DOCUMENTS_FILE):
        print(f"Error: {DOCUMENTS_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    strategies = load_strategies_from_config(CONFIG_FILE)
    print(f"Loaded strategies: {len(strategies)}")
    print(f"Domain: {ACTIVE_DOMAIN}")
    print(f"Documents: {DOCUMENTS_FILE}")
    print(f"Chunks  -> {CHUNKS_DIR}")
    print(f"Stats   -> {STATS_DIR}")

    workers = max(1, CHUNKING_WORKERS)

    if workers == 1:
        for strategy_name, spec in strategies:
            run_strategy(strategy_name, spec)
    else:
        print(
            f"Running {workers} strategies concurrently.\n"
            "  WARNING: resource metrics are no longer isolated. Concurrent children\n"
            "  share the CPU and GPU, so ram_peak_mb / gpu_peak_mem_mb and the\n"
            "  elapsed times are not comparable to a sequential run. Use\n"
            "  CHUNKBENCH_WORKERS=1 when the resource numbers matter.",
            file=sys.stderr,
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(run_strategy, name, spec): name
                for name, spec in strategies
            }
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"[ERROR] {name} raised: {e}", file=sys.stderr)

    print("\nAll strategies completed.")

if __name__ == "__main__":
    main()
