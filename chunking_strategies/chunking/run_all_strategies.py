
# run_all_strategies.py
import os
import sys
import json
import time
import threading
import subprocess
import shutil
from statistics import median
from typing import Dict, Any, List, Optional, Tuple

# Security/Environment fix for potential SSL certificate issues on Windows/Conda
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    # We print to stderr so it doesn't interfere with potential JSON output if any, 
    # though run_all_strategies is mostly a CLI tool.
    print(f"Warning: SSL_CERT_FILE is set to a non-existent path: {os.environ['SSL_CERT_FILE']}. Unsetting.", file=sys.stderr)
    del os.environ["SSL_CERT_FILE"]

# Paths
DOCUMENTS_FILE = "chunking_strategies/data/domain_context/agriculture_unique_contexts.json"
CHUNKS_DIR = "chunking_strategies/chunking/chunks"
STATS_DIR = "chunking_strategies/chunking/stats"

# -------------------------------
# Parent-side monitors (per child PID)
# -------------------------------

def _safe_import_psutil():
    try:
        import psutil
        return psutil
    except Exception:
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
    def __init__(self, pid: int, interval: float = 0.5):
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

def monitor_child_process(pid: int, interval: float = 0.25) -> Dict[str, Any]:
    """
    Tracks baseline and peak RSS MB of the child PID and GPU memory/util during its lifetime.
    """
    rss_baseline_mb = get_child_rss_mb(pid)
    peak_rss_mb = rss_baseline_mb or 0.0

    gpu_mon = PerProcessGPUMonitor(pid=pid, interval=0.5)
    gpu_mon.start()

    try:
        while True:
            # Check child still alive
            if PSUTIL:
                try:
                    proc = PSUTIL.Process(pid)
                    if not proc.is_running():
                        break
                except Exception:
                    break
            else:
                # Without psutil, we rely on parent Popen wait
                pass

            rss_now_mb = get_child_rss_mb(pid)
            if rss_now_mb is not None:
                peak_rss_mb = max(peak_rss_mb, rss_now_mb)

            time.sleep(interval)
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

CONFIG_FILE = "chunking_strategies/chunking/chunker_config.json"

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

def main():
    if not os.path.exists(DOCUMENTS_FILE):
        print(f"Error: {DOCUMENTS_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    strategies = load_strategies_from_config(CONFIG_FILE)
    print(f"Loaded strategies: {len(strategies)}")

    for strategy_name, spec in strategies:
        print(f"\n=== Running strategy in subprocess: {strategy_name} ===")

        # Environment isolation hint: pin to GPU 0 if desired
        env = os.environ.copy()
        # env["CUDA_VISIBLE_DEVICES"] = "0"  # optionally uncomment to pin GPU

        # Spawn child process
        proc = subprocess.Popen(
            [sys.executable, "chunking_strategies/chunking/run_single_strategy.py", strategy_name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Start parent-side monitoring for child PID
        monitor_thread_result: Dict[str, Any] = {}
        def _monitor():
            res = monitor_child_process(proc.pid, interval=0.25)
            monitor_thread_result.update(res)

        t0 = time.perf_counter()
        mon_thread = threading.Thread(target=_monitor, daemon=True)
        mon_thread.start()

        # Send spec JSON to child via stdin
        spec_json = json.dumps(spec)
        out, err = proc.communicate(input=spec_json.encode("utf-8"))
        mon_thread.join()
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
            continue

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

    print("\nAll strategies completed.")

if __name__ == "__main__":
    main()
