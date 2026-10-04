"""
Edge inference energy measurement harness.

Measures GPU-device power with NVML (nvmlDeviceGetPowerUsage) sampled at 20 Hz
and integrated over long fixed-duration windows. The NVML sensor refreshes at
roughly 5-10 Hz, so a 30 s window contains >150 independent sensor updates.
The cumulative energy counter (nvmlDeviceGetTotalEnergyConsumption) is also
recorded, but only as a diagnostic: under WSL2 it was found to be inconsistent
with the power sensor (reporting several hundred watts at idle), so it is never
used for the reported values. Each stage runs at the line capture rate (default 30 FPS) and
is repeated with a seeded, shuffled stage order so that thermal / clock-state
drift is not confounded with stage cost. An idle window precedes every repeat,
so active power is always referenced to a nearby idle level.

The inference workload is a PatchCore-style detector: a ResNet-18 backbone
(randomly initialised weights -- compute cost is independent of weight values),
layer2+layer3 patch features at 28x28, and 1-NN search against a synthetic
memory bank. No measurement value is ever substituted: if NVML is unavailable
the harness raises instead of fabricating a fallback.

Boundary: GPU device only. Host CPU/RAM, camera and lighting power are not
observable through NVML; the accounting model adds them through the explicit
P_host assumption in the source registry.
"""

from __future__ import annotations

import json
import platform
import random
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.paths import ENERGY_SUMMARY, ENERGY_TRACE_DIR, REPO_ROOT

STAGES = ("STAGE_MODEL_INFER", "STAGE_MODEL_THRESHOLD", "STAGE_MODEL_POLICY", "STAGE_FULL_PIPELINE")


class DualEMAPolicy:
    """Dual-EMA divergence gate + k-of-N persistence + cooldown (temporal alert filter)."""

    def __init__(self, alpha_fast=0.3, alpha_slow=0.05, k=3, n=5, cooldown=15):
        self.alpha_fast, self.alpha_slow = alpha_fast, alpha_slow
        self.k, self.cooldown = k, cooldown
        self.ema_fast = self.ema_slow = 0.0
        self.window = deque(maxlen=n)
        self.cooldown_rem = 0

    def update(self, score: float, threshold: float) -> bool:
        self.ema_fast = self.alpha_fast * score + (1.0 - self.alpha_fast) * self.ema_fast
        self.ema_slow = self.alpha_slow * score + (1.0 - self.alpha_slow) * self.ema_slow
        self.window.append(score > threshold and self.ema_fast > self.ema_slow)
        if self.cooldown_rem > 0:
            self.cooldown_rem -= 1
            return False
        if sum(self.window) >= self.k:
            self.cooldown_rem = self.cooldown
            return True
        return False


MEMORY_BANK_SIZE = 13798  # coreset size of the PatchCore model in Paper A (10% greedy k-center)


def build_patchcore_proxy(device, memory_bank_size: int = MEMORY_BANK_SIZE, seed: int = 0):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models import resnet18

    torch.manual_seed(seed)

    class PatchCoreProxy(nn.Module):
        def __init__(self):
            super().__init__()
            bb = resnet18()  # random init; default is untrained on all torchvision versions
            self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool, bb.layer1)
            self.layer2, self.layer3 = bb.layer2, bb.layer3
            self.register_buffer("bank", torch.randn(memory_bank_size, 384))

        def forward(self, x):
            f2 = self.layer2(self.stem(x))
            f3 = self.layer3(f2)
            f2 = F.avg_pool2d(f2, 3, 1, 1)
            f3 = F.interpolate(F.avg_pool2d(f3, 3, 1, 1), size=f2.shape[-2:], mode="bilinear", align_corners=False)
            patches = torch.cat([f2, f3], dim=1).flatten(2).transpose(1, 2)[0]  # (784, 384)
            d = torch.cdist(patches, self.bank).min(dim=1).values
            return d.max()

    return PatchCoreProxy().to(device).eval()


class NvmlMeter:
    """Integrates sampled NVML power over a window; also logs the energy counter as a diagnostic."""

    def __init__(self, index: int = 0, sample_hz: float = 20.0):
        import pynvml

        self.nv = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.nv.nvmlDeviceGetPowerUsage(self.handle)  # raises if unsupported; no fallback values
        self.dt = 1.0 / sample_hz
        self._stop = threading.Event()
        self._trace: List[tuple] = []
        self._error: Optional[BaseException] = None
        self._thread = None

    def _counter_mj(self):
        try:
            return int(self.nv.nvmlDeviceGetTotalEnergyConsumption(self.handle))
        except Exception:
            return None

    def _loop(self, t0):
        try:
            while not self._stop.is_set():
                self._trace.append((time.perf_counter() - t0, self.nv.nvmlDeviceGetPowerUsage(self.handle) / 1000.0))
                time.sleep(self.dt)
        except BaseException as exc:  # surfaced by stop(); a truncated trace must not be integrated
            self._error = exc

    def start(self):
        self._trace = []
        self._error = None
        self._stop.clear()
        self._t0 = time.perf_counter()
        self._c0 = self._counter_mj()
        self._thread = threading.Thread(target=self._loop, args=(self._t0,), daemon=True)
        self._thread.start()

    def stop(self):
        c1 = self._counter_mj()
        t1 = time.perf_counter()
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._error is not None:
            raise RuntimeError("NVML power sampling failed during the window") from self._error
        tr = list(self._trace)
        if len(tr) < 2:
            raise RuntimeError(f"only {len(tr)} NVML power samples in the window; cannot integrate")
        t = np.array([x[0] for x in tr])
        p = np.array([x[1] for x in tr])
        mean_p = float(np.trapezoid(p, t) / (t[-1] - t[0]))
        counter_w = ((c1 - self._c0) / 1000.0) / (t1 - self._t0) if (c1 is not None and self._c0 is not None) else None
        return mean_p, t1 - self._t0, tr, counter_w

    def metadata(self) -> Dict[str, str]:
        name = self.nv.nvmlDeviceGetName(self.handle)
        return {
            "gpu_name": name.decode() if isinstance(name, bytes) else name,
            "driver_version": str(self.nv.nvmlSystemGetDriverVersion()),
        }


@dataclass
class StageRunner:
    model: object
    device: object
    db_path: Path
    threshold: float = 0.5

    def run(self, stage: str, duration_s: float, fps: float | None) -> Dict[str, float]:
        """Run a stage for duration_s; fps=None means unthrottled. Returns frame count and latencies."""
        import torch

        if stage not in STAGES:
            raise ValueError(f"unknown stage '{stage}'")
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        policy = DualEMAPolicy()
        conn = None
        if stage == "STAGE_FULL_PIPELINE":
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("CREATE TABLE IF NOT EXISTS events (frame_id INT, score REAL, escalated INT, ts REAL)")
        frame = torch.rand(1, 3, 224, 224, device=self.device)
        lat: List[float] = []
        t_start = time.perf_counter()
        i = 0
        with torch.no_grad():
            while True:
                if fps is not None:
                    target = t_start + i / fps
                    now = time.perf_counter()
                    if target > now:
                        time.sleep(target - now)
                t0 = time.perf_counter()
                if t0 - t_start >= duration_s:
                    break
                score = float(self.model(frame))  # .item() synchronises the device
                if stage == "STAGE_MODEL_THRESHOLD":
                    _ = score > self.threshold
                elif stage == "STAGE_MODEL_POLICY":
                    _ = policy.update(score, self.threshold)
                elif stage == "STAGE_FULL_PIPELINE":
                    esc = policy.update(score, self.threshold)
                    conn.execute("INSERT INTO events VALUES (?, ?, ?, ?)", (i, score, int(esc), time.time()))
                    _ = json.dumps({"frame": i, "score": score, "alert": bool(esc)})
                    if i % 30 == 29:
                        conn.commit()
                lat.append((time.perf_counter() - t0) * 1000.0)
                i += 1
        if conn is not None:
            conn.commit()
            conn.close()
        wall = time.perf_counter() - t_start
        return {
            "frames": i,
            "wall_s": wall,
            "achieved_fps": i / wall,
            "latency_ms_p50": float(np.percentile(lat, 50)),
            "latency_ms_p95": float(np.percentile(lat, 95)),
        }


def _stats(values: List[float]) -> Dict[str, float]:
    """Robust per-repeat statistics. The median is the reported central value and
    [min, max] the uncertainty bounds: the GPU is shared with the host desktop, so
    occasional contaminated windows are kept in the raw data but cannot move the median."""
    arr = np.asarray(values, dtype=float)
    return {"median": float(np.median(arr)), "min": float(arr.min()), "max": float(arr.max()),
            "mean": float(arr.mean()), "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, "n": int(len(arr))}


def summarize_run(trace_dir: Path) -> Dict:
    """Build the energy summary from a run directory (windows.csv + run_meta.json)."""
    import pandas as pd

    meta = json.loads((trace_dir / "run_meta.json").read_text(encoding="utf-8"))
    df = pd.read_csv(trace_dir / "windows.csv")
    stages_out = {}
    for st in ("BASELINE_IDLE",) + STAGES:
        sub = df[df.stage == st]
        out = {"p_total_w": _stats(sub.p_total_w.tolist()), "p_active_w": _stats(sub.p_active_w.tolist())}
        if st != "BASELINE_IDLE":
            for col in ("achieved_fps", "e_frame_total_j", "e_frame_active_j", "latency_ms_p50", "latency_ms_p95"):
                out[col] = _stats(sub[col].tolist())
            out["unthrottled"] = meta["unthrottled"][st]
        stages_out[st] = out
    return {**{k: v for k, v in meta.items() if k != "unthrottled"},
            "trace_dir": trace_dir.resolve().relative_to(REPO_ROOT).as_posix(), "stages": stages_out}


def measured_registry_bounds(summary: Dict) -> Dict[str, tuple]:
    """(low, central, high) = (min, median, max) across repeats, rounded as stored in the registry."""
    stats = {"gpu_idle_power": summary["stages"]["BASELINE_IDLE"]["p_total_w"],
             "gpu_active_power": summary["stages"]["STAGE_FULL_PIPELINE"]["p_active_w"]}
    return {key: tuple(round(st[c], 3) for c in ("min", "median", "max")) for key, st in stats.items()}


def run_benchmark(
    repeats: int = 5,
    window_s: float = 30.0,
    idle_s: float = 15.0,
    fps: float = 30.0,
    throughput_window_s: float = 5.0,
    warmup_frames: int = 100,
    seed: int = 2026,
    trace_root: Path = ENERGY_TRACE_DIR,
    summary_path: Path = ENERGY_SUMMARY,
) -> Dict:
    """Measure all stages; raw traces go to trace_root/<run_id>/ and the summary to summary_path.
    trace_root must lie inside the repository (the summary stores it repository-relative)."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for the GPU energy benchmark.")
    device = torch.device("cuda:0")
    meter = NvmlMeter()
    model = build_patchcore_proxy(device)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace_dir = trace_root / run_id
    trace_dir.mkdir(parents=True, exist_ok=True)
    runner = StageRunner(model=model, device=device, db_path=trace_dir / "spool.db")

    with torch.no_grad():
        x = torch.rand(1, 3, 224, 224, device=device)
        for _ in range(warmup_frames):
            float(model(x))
    torch.cuda.synchronize()

    # Unthrottled throughput / latency (not used for energy).
    throughput = {st: runner.run(st, throughput_window_s, fps=None) for st in STAGES}

    rng = random.Random(seed)
    records = []
    traces = []
    for rep in range(repeats):
        time.sleep(3.0)
        meter.start()
        time.sleep(idle_s)
        p_idle, t_idle, tr, cw = meter.stop()
        traces += [("BASELINE_IDLE", rep, t, p) for t, p in tr]
        records.append({"repeat": rep, "stage": "BASELINE_IDLE", "duration_s": t_idle, "p_total_w": p_idle,
                        "p_active_w": 0.0, "frames": 0, "achieved_fps": 0.0, "n_samples": len(tr),
                        "n_distinct_readings": len({p for _, p in tr}), "counter_diag_w": cw})
        order = list(STAGES)
        rng.shuffle(order)
        for position, st in enumerate(order):
            meter.start()
            info = runner.run(st, window_s, fps=fps)
            p_total, dt, tr, cw = meter.stop()
            traces += [(st, rep, t, p) for t, p in tr]
            records.append({
                "repeat": rep, "stage": st, "duration_s": dt, "p_total_w": p_total,
                "p_active_w": p_total - p_idle, "frames": info["frames"], "achieved_fps": info["achieved_fps"],
                "e_frame_total_j": p_total * dt / info["frames"],
                "e_frame_active_j": (p_total - p_idle) * dt / info["frames"],
                "order_position": position, "n_samples": len(tr),
                "n_distinct_readings": len({p for _, p in tr}), "counter_diag_w": cw,
                "latency_ms_p50": info["latency_ms_p50"], "latency_ms_p95": info["latency_ms_p95"],
            })
            print(f"  rep {rep} {st:22s} P_total={p_total:6.2f} W  P_active={p_total - p_idle:6.2f} W  fps={info['achieved_fps']:.2f}")

    import pandas as pd

    pd.DataFrame(records).to_csv(trace_dir / "windows.csv", index=False)
    pd.DataFrame(traces, columns=["stage", "repeat", "time_s", "power_w"]).to_csv(trace_dir / "power_trace_20hz.csv", index=False)
    meta = {
        "run_id": run_id,
        "method": "NVML power sensor sampled at 20 Hz, trapezoid-integrated over fixed windows at target FPS; "
                  "shuffled stage order; idle reference per repeat; median [min, max] across repeats",
        "target_fps": fps, "repeats": repeats, "window_s": window_s, "idle_window_s": idle_s, "seed": seed,
        "workload": "PatchCore-style: ResNet-18 layer2+3 features (28x28x384), 1-NN vs 13798x384 memory bank, batch 1, 224x224, FP32",
        "hardware": {**meter.metadata(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                     "kernel": platform.release(), "python": platform.python_version()},
        "unthrottled": throughput,
    }
    (trace_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    for extra in ("spool.db", "spool.db-wal", "spool.db-shm"):
        (trace_dir / extra).unlink(missing_ok=True)
    summary = summarize_run(trace_dir)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary


if __name__ == "__main__":
    s = run_benchmark()
    print(json.dumps(s["stages"]["STAGE_FULL_PIPELINE"], indent=2))
