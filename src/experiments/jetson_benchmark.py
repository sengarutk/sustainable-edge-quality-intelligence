"""
Embedded edge measurement on an NVIDIA Jetson (runs on the device; needs numpy, pandas and TensorRT
only -- no PyTorch).

Power: the module's INA3221 monitor (hwmon 'ina3221'), read through sysfs at 20 Hz. Channel VDD_IN
is the total input power of the Jetson module (CPU, GPU, memory, SoC), so unlike NVML on a
workstation it covers the whole edge computer except the carrier board, camera and lighting. The
VDD_CPU_GPU_CV and VDD_SOC rails are logged alongside. Windows are trapezoid-integrated exactly
as in the workstation protocol (src.experiments.energy_benchmark.measure_stages).

Workload: the single-graph ONNX detectors of src.experiments.edge_export, compiled to TensorRT
engines on the device. Each frame is copied host-to-device, scored by one engine execution, and
the scalar score copied back, so per-frame transfer cost is included.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.experiments.energy_benchmark import StageRunner, measure_stages, summarize_run

RAILS = ("VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC")


class Cudart:
    H2D, D2H = 1, 2

    def __init__(self, lib: str = "libcudart.so"):
        self.lib = ctypes.CDLL(lib)
        self.stream = ctypes.c_void_p()
        self._ok(self.lib.cudaStreamCreate(ctypes.byref(self.stream)), "cudaStreamCreate")

    @staticmethod
    def _ok(code: int, what: str):
        if code != 0:
            raise RuntimeError(f"{what} failed with CUDA error {code}")

    def malloc(self, nbytes: int) -> int:
        ptr = ctypes.c_void_p()
        self._ok(self.lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)), "cudaMalloc")
        return ptr.value

    def copy(self, dst: int, src: int, nbytes: int, kind: int):
        self._ok(self.lib.cudaMemcpyAsync(ctypes.c_void_p(dst), ctypes.c_void_p(src), ctypes.c_size_t(nbytes),
                                          ctypes.c_int(kind), self.stream), "cudaMemcpyAsync")

    def sync(self):
        self._ok(self.lib.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")


class TrtScorer:
    """score_fn(i): score frame i % n with one TensorRT execution (H2D frame, D2H scalar)."""

    def __init__(self, engine_path: Path, frames: np.ndarray):
        import tensorrt as trt

        self.cuda = Cudart()
        logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        inp = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        out = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if len(inp) != 1 or len(out) != 1:
            raise RuntimeError(f"expected one input and one output, got {names}")
        self.frames = np.ascontiguousarray(frames, dtype=np.float32)
        self.frame_bytes = self.frames[0].nbytes
        self.out = np.zeros(1, dtype=np.float32)
        self.d_in = self.cuda.malloc(self.frame_bytes)
        self.d_out = self.cuda.malloc(self.out.nbytes)
        self.ctx.set_tensor_address(inp[0], self.d_in)
        self.ctx.set_tensor_address(out[0], self.d_out)

    def __call__(self, i: int) -> float:
        f = self.frames[i % len(self.frames)]
        self.cuda.copy(self.d_in, f.ctypes.data, self.frame_bytes, Cudart.H2D)
        if not self.ctx.execute_async_v3(self.cuda.stream.value):
            raise RuntimeError("TensorRT execution failed")
        self.cuda.copy(self.out.ctypes.data, self.d_out, self.out.nbytes, Cudart.D2H)
        self.cuda.sync()
        return float(self.out[0])


def find_ina3221() -> Path:
    for hw in Path("/sys/class/hwmon").iterdir():
        if (hw / "name").read_text().strip() == "ina3221":
            return hw.resolve()
    raise FileNotFoundError("no ina3221 hwmon device (not a Jetson?)")


class InaMeter:
    """Same interface as NvmlMeter: start(); stop() -> (mean VDD_IN W, duration, trace, None)."""

    def __init__(self, sample_hz: float = 20.0):
        hw = find_ina3221()
        labels = {(hw / f"in{k}_label").read_text().strip(): k for k in (1, 2, 3)}
        missing = [r for r in RAILS if r not in labels]
        if missing:
            raise RuntimeError(f"INA3221 rails missing: {missing}")
        self.files = [(hw / f"in{labels[r]}_input", hw / f"curr{labels[r]}_input") for r in RAILS]
        self.hw = hw
        self.read()  # raises if unreadable; no fallback values
        self.dt = 1.0 / sample_hz
        self._stop = threading.Event()
        self._trace: List[tuple] = []
        self.rail_trace: List[tuple] = []
        self._error: Optional[BaseException] = None

    def read(self) -> List[float]:
        return [int(v.read_text()) * int(c.read_text()) / 1e6 for v, c in self.files]  # mV * mA -> W

    def _loop(self, t0):
        try:
            while not self._stop.is_set():
                t = time.perf_counter() - t0
                p = self.read()
                self._trace.append((t, p[0]))
                self._rails.append((t, *p))
                time.sleep(self.dt)
        except BaseException as exc:
            self._error = exc

    def start(self):
        self._trace, self._rails, self._error = [], [], None
        self._stop.clear()
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, args=(self._t0,), daemon=True)
        self._thread.start()

    def stop(self):
        t1 = time.perf_counter()
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._error is not None:
            raise RuntimeError("INA3221 sampling failed during the window") from self._error
        tr = list(self._trace)
        if len(tr) < 2:
            raise RuntimeError(f"only {len(tr)} INA3221 samples in the window; cannot integrate")
        t = np.array([x[0] for x in tr])
        p = np.array([x[1] for x in tr])
        self.rail_trace = list(self._rails)
        trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2 on JetPack
        return float(trapezoid(p, t) / (t[-1] - t[0])), t1 - self._t0, tr, None

    def metadata(self) -> Dict[str, str]:
        def sh(cmd):
            try:
                return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
            except Exception:
                return "unavailable"

        return {"device_model": Path("/proc/device-tree/model").read_text().strip("\x00\n "),
                "l4t_release": Path("/etc/nv_tegra_release").read_text().splitlines()[0],
                "nvpmodel": sh(["nvpmodel", "-q"]).replace("\n", " | "),
                "power_sensor": f"INA3221 ({self.hw}), rails {', '.join(RAILS)}; reported: VDD_IN"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fidelity(score_fn, reference: List[float], threshold: float) -> Dict[str, float]:
    """Engine scores against the FP32 reference scorer on the same frames."""
    ref = np.asarray(reference)
    got = np.array([score_fn(i) for i in range(len(ref))])
    rank = lambda x: np.argsort(np.argsort(x))  # noqa: E731
    return {"max_rel_error": float(np.max(np.abs(got - ref) / np.abs(ref))),
            "spearman": float(np.corrcoef(rank(got), rank(ref))[0, 1]),
            "decision_agreement": float(np.mean((got > threshold) == (ref > threshold)))}


def run(export_dir: Path, detector: str, precision: str, out_root: Path, repeats: int = 5, window_s: float = 30.0,
        idle_s: float = 15.0, fps: float = 30.0, seed: int = 2026) -> Dict:
    import pandas as pd
    import tensorrt as trt

    workload = json.loads((export_dir / "workload.json").read_text())
    det = workload["detectors"][detector]
    engine = export_dir / f"{detector}_{precision}.engine"
    score_fn = TrtScorer(engine, np.load(export_dir / det.get("frames_file", "frames.npy")))
    meter = InaMeter()
    fid = fidelity(score_fn, det["reference_scores"], det["threshold"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{detector}_{precision}"
    trace_dir = out_root / run_id
    trace_dir.mkdir(parents=True, exist_ok=True)
    runner = StageRunner(model=score_fn, device="jetson", db_path=trace_dir / "spool.db", threshold=det["threshold"])
    rail_rows = []

    class RailLogger:  # wraps the meter to keep the per-rail trace of every window
        def start(self):
            meter.start()

        def stop(self):
            out = meter.stop()
            rail_rows.extend(meter.rail_trace)
            return out

    throughput, records, traces = measure_stages(RailLogger(), runner, score_fn, repeats, window_s, idle_s, fps,
                                                 5.0, 100, seed)
    pd.DataFrame(records).to_csv(trace_dir / "windows.csv", index=False)
    pd.DataFrame(traces, columns=["stage", "repeat", "time_s", "power_w"]).to_csv(trace_dir / "power_trace_20hz.csv",
                                                                                  index=False)
    pd.DataFrame(rail_rows, columns=["time_s", *[f"{r}_w" for r in RAILS]]).to_csv(trace_dir / "rail_trace_20hz.csv",
                                                                                     index=False)
    meta = {
        "run_id": run_id,
        "method": "Jetson INA3221 VDD_IN (module input power) sampled at 20 Hz, trapezoid-integrated over fixed "
                  "windows at target FPS; shuffled stage order; idle reference per repeat; median [min, max] across repeats",
        "target_fps": fps, "repeats": repeats, "window_s": window_s, "idle_window_s": idle_s, "seed": seed,
        "workload": f"{detector} single-graph TensorRT engine ({precision}), batch 1, {det.get('img_size', 224)}x{det.get('img_size', 224)}, MVTec AD "
                    f"{workload['category']} frames; host-to-device copy per frame",
        "workload_detail": {"detector": detector, "precision": precision, "category": workload["category"],
                            "state_size": det["state_size"], "threshold": det["threshold"],
                            "onnx_sha256": det["onnx_sha256"], "engine_sha256": _sha256(engine),
                            "fidelity_vs_fp32_reference": fid},
        "hardware": {**meter.metadata(), "tensorrt": trt.__version__, "kernel": platform.release(),
                     "python": platform.python_version()},
        "unthrottled": throughput,
    }
    (trace_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    for extra in ("spool.db", "spool.db-wal", "spool.db-shm"):
        (trace_dir / extra).unlink(missing_ok=True)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--detector", choices=("patchcore", "padim", "patchcore_448", "patchcore_wrn50"), required=True)
    ap.add_argument("--precision", choices=("fp32", "fp16"), required=True)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    m = run(a.export_dir, a.detector, a.precision, a.out, repeats=a.repeats)
    print(json.dumps(m["workload_detail"]["fidelity_vs_fp32_reference"]))
