"""
Single-graph ONNX export of the measured detectors for embedded (TensorRT) deployment.

The whole per-frame scoring path runs inside one graph, so an edge device executes it as one
TensorRT engine: ResNet-18 features, the detector head (PatchCore 1-NN distance to the memory
bank, or PaDiM per-position Mahalanobis distance), bilinear upsampling to 224x224, a separable
Gaussian filter (sigma 4, truncated at 4 sigma, symmetric boundary as in scipy.ndimage's
'reflect' mode) and the max. `export_workload` fits both detectors on the same MVTec AD category
and frames as the workstation benchmark (hazelnut, bank of 13,798 patches, 80/20 fit/calibration
split, 120 cycled test frames), checks that the graph reproduces the reference scorer, and writes

  <out>/<detector>.onnx   graph with the fitted state as constants (not committed: large)
  <out>/frames.npy        normalised test frames, float32 (not committed: derived from MVTec AD)
  <out>/workload.json     thresholds, reference scores and provenance (committed with the run)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict

import numpy as np

from src.experiments.detector_eval import (IMG_SIZE, PaDiM, PatchCore, default_mvtec_root, load_images, make_detector,
                                           split_good,
                                           weights_digest)
from src.experiments.energy_benchmark import MEMORY_BANK_SIZE

SIGMA = 4.0
RADIUS = int(4.0 * SIGMA + 0.5)  # scipy.ndimage default truncate=4.0


def _gaussian_kernel():
    x = np.arange(-RADIUS, RADIUS + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / SIGMA) ** 2)
    return (k / k.sum()).astype(np.float32)


def _graph_module(det):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    size = det.img_size
    sym = torch.tensor(list(range(RADIUS - 1, -1, -1)) + list(range(size))
                       + list(range(size - 1, size - 1 - RADIUS, -1)))
    k = torch.from_numpy(_gaussian_kernel())

    class Graph(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = det.net
            self.pool = det.pool
            self.register_buffer("sym", sym)
            self.register_buffer("kx", k.view(1, 1, 1, -1))
            self.register_buffer("ky", k.view(1, 1, -1, 1))
            self.wide = getattr(det, "arch", "resnet18") == "wide_resnet50_2"
            if isinstance(det, PatchCore):
                self.kind = "patchcore"
                self.register_buffer("bank_t", det.bank.T.contiguous())
                self.register_buffer("bank_sq", (det.bank * det.bank).sum(1)[None, :])
            else:
                self.kind = "padim"
                self.register_buffer("idx", det.idx)
                self.register_buffer("mean", det.mean[:, None, :])  # (P, 1, C)
                self.register_buffer("cov_inv", det.cov_inv)  # (P, C, C)

        def blur_max(self, d, h, w):
            a = F.interpolate(d.reshape(1, 1, h, w), size=(size, size), mode="bilinear", align_corners=False)
            a = F.conv2d(a.index_select(3, self.sym), self.kx)
            a = F.conv2d(a.index_select(2, self.sym), self.ky)
            return a.amax(dim=(1, 2, 3))

        def forward(self, x):  # x: (1, 3, 224, 224) normalised
            n = self.net
            l1 = n.layer1(n.maxpool(n.relu(n.bn1(n.conv1(x)))))
            l2 = n.layer2(l1)
            l3 = n.layer3(l2)
            if self.kind == "patchcore":
                p2, p3 = self.pool(l2), self.pool(l3)
                p3 = F.interpolate(p3, size=p2.shape[-2:], mode="bilinear", align_corners=False)
                f = torch.cat([p2, p3], dim=1)
                if self.wide:
                    f = f.reshape(1, f.shape[1] // 2, 2, f.shape[2], f.shape[3]).mean(2)
                h, w = f.shape[-2:]
                q = f.flatten(2)[0].T  # (784, 384)
                d2 = (q * q).sum(1, keepdim=True) - 2.0 * (q @ self.bank_t) + self.bank_sq
                d = torch.sqrt(torch.clamp(d2.amin(1), min=0.0))
            else:
                up = lambda t: F.interpolate(t, size=l1.shape[-2:], mode="nearest")  # noqa: E731
                f = torch.cat([l1, up(l2), up(l3)], dim=1).index_select(1, self.idx)
                h, w = f.shape[-2:]
                diff = f.flatten(2)[0].T[:, None, :] - self.mean  # (P, 1, C)
                d = torch.sqrt(torch.clamp(((diff @ self.cov_inv) * diff).sum(-1)[:, 0], min=0.0))
            return self.blur_max(d, h, w)

    return Graph().eval()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _make(name: str, device, seed: int):
    """Edge configurations: the primary PatchCore with the companion study's bank size, and the detector
    variants of Section V in the configuration in which their recall was measured."""
    if name == "patchcore":
        return PatchCore(device, seed=seed, bank_size=MEMORY_BANK_SIZE)
    return make_detector(name, device, seed)


def export_workload(out_dir: Path, detectors=("patchcore", "padim"), category: str = "hazelnut", seed: int = 0,
                    n_frames: int = 120, mvtec_root: Path | None = None, device=None) -> Dict:
    """Export each detector's single graph; existing entries of out_dir/workload.json are kept."""
    import torch

    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = False  # exact FP32 so graph and reference scorer are comparable
    torch.backends.cuda.matmul.allow_tf32 = False
    root = mvtec_root or default_mvtec_root()
    train = sorted((root / category / "train" / "good").glob("*.png"))
    fit, cal, held = split_good(train, seed)
    fit, cal = fit + cal, held  # 80 % fit, 20 % threshold calibration (as the workstation benchmark)
    test = sorted((root / category / "test").glob("*/*.png"))
    rng = np.random.default_rng(seed)
    pick = [test[i] for i in rng.choice(len(test), size=min(n_frames, len(test)), replace=False)]
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "workload.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"detectors": {}}
    meta.update({"category": category, "seed": seed, "n_frames": len(pick), "n_fit_images": len(fit),
                 "frames": [str(p.relative_to(root)) for p in pick]})
    for name in detectors:
        det = _make(name, device, seed)
        size = det.img_size
        frames_file = "frames.npy" if size == IMG_SIZE else f"frames_{size}.npy"
        frames = load_images(pick, device, size)
        np.save(out_dir / frames_file, frames.cpu().numpy().astype(np.float32))
        det.fit(load_images(fit, device) if size == IMG_SIZE else fit, **({} if size == IMG_SIZE else {"batch": 8}))
        threshold = float(np.quantile(det.score(load_images(cal, device, size)), 0.99))
        ref = det.score(frames)
        graph = _graph_module(det).to(device)
        with torch.no_grad():
            got = np.array([float(graph(frames[i:i + 1])) for i in range(len(frames))])
        rel = float(np.max(np.abs(got - ref) / np.abs(ref)))
        if rel > 1e-3:
            raise RuntimeError(f"{name}: single-graph scores deviate from the reference scorer by {rel:.2e}")
        path = out_dir / f"{name}.onnx"
        torch.onnx.export(graph.cpu(), frames[:1].cpu(), str(path), input_names=["frame"], output_names=["score"],
                          opset_version=17, dynamo=False)
        meta["detectors"][name] = {
            "threshold": threshold, "state_size": det.state_size, "reference_scores": ref.tolist(), "img_size": size,
            "frames_file": frames_file, "graph_max_rel_error": rel, "onnx_sha256": _sha256(path),
            "onnx_bytes": path.stat().st_size, "backbone": det.arch, "backbone_sha256": weights_digest(det.net)}
        del graph, det
        if device.type == "cuda":
            torch.cuda.empty_cache()
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    return meta
