"""
Measured detector operating points: two detectors on two public industrial datasets.

Detectors (ImageNet backbones, image size 224):
  PatchCore -- as in the companion alert-policy study: layer2 + layer3 features average-pooled
               3x3 and concatenated (28x28x384), greedy k-center coreset of 10 % of the patches,
               1-NN distance to the memory bank. Two higher-compute variants test whether compute buys
               recall: a WideResNet-50 backbone (patchcore_wrn50, 768-d patches) and 448x448 input
               (patchcore_448; a seeded 300k-patch subset enters the coreset selection);
  PaDiM     -- layer1-3 features at 56x56 (448 channels, 100 kept by a seeded random choice),
               one Gaussian per patch position, Mahalanobis distance.
Image score = max of the Gaussian-smoothed (sigma 4) anomaly map at 224x224.

Datasets: MVTec AD (15 categories) and VisA (12 categories, official 1-class split).
For every detector x dataset x category and split seed, the good training images are
shuffled and split into
  fit (60 %)          -> memory bank / Gaussian statistics
  calibration (20 %)  -> operating thresholds (e.g. the 99th percentile of good scores)
  held-out (20 %)     -> extra nominal evaluation images, together with the good test images
The defective test images give per-part recall. All image scores are written to
data/raw/detector_scores/<detector>/<dataset>/ (scores only; images are not redistributed).
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import numpy as np

from src.paths import RAW_DATA_DIR, REPO_ROOT

SCORE_DIR = RAW_DATA_DIR / "detector_scores"
DATASETS = {
    "mvtec": ("bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill",
              "screw", "tile", "toothbrush", "transistor", "wood", "zipper"),
    "visa": ("candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1", "pcb2",
             "pcb3", "pcb4", "pipe_fryum"),
}
DETECTORS = ("patchcore", "padim", "patchcore_wrn50", "patchcore_448")
HIRES_SIZE, HIRES_MAX_PATCHES = 448, 300_000
PRIMARY_DETECTOR = "patchcore"
SEEDS = (0, 1, 2)
IMG_SIZE = 224
CORESET_RATIO = 0.10
PROJECTION_DIM = 128
PADIM_DIM = 100
PADIM_EPS = 0.01
SPLIT = (0.6, 0.2, 0.2)  # fit / calibration / held-out good
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Unit(NamedTuple):
    detector: str
    dataset: str
    category: str

    @property
    def label(self) -> str:
        return f"{self.detector}/{self.dataset}/{self.category}"


def units(detector: str | None = None, dataset: str | None = None) -> List[Unit]:
    return [Unit(det, ds, cat) for det in DETECTORS if detector in (None, det)
            for ds, cats in DATASETS.items() if dataset in (None, ds) for cat in cats]


def score_path(unit: Unit, seed: int) -> Path:
    return SCORE_DIR / unit.detector / unit.dataset / f"{unit.category}_seed{seed}.npz"


def default_mvtec_root() -> Path:
    return REPO_ROOT.parent / "industrial-defect-anomaly-benchmark" / "data" / "mvtec_ad"


def default_root(dataset: str) -> Path:
    return {"mvtec": REPO_ROOT.parent / "datasets" / "mvtec" / "full",
            "visa": REPO_ROOT.parent / "datasets" / "visa" / "VisA"}[dataset]


def dataset_paths(dataset: str, root: Path, category: str) -> Tuple[List[Path], List[Path], List[Path]]:
    """(good training, good test, defective test) image paths of one category."""
    if dataset == "mvtec":
        cdir = root / category
        train_good = sorted((cdir / "train" / "good").glob("*.png"))
        test_good = sorted((cdir / "test" / "good").glob("*.png"))
        test_def = sorted(p for p in (cdir / "test").glob("*/*.png") if p.parent.name != "good")
        return train_good, test_good, test_def
    if dataset == "visa":
        out = {("train", "normal"): [], ("test", "normal"): [], ("test", "anomaly"): []}
        with open(root / "split_csv" / "1cls.csv", newline="") as fh:
            for row in csv.DictReader(fh):
                if row["object"] == category:
                    out[(row["split"], row["label"])].append(root / row["image"])
        return tuple(sorted(out[k]) for k in (("train", "normal"), ("test", "normal"), ("test", "anomaly")))
    raise KeyError(dataset)


def load_images(paths: List[Path], device, size: int = IMG_SIZE):
    import torch
    from PIL import Image

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    out = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
        t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        out.append((t - mean) / std)
    return torch.stack(out).to(device)


BACKBONES = ("resnet18", "wide_resnet50_2")
WRN50_URL = "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth"  # torchvision IMAGENET1K_V1


def wide_resnet50_2():
    """WideResNet-50-2 with torchvision's parameter names and the official ImageNet weights. Defined here
    because old torchvision releases lack it; the checkpoint is verified against the hash prefix in its name."""
    import hashlib as _h

    import torch
    import torch.nn as nn

    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, inp, planes, stride=1, downsample=None):
            super().__init__()
            width = planes * 2  # width_per_group = 128
            self.conv1 = nn.Conv2d(inp, width, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(width)
            self.conv2 = nn.Conv2d(width, width, 3, stride, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(width)
            self.conv3 = nn.Conv2d(width, planes * 4, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * 4)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample

        def forward(self, x):
            idt = x if self.downsample is None else self.downsample(x)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.relu(self.bn2(self.conv2(out)))
            return self.relu(self.bn3(self.conv3(out)) + idt)

    class WRN(nn.Module):
        def __init__(self):
            super().__init__()
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, 2, 1)
            self.layer1 = self._layer(64, 3)
            self.layer2 = self._layer(128, 4, 2)
            self.layer3 = self._layer(256, 6, 2)
            self.layer4 = self._layer(512, 3, 2)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(2048, 1000)

        def _layer(self, planes, blocks, stride=1):
            ds = None
            if stride != 1 or self.inplanes != planes * 4:
                ds = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, 1, stride, bias=False), nn.BatchNorm2d(planes * 4))
            layers = [Bottleneck(self.inplanes, planes, stride, ds)]
            self.inplanes = planes * 4
            layers += [Bottleneck(self.inplanes, planes) for _ in range(1, blocks)]
            return nn.Sequential(*layers)

    net = WRN()
    path = torch.hub.get_dir() + "/checkpoints/" + WRN50_URL.rsplit("/", 1)[1]
    import os
    if not os.path.exists(path):
        torch.hub.download_url_to_file(WRN50_URL, path, progress=False)
    with open(path, "rb") as fh:
        if not _h.sha256(fh.read()).hexdigest().startswith("95faca4d"):
            raise RuntimeError("WideResNet-50-2 checkpoint hash mismatch")
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return net


class Backbone:
    def __init__(self, device, seed: int = 0, arch: str = "resnet18", img_size: int = IMG_SIZE):
        import torch
        import torch.nn as nn
        import torchvision.models as tvm

        if arch not in BACKBONES:
            raise ValueError(f"unknown backbone '{arch}'")
        self.torch = torch
        self.device = device
        self.seed = seed
        self.arch = arch
        self.img_size = img_size
        net = tvm.resnet18(pretrained=True) if arch == "resnet18" else wide_resnet50_2()  # ImageNet weights
        self.net = net.to(device).eval()
        self.pool = nn.AvgPool2d(3, 1, 1)

    def anomaly_maps_to_scores(self, d, b: int, h: int, w: int) -> List[float]:
        import torch.nn.functional as F
        from scipy.ndimage import gaussian_filter

        amap = F.interpolate(d.reshape(b, 1, h, w), size=(self.img_size, self.img_size), mode="bilinear",
                             align_corners=False)[:, 0].cpu().numpy()
        return [float(gaussian_filter(a, sigma=4).max()) for a in amap]


class PatchCore(Backbone):
    """PatchCore with ResNet-18 (384-d patches, as in the companion study) or WideResNet-50 (the backbone of the
    original paper; its 1536 layer-2/3 channels are averaged in adjacent pairs to 768-d patches)."""

    def __init__(self, device, seed: int = 0, coreset_ratio: float = CORESET_RATIO, bank_size: int | None = None,
                 arch: str = "resnet18", img_size: int = IMG_SIZE, max_patches: int | None = None):
        super().__init__(device, seed, arch, img_size)
        self.max_patches = max_patches
        self.coreset_ratio = coreset_ratio
        self.bank_size = bank_size
        self.bank = None

    def features(self, x):
        torch = self.torch
        import torch.nn.functional as F

        with torch.no_grad():
            n = self.net
            l2 = n.layer2(n.layer1(n.maxpool(n.relu(n.bn1(n.conv1(x))))))
            l3 = n.layer3(l2)
            p2, p3 = self.pool(l2), self.pool(l3)
            p3 = F.interpolate(p3, size=p2.shape[-2:], mode="bilinear", align_corners=False)
            f = torch.cat([p2, p3], dim=1)  # (B, 384, 28, 28) for ResNet-18
            if self.arch == "wide_resnet50_2":
                b, c, h, w = f.shape
                f = f.reshape(b, c // 2, 2, h, w).mean(2)  # (B, 768, 28, 28)
        return f

    def _coreset(self, patches):
        torch = self.torch
        n = patches.shape[0]
        m = self.bank_size if self.bank_size is not None else max(1, int(n * self.coreset_ratio))
        if m >= n:
            return patches
        g = torch.Generator(device=self.device).manual_seed(self.seed)
        proj, _ = torch.linalg.qr(torch.randn(patches.shape[1], PROJECTION_DIM, device=self.device, generator=g))
        z = patches @ proj
        zsq = (z * z).sum(1)
        start = int(torch.argmax(torch.norm(z - z.mean(0, keepdim=True), dim=1)))
        sel = [start]
        mind = torch.clamp(zsq + zsq[start] - 2 * (z @ z[start]), min=0)
        while len(sel) < m:
            k = min(50, m - len(sel))
            _, idx = torch.topk(mind, k)
            sel.extend(idx.tolist())
            q = z[idx]
            d = torch.clamp(zsq[:, None] + (q * q).sum(1)[None, :] - 2 * (z @ q.T), min=0)
            mind = torch.minimum(mind, d.min(1).values)
        return patches[sel[:m]]

    def fit(self, images, batch: int = 32):
        """images: a tensor, or a list of paths loaded batch by batch (high resolutions). With max_patches, a
        seeded random subset of the patches enters the coreset selection (approximate greedy coreset)."""
        torch = self.torch
        feats = []
        for i in range(0, len(images), batch):
            x = images[i:i + batch]
            x = load_images(x, self.device, self.img_size) if isinstance(x, list) else x
            f = self.features(x)
            feats.append(f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]))
        patches = torch.cat(feats)
        if self.max_patches is not None and len(patches) > self.max_patches:
            g = torch.Generator(device=self.device).manual_seed(self.seed)
            patches = patches[torch.randperm(len(patches), generator=g, device=self.device)[:self.max_patches]]
        self.bank = self._coreset(patches)

    def score(self, images, batch: int = 16) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(images), batch):
                f = self.features(images[i:i + batch])
                b, c, h, w = f.shape
                q = f.permute(0, 2, 3, 1).reshape(-1, c)
                d = torch.cat([torch.cdist(q[j:j + 4096], self.bank).min(1).values for j in range(0, len(q), 4096)])
                out.extend(self.anomaly_maps_to_scores(d, b, h, w))
        return np.asarray(out)

    @property
    def state_size(self) -> int:
        return int(self.bank.shape[0])


class PaDiM(Backbone):
    """Defard et al. (2021) with ResNet-18: layer1-3 at 56x56, 100 random channels, per-position Gaussian."""

    def __init__(self, device, seed: int = 0, dim: int = PADIM_DIM):
        super().__init__(device, seed)
        g = self.torch.Generator().manual_seed(seed)
        self.idx = self.torch.randperm(448, generator=g)[:dim].to(device)
        self.mean = self.cov_inv = None

    def features(self, x):
        torch = self.torch
        import torch.nn.functional as F

        with torch.no_grad():
            n = self.net
            l1 = n.layer1(n.maxpool(n.relu(n.bn1(n.conv1(x)))))
            l2 = n.layer2(l1)
            l3 = n.layer3(l2)
            up = lambda t: F.interpolate(t, size=l1.shape[-2:], mode="nearest")  # noqa: E731
            f = torch.cat([l1, up(l2), up(l3)], dim=1)  # (B, 448, 56, 56)
        return f.index_select(1, self.idx)

    def fit(self, images, batch: int = 32):
        torch = self.torch
        emb = torch.cat([self.features(images[i:i + batch]) for i in range(0, len(images), batch)])
        n, c, h, w = emb.shape
        x = emb.reshape(n, c, h * w).permute(2, 0, 1).double()  # (P, N, C)
        self.mean = x.mean(1)
        xc = x - self.mean[:, None, :]
        cov = xc.transpose(1, 2) @ xc / (n - 1) + PADIM_EPS * torch.eye(c, device=self.device, dtype=x.dtype)
        self.cov_inv = torch.linalg.inv(cov).float()
        self.mean = self.mean.float()
        self.hw = (h, w)

    def score(self, images, batch: int = 16) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(images), batch):
                f = self.features(images[i:i + batch])
                b, c, h, w = f.shape
                diff = f.reshape(b, c, h * w).permute(2, 0, 1) - self.mean[:, None, :]  # (P, B, C)
                m = torch.sqrt(torch.clamp(((diff @ self.cov_inv) * diff).sum(-1), min=0))  # (P, B)
                out.extend(self.anomaly_maps_to_scores(m.T.contiguous(), b, h, w))
        return np.asarray(out)

    @property
    def state_size(self) -> int:
        return int(self.mean.shape[0])


def make_detector(name: str, device, seed: int):
    if name == "patchcore_wrn50":
        return PatchCore(device, seed=seed, arch="wide_resnet50_2")
    if name == "patchcore_448":
        return PatchCore(device, seed=seed, img_size=HIRES_SIZE, max_patches=HIRES_MAX_PATCHES)
    return {"patchcore": PatchCore, "padim": PaDiM}[name](device, seed=seed)


def weights_digest(net) -> str:
    """SHA-256 over the loaded backbone parameters (independent of the checkpoint file name)."""
    h = hashlib.sha256()
    for name, t in sorted(net.state_dict().items()):
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def split_good(paths: List[Path], seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(paths))
    n_fit = int(round(SPLIT[0] * len(paths)))
    n_cal = int(round(SPLIT[1] * len(paths)))
    pick = lambda ii: [paths[i] for i in sorted(ii)]  # noqa: E731
    return pick(idx[:n_fit]), pick(idx[n_fit:n_fit + n_cal]), pick(idx[n_fit + n_cal:])


def evaluate_unit(unit: Unit, root: Path, seed: int, device) -> Dict:
    train_good, test_good, test_def = dataset_paths(unit.dataset, root, unit.category)
    if not train_good or not test_good or not test_def:
        raise FileNotFoundError(f"{unit.label}: missing images under {root}")
    fit, cal, held = split_good(train_good, seed)
    model = make_detector(unit.detector, device, seed)
    if model.img_size == IMG_SIZE:
        model.fit(load_images(fit, device))
    else:  # high resolution: stream the fit images in small batches
        model.fit(fit, batch=8)
    roles = {"calibration": cal, "heldout_good": held, "test_good": test_good, "test_defect": test_def}
    scores, role, dtype, files = [], [], [], []
    for r, ps in roles.items():
        for j in range(0, len(ps), 64):  # bounded GPU memory for large categories
            scores.extend(model.score(load_images(ps[j:j + 64], device, model.img_size)))
        role.extend([r] * len(ps))
        dtype.extend([p.parent.name for p in ps])
        files.extend([str(p.relative_to(root)) for p in ps])
    return {"scores": np.asarray(scores), "role": np.asarray(role), "defect_type": np.asarray(dtype),
            "file": np.asarray(files), "state_size": model.state_size, "n_fit": len(fit)}


def run_detector_eval(roots: Dict[str, Path] | None = None, out_dir: Path = SCORE_DIR,
                      detectors=DETECTORS, datasets=tuple(DATASETS), skip_existing: bool = False) -> Path:
    import torch
    import torchvision

    roots = {ds: (roots or {}).get(ds) or default_root(ds) for ds in datasets}
    for ds, root in roots.items():
        if not root.exists():
            raise FileNotFoundError(f"dataset '{ds}' not found at {root}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    meta_path = out_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"runs": {}}
    meta.update({"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "device": str(device),
                 "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                 "torch": torch.__version__, "torchvision": getattr(torchvision, "__version__", "unknown"),
                 "python": platform.python_version(), "img_size": IMG_SIZE, "coreset_ratio": CORESET_RATIO,
                 "projection_dim": PROJECTION_DIM, "padim_dim": PADIM_DIM, "padim_eps": PADIM_EPS,
                 "split_fit_cal_heldout": SPLIT, "seeds": SEEDS})
    meta["resnet18_weights_sha256"] = weights_digest(Backbone(device).net)
    for det in detectors:
        for ds in datasets:
            for unit in units(det, ds):
                for seed in SEEDS:
                    if skip_existing and score_path(unit, seed).exists() and f"{unit.label}/seed{seed}" in meta["runs"]:
                        continue
                    r = evaluate_unit(unit, roots[ds], seed, device)
                    path = score_path(unit, seed)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, scores=r["scores"], role=r["role"], defect_type=r["defect_type"],
                                        image=r["file"])
                    meta["runs"][f"{unit.label}/seed{seed}"] = {"state_size": r["state_size"], "n_fit": r["n_fit"],
                                                                "n_scores": int(len(r["scores"]))}
                    print(f"{unit.label} seed {seed}: state {r['state_size']}, {len(r['scores'])} scored", flush=True)
                    torch.cuda.empty_cache()
                meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return out_dir


if __name__ == "__main__":
    run_detector_eval()
