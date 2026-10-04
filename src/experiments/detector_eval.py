"""
Measured detector operating points: PatchCore on MVTec AD.

For every category and split seed the good training images are shuffled and split into
  fit (60 %)          -> memory bank (greedy k-center coreset, 10 % of patches)
  calibration (20 %)  -> operating thresholds (e.g. the 99th percentile of good scores)
  held-out (20 %)     -> extra nominal evaluation images, together with test/good
The test set's defective images give per-part recall. All image scores are written to
data/raw/detector_scores/ (scores only; MVTec images are not redistributed).

Detector (as in the companion alert-policy study): ResNet-18 with ImageNet weights,
layer2 + layer3 features average-pooled 3x3 and concatenated (28x28x384 at 224x224 input),
1-NN distance to the memory bank, image score = max of the Gaussian-smoothed (sigma 4)
anomaly map.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.paths import RAW_DATA_DIR, REPO_ROOT

SCORE_DIR = RAW_DATA_DIR / "detector_scores"
CATEGORIES = ("bottle", "cable", "carpet", "grid", "hazelnut", "leather", "metal_nut")
SEEDS = (0, 1, 2)
IMG_SIZE = 224
CORESET_RATIO = 0.10
PROJECTION_DIM = 128
SPLIT = (0.6, 0.2, 0.2)  # fit / calibration / held-out good
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_mvtec_root() -> Path:
    return REPO_ROOT.parent / "industrial-defect-anomaly-benchmark" / "data" / "mvtec_ad"


def load_images(paths: List[Path], device):
    import torch
    from PIL import Image

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    out = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        out.append((t - mean) / std)
    return torch.stack(out).to(device)


class PatchCore:
    def __init__(self, device, seed: int = 0, coreset_ratio: float = CORESET_RATIO, bank_size: int | None = None):
        import torch
        import torch.nn as nn
        from torchvision.models import resnet18

        self.torch = torch
        self.device = device
        self.seed = seed
        self.coreset_ratio = coreset_ratio
        self.bank_size = bank_size
        net = resnet18(pretrained=True)  # ImageNet weights (torchvision cache); API valid across torchvision versions
        self.net = net.to(device).eval()
        self.pool = nn.AvgPool2d(3, 1, 1)
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
            f = torch.cat([p2, p3], dim=1)  # (B, 384, 28, 28)
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
        feats = []
        for i in range(0, len(images), batch):
            f = self.features(images[i:i + batch])
            feats.append(f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]))
        self.bank = self._coreset(self.torch.cat(feats))

    def score(self, images, batch: int = 16) -> np.ndarray:
        import torch.nn.functional as F
        from scipy.ndimage import gaussian_filter

        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(images), batch):
                f = self.features(images[i:i + batch])
                b, c, h, w = f.shape
                q = f.permute(0, 2, 3, 1).reshape(-1, c)
                d = torch.cat([torch.cdist(q[j:j + 4096], self.bank).min(1).values for j in range(0, len(q), 4096)])
                amap = F.interpolate(d.reshape(b, 1, h, w), size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                                     align_corners=False)[:, 0].cpu().numpy()
                out.extend(float(gaussian_filter(a, sigma=4).max()) for a in amap)
        return np.asarray(out)


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


def evaluate_category(root: Path, category: str, seed: int, device) -> Dict:
    cdir = root / category
    train_good = sorted((cdir / "train" / "good").glob("*.png"))
    fit, cal, held = split_good(train_good, seed)
    test_good = sorted((cdir / "test" / "good").glob("*.png"))
    test_def = sorted(p for p in (cdir / "test").glob("*/*.png") if p.parent.name != "good")
    model = PatchCore(device, seed=seed)
    model.fit(load_images(fit, device))
    roles = {"calibration": cal, "heldout_good": held, "test_good": test_good, "test_defect": test_def}
    scores, role, dtype, files = [], [], [], []
    for r, ps in roles.items():
        s = model.score(load_images(ps, device))
        scores.extend(s)
        role.extend([r] * len(ps))
        dtype.extend([p.parent.name for p in ps])
        files.extend([f"{category}/{p.parent.parent.name}/{p.parent.name}/{p.name}" for p in ps])
    return {"scores": np.asarray(scores), "role": np.asarray(role), "defect_type": np.asarray(dtype),
            "file": np.asarray(files), "bank_size": int(model.bank.shape[0]), "n_fit": len(fit)}


def run_detector_eval(root: Path | None = None, out_dir: Path = SCORE_DIR) -> Path:
    import torch
    import torchvision

    root = root or default_mvtec_root()
    if not root.exists():
        raise FileNotFoundError(f"MVTec AD not found at {root}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "device": str(device),
            "torch": torch.__version__, "torchvision": getattr(torchvision, "__version__", "unknown"),
            "python": platform.python_version(), "img_size": IMG_SIZE, "coreset_ratio": CORESET_RATIO,
            "projection_dim": PROJECTION_DIM, "split_fit_cal_heldout": SPLIT, "seeds": SEEDS, "runs": {}}
    meta["resnet18_weights_sha256"] = weights_digest(PatchCore(device).net)
    for cat in CATEGORIES:
        for seed in SEEDS:
            r = evaluate_category(root, cat, seed, device)
            np.savez_compressed(out_dir / f"{cat}_seed{seed}.npz", scores=r["scores"], role=r["role"],
                                defect_type=r["defect_type"], image=r["file"])
            meta["runs"][f"{cat}_seed{seed}"] = {"bank_size": r["bank_size"], "n_fit": r["n_fit"],
                                                 "n_scores": int(len(r["scores"]))}
            print(f"{cat} seed {seed}: bank {r['bank_size']}, {len(r['scores'])} scored")
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    return out_dir


if __name__ == "__main__":
    run_detector_eval()
