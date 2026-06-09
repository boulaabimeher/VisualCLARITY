"""Step 1 — Environment smoke test.

Loads the DINOv2 ViT-B/14 backbone from a local checkpoint and asserts that
it produces patch tokens of shape exactly (1, 256, 768) for a single image.
This must pass before any other step runs.

Usage:
    python scripts/step1_smoke_test.py  
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clarity_vision.models import DINOv2Backbone


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config))
    weights_path = ROOT / cfg["dinov2_weights_path"]

    print(f"[step1] Loading DINOv2 from {weights_path} ...")
    backbone = DINOv2Backbone(str(weights_path))
    backbone.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = backbone.to(device)

    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        patch_tokens = backbone(dummy)

    shape = tuple(patch_tokens.shape)
    expected = (1, 256, 768)
    assert shape == expected, f"FAIL: expected {expected}, got {shape}"
    print(f"[step1] PASS — patch token shape: {shape}  (1=batch, 256=16x16 patches, 768=embed_dim)")


if __name__ == "__main__":
    main()
