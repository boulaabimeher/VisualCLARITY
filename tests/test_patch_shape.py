"""Patch-shape regression test for the DINOv2 backbone.

Asserts that DINOv2Backbone.forward returns patch tokens of shape
(B, 256, 768) for a batch of B images at 224x224 (16x16 = 256 patches,
embed_dim 768), with the CLS token dropped.

The backbone __init__ requires the real local DINOv2 checkpoint, which may not
be present where tests run (a laptop without weights, or before the checkpoint
is synced from the cluster). We therefore skipif the weights file is absent,
rather than mocking: when the file IS present the test exercises the REAL load
path — checkpoint load + pos_embed interpolation + forward — and when it is
absent it skips cleanly instead of giving a false pass on random weights.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clarity_vision.models import DINOv2Backbone

WEIGHTS_PATH = ROOT / "weights" / "dinov2_vitb14_pretrain.pth"

B = 2
IMG_SIZE = 224
NUM_PATCHES = 256   # 16x16
EMBED_DIM = 768


@pytest.mark.skipif(
    not WEIGHTS_PATH.exists(),
    reason=f"DINOv2 weights not found at {WEIGHTS_PATH}; "
           "runs on the cluster / any host with the checkpoint, skips otherwise.",
)
def test_backbone_patch_shape():
    """forward() returns (B, 256, 768) for a (B, 3, 224, 224) batch."""
    backbone = DINOv2Backbone(str(WEIGHTS_PATH))
    images = torch.randn(B, 3, IMG_SIZE, IMG_SIZE)

    with torch.no_grad():
        patch_tokens = backbone(images)

    assert patch_tokens.shape == (B, NUM_PATCHES, EMBED_DIM), (
        f"expected ({B}, {NUM_PATCHES}, {EMBED_DIM}), got {tuple(patch_tokens.shape)}"
    )
