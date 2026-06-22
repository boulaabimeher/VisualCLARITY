"""Step 7b — visualize which patches VisualCLARITY selects per concept.

For a trained VisualCLARITY checkpoint, this renders, for a handful of test
images, the top-k patches each concept selects — overlaid on the actual bird
photo. It is a qualitative view of WHERE each concept looks.

It does NOT draw ground-truth part annotations. Part annotations are
evaluation-only and belong to the pointing game in step8. This script reuses only
the pixel<->patch coordinate math, never part-location data — so `make guard`
stays green.

Inputs : a trained checkpoint (outputs/checkpoints/visualclarity_seed{N}.pt),
         cached test features + image_ids, the raw CUB images, concepts.json.
Outputs: one PNG per (image, concept) pair into outputs/viz/, plus a small
         grid summary per image.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")  # headless / offline cluster — no display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# Repo-root bootstrap (same idiom step6/step7 use, since there's no editable install)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clarity_vision.models import VisualCLARITY
from clarity_vision.data import cub_root, load_image_list
from clarity_vision.train_utils import make_loaders


def patch_idx_to_box(patch_idx: int, img_size: int = 224, patch_size: int = 14):
    """Patch index -> (corner_x, corner_y) for a 224-space rectangle.

    Row-major convention: row = idx // 16, col = idx % 16 — matches the
    (B, C, N=256) mask layout and step2's overlay math. Corner offset by -0.5
    to sit on pixel boundaries, exactly as step2's draw_overlay does.
    """
    n_cols = img_size // patch_size  # 16
    row = patch_idx // n_cols
    col = patch_idx % n_cols
    corner_x = col * patch_size - 0.5
    corner_y = row * patch_size - 0.5
    return corner_x, corner_y


def load_model_from_ckpt(ckpt_path: str, device: str) -> VisualCLARITY:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = VisualCLARITY(
        num_concepts=ckpt["num_concepts"],
        num_classes=ckpt["num_classes"],
        embed_dim=ckpt["embed_dim"],
        top_k=ckpt["top_k"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def resolve_image(img_id: int, img_list: dict, images_dir: Path, img_size: int = 224):
    """image_id -> (img_224 ndarray, orig_w, orig_h). Opens the actual JPEG.

    image_ids from the cache are CUB 1-indexed and index load_image_list directly
    with no offset (confirmed in recon).
    """
    rel_path = img_list[int(img_id)]
    img_path = images_dir / rel_path
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    img_224 = np.array(img.resize((img_size, img_size), Image.BILINEAR))
    return img_224, orig_w, orig_h


def overlay_concept(ax, img_224, selected_patches, color, img_size=224, patch_size=14):
    """Draw the bird + the selected patches for ONE concept as outlined cells."""
    n = img_size // patch_size  # 16
    ax.imshow(img_224)
    # faint patch grid, like step2
    for i in range(1, n):
        ax.axhline(i * patch_size - 0.5, color="white", lw=0.3, alpha=0.3)
        ax.axvline(i * patch_size - 0.5, color="white", lw=0.3, alpha=0.3)
    # highlight each selected patch
    for patch_idx in selected_patches:
        cx, cy = patch_idx_to_box(int(patch_idx), img_size, patch_size)
        rect = mpatches.Rectangle(
            (cx, cy), patch_size, patch_size,
            linewidth=1.6, edgecolor=color, facecolor=color, alpha=0.30,
        )
        ax.add_patch(rect)
    ax.set_xlim(0, img_size)
    ax.set_ylim(img_size, 0)  # inverted y so origin is top-left — MUST match step2
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize VisualCLARITY patch selection.")
    parser.add_argument("--config", default="configs/gate.yaml")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to visualclarity_seed{N}.pt")
    parser.add_argument("--num-images", type=int, default=6,
                        help="How many test images to visualize.")
    parser.add_argument("--concepts", type=int, nargs="+", default=None,
                        help="Concept indices to show. Default: a few spread-out ones.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset_path = cfg["dataset_path"]
    img_list = load_image_list(dataset_path)
    images_dir = cub_root(dataset_path) / "images"

    # concept_idx -> human-readable name (concepts.json uses key "name")
    concepts_json = json.load(open(cfg["concepts_json"]))
    names = {c["concept_id"]: c["name"] for c in concepts_json["concepts"]}

    model = load_model_from_ckpt(args.checkpoint, args.device)

    # cached test features + image_ids (no backbone needed)
    _, test_loader = make_loaders(cfg["cache_dir"], batch_size=args.num_images, num_workers=0)
    features, labels, _concepts, image_ids = next(iter(test_loader))
    features = features.to(args.device)

    # binary (B, C, N) top-k mask — the selector's choices
    masks = model.concept_patch_map(features).cpu().numpy()  # (B, C, N)

    # default concept selection: a few spread across the range, skipping the
    # 6 part-less concepts (size/shape/color groups) which have no spatial target
    if args.concepts is not None:
        concept_ids = args.concepts
    else:
        concept_ids = [0, 4, 30, 60, 90, 107]

    out_dir = Path(cfg["output_dir"]) / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    palette = plt.get_cmap("tab10")

    for b in range(features.shape[0]):
        img_id = int(image_ids[b])
        img_224, ow, oh = resolve_image(img_id, img_list, images_dir)

        ncols = len(concept_ids)
        fig, axes = plt.subplots(1, ncols, figsize=(3 * ncols, 3.4))
        if ncols == 1:
            axes = [axes]
        for j, c in enumerate(concept_ids):
            selected = np.nonzero(masks[b, c])[0]  # patch indices for concept c
            overlay_concept(axes[j], img_224, selected, palette(j % 10))
            axes[j].set_title(f"{names.get(c, f'concept_{c}')}", fontsize=7)
        fig.suptitle(f"image_id {img_id} — VisualCLARITY selected patches", fontsize=9)
        fig.tight_layout()
        out_path = out_dir / f"viz_img{img_id}.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_path}", flush=True)

    print(f"\nDone. {features.shape[0]} images x {len(concept_ids)} concepts -> {out_dir}",
          flush=True)


if __name__ == "__main__":
    main()
