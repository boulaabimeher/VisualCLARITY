"""Step 2 — Data verification: produce 3 overlay diagnostic PNGs.

For each of 3 selected images, renders the bird photo at 224x224 with:
  - The 16x16 DINOv2 patch grid drawn in light grey.
  - A red dot on every VISIBLE part annotation (beak, eye, wing, etc.).
  - The patch cell containing each annotation highlighted with a coloured border.

This is VISUAL PROOF that the pixel → patch coordinate math is correct.
Part annotations are read here for diagnostic purposes only — this script
is not a training component (Rule 1).

Usage:
    python scripts/step2_verify_data.py
Outputs:
    outputs/overlays/overlay_<image_id>.png  (3 files)
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clarity_vision.data import (
    cub_root,
    load_image_list,
    load_part_locs_EVAL_ONLY,
    load_part_names_EVAL_ONLY,
    load_train_test_split,
    pixel_to_patch,
)

# Part colours for visual distinction
PART_COLORS = [
    "#FF0000", "#FF6600", "#FFCC00", "#33CC33", "#0066FF",
    "#9900CC", "#FF66CC", "#00CCCC", "#996633", "#666666",
    "#FF3333", "#FF9933", "#99FF33", "#3399FF", "#CC33FF",
]


def draw_overlay(ax, img_224: np.ndarray, part_locs_image, part_names: dict,
                 orig_w: int, orig_h: int, img_size: int = 224, patch_size: int = 14):
    n = img_size // patch_size  # 16

    ax.imshow(img_224)

    # Draw patch grid
    for i in range(1, n):
        ax.axhline(i * patch_size - 0.5, color="white", lw=0.3, alpha=0.4)
        ax.axvline(i * patch_size - 0.5, color="white", lw=0.3, alpha=0.4)

    legend_handles = []
    for part_id, px, py, visible in part_locs_image:
        if not visible:
            continue
        # Scale pixel to 224x224
        sx = px * img_size / orig_w
        sy = py * img_size / orig_h
        color = PART_COLORS[(part_id - 1) % len(PART_COLORS)]

        # Red dot at the exact annotation point
        ax.plot(sx, sy, "o", color=color, markersize=6, markeredgecolor="black", markeredgewidth=0.5)

        # Highlight the patch cell that contains this annotation
        row, col, _ = pixel_to_patch(px, py, orig_w, orig_h, img_size, patch_size)
        rect = mpatches.Rectangle(
            (col * patch_size - 0.5, row * patch_size - 0.5),
            patch_size, patch_size,
            linewidth=1.5, edgecolor=color, facecolor="none", alpha=0.8,
        )
        ax.add_patch(rect)

        pname = part_names.get(part_id, f"part{part_id}")
        legend_handles.append(mpatches.Patch(color=color, label=pname))

    ax.set_xlim(0, img_size)
    ax.set_ylim(img_size, 0)
    ax.axis("off")
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right", fontsize=5,
                  framealpha=0.7, ncol=2)


def pick_representative_images(dataset_path: str, n: int = 3) -> list:
    """Pick n images from different species that have many visible parts."""
    img_list = load_image_list(dataset_path)
    split_map = load_train_test_split(dataset_path)
    part_locs = load_part_locs_EVAL_ONLY(dataset_path)

    # Score each image by number of visible parts; pick diverse species
    scored = []
    seen_classes = set()
    for img_id in sorted(img_list.keys()):
        if split_map[img_id] != 1:  # train split only
            continue
        cls = img_list[img_id].split("/")[0]
        locs = part_locs.get(img_id, [])
        n_visible = sum(1 for _, _, _, v in locs if v == 1)
        if n_visible >= 5:
            scored.append((n_visible, img_id, cls))

    scored.sort(reverse=True)
    chosen = []
    for _, img_id, cls in scored:
        if cls not in seen_classes:
            seen_classes.add(cls)
            chosen.append(img_id)
        if len(chosen) == n:
            break

    # Fallback: just take first n if not enough variety
    if len(chosen) < n:
        chosen = [img_id for _, img_id, _ in scored[:n]]
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate.yaml")
    parser.add_argument("--n", type=int, default=3, help="number of overlay PNGs")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config))
    dataset_path = str(ROOT / cfg["dataset_path"])
    out_dir = ROOT / cfg["output_dir"] / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_size = cfg.get("img_size", 224)
    patch_size = cfg.get("patch_size", 14)

    print("[step2] Loading CUB annotation files ...")
    img_list = load_image_list(dataset_path)
    part_names = load_part_names_EVAL_ONLY(dataset_path)
    part_locs = load_part_locs_EVAL_ONLY(dataset_path)
    images_dir = cub_root(dataset_path) / "images"

    image_ids = pick_representative_images(dataset_path, n=args.n)
    print(f"[step2] Selected images: {image_ids}")

    for img_id in image_ids:
        rel_path = img_list[img_id]
        img_path = images_dir / rel_path
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img_224 = np.array(img.resize((img_size, img_size), Image.BILINEAR))

        locs = part_locs.get(img_id, [])
        n_visible = sum(1 for _, _, _, v in locs if v == 1)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=120)
        fig.suptitle(
            f"Image {img_id}: {rel_path.split('/')[0]}\n"
            f"orig {orig_w}×{orig_h}  →  {img_size}×{img_size}  |  "
            f"{n_visible} visible parts",
            fontsize=6,
        )
        draw_overlay(ax, img_224, locs, part_names, orig_w, orig_h, img_size, patch_size)

        out_path = out_dir / f"overlay_{img_id:05d}.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"[step2] Saved {out_path}")

    print(f"[step2] Done — {args.n} overlay PNGs in {out_dir}")


if __name__ == "__main__":
    main()
