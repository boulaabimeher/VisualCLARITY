"""Step 8 — gate evaluation: accuracy + pointing game.

Produces the two headline numbers:
  * Top-1 accuracy for BOTH models (GlobalCBM and VisualCLARITY).
  * Pointing-game localization for VisualCLARITY — do each concept's selected
    patches land on the correct bird part?

Why there is a random baseline and not a GlobalCBM comparison for localization:
GlobalCBM mean-pools all patches — it has NO per-concept spatial selection, so
there is nothing to "point" with. The pointing game therefore cannot compare
VisualCLARITY against GlobalCBM. The comparison it can support is VisualCLARITY vs
RANDOM patch selection: does the learned selector localize concepts better than
picking top_k patches at random? This script reports both.

Eval-only boundary: this is the ONE script permitted to read part annotations. It
imports load_part_locs_EVAL_ONLY from clarity_vision.data and pointing_game from
clarity_vision.evaluation. All part-location code must stay in this file or in
clarity_vision.evaluation, never in a training script; `make guard` enforces that
boundary by scanning the training files. The Makefile runs this as `step8: guard`
so the guard still runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clarity_vision.models import GlobalCBM, VisualCLARITY
from clarity_vision.evaluation import pointing_game, accuracy
from clarity_vision.data import cub_root, load_image_list, load_part_locs_EVAL_ONLY
from clarity_vision.train_utils import make_loaders, load_checkpoint


def rebuild_model(ckpt: dict, device: str):
    """Rebuild the right model from a checkpoint dict. Branches on ckpt['model']
    because GlobalCBM has no top_k and VisualCLARITY does."""
    if ckpt["model"] == "VisualCLARITY":
        model = VisualCLARITY(
            num_concepts=ckpt["num_concepts"],
            num_classes=ckpt["num_classes"],
            embed_dim=ckpt["embed_dim"],
            top_k=ckpt["top_k"],
            # grid_size / dropout_rate not saved -> model defaults (16, 0.1).
            # Fine for eval: concept_patch_map uses neither.
        ).to(device)
    elif ckpt["model"] == "GlobalCBM":
        model = GlobalCBM(
            num_concepts=ckpt["num_concepts"],
            num_classes=ckpt["num_classes"],
            embed_dim=ckpt["embed_dim"],
        ).to(device)
    else:
        raise ValueError(f"Unknown model in checkpoint: {ckpt['model']}")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def eval_accuracy(model, loader, device: str) -> float:
    """Top-1 over the whole loader. Handles 2-tuple (GlobalCBM) and 3-tuple
    (VisualCLARITY) model outputs by indexing [0] for logits."""
    correct = 0
    total = 0
    for features, labels, _concepts, _image_ids in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)[0]
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def build_img_sizes(image_ids, img_list, images_dir):
    """Open each JPEG to get original (W, H) — no cached sizes exist. Positionally
    aligned with image_ids (and therefore with the mask batch)."""
    sizes = []
    for iid in image_ids:
        with Image.open(images_dir / img_list[int(iid)]) as im:
            sizes.append(im.size)  # (W, H)
    return sizes


def random_mask_like(mask: torch.Tensor, top_k: int, seed: int = 0) -> torch.Tensor:
    """Random-selection baseline: for each (image, concept) pick top_k random
    patches. Same sparsity as the real selector, no learning — the honest 'chance'
    floor for the pointing game."""
    B, C, N = mask.shape
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(B, C, N)
    for b in range(B):
        for c in range(C):
            idx = torch.randperm(N, generator=g)[:top_k]
            out[b, c, idx] = 1.0
    return out


def run_pointing_game(model, loader, cfg, device, max_batches=None):
    """Run the pointing game for a VisualCLARITY model over the loader.
    Returns (overall_acc, per_concept, random_overall)."""
    dataset_path = cfg["dataset_path"]
    img_list = load_image_list(dataset_path)
    images_dir = cub_root(dataset_path) / "images"

    # Flatten the nested concept_part_map.json -> {concept_id: [part_id,...]}
    cpm_data = json.load(open(ROOT / cfg["concept_part_map_json"]))
    concept_part_map = {e["concept_id"]: e["part_ids"] for e in cpm_data["concepts"]}

    # Part locations (EVAL ONLY).
    part_locs = load_part_locs_EVAL_ONLY(dataset_path)

    top_k = model.top_k

    # Accumulate across batches by collecting masks + ids + sizes, then one call.
    all_masks = []
    all_ids = []
    all_sizes = []
    for b_idx, (features, _labels, _concepts, image_ids) in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        features = features.to(device)
        masks = model.concept_patch_map(features).cpu()  # (B, C, N) {0,1}
        ids = [int(x) for x in image_ids]
        sizes = build_img_sizes(ids, img_list, images_dir)
        all_masks.append(masks)
        all_ids.extend(ids)
        all_sizes.extend(sizes)

    masks = torch.cat(all_masks, dim=0)  # (Total, C, N)

    overall, per_concept = pointing_game(
        concept_patch_masks=masks,
        concept_part_map=concept_part_map,
        part_locs=part_locs,
        image_ids=all_ids,
        img_sizes=all_sizes,
        img_size=cfg.get("img_size", 224),
        patch_size=cfg.get("patch_size", 14),
        tolerance=cfg.get("pointing_game_tolerance", 1),
    )

    # Random-selection baseline on the SAME images.
    rand_masks = random_mask_like(masks, top_k, seed=0)
    rand_overall, _ = pointing_game(
        concept_patch_masks=rand_masks,
        concept_part_map=concept_part_map,
        part_locs=part_locs,
        image_ids=all_ids,
        img_sizes=all_sizes,
        img_size=cfg.get("img_size", 224),
        patch_size=cfg.get("patch_size", 14),
        tolerance=cfg.get("pointing_game_tolerance", 1),
    )

    return overall, per_concept, rand_overall


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 8 — gate evaluation.")
    parser.add_argument("--config", default="configs/gate.yaml")
    parser.add_argument("--clarity-ckpt", default=None,
                        help="VisualCLARITY checkpoint (.pt). Default: seed1.")
    parser.add_argument("--global-ckpt", default=None,
                        help="GlobalCBM checkpoint (.pt). Default: seed1.")
    parser.add_argument("--smoke", action="store_true",
                        help="Pointing game on a few batches only — fast wiring check.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config))
    ckpt_dir = ROOT / cfg["checkpoint_dir"]
    _, test_loader = make_loaders(cfg["cache_dir"], batch_size=cfg["batch_size"], num_workers=0)

    clarity_path = args.clarity_ckpt or str(ckpt_dir / "visualclarity_seed1.pt")
    global_path = args.global_ckpt or str(ckpt_dir / "globalcbm_seed1.pt")
    max_batches = 2 if args.smoke else None

    print("=" * 60)
    print("STEP 8 — GATE EVALUATION")
    print("=" * 60)

    # --- Accuracy: both models ---
    if Path(global_path).exists():
        gck = load_checkpoint(global_path, device=args.device)
        gmodel = rebuild_model(gck, args.device)
        gacc = eval_accuracy(gmodel, test_loader, args.device)
        print(f"GlobalCBM      top-1 accuracy: {gacc:.4f}")
    else:
        print(f"GlobalCBM checkpoint not found ({global_path}) — skipping.")

    cck = load_checkpoint(clarity_path, device=args.device)
    cmodel = rebuild_model(cck, args.device)
    cacc = eval_accuracy(cmodel, test_loader, args.device)
    print(f"VisualCLARITY  top-1 accuracy: {cacc:.4f}")

    # --- Pointing game: VisualCLARITY vs random baseline ---
    print("-" * 60)
    print("Pointing game (VisualCLARITY only — GlobalCBM has no spatial selection)")
    overall, per_concept, rand_overall = run_pointing_game(
        cmodel, test_loader, cfg, args.device, max_batches=max_batches,
    )
    print(f"  VisualCLARITY pointing-game acc: {overall:.4f}")
    print(f"  Random-selection baseline:       {rand_overall:.4f}")
    print(f"  Lift over random:                {overall - rand_overall:+.4f}")

    # A few best/worst localized concepts (drop NaN concepts)
    valid = {c: v for c, v in per_concept.items() if v == v}  # v==v filters NaN
    if valid:
        ranked = sorted(valid.items(), key=lambda kv: kv[1], reverse=True)
        print("  Best-localized concepts:",
              [(c, round(v, 3)) for c, v in ranked[:5]])
        print("  Worst-localized concepts:",
              [(c, round(v, 3)) for c, v in ranked[-5:]])

    print("=" * 60)
    if args.smoke:
        print("(smoke run — pointing game on a few batches only, not the full test set)")


if __name__ == "__main__":
    main()
