"""Pointing-game localization metric and accuracy helpers.

EVAL_ONLY: this module may freely read part_locs — it is never imported
by training scripts (step6, step7).  The Makefile guard (make guard) checks
that training scripts do not import clarity_vision.evaluation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy from (B, K) logits and (B,) integer labels."""
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def concept_accuracy(concept_scores: torch.Tensor, concept_labels: torch.Tensor,
                     threshold: float = 0.0) -> float:
    """Binary accuracy of concept predictions (threshold on raw logit)."""
    preds = (concept_scores > threshold).float()
    return (preds == concept_labels).float().mean().item()


# ---------------------------------------------------------------------------
# Pointing game
# ---------------------------------------------------------------------------

def pointing_game(
    concept_patch_masks: torch.Tensor,
    concept_part_map: Dict[int, List[int]],
    part_locs: Dict[int, List[Tuple[int, float, float, int]]],
    image_ids: List[int],
    img_sizes: List[Tuple[int, int]],
    img_size: int = 224,
    patch_size: int = 14,
    tolerance: int = 1,
) -> Tuple[float, Dict[int, float]]:
    """Compute pointing-game accuracy across a batch.

    For each concept c and each image i:
      1. Find the patch with the highest attention (argmax of the mask).
      2. Convert that patch to pixel center coordinates in the 224x224 space.
      3. Check if the ground-truth part annotation for the expected part falls
         within `tolerance` patches of the selected patch.

    Args:
        concept_patch_masks: (B, C, N) float tensor, 1 on selected patches.
        concept_part_map: {concept_idx: [part_id, ...]} (0-indexed concept ids).
        part_locs: {image_id: [(part_id, x, y, visible), ...]} — EVAL_ONLY data.
        image_ids: list of length B with original CUB image IDs (1-indexed).
        img_sizes: list of (width, height) of original images before resize.
        img_size: DINOv2 input resolution (224).
        patch_size: ViT patch stride (14).
        tolerance: hit if selected patch is within this many patches of GT.
    Returns:
        (overall_accuracy, per_concept_accuracy dict)
    """
    from clarity_vision.data import pixel_to_patch, patch_to_pixel_center

    B, C, N = concept_patch_masks.shape
    n_cols = img_size // patch_size
    hits: Dict[int, int] = {c: 0 for c in range(C)}
    totals: Dict[int, int] = {c: 0 for c in range(C)}

    masks_np = concept_patch_masks.cpu().numpy()

    for b_idx in range(B):
        img_id = image_ids[b_idx]
        w, h = img_sizes[b_idx]
        locs = {pid: (px, py) for pid, px, py, vis in part_locs.get(img_id, []) if vis == 1}

        for c_idx in range(C):
            part_ids = concept_part_map.get(c_idx, [])
            if not part_ids:
                continue

            # Collect visible GT annotations for this concept's parts
            gt_patches = []
            for pid in part_ids:
                if pid in locs:
                    px, py = locs[pid]
                    pr, pc, pidx = pixel_to_patch(px, py, int(w), int(h), img_size, patch_size)
                    gt_patches.append((pr, pc))

            if not gt_patches:
                continue

            totals[c_idx] += 1

            # Predicted patch: argmax of mask (all 1s tie-broken by first occurrence)
            pred_patch = int(masks_np[b_idx, c_idx].argmax())
            pred_row = pred_patch // n_cols
            pred_col = pred_patch % n_cols

            # Hit if within tolerance patches of ANY ground-truth part annotation
            hit = any(
                abs(pred_row - gt_row) <= tolerance and abs(pred_col - gt_col) <= tolerance
                for gt_row, gt_col in gt_patches
            )
            if hit:
                hits[c_idx] += 1

    per_concept = {
        c: hits[c] / totals[c] if totals[c] > 0 else float("nan")
        for c in range(C)
    }
    valid = [v for v in per_concept.values() if not np.isnan(v)]
    overall = float(np.mean(valid)) if valid else float("nan")
    return overall, per_concept
