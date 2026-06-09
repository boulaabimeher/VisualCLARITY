"""CUB-200-2011 data loading, concept processing, and patch-coord helpers.

Dataset layout (set dataset_path in gate.yaml to the outer dir):
  {dataset_path}/
    attributes.txt                          <- 312 attribute name definitions
    CUB_200_2011/
      images.txt                            <- image_id -> relative path
      image_class_labels.txt               <- image_id -> class_id (1-indexed)
      train_test_split.txt                 <- image_id -> 1=train / 0=test
      bounding_boxes.txt                   <- image_id -> x y w h
      classes.txt
      images/
      attributes/
        image_attribute_labels.txt         <- image_id attr_id is_present certainty time
        class_attribute_labels_continuous.txt  <- 200x312 float matrix
      parts/
        parts.txt                          <- part_id -> name  (EVAL ONLY)
        part_locs.txt                      <- image_id part_id x y visible  (EVAL ONLY)

IMPORTANT: part_locs / part annotations are NEVER read by training code (Rule 1).
They are exposed here only through eval-gated helpers clearly labelled EVAL_ONLY.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def cub_root(dataset_path: str) -> Path:
    """Return the inner CUB_200_2011/ directory given the outer dataset_path."""
    return Path(dataset_path) / "CUB_200_2011"


# ---------------------------------------------------------------------------
# Raw file parsers
# ---------------------------------------------------------------------------

def load_image_list(dataset_path: str) -> Dict[int, str]:
    """Return {image_id: relative_image_path} from images.txt."""
    root = cub_root(dataset_path)
    result: Dict[int, str] = {}
    with open(root / "images.txt") as f:
        for line in f:
            img_id, rel_path = line.strip().split()
            result[int(img_id)] = rel_path
    return result


def load_class_labels(dataset_path: str) -> Dict[int, int]:
    """Return {image_id: class_id} (class_id is 1-indexed)."""
    root = cub_root(dataset_path)
    result: Dict[int, int] = {}
    with open(root / "image_class_labels.txt") as f:
        for line in f:
            img_id, cls = line.strip().split()
            result[int(img_id)] = int(cls)
    return result


def load_train_test_split(dataset_path: str) -> Dict[int, int]:
    """Return {image_id: 1=train / 0=test}."""
    root = cub_root(dataset_path)
    result: Dict[int, int] = {}
    with open(root / "train_test_split.txt") as f:
        for line in f:
            img_id, split = line.strip().split()
            result[int(img_id)] = int(split)
    return result


def load_bounding_boxes(dataset_path: str) -> Dict[int, Tuple[float, float, float, float]]:
    """Return {image_id: (x, y, width, height)}."""
    root = cub_root(dataset_path)
    result: Dict[int, Tuple[float, float, float, float]] = {}
    with open(root / "bounding_boxes.txt") as f:
        for line in f:
            parts = line.strip().split()
            img_id = int(parts[0])
            result[img_id] = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
    return result


def load_attribute_names(dataset_path: str) -> Dict[int, str]:
    """Return {attr_id: attr_name} from the top-level attributes.txt."""
    attr_file = Path(dataset_path) / "attributes.txt"
    result: Dict[int, str] = {}
    with open(attr_file) as f:
        for line in f:
            attr_id, name = line.strip().split(" ", 1)
            result[int(attr_id)] = name
    return result


def load_class_attribute_matrix(dataset_path: str) -> np.ndarray:
    """Return float32 array of shape (200, 312): % annotators saying attr present per class."""
    path = cub_root(dataset_path) / "attributes" / "class_attribute_labels_continuous.txt"
    rows = []
    with open(path) as f:
        for line in f:
            rows.append([float(v) for v in line.strip().split()])
    return np.array(rows, dtype=np.float32)


def load_image_attribute_labels(dataset_path: str) -> np.ndarray:
    """Return int8 array of shape (N_images, 312) with binarised attribute labels.

    Uses certainty >= 3 (probably / definitely) and is_present == 1.
    Returns -1 where certainty < 3 (uncertain annotation).
    """
    root = cub_root(dataset_path)
    img_list = load_image_list(dataset_path)
    n_images = len(img_list)
    n_attrs = 312
    labels = np.full((n_images, n_attrs), -1, dtype=np.int8)

    path = root / "attributes" / "image_attribute_labels.txt"
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            img_id = int(parts[0]) - 1       # 0-indexed
            attr_id = int(parts[1]) - 1      # 0-indexed
            is_present = int(parts[2])
            certainty = int(parts[3])
            if certainty >= 3:
                labels[img_id, attr_id] = is_present
    return labels


# ---------------------------------------------------------------------------
# EVAL_ONLY: part-location helpers  (never imported by training code)
# ---------------------------------------------------------------------------

def load_part_names_EVAL_ONLY(dataset_path: str) -> Dict[int, str]:
    """Return {part_id: part_name}.  EVAL_ONLY — do not import in training scripts."""
    root = cub_root(dataset_path)
    result: Dict[int, str] = {}
    with open(root / "parts" / "parts.txt") as f:
        for line in f:
            part_id, name = line.strip().split(" ", 1)
            result[int(part_id)] = name
    return result


def load_part_locs_EVAL_ONLY(dataset_path: str) -> Dict[int, List[Tuple[int, float, float, int]]]:
    """Return {image_id: [(part_id, x, y, visible), ...]}.  EVAL_ONLY."""
    root = cub_root(dataset_path)
    result: Dict[int, List[Tuple[int, float, float, int]]] = {}
    with open(root / "parts" / "part_locs.txt") as f:
        for line in f:
            parts = line.strip().split()
            img_id = int(parts[0])
            part_id = int(parts[1])
            x, y = float(parts[2]), float(parts[3])
            visible = int(parts[4])
            result.setdefault(img_id, []).append((part_id, x, y, visible))
    return result


# ---------------------------------------------------------------------------
# Patch-coordinate helpers
# ---------------------------------------------------------------------------

def pixel_to_patch(px: float, py: float, orig_w: int, orig_h: int,
                   img_size: int = 224, patch_size: int = 14) -> Tuple[int, int, int]:
    """Map a pixel coordinate in the original image to a DINOv2 patch cell.

    Returns (patch_row, patch_col, patch_index) where patch_index = row*n_cols + col.
    patch_index is the position in the flattened (N=256) token sequence.
    """
    n_cols = img_size // patch_size   # 16
    # Scale pixel to 224x224 coordinate space
    sx = px * img_size / orig_w
    sy = py * img_size / orig_h
    col = min(int(sx / patch_size), n_cols - 1)
    row = min(int(sy / patch_size), n_cols - 1)
    return row, col, row * n_cols + col


def patch_to_pixel_center(patch_idx: int, img_size: int = 224,
                           patch_size: int = 14) -> Tuple[int, int]:
    """Return the center pixel (x, y) of a patch in the 224x224 coordinate space."""
    n_cols = img_size // patch_size
    row = patch_idx // n_cols
    col = patch_idx % n_cols
    cx = col * patch_size + patch_size // 2
    cy = row * patch_size + patch_size // 2
    return cx, cy


# ---------------------------------------------------------------------------
# Standard image transform for DINOv2
# ---------------------------------------------------------------------------

DINO_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class CUBDataset(Dataset):
    """CUB-200-2011 image dataset with concept labels.

    Returns (image_tensor, class_label_0indexed, concept_vector, image_id).
    concept_vector is a float32 tensor of length num_concepts with values in {0, 1}.
    Images with uncertain annotations use the class-level mean as fallback.
    """

    def __init__(
        self,
        dataset_path: str,
        concept_ids: List[int],
        split: str = "train",
        transform=None,
    ):
        """
        Args:
            dataset_path: path to outer CUB dir (containing attributes.txt).
            concept_ids: list of 0-indexed attribute IDs to use as concepts.
            split: "train" or "test".
            transform: torchvision transform; defaults to DINO_TRANSFORM.
        """
        self.dataset_path = dataset_path
        self.concept_ids = concept_ids
        self.transform = transform or DINO_TRANSFORM
        self.images_dir = cub_root(dataset_path) / "images"

        img_list = load_image_list(dataset_path)
        class_labels = load_class_labels(dataset_path)
        split_map = load_train_test_split(dataset_path)

        target_split = 1 if split == "train" else 0
        self.samples: List[Tuple[int, str, int]] = [
            (img_id, rel_path, class_labels[img_id] - 1)   # 0-indexed class
            for img_id, rel_path in sorted(img_list.items())
            if split_map[img_id] == target_split
        ]

        # Per-image attribute labels (N_images x 312), -1 = uncertain
        img_attrs = load_image_attribute_labels(dataset_path)
        # Class-level fallback (200 x 312), binarised at 50%
        class_attrs = load_class_attribute_matrix(dataset_path)
        class_attrs_binary = (class_attrs > 50.0).astype(np.float32)

        # Build concept matrix for this split
        self.concepts = np.zeros((len(self.samples), len(concept_ids)), dtype=np.float32)
        for i, (img_id, _, cls_id) in enumerate(self.samples):
            for j, attr_id in enumerate(concept_ids):
                label = img_attrs[img_id - 1, attr_id]
                if label >= 0:
                    self.concepts[i, j] = float(label)
                else:
                    self.concepts[i, j] = class_attrs_binary[cls_id, attr_id]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_id, rel_path, cls = self.samples[idx]
        img = Image.open(self.images_dir / rel_path).convert("RGB")
        img_tensor = self.transform(img)
        concept_vec = torch.from_numpy(self.concepts[idx])
        return img_tensor, cls, concept_vec, img_id
