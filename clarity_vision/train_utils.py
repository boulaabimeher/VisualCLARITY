"""Cached-feature dataset and shared training utilities.

Steps 6 and 7 use pre-extracted DINOv2 patch features stored as numpy .npy
files (written by step5) to avoid re-running the backbone each epoch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# Fraction of the cached TRAIN split held out for validation, and the seed that
# fixes which samples are held out. VAL_SPLIT_SEED is a constant independent of
# the model training seed, so the validation set is identical on every run and
# identical across both models — the only thing that differs between GlobalCBM
# and VisualCLARITY remains the patch-selection mechanism. The cached TEST split
# is never touched and stays the final held-out evaluation.
VAL_FRACTION = 0.15
VAL_SPLIT_SEED = 0


class CachedFeatureDataset(Dataset):
    """Dataset that loads pre-extracted DINOv2 patch features from disk.

    Expected files in cache_dir:
        {split}_features.npy   shape (N, 256, 768)  float32
        {split}_labels.npy     shape (N,)            int64
        {split}_concepts.npy   shape (N, C)          float32
        {split}_image_ids.npy  shape (N,)            int64
    """

    def __init__(self, cache_dir: str, split: str = "train"):
        root = Path(cache_dir)
        self.features = np.load(root / f"{split}_features.npy", mmap_mode="r")
        self.labels = np.load(root / f"{split}_labels.npy")
        self.concepts = np.load(root / f"{split}_concepts.npy")
        self.image_ids = np.load(root / f"{split}_image_ids.npy")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        features = torch.from_numpy(self.features[idx].copy())
        label = int(self.labels[idx])
        concepts = torch.from_numpy(self.concepts[idx].copy())
        image_id = int(self.image_ids[idx])
        return features, label, concepts, image_id


def make_loaders(
    cache_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) from cached features."""
    train_ds = CachedFeatureDataset(cache_dir, "train")
    test_ds = CachedFeatureDataset(cache_dir, "test")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def train_val_split_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic (train_idx, val_idx) over n cached-train samples.

    The permutation is seeded by VAL_SPLIT_SEED, so the held-out validation
    indices are reproducible on every construction and for every model. The
    last VAL_FRACTION of the permutation is the validation set.
    """
    rng = np.random.RandomState(VAL_SPLIT_SEED)
    perm = rng.permutation(n)
    n_val = int(round(n * VAL_FRACTION))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def make_loaders_with_val(
    cache_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader) from cached features.

    The validation set is a deterministic VAL_FRACTION slice carved from the
    cached TRAIN split (see train_val_split_indices); the cached TEST split is
    left untouched as the final held-out evaluation. All three loaders yield the
    same batch structure as make_loaders: (features, labels, concepts,
    image_ids).
    """
    train_full = CachedFeatureDataset(cache_dir, "train")
    test_ds = CachedFeatureDataset(cache_dir, "test")

    train_idx, val_idx = train_val_split_indices(len(train_full))
    train_ds = Subset(train_full, train_idx.tolist())
    val_ds = Subset(train_full, val_idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def num_concepts_of(loader: DataLoader) -> int:
    """Concept-vector width for a loader built by make_loaders[_with_val].

    Handles both a raw CachedFeatureDataset and a Subset wrapping one, so the
    concept count can be inferred regardless of whether the loader carries a
    validation split.
    """
    ds = loader.dataset
    if isinstance(ds, Subset):
        ds = ds.dataset
    return int(ds.concepts.shape[1])


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(state: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, device: str = "cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=True)
