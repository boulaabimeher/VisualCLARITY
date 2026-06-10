"""Step 5 — Cache DINOv2 patch features for the full dataset.

Runs DINOv2 over every training and test image once, saves patch token
arrays as .npy files under outputs/cache/. Training scripts load from
these cached arrays so the backbone is not re-run each epoch.

Usage:
    python scripts/step5_cache_features.py [--config configs/gate.yaml]
    python scripts/step5_cache_features.py --limit 4   # tiny smoke-test

IMPORTANT (Rule 1): this script reads ONLY images and image_ids.
It never reads part annotations or body-joint locations of any kind.

NOTE: the full run requires a GPU and is an overnight cluster job.
      Use --limit N to smoke-test on N images per split locally.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clarity_vision.data import CUBDataset
from clarity_vision.models import DINOv2Backbone


def cache_split(
    backbone: DINOv2Backbone,
    dataset: CUBDataset,
    split_name: str,
    cache_dir: Path,
    batch_size: int,
    device: torch.device,
    limit: int | None,
) -> None:
    """Extract and save features for one split (train or test).

    Skips images that are already cached (resumable).
    """
    done_file = cache_dir / f"{split_name}_done.flag"
    if done_file.exists() and limit is None:
        print(f"[step5] {split_name}: already cached, skipping.")
        return

    if limit is not None:
        indices = list(range(min(limit, len(dataset))))
        dataset = Subset(dataset, indices)

    n = len(dataset)
    print(f"[step5] {split_name}: caching {n} images ...")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    all_features = []
    all_labels = []
    all_concepts = []
    all_image_ids = []

    backbone.eval()
    with torch.no_grad():
        for batch_idx, (images, labels, concepts, image_ids) in enumerate(loader):
            images = images.to(device)
            patch_tokens = backbone(images)          # (B, 256, 768)
            all_features.append(patch_tokens.cpu().numpy().astype(np.float32))
            all_labels.append(labels.numpy().astype(np.int64))
            all_concepts.append(concepts.numpy().astype(np.float32))
            all_image_ids.append(
                image_ids.numpy().astype(np.int64)
                if isinstance(image_ids, torch.Tensor)
                else np.array(image_ids, dtype=np.int64)
            )
            done = min((batch_idx + 1) * loader.batch_size, n)
            print(f"  {done}/{n}", end="\r", flush=True)

    print()

    features_arr = np.concatenate(all_features, axis=0)    # (N, 256, 768)
    labels_arr = np.concatenate(all_labels, axis=0)        # (N,)
    concepts_arr = np.concatenate(all_concepts, axis=0)    # (N, C)
    ids_arr = np.concatenate(all_image_ids, axis=0)        # (N,)

    suffix = "" if limit is None else f"_limit{limit}"
    np.save(cache_dir / f"{split_name}{suffix}_features.npy", features_arr)
    np.save(cache_dir / f"{split_name}{suffix}_labels.npy", labels_arr)
    np.save(cache_dir / f"{split_name}{suffix}_concepts.npy", concepts_arr)
    np.save(cache_dir / f"{split_name}{suffix}_image_ids.npy", ids_arr)

    print(f"[step5] {split_name}: saved {features_arr.shape} features to {cache_dir}/")

    # Write completion flag only for the full (non-limited) run
    if limit is None:
        done_file.touch()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache DINOv2 patch features.")
    parser.add_argument("--config", default="configs/gate.yaml")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Cache only N images per split (smoke-test mode).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config))

    concept_map_path = ROOT / cfg["concept_part_map_json"]
    if not concept_map_path.exists():
        print(f"[step5] ERROR: {concept_map_path} not found. Run step4 first.")
        sys.exit(1)

    concepts_path = ROOT / cfg["concepts_json"]
    if not concepts_path.exists():
        print(f"[step5] ERROR: {concepts_path} not found. Run step3 first.")
        sys.exit(1)

    with open(concepts_path) as f:
        concepts_data = json.load(f)
    concept_ids = [c["attr_id_0indexed"] for c in concepts_data["concepts"]]
    print(f"[step5] Using {len(concept_ids)} concepts.")

    weights_path = ROOT / cfg["dinov2_weights_path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[step5] Loading backbone from {weights_path} on {device} ...")
    backbone = DINOv2Backbone(str(weights_path)).to(device)

    cache_dir = ROOT / cfg["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = str(ROOT / cfg["dataset_path"])

    for split in ("train", "test"):
        ds = CUBDataset(dataset_path, concept_ids=concept_ids, split=split)
        cache_split(
            backbone=backbone,
            dataset=ds,
            split_name=split,
            cache_dir=cache_dir,
            batch_size=args.batch_size,
            device=device,
            limit=args.limit,
        )

    print("[step5] Done.")


if __name__ == "__main__":
    main()
