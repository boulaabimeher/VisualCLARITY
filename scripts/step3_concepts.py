"""Step 3 — Process CUB attributes into the ~112 visual concept set.

Follows Koh et al. (CBM, 2020): filter the 312 raw CUB attributes by
class-level prevalence to obtain a semantically clean set of ~112 concepts.
An attribute is kept if it is the dominant value (>50% of annotators) for
at least `min_class_prevalence` classes and at most `max_class_prevalence`
classes. This removes near-universal and near-absent attributes.

Writes:
    outputs/concepts.json — structured concept list with attr IDs and groups.

Usage:
    python scripts/step3_concepts.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clarity_vision.data import load_attribute_names, load_class_attribute_matrix


def parse_attribute(raw_name: str):
    """Split 'has_wing_color::blue' into group='has_wing_color', value='blue'."""
    if "::" in raw_name:
        group, value = raw_name.split("::", 1)
    else:
        group, value = raw_name, raw_name
    return group.strip(), value.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config))
    dataset_path = str(ROOT / cfg["dataset_path"])
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = ROOT / cfg["concepts_json"]

    min_cls = cfg.get("concept_min_class_prevalence", 10)
    max_cls = cfg.get("concept_max_class_prevalence", 190)

    print("[step3] Loading attributes ...")
    attr_names = load_attribute_names(dataset_path)         # {attr_id: name}  1-indexed
    class_attr = load_class_attribute_matrix(dataset_path) # (200, 312) float32

    # Binarise: attribute is "present" for a class if >50% of annotators said so
    binary = (class_attr > 50.0).astype(int)  # (200, 312)
    class_counts = binary.sum(axis=0)          # (312,) — how many classes have attr

    print(f"[step3] Total attributes: 312")
    print(f"[step3] Filtering: min_class_prevalence={min_cls}, max_class_prevalence={max_cls}")

    concepts = []
    concept_idx = 0
    for attr_id_1indexed, name in sorted(attr_names.items()):
        attr_id_0indexed = attr_id_1indexed - 1
        count = int(class_counts[attr_id_0indexed])
        if count < min_cls or count > max_cls:
            continue
        group, value = parse_attribute(name)
        concepts.append({
            "concept_id": concept_idx,
            "attr_id_1indexed": attr_id_1indexed,
            "attr_id_0indexed": attr_id_0indexed,
            "name": name,
            "group": group,
            "value": value,
            "n_classes_present": count,
        })
        concept_idx += 1

    print(f"[step3] Retained {len(concepts)} concepts after filtering.")

    # Sanity-check: print sample of group distribution
    from collections import Counter
    group_counts = Counter(c["group"] for c in concepts)
    print("[step3] Groups and concept counts:")
    for grp, cnt in sorted(group_counts.items()):
        print(f"         {grp:40s} {cnt:3d}")

    output = {
        "num_concepts": len(concepts),
        "filter": {
            "min_class_prevalence": min_cls,
            "max_class_prevalence": max_cls,
            "description": (
                "Attribute retained if it is the dominant value (>50% annotators) "
                f"for >= {min_cls} and <= {max_cls} out of 200 classes."
            ),
        },
        "concepts": concepts,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[step3] Written {out_path}")


if __name__ == "__main__":
    main()
