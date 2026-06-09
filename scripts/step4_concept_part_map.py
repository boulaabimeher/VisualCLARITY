"""Step 4 — Generate heuristic concept→part mapping.

Reads concepts.json (from step3) and maps each concept to the most likely
CUB body-part annotation(s) based on the attribute group name.

Writes:
    outputs/concept_part_map.json

Usage:
    python scripts/step4_concept_part_map.py
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# CUB part IDs (from parts/parts.txt)
# 1=back, 2=beak, 3=belly, 4=breast, 5=crown, 6=forehead,
# 7=left_eye, 8=left_leg, 9=left_wing, 10=nape, 11=right_eye,
# 12=right_leg, 13=right_wing, 14=tail, 15=throat
PART_ID = {
    "back": 1, "beak": 2, "belly": 3, "breast": 4, "crown": 5,
    "forehead": 6, "left_eye": 7, "left_leg": 8, "left_wing": 9,
    "nape": 10, "right_eye": 11, "right_leg": 12, "right_wing": 13,
    "tail": 14, "throat": 15,
}

# Heuristic mapping: attribute group -> list of (part_id, part_name) pairs
# Groups that span the whole bird or have no clear localisation get an empty list.
GROUP_TO_PARTS = {
    "has_bill_shape":        [(2, "beak")],
    "has_bill_length":       [(2, "beak")],
    "has_bill_color":        [(2, "beak")],
    "has_wing_color":        [(9, "left_wing"), (13, "right_wing")],
    "has_wing_shape":        [(9, "left_wing"), (13, "right_wing")],
    "has_wing_pattern":      [(9, "left_wing"), (13, "right_wing")],
    "has_upperparts_color":  [(1, "back")],
    "has_underparts_color":  [(3, "belly")],
    "has_breast_pattern":    [(4, "breast")],
    "has_breast_color":      [(4, "breast")],
    "has_back_color":        [(1, "back")],
    "has_back_pattern":      [(1, "back")],
    "has_tail_shape":        [(14, "tail")],
    "has_tail_pattern":      [(14, "tail")],
    "has_upper_tail_color":  [(14, "tail")],
    "has_under_tail_color":  [(14, "tail")],
    "has_head_pattern":      [(5, "crown"), (6, "forehead")],
    "has_eye_color":         [(7, "left_eye"), (11, "right_eye")],
    "has_forehead_color":    [(6, "forehead")],
    "has_nape_color":        [(10, "nape")],
    "has_belly_color":       [(3, "belly")],
    "has_belly_pattern":     [(3, "belly")],
    "has_throat_color":      [(15, "throat")],
    "has_crown_color":       [(5, "crown")],
    "has_leg_color":         [(8, "left_leg"), (12, "right_leg")],
    # These groups span the whole bird with no single localised part
    "has_primary_color":     [],
    "has_size":              [],
    "has_shape":             [],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config))
    concepts_path = ROOT / cfg["concepts_json"]
    out_path = ROOT / cfg["concept_part_map_json"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not concepts_path.exists():
        print(f"[step4] ERROR: {concepts_path} not found. Run step3 first.")
        sys.exit(1)

    with open(concepts_path) as f:
        concepts_data = json.load(f)

    concepts = concepts_data["concepts"]
    n_unmapped = 0
    mapping = []

    for c in concepts:
        group = c["group"]
        parts = GROUP_TO_PARTS.get(group, [])
        if not parts:
            n_unmapped += 1
        mapping.append({
            "concept_id": c["concept_id"],
            "concept_name": c["name"],
            "group": group,
            "part_ids": [p[0] for p in parts],
            "part_names": [p[1] for p in parts],
        })

    output = {
        "part_legend": {str(v): k for k, v in PART_ID.items()},
        "num_concepts": len(mapping),
        "concepts": mapping,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[step4] Written {out_path}")
    print(f"[step4] {len(mapping)} concepts total, {n_unmapped} with no specific part mapping.")


if __name__ == "__main__":
    main()
