# VisualCLARITY — Code Walkthrough

This document explains every file in the project: what it does, why it exists, and how all the pieces connect.

---

## Big Picture

VisualCLARITY is a research project that trains an **interpretable bird-species classifier** on the CUB-200-2011 dataset.  
The key idea: instead of letting a neural network explain its predictions only with raw pixel gradients, we force it to first predict human-understandable **concepts** (e.g. "has blue wings", "has a red beak") and only then decide on the species. This family of models is called a **Concept Bottleneck Model (CBM)**.

The project builds and compares two variants:

| Model | How it reads concepts |
|---|---|
| **GlobalCBM** (baseline) | Averages all 256 patch tokens → one concept score |
| **VisualCLARITY** (proposed) | Each concept selects its own top-8 most relevant patches → more spatially specific |

The backbone is a frozen **DINOv2 ViT-B/14** (Vision Transformer), which turns a 224×224 image into 256 patch tokens of dimension 768.

---

## Directory Layout

```
VisualCLARITY/
├── clarity_vision/          ← Python package (library code)
│   ├── __init__.py
│   ├── data.py              ← dataset loading & patch-coord helpers
│   ├── models.py            ← neural network architectures & loss
│   ├── train_utils.py       ← cached-feature dataset & training utilities (not yet implemented)
│   └── evaluation.py        ← metrics: accuracy + pointing game (not yet implemented)
├── scripts/                 ← numbered pipeline steps (run in order)
│   ├── step1_smoke_test.py
│   ├── step2_verify_data.py
│   ├── step3_concepts.py
│   ├── step4_concept_part_map.py
│   ├── step5_cache_features.py   ← not yet implemented
│   ├── step6_train_baseline.py   ← not yet implemented
│   ├── step7_train_clarity.py    ← not yet implemented
│   └── step8_gate_eval.py        ← not yet implemented
├── configs/
│   └── gate.yaml            ← all hyperparameters in one place
├── outputs/                 ← generated files (JSON, cache, checkpoints)
├── dataset/                 ← CUB-200-2011 raw data
├── Makefile                 ← pipeline runner with integrity guards
└── requirements.txt
```

---

## `clarity_vision/` — The Library

### `__init__.py`
One line: the package docstring. Its only job is to make `clarity_vision` a Python package so other files can do `from clarity_vision.models import ...`.

---

### `clarity_vision/data.py`

**Purpose:** Everything related to reading the CUB-200-2011 dataset and converting pixel coordinates to DINOv2 patch indices.

#### File parsers (lines 50–138)

These are simple functions that read the raw CUB text files:

| Function | What it reads | Returns |
|---|---|---|
| `load_image_list` | `images.txt` | `{image_id → relative_path}` |
| `load_class_labels` | `image_class_labels.txt` | `{image_id → class_id}` (1-indexed) |
| `load_train_test_split` | `train_test_split.txt` | `{image_id → 1=train / 0=test}` |
| `load_bounding_boxes` | `bounding_boxes.txt` | `{image_id → (x, y, w, h)}` |
| `load_attribute_names` | `attributes.txt` | `{attr_id → name}` e.g. `"has_wing_color::blue"` |
| `load_class_attribute_matrix` | `class_attribute_labels_continuous.txt` | `float32 (200, 312)` — % of annotators who said attr is present for each class |
| `load_image_attribute_labels` | `image_attribute_labels.txt` | `int8 (N_images, 312)` — per-image binary labels; `-1` if annotator was uncertain |

#### `EVAL_ONLY` part-location helpers (lines 145–168)

`load_part_names_EVAL_ONLY` and `load_part_locs_EVAL_ONLY` read where each bird body part (beak, eye, wing…) is located in each photo.  
**These are only used for evaluation** (pointing-game metric). A Makefile guard enforces that no training script ever imports them — this prevents the model from "cheating" by seeing the answer during training.

#### Patch-coordinate math (lines 175–199)

DINOv2 splits each 224×224 image into a **16×16 grid of 14×14 pixel patches** (256 patches total). Two helper functions handle coordinate conversion:

- `pixel_to_patch(px, py, orig_w, orig_h)` — takes a pixel position in the original (possibly non-square) image and returns the patch `(row, col, flat_index)` it falls in after resizing to 224×224.
- `patch_to_pixel_center(patch_idx)` — the reverse: given a flat patch index (0–255), returns the center pixel in 224×224 space.

These are used in the pointing-game metric to check whether the model's predicted patch is close to the ground-truth part annotation.

#### `DINO_TRANSFORM` (line 206)

The standard ImageNet normalization transform applied to every image before feeding it to DINOv2. Resizes to 224×224, converts to tensor, normalizes with ImageNet mean/std.

#### `CUBDataset` (lines 218–280)

A PyTorch `Dataset` that yields `(image_tensor, class_label, concept_vector, image_id)` for each bird photo.

Key detail: CUB annotators sometimes marked an attribute as "uncertain". When `certainty < 3`, the per-image label is set to `-1`. `CUBDataset` handles this by **falling back to the class-level average** — if the annotator was unsure whether *this specific bird* has blue wings, the dataset uses the known average for its species.

---

### `clarity_vision/models.py`

**Purpose:** All neural network code — the backbone, both CBM models, the loss function, and a GradCAM helper.

#### `DINOv2Backbone` (lines 21–75)

Loads a pre-trained DINOv2 ViT-B/14 from a **local `.pth` file** (no internet access at runtime).

Important detail — **positional embedding interpolation** (lines 52–61): the official DINOv2 weights were trained at 518×518 resolution (37×37 = 1369 patches). We use 224×224 (16×16 = 256 patches). The code detects the mismatch and uses bicubic interpolation to resize the positional embeddings from 37×37 to 16×16 before loading the weights. Without this, the model would fail to load.

- `forward(x)` returns only the **patch tokens** — shape `(B, 256, 768)`. The CLS token (position 0) is dropped.
- `forward_with_cls(x)` returns both if needed.

#### `GlobalCBM` (lines 82–111) — Baseline

The control model. Architecture:

```
patch_tokens (B, 256, 768)
    → mean over all 256 patches → (B, 768)      [spatial info lost here]
    → linear → (B, C) concept scores
    → linear → (B, 200) class logits
```

Because it mean-pools all patches, it has no spatial awareness. A concept like "blue wing" gets the same score regardless of whether blue pixels are on the wing or the beak.

#### `VisualCLARITY` (lines 114–187) — Proposed Model

The main contribution. Each of the `C` concepts has its own learned **attention weight vector** `w_c` (shape `D=768`). For a given image:

1. **Compute attention scores** for all patches: `a_c(i) = w_c · token_i` → shape `(B, C, 256)`
2. **Select top-k patches** per concept (default k=8) → `(B, C, 8)`
3. **Project only those k patches** to concept space and average them → `(B, C)` concept scores
4. **Classify** from concept scores → `(B, 200)` logits

The diagonal trick at line 169: `concept_proj` is a `(768 → C)` linear layer, but for concept `c` we only want output `c`, not all C outputs. After computing `proj` of shape `(B, C, k, C)`, `.diagonal(dim1=1, dim2=3)` extracts the `c`-th output for the `c`-th concept efficiently without rewriting the projection.

`concept_patch_map(patch_tokens)` — eval-only method that returns a binary `(B, C, 256)` mask marking which patches each concept selected. Used in the pointing game.

#### `loss_fn` (lines 194–215)

Shared loss for both models:

```
total_loss = cross_entropy(logits, class_labels)
           + λ × binary_cross_entropy(concept_scores, concept_labels)
```

`λ = 0.01` (from `gate.yaml`). The concept loss is an auxiliary signal that encourages concept scores to be semantically meaningful even if the classifier could technically ignore them.

#### `gradcam_concept_maps` (lines 222–254)

A GradCAM-style visualization helper. For a given concept index, it backpropagates the concept score through the backbone and uses the gradient magnitude to weight patch tokens, producing a 16×16 heatmap. Works for both GlobalCBM and VisualCLARITY.

## `scripts/` — The Pipeline

Each script is a self-contained step. Run them in order via `make stepN`.

### `step1_smoke_test.py`

Loads the DINOv2 backbone from `weights/dinov2_vitb14_pretrain.pth` and passes a zero-valued dummy image through it. Asserts that the output shape is exactly `(1, 256, 768)`. If this fails, nothing else will work.

### `step2_verify_data.py`

Picks 3 bird images (from different species, each with ≥5 visible part annotations) and saves diagnostic PNGs to `outputs/overlays/`. Each PNG shows the bird at 224×224 with:
- A faint 16×16 patch grid.
- A colored dot at each annotated body part (beak, eye, wing, etc.).
- A colored rectangle highlighting which patch cell contains the annotation.

This is visual proof that `pixel_to_patch` is mathematically correct. If the dots are not in the highlighted cells, the coordinate math is wrong.

### `step3_concepts.py`

Filters the raw 312 CUB attributes down to a clean concept set (~112 concepts).

The filter: an attribute is kept only if more than 50% of annotators agreed it is present in **at least 10 but no more than 190** of the 200 bird species. This removes:
- Near-universal attributes (present in almost all species — not discriminative).
- Near-absent attributes (present in almost no species — too rare to train on).

Output: `outputs/concepts.json` — a list of concept objects, each with its 0-indexed ID, the original attribute name, parsed group and value, and how many species have it.

### `step4_concept_part_map.py`

Reads `concepts.json` and creates a heuristic mapping from each concept to the CUB body part(s) it should localize to.

The mapping is based on the **attribute group name** (the part before `::` in `"has_wing_color::blue"`):

| Attribute group | Mapped parts |
|---|---|
| `has_bill_color` | beak (part 2) |
| `has_wing_color` | left_wing (9), right_wing (13) |
| `has_eye_color` | left_eye (7), right_eye (11) |
| `has_tail_shape` | tail (14) |
| `has_size`, `has_shape` | (none — whole-bird attributes) |

Output: `outputs/concept_part_map.json` — used only in step 8 evaluation to run the pointing game.

**`REVIEW : outputs/concept_part_map.json`** 

## `configs/gate.yaml`

The single source of truth for all numbers. Both training scripts and evaluation read from here, so changing a value once updates the whole pipeline.

Key settings:

| Key | Value | Meaning |
|---|---|---|
| `top_k_patches` | 8 | VisualCLARITY selects 8 patches per concept |
| `concept_loss_weight` | 0.01 | λ in the combined loss |
| `seeds` | [1, 2, 3] | Three training runs for statistical stability |
| `epochs` | 100 | Training length |
| `lr` | 0.001 | Adam learning rate |
| `concept_min_class_prevalence` | 10 | Concept filter lower bound |
| `concept_max_class_prevalence` | 190 | Concept filter upper bound |
| `pointing_game_tolerance` | 1 | A hit is allowed ±1 patch from GT |

---

## `Makefile`

Two important checks run before every step:

**`make guard`** — greps training files for any reference to `part_locs`, `load_part`, or `keypoint`. If found, the build fails. This ensures the model cannot accidentally use part annotations during training (that would make the pointing-game evaluation meaningless).

**`make structure-check`** — verifies that all required source files exist.

Step dependencies are encoded as Makefile rules:
- Steps 1–4 can run locally (no GPU needed).
- Steps 5–8 require a GPU and depend on `concept_part_map.json` being frozen first.

---

## Data Flow Summary

```
CUB raw files
    → step3: 312 attributes → 108 concepts.json
    → step4: concepts.json → concept_part_map.json
    → step5-9: not yet implemented
```


**`Provided by : Meher Boulaabi`** 