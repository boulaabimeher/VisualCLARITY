# VisualCLARITY — Report v1 (baseline snapshot)

**Date:** 2026-06-22
**Purpose:** Record exactly what exists *now*, before the next round of changes,
so the starting point is unambiguous. This is the hand-maintained write-up.
`step9_metrics.py` produces only `outputs/metrics/metrics.json` and the figures;
the report itself lives here, by hand.

This report is deliberately critical. The goal is not to flatter the current
result but to mark precisely where it is strong, where it is thin, and what must
not break when we start fixing things.

---

## 1. What the contribution actually is (as implemented)

Stripped of framing, the code implements:

- **Backbone (frozen, both models):** DINOv2 ViT-B/14 → 256 patch tokens × 768-d,
  CLS dropped. Features cached to `.npy`; only the lightweight head trains.
- **GlobalCBM (baseline):** mean over 256 tokens → shared 2-layer MLP
  `Linear(768,384)→LN→ReLU→Dropout→Linear(384,108)` → sigmoid → `Linear(108,200)`.
- **VisualCLARITY (proposed):**
  - A single learnable matrix `attn_weights` of shape `(108, 768)`. Per concept,
    `a_c(i) = w_c · token_i` scores all 256 patches.
  - **Hard top-k (k=8)** per concept → binary `{0,1}` mask (non-differentiable).
  - **Plain average** of the 8 selected tokens → one 768-vector *per concept*.
  - The **same** shared MLP runs on each concept's pooled vector; the **diagonal**
    of the resulting `(B,108,108)` gives the 108 concept logits → sigmoid →
    `Linear(108,200)`.
  - The selector gets **zero gradient from `cls_loss`** (hard top-k blocks it) and
    is trained **only** by sparsity + continuity aux losses on the continuous
    `attn`. Enforced by the `step7` hard guard (`weights > 0`) and pinned both
    directions by `tests/test_gradient_flow.py`.

**Precise statement of the contribution:** *a per-concept hard top-k patch
selector, supervised entirely by sparsity + continuity priors (never by the class
loss or by part labels), feeding an otherwise-identical concept bottleneck — plus
a pointing-game protocol that tests whether the selected patches land on the
anatomically correct CUB part.*

### Notes where the implementation differs from the loose description

- **A second architectural difference exists** beyond "mean-pool vs top-k."
  GlobalCBM forces all 108 concepts to read from one shared pooled vector;
  VisualCLARITY gives each concept its **own** input vector (its 8 patches). That
  is a per-concept input-capacity increase independent of spatial grounding. So
  "the accuracy gap is attributable to spatial selection" does **not** strictly
  follow from this comparison — see §3c.
- **Pointing-game readout is weaker than intended.** `evaluation.py` does `argmax`
  on the **binary** mask, which returns the *first selected patch in row-major
  order*, not the highest-attention patch (`concept_patch_map` discards attention
  magnitude). So the metric scores one arbitrary patch of the 8, not the most
  salient. This both understates localization and is an attackable artifact —
  first fix on the roadmap.
- **Coordinate mapping is correct.** Preprocessing is `Resize((224,224))`
  (anisotropic squash) and `pixel_to_patch` scales by matching anisotropic
  factors — no aspect-ratio bug.

---

## 2. Results (3 seeds, 100 epochs, CUB-200 test)

### Accuracy

| Model | Top-1 (mean ± std) | Per-seed |
|---|---|---|
| GlobalCBM (baseline) | **0.7221 ± 0.0014** | 0.7209 / 0.7240 / 0.7213 |
| VisualCLARITY (proposed) | **0.8607 ± 0.0013** | 0.8605 / 0.8624 / 0.8592 |
| **Gap** | **+0.1386 (+13.9 pts)** | |

Std ≈ 0.13%, two orders of magnitude below the gap — the ordering is stable, not
seed noise.

### Localization (pointing game, VisualCLARITY only)

| Metric | Value |
|---|---|
| VisualCLARITY (mean ± std) | 0.1277 ± 0.0189 |
| Random-selection baseline | 0.0161 |
| Multiple above chance | ~7.9× |

Per-seed: 0.117 / 0.154 / 0.112 (relative std ~15%).

### Training dynamics

- Both models reach **train_acc = 1.0** (full memorization on frozen features).
- VisualCLARITY val: loss flat ~0.73, val_acc ~0.86–0.88, val_cls ~0.62.
- GlobalCBM val: loss **drifts up 1.10→1.15**, val_acc flat ~0.74, val_cls ~1.06.
- Concept BCE rises ~9 epochs, then falls, settling **~0.90–0.93 on both train and
  val** for both models.
- Sparsity loss drops 1.46 → ~0.20 and holds — the selector demonstrably
  concentrates.

### Strongest vs. weakest result

- **Strongest:** the accuracy gap — tight std, clean per-seed separation,
  reproducible.
- **Weakest / first to be attacked:** the pointing game — absolute 0.13 is low,
  the random floor (0.016) is a trivial comparator, the relative std is ~15%, and
  the readout artifact (§1) means we are not even scoring the best patch.

---

## 3. Honest weaknesses

**a) Pointing-game absolute is weak and undersold.** At `tolerance=1` (a 3×3 ≈
3.5%-area window), 0.13 means concepts mostly do *not* land on their nominal part.
"~8× chance" rides on a near-floor denominator. The argmax-of-binary artifact makes
it worse and arbitrary.

**b) The selector is never told where parts are.** Its only supervision is sparsity
+ continuity (generic priors on DINOv2 feature geometry). Nothing links concept *c*
to part *p*. Any localization is emergent, not learned toward anatomy — which is
exactly why 0.13 is low. This is a genuine tension between the mechanism and the
honesty claim. It cannot be "fixed" by supervising with parts (that breaks the
eval-only guard); it is a framing problem to be addressed by honest reporting and
stronger comparators, not by leaking part labels into training.

**c) The baseline is too weak; the accuracy gap is confounded.** Flat mean-pool is
the weakest possible CBM. Two confounds inflate +13.9: (i) selective vs averaged
pooling, and (ii) per-concept input diversity (§1). Neither is "grounding makes it
honest." A **soft-attention-pooling** baseline (same per-concept attention, soft
differentiable pool instead of hard top-k) would hold (ii) fixed and isolate the
hard-selection effect.

**d) Random selection is a strawman localization floor.** "Beats random 8×" is the
weakest comparator. The honest one is a localization method on the *baseline* — and
`gradcam_concept_maps` already exists and works on GlobalCBM.

**e) Both models memorize; the bottleneck is soft.** Concept BCE settles ~0.91
nats — concepts are weakly predicted, the concept weight is only 0.1, and the
classifier reads a 108-dim sigmoid vector that can route around poor concepts. The
"VisualCLARITY regularizes / overfits less" observation is real in the val curves
but its *mechanistic attribution* is speculative (both still fit train fully; VC's
lower val loss largely reflects that it simply classifies better).

**f) Novelty is incremental at the mechanism level.** As built, this is "CBM +
per-concept hard top-k patch selection + average pool." The distinctive moves are
the clean decoupling (selector trained solely by aux losses, provably zero cls
gradient) and the testable-honesty framing via the CUB pointing game. The framing
is the stronger hook — but it is currently undercut by the weak pointing numbers.

---

## 4. Invariants — what must NOT break in any future change

1. **Architecture-matched comparison.** GlobalCBM and VisualCLARITY share
   `_build_concept_mapper` and the classifier verbatim; only patch-reading
   differs. New baselines are **added**, never folded into GlobalCBM's control
   role.
2. **Selector trained solely by aux losses.** The hard top-k → zero-cls-gradient
   property, pinned by `test_gradient_flow.py` (both directions) and the `step7`
   `weights > 0` guard. Do **not** add a differentiable selection path *into
   VisualCLARITY itself* — a soft-attention model must be a **separate** model.
3. **Eval-only guard boundary (Rule 1).** Part annotations are read only in
   `evaluation.py` / `step8`; training scripts never import evaluation or read
   `part_locs`; `make guard` enforces it. No change may push part/localization
   signal into training.
4. **Deterministic val split.** `VAL_SPLIT_SEED=0`, `VAL_FRACTION=0.15`, identical
   across seeds and both models; test split untouched. Never make val depend on
   the model training seed.
5. **Coordinate-mapping consistency.** `Resize((224,224))` squash ↔ matching
   anisotropic `pixel_to_patch`. If preprocessing changes, the pointing-game
   mapping changes in lockstep.
6. **Identical frozen, cached features for both models.** Fairness of the accuracy
   comparison depends on it.

---

## 5. Roadmap (proposals, ranked by impact-to-risk) — NOT yet implemented

1. **Fix/augment the pointing-game readout** [highest — eval-only, cheap]. Score
   the true max-attention patch and additionally report **hit@k**. No retraining;
   touches no accuracy. Report both honestly (not metric-hacking).
2. **GradCAM-on-GlobalCBM as a localization comparator** [high — eval-only].
   Replaces the strawman random floor with a real comparator using existing
   `gradcam_concept_maps`.
3. **Soft-attention-pooling baseline** [high impact, controlled risk]. A separate
   third model that isolates hard spatial selection from per-concept pooling
   capacity. May match VC accuracy — that is the point of a control.
4. **Ablations on `top_k` and aux weights** [high for a paper, additive]. Plot
   accuracy *and* pointing game vs k.
5. **Report concept accuracy + per-concept localization breakdown** [cheap,
   additive]. Will expose the soft bottleneck — pair with honest framing.
6. **Stronger concept supervision** [medium — handle last]. Raising
   `concept_loss_weight` risks the clean accuracy gap; only if the
   "is-it-really-a-CBM" critique becomes central.

**Recommended order:** 1 → 2 → 3 → 4, with 5 alongside; 6 only if needed. Items 1
and 2 are pure upside (eval-only, no retraining). Item 3 is the scientifically
decisive control but must stay a *separate* model to preserve invariant #2.

---

## 6. File map (for whoever picks this up next)

- `clarity_vision/models.py` — `DINOv2Backbone`, `GlobalCBM`, `VisualCLARITY`,
  `loss_fn`, `gradcam_concept_maps`.
- `clarity_vision/evaluation.py` — `pointing_game` (EVAL-ONLY), `accuracy`,
  `concept_accuracy`.
- `clarity_vision/train_utils.py` — cached-feature dataset, deterministic
  train/val split, seeding, checkpoint I/O.
- `scripts/step6_train_baseline.py` / `step7_train_clarity.py` — training.
- `scripts/step8_gate_eval.py` — accuracy + pointing game (only script allowed to
  read part locs).
- `scripts/step9_metrics.py` — metrics JSON and figures.
- `tests/test_gradient_flow.py` — pins the selector-supervision invariant.
- `configs/gate.yaml` — all shared hyperparameters.
- `outputs/metrics/metrics.json` — the numbers quoted in §2.
