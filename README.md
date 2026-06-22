# VisualCLARITY

A spatially-grounded concept bottleneck model for interpretable bird-species
classification on CUB-200-2011, built on a frozen DINOv2 backbone.

> **Status:** baseline snapshot (v1). The mechanism works and the pipeline is
> reproducible end-to-end, but the scientific framing is still being hardened —
> see `report/report_v1.md` for an honest assessment of what is and isn't yet
> defensible, and the roadmap at the bottom of that file for what changes next.

---

## 1. The idea

Concept bottleneck models (CBMs) classify in two stages: first predict
human-interpretable concepts (e.g. *has_bill_shape::dagger*,
*has_wing_color::brown*), then predict the class from those concepts. The promise
is interpretability — you can read *why* the model decided, and intervene on
concepts.

The problem this project attacks: **a concept can be right for the wrong reason.**
A model can output a high score for "red crown" while actually reading from the
background, a watermark, or any spurious feature that happens to correlate with
the label. Accuracy alone cannot detect this — the model classifies well either
way, and the interpretability becomes a fiction.

**VisualCLARITY's premise:** *a concept is only honest if it reads from the right
place.* If "red crown" is a real concept, the image patches it depends on should
sit on the bird's crown. So we make each concept read from a small,
spatially-selected set of image patches — and then we **measure**, with a
pointing-game metric against CUB's ground-truth part annotations, whether those
patches actually land on the correct anatomical part.

The name comes from porting the text-based **CLARITY** rationale model to vision:
instead of selecting rationale *tokens* from text, VisualCLARITY selects rationale
*patches* from an image.

---

## 2. The final goal

Turn interpretability from a *claim* into a *test*. Concretely, the project aims
to establish — with controlled, reproducible experiments — that:

1. **Spatial grounding is measurable.** Each concept's spatial origin is
   well-defined (the patches it selects), so it can be scored against ground-truth
   bird parts. We can say *how often* a concept reads from its anatomically
   correct location, not just assert that it is interpretable.

2. **Grounding is the cause of the improvement, not a confound.** The headline
   comparison must isolate spatial selection from incidental advantages (pooling
   capacity, per-concept input diversity). This requires controls stronger than a
   flat mean-pool baseline — a soft-attention-pooling control and a
   localization comparator on the baseline (e.g. GradCAM) — so that any win is
   attributable to *grounding*, not to "anything beats averaging."

3. **The localization claim is strong in absolute terms**, not only relative to a
   trivial random floor. The pointing-game readout and protocol must be measured
   honestly (best/most-salient patch, hit@k) and reported with its limitations.

The end state is a clean, defensible scientific contribution: *spatial grounding
makes a concept bottleneck testably honest*, supported by an architecture-matched
accuracy comparison and a localization metric that survives a skeptical reviewer.

---

## 3. The mechanism (what the model does)

Each concept independently, for one image's 256 patch tokens:

1. **Score every patch.** A learned per-concept weight vector scores all 256
   patches.
2. **Select the top-k.** Keep only the k highest-scoring patches (k = 8). Hard,
   non-differentiable selection.
3. **Read the concept from only those k patches.** The concept's value is computed
   from the selected patches alone; the other 248 contribute nothing to it.

Each concept selects its **own** patches — "bill shape" can attend to the beak
while "wing color" attends to the wing. This per-concept selection is exactly what
the pointing game evaluates.

**Baseline — GlobalCBM (the un-grounded control).** Mean-pools **all** 256 patches
into one vector before predicting concepts. It has no notion of *where* a concept
comes from. (This baseline is intentionally minimal in v1 and will be strengthened
— see the roadmap.)

### How the selector is trained — and why it matters

The top-k selection is non-differentiable, so the classification loss gives the
selector **zero gradient**. If nothing else trained it, the selector would stay at
its random initialization and the model would classify from randomly-selected
patches — a *silent failure* that still produces plausible accuracy while the
spatial-grounding contribution does nothing.

The selector is therefore trained **entirely by auxiliary losses** on the
continuous attention scores (before selection):

- **Sparsity loss** — concentrate each concept's attention on a few patches.
- **Spatial-continuity loss** — encourage selected patches to form a coherent
  region on the 16×16 grid.

These are the **sole supervision** for the selector. A regression test
(`tests/test_gradient_flow.py`) asserts both directions: non-zero selector
gradient when the aux weights are positive, and **exactly zero** when they are
zero — proving no gradient leaks through the classification path.

---

## 4. Architecture

**Backbone (frozen).** DINOv2 ViT-B/14 loaded from a local checkpoint (no network
at runtime). An image → 256 patch tokens (16×16 grid) of dimension 768. Weights
are never updated; features are cached once and reused. Positional embeddings are
bicubic-interpolated from the checkpoint's 37×37 grid down to 16×16.

**Two models, sharing an identical concept head and classifier** — only the
patch-reading mechanism differs (a hard rule that keeps the comparison clean):

```
GlobalCBM (baseline)
  patch tokens (256, 768)
    -> mean-pool all patches -> (768,)
    -> concept head (MLP) -> concept scores (108,)
    -> sigmoid -> classifier -> class logits (200,)

VisualCLARITY (proposed)
  patch tokens (256, 768)
    -> per-concept attention  a_c(i) = w_c . token_i   (108, 256)
    -> hard top-k (k=8) per concept -> binary mask      (108, 256)
    -> plain average of each concept's 8 patches        (108, 768)
    -> concept head (MLP) -> concept scores (108,)
    -> sigmoid -> classifier -> class logits (200,)
```

**Concept head (shared).** `Linear(768, 384) -> LayerNorm -> ReLU -> Dropout ->
Linear(384, 108)`. Concept scores are kept as **logits** for the BCE loss; the
sigmoid is applied only on the path to the classifier.

**Loss.**
```
total = cls_loss
      + concept_loss_weight * concept_BCE     (concepts vs ground-truth concept labels)
      + sparsity_weight     * sparsity_loss   (selector supervision)
      + continuity_weight   * continuity_loss (selector supervision)
```
GlobalCBM uses the same loss with the selector terms set to zero.

---

## 5. Pipeline

Part annotations are **evaluation-only** — never touched during training, enforced
by a `make guard` check that fails the build if any training file references
part-location data.

1. **step1–4** — backbone smoke test, coordinate-math verification, concept
   construction (108 concepts from CUB attributes), concept→part map (heuristic).
2. **step5** — cache DINOv2 features for train/test to `.npy` (~8.7 GB,
   git-ignored).
3. **step6** — train GlobalCBM (baseline), 3 seeds, 100 epochs.
4. **step7** — train VisualCLARITY (proposed), 3 seeds, 100 epochs.
5. **step7b** — visualize each concept's selected patches on the bird photo.
6. **step8** — gate evaluation: top-1 accuracy for both models, pointing-game
   localization for VisualCLARITY vs a random-selection floor. **This is the only
   script permitted to read part annotations.**
7. **step9** — re-evaluate checkpoints, parse logs, emit `outputs/metrics/metrics.json`
   and the figures. The written report is maintained by hand in `report/`
   (`report/report_v1.md`), which cites these numbers and figures.

**Data splits.** A 15% validation split is carved deterministically from the train
set (`VAL_SPLIT_SEED=0`), identically for both models and across seeds. The test
set is fully held out for final numbers.

**Environment.** Offline cluster (no internet), `venv` (not conda), DINOv2 weights
and dataset present locally, H100 GPU.

---

## 6. Reproducing

```bash
# verification (small scale): clean results, keep cache, run guard + tests + a
# smoke pass of the whole pipeline
sbatch run_verify.slurm

# full run + metrics: both models, 3 seeds, 100 epochs, evaluate, visualize, report
sbatch run_full_metrics.slurm

# metrics only, from existing checkpoints/logs (no retraining)
sbatch run_metrics.slurm
```

Key config (`configs/gate.yaml`): `num_classes=200`, `top_k_patches=8`,
`sparsity_weight=0.1`, `continuity_weight=0.05`, `concept_loss_weight=0.1`,
`seeds=[1,2,3]`, `epochs=100`. `num_concepts` (108) is inferred from the dataset.

---

## 7. Where things stand

Current results, the honest read on what is and isn't defensible, and the roadmap
for the next round of changes live in **`report/report_v1.md`**. Read that before
modifying anything — it records the invariants that must not break (the
architecture-matched comparison, the selector-trained-solely-by-aux-losses
property, the eval-only guard boundary, and the deterministic val split).
