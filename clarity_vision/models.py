"""GlobalCBM, VisualCLARITY models, loss function, and GradCAM concept maps.

Backbone loading is ALWAYS from a local file path — no network access at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class DINOv2Backbone(nn.Module):
    """DINOv2 ViT-B/14 feature extractor — loads weights from a local checkpoint.

    forward() returns patch token tensor of shape (B, 256, 768).
    The CLS token is discarded; callers that need it can use forward_with_cls().
    """

    TIMM_NAME = "vit_base_patch14_dinov2"
    IMG_SIZE = 224
    PATCH_SIZE = 14
    NUM_PATCHES = 256   # 16x16
    EMBED_DIM = 768

    def __init__(self, weights_path: str):
        super().__init__()
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"DINOv2 weights not found at '{weights_path}'. "
                f"Run scripts/fetch_weights.py on the laptop first."
            )
        self.vit = timm.create_model(
            self.TIMM_NAME,
            pretrained=False,
            img_size=self.IMG_SIZE,
        )
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        # The pretrain checkpoint may be wrapped under a key
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        # DINOv2 pretrain weights use 518x518 (37x37 patches); interpolate pos_embed
        # to match the 224x224 (16x16) grid we use here.
        if "pos_embed" in state and state["pos_embed"].shape[1] != self.NUM_PATCHES + 1:
            pe = state["pos_embed"]                        # (1, 1370, 768)
            cls_pe, patch_pe = pe[:, :1], pe[:, 1:]       # split CLS from patches
            old_n = int(patch_pe.shape[1] ** 0.5)         # 37
            new_n = self.IMG_SIZE // self.PATCH_SIZE       # 16
            patch_pe = patch_pe.reshape(1, old_n, old_n, self.EMBED_DIM).permute(0, 3, 1, 2)
            patch_pe = F.interpolate(patch_pe, size=(new_n, new_n),
                                     mode="bicubic", align_corners=False)
            patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, new_n * new_n, self.EMBED_DIM)
            state["pos_embed"] = torch.cat([cls_pe, patch_pe], dim=1)  # (1, 257, 768)
        missing, unexpected = self.vit.load_state_dict(state, strict=False)
        if missing:
            print(f"[DINOv2Backbone] missing keys ({len(missing)}): {missing[:5]} ...")
        self.vit.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return patch tokens (B, 256, 768)."""
        feats = self.vit.forward_features(x)   # (B, 257, 768) — CLS + 256 patches
        return feats[:, 1:, :]                  # drop CLS token

    def forward_with_cls(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (cls_token, patch_tokens) both as (B, 768) and (B, 256, 768)."""
        feats = self.vit.forward_features(x)
        return feats[:, 0, :], feats[:, 1:, :]


# ---------------------------------------------------------------------------
# Concept bottleneck models
# ---------------------------------------------------------------------------

class GlobalCBM(nn.Module):
    """Baseline concept bottleneck model.

    Concept scores come from mean-pooling ALL 256 patch tokens, then a linear
    projection.  The spatial origin of each concept is therefore undefined —
    this is the control model for the pointing-game comparison.

    Architecture:
        patch_tokens (B,256,768)
            -> mean_pool -> (B,768)
            -> concept_proj -> (B,C)  [concept bottleneck]
            -> classifier  -> (B,K)   [class logits]
    """

    def __init__(self, num_concepts: int, num_classes: int, embed_dim: int = 768):
        super().__init__()
        self.concept_proj = nn.Linear(embed_dim, num_concepts)
        self.classifier = nn.Linear(num_concepts, num_classes)

    def forward(self, patch_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patch_tokens: (B, N, D) pre-extracted DINOv2 patch tokens.
        Returns:
            (logits, concept_scores): (B, K) and (B, C).
        """
        pooled = patch_tokens.mean(dim=1)          # (B, D)
        concept_scores = self.concept_proj(pooled)  # (B, C)
        logits = self.classifier(concept_scores)    # (B, K)
        return logits, concept_scores


class VisualCLARITY(nn.Module):
    """Spatially-grounded concept bottleneck model.

    Pipeline:
        patch_tokens (B,N,D)
          -> per-concept attention scores a_c(i) = w_c . token_i      (B,C,N)
          -> HARD top-k selection (0/1 mask, non-differentiable)        (B,C,N)
          -> PLAIN AVERAGE of selected tokens                          (B,C,D)
          -> per-concept linear scorer -> concept score                (B,C)
          -> classifier                                                 (B,K)

    The selector (`attn_weights`) is NOT trained through the main classification
    loss — the classification path runs through a HARD top-k selection
    (non-differentiable), so the selector gets ZERO gradient from cls_loss. It is
    trained ENTIRELY by the AUXILIARY losses on the continuous `attn` scores
    (sparsity + spatial continuity). forward() therefore RETURNS `attn` so
    loss_fn can compute those terms; if their weights are 0, the selector never
    receives gradient and stays frozen at its random init — so the gradient-flow
    test must FAIL (grad == 0) in that case.

    concept_patch_map() returns the hard top-k binary mask for pointing-game
    evaluation — that path is index-only and intentionally has no gradient.
    """

    def __init__(self, num_concepts: int, num_classes: int,
                 embed_dim: int = 768, top_k: int = 8, grid_size: int = 16):
        super().__init__()
        self.num_concepts = num_concepts
        self.top_k = top_k
        self.grid_size = grid_size  # 16x16 = 256 patches; used by continuity loss

        # Per-concept patch attention weights: (C, D) — the learnable selector.
        self.attn_weights = nn.Parameter(torch.randn(num_concepts, embed_dim) * 0.02)

        # Per-concept linear scorer applied to the pooled selected patches.
        self.concept_proj = nn.Linear(embed_dim, num_concepts)

        self.classifier = nn.Linear(num_concepts, num_classes)

    def forward(self, patch_tokens: torch.Tensor):
        """
        Args:
            patch_tokens: (B, N, D) pre-extracted DINOv2 patch tokens.
        Returns:
            logits:         (B, K)
            concept_scores: (B, C)
            attn:           (B, C, N)  continuous selector scores (for aux losses)
        """
        B, N, D = patch_tokens.shape
        C = self.num_concepts

        # Continuous attention score for each patch per concept: (B, C, N).
        # This is the gradient-carrying tensor — the aux losses act on it.
        attn = torch.einsum("cd,bnd->bcn", self.attn_weights, patch_tokens)

        # --- HARD top-k selection (non-differentiable) ---
        # We take indices only and build a 0/1 mask. The main loss does NOT flow
        # to attn_weights through here — that is intentional: selection is a hard
        # mask feeding a plain-average pool, so it carries no gradient.
        _, topk_idx = attn.topk(self.top_k, dim=-1)          # (B, C, k)
        mask = torch.zeros(B, C, N, device=patch_tokens.device, dtype=patch_tokens.dtype)
        mask.scatter_(2, topk_idx, 1.0)                       # (B, C, N) in {0,1}

        # --- PLAIN AVERAGE pool of selected patches ---
        # masked sum over patches / number selected. top_k is constant (=8) so
        # the denominator is just top_k, but we divide by the actual mask sum to
        # stay robust if N < top_k for any concept.
        # pooled: (B, C, D)
        masked_sum = torch.einsum("bcn,bnd->bcd", mask, patch_tokens)
        counts = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)  # (B, C, 1)
        pooled = masked_sum / counts                           # (B, C, D)

        # --- Per-concept score from its own pooled vector ---
        # concept_proj.weight is (C, D); row c scores concept c. We want, for
        # each concept c, score = pooled[:, c, :] . weight[c, :] + bias[c].
        # einsum gives the per-concept diagonal directly (no (B,C,k,C) blowup).
        concept_scores = torch.einsum("bcd,cd->bc", pooled, self.concept_proj.weight)
        concept_scores = concept_scores + self.concept_proj.bias  # (B, C)

        logits = self.classifier(concept_scores)               # (B, K)
        return logits, concept_scores, attn

    @torch.no_grad()
    def concept_patch_map(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Return binary mask (B, C, N) marking the top-k patches per concept.

        Eval-only — used for the pointing-game localization metric. Index-only,
        intentionally no gradient.
        """
        B, N, D = patch_tokens.shape
        attn = torch.einsum("cd,bnd->bcn", self.attn_weights, patch_tokens)
        _, topk_idx = attn.topk(self.top_k, dim=-1)
        mask = torch.zeros(B, self.num_concepts, N, device=patch_tokens.device)
        mask.scatter_(2, topk_idx, 1.0)
        return mask


# ---------------------------------------------------------------------------
# Loss function (shared by GlobalCBM and VisualCLARITY)
# ---------------------------------------------------------------------------

def _sparsity_loss(attn: torch.Tensor) -> torch.Tensor:
    """Entropy-based sparsity on the FULL per-concept patch distribution.

    attn: (B, C, N). Lower entropy => mass concentrated on few patches => more
    sparse/localized. We softmax over the N patch dimension and return mean
    entropy (to be MINIMIZED). Operates on all N scores so gradient can move the
    selection, not just sharpen the already-selected patches.
    """
    probs = F.softmax(attn, dim=-1)                      # (B, C, N)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # (B, C)
    return entropy.mean()


def _continuity_loss(attn: torch.Tensor, grid_size: int = 16) -> torch.Tensor:
    """Spatial-contiguity loss on the 2D patch grid.

    attn: (B, C, N) with N == grid_size**2. We softmax to a soft spatial map,
    reshape to (B, C, H, W), and penalize differences between each patch and its
    right and down neighbors. Encourages selected patches to form a coherent
    region — what the pointing-game rewards. Differentiable (acts on softmax),
    so it contributes selector gradient too.
    """
    B, C, N = attn.shape
    H = W = grid_size
    if N != H * W:
        # Grid assumption broken — skip rather than mis-compute. (Flag, don't hide.)
        return attn.new_zeros(())
    probs = F.softmax(attn, dim=-1).reshape(B, C, H, W)  # (B, C, H, W)
    dh = (probs[:, :, 1:, :] - probs[:, :, :-1, :]).abs().mean()
    dw = (probs[:, :, :, 1:] - probs[:, :, :, :-1]).abs().mean()
    return dh + dw


def loss_fn(
    logits: torch.Tensor,
    concept_scores: torch.Tensor,
    class_labels: torch.Tensor,
    concept_labels: torch.Tensor,
    attn: Optional[torch.Tensor] = None,
    concept_loss_weight: float = 0.01,
    sparsity_weight: float = 0.0,
    continuity_weight: float = 0.0,
    grid_size: int = 16,
) -> Tuple[torch.Tensor, dict]:
    """Combined classification + concept + selector-auxiliary loss.

    Args:
        logits:            (B, K) class predictions.
        concept_scores:    (B, C) predicted concept activations (logits).
        class_labels:      (B,) ground-truth class indices.
        concept_labels:    (B, C) ground-truth concept labels in {0, 1}.
        attn:              (B, C, N) continuous selector scores from
                           VisualCLARITY. Pass None for GlobalCBM (aux = 0).
        concept_loss_weight:  lambda on the concept BCE.
        sparsity_weight:      lambda on the selector sparsity loss. MUST be > 0
                              for VisualCLARITY or the selector never learns.
        continuity_weight:    lambda on the spatial-continuity loss.
        grid_size:            patch grid side (16 for 16x16 = 256 patches).
    Returns:
        (total_loss, parts) where parts is a dict of the individual scalar terms
        for logging — so you can SEE each term, not just the sum.
    """
    cls_loss = F.cross_entropy(logits, class_labels)
    concept_loss = F.binary_cross_entropy_with_logits(concept_scores, concept_labels)

    if attn is not None:
        sparsity = _sparsity_loss(attn)
        continuity = _continuity_loss(attn, grid_size=grid_size)
    else:
        sparsity = concept_scores.new_zeros(())
        continuity = concept_scores.new_zeros(())


    total = (
        cls_loss
        + concept_loss_weight * concept_loss
        + sparsity_weight * sparsity
        + continuity_weight * continuity
    )

    parts = {
        "total": total.detach(),
        "cls": cls_loss.detach(),
        "concept": concept_loss.detach(),
        "sparsity": sparsity.detach(),
        "continuity": continuity.detach(),
    }
    return total, parts


# ---------------------------------------------------------------------------
# GradCAM-style concept activation maps  (eval helper, not used in training)
# ---------------------------------------------------------------------------

def gradcam_concept_maps(
    backbone: DINOv2Backbone,
    model: nn.Module,
    images: torch.Tensor,
    concept_idx: int,
) -> torch.Tensor:
    """Compute a GradCAM-style saliency map for a single concept.

    Returns a (B, 16, 16) float tensor in [0, 1] — one heatmap per image.
    Works for both GlobalCBM and VisualCLARITY.
    """
    backbone.eval()
    model.eval()
    images.requires_grad_(False)

    with torch.enable_grad():
        patch_tokens = backbone(images)
        patch_tokens.retain_grad()
        # GlobalCBM returns (logits, concept_scores); VisualCLARITY returns
        # (logits, concept_scores, attn). Index by position to support both.
        outputs = model(patch_tokens)
        concept_scores = outputs[1]
        score = concept_scores[:, concept_idx].sum()
        score.backward()

    grad = patch_tokens.grad                   # (B, 256, D)
    weights = grad.mean(dim=-1, keepdim=True)  # (B, 256, 1)
    cam = (weights * patch_tokens).sum(dim=-1) # (B, 256)
    cam = F.relu(cam)
    cam = cam.reshape(-1, 16, 16)

    # Normalise per image
    mn = cam.flatten(1).min(dim=1).values.view(-1, 1, 1)
    mx = cam.flatten(1).max(dim=1).values.view(-1, 1, 1)
    cam = (cam - mn) / (mx - mn + 1e-8)
    return cam
