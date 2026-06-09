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

    Each concept reads its score from its TOP-K most relevant patch tokens
    (selected by a learned per-concept attention weight vector), rather than
    mean-pooling all patches.  This is the proposed model.

    Architecture:
        patch_tokens (B,256,768)
            -> per-concept attention scores a_c(i) = w_c · token_i  (B,C,N)
            -> top-k mask per concept                                (B,C,k)
            -> concept_score_c = mean over top-k of linear(token_i) (B,C)
            -> classifier -> (B,K)
    """

    def __init__(self, num_concepts: int, num_classes: int,
                 embed_dim: int = 768, top_k: int = 8):
        super().__init__()
        self.num_concepts = num_concepts
        self.top_k = top_k

        # Per-concept patch attention weights: (C, D)
        self.attn_weights = nn.Parameter(torch.randn(num_concepts, embed_dim) * 0.02)

        # Per-concept linear scorer applied to each selected patch
        self.concept_proj = nn.Linear(embed_dim, num_concepts)

        self.classifier = nn.Linear(num_concepts, num_classes)

    def forward(self, patch_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patch_tokens: (B, N, D) pre-extracted DINOv2 patch tokens.
        Returns:
            (logits, concept_scores): (B, K) and (B, C).
        """
        B, N, D = patch_tokens.shape
        C = self.num_concepts

        # Attention score for each patch per concept: (B, C, N)
        attn = torch.einsum("cd,bnd->bcn", self.attn_weights, patch_tokens)

        # Top-k patch indices per concept: (B, C, k)
        _, topk_idx = attn.topk(self.top_k, dim=-1)

        # Gather top-k tokens: (B, C, k, D)
        idx_expanded = topk_idx.unsqueeze(-1).expand(B, C, self.top_k, D)
        tokens_expanded = patch_tokens.unsqueeze(1).expand(B, C, N, D)
        topk_tokens = tokens_expanded.gather(2, idx_expanded)  # (B, C, k, D)

        # Project each token to concept space then mean over k
        # concept_proj maps D -> C, but we only care about the diagonal
        # So: proj = topk_tokens @ concept_proj.weight.T  (B, C, k, C)
        # then select the c-th output for concept c
        proj = topk_tokens @ self.concept_proj.weight.T  # (B, C, k, C)
        diag_scores = proj.diagonal(dim1=1, dim2=3)       # (B, k, C) — diagonal over (C,C)
        concept_scores = diag_scores.mean(dim=1)           # (B, C)
        concept_scores = concept_scores + self.concept_proj.bias

        logits = self.classifier(concept_scores)
        return logits, concept_scores

    @torch.no_grad()
    def concept_patch_map(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Return binary mask (B, C, N) marking the top-k patches per concept.

        Used at evaluation time to compute pointing-game localization score.
        """
        B, N, D = patch_tokens.shape
        attn = torch.einsum("cd,bnd->bcn", self.attn_weights, patch_tokens)
        _, topk_idx = attn.topk(self.top_k, dim=-1)
        mask = torch.zeros(B, self.num_concepts, N, device=patch_tokens.device)
        mask.scatter_(2, topk_idx, 1.0)
        return mask


# ---------------------------------------------------------------------------
# Loss function (shared — Rule 4)
# ---------------------------------------------------------------------------

def loss_fn(
    logits: torch.Tensor,
    concept_scores: torch.Tensor,
    class_labels: torch.Tensor,
    concept_labels: torch.Tensor,
    concept_loss_weight: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined classification + concept alignment loss.

    Args:
        logits: (B, K) class predictions.
        concept_scores: (B, C) predicted concept activations.
        class_labels: (B,) ground-truth class indices.
        concept_labels: (B, C) ground-truth concept labels in {0, 1}.
        concept_loss_weight: lambda weighting the concept auxiliary loss.
    Returns:
        (total_loss, cls_loss, concept_loss)
    """
    cls_loss = F.cross_entropy(logits, class_labels)
    concept_loss = F.binary_cross_entropy_with_logits(concept_scores, concept_labels)
    total = cls_loss + concept_loss_weight * concept_loss
    return total, cls_loss, concept_loss


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
        _, concept_scores = model(patch_tokens)
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
