"""Gradient-flow regression tests for VisualCLARITY.

The selector (`attn_weights`) is trained ENTIRELY by the auxiliary losses
(sparsity + continuity) on the continuous attention scores. The classification
path runs through a HARD, non-differentiable top-k selection, so it leaks NO
gradient to the selector. These tests pin down BOTH halves of that claim:

Test 1 (direction a): with sparsity_weight and continuity_weight NONZERO,
    attn_weights.grad.abs().sum() > 0 — the aux losses do train the selector.

Test 2 (direction b): with BOTH weights set to 0.0, attn_weights.grad is None
    or its abs().sum() == 0 — proof that NO gradient leaks through the
    classification path. If this ever fails, the selector is being trained by
    something other than the aux losses, and the localization claim is unsound.

Test 3: forward() returns (logits, concept_scores, attn) with the right shapes.
"""

import torch
import torch.nn.functional as F
import pytest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clarity_vision.models import VisualCLARITY, loss_fn


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

B, N, D = 2, 256, 768
NUM_CONCEPTS = 10
NUM_CLASSES = 5
TOP_K = 4
GRID_SIZE = 16   # 16x16 == N == 256

SPARSITY_W = 0.1
CONTINUITY_W = 0.05


def _make_batch():
    torch.manual_seed(0)
    return torch.randn(B, N, D)


def _make_labels():
    return (
        torch.randint(0, NUM_CLASSES, (B,)),
        torch.randint(0, 2, (B, NUM_CONCEPTS)).float(),
    )


def _make_model():
    return VisualCLARITY(num_concepts=NUM_CONCEPTS, num_classes=NUM_CLASSES,
                         embed_dim=D, top_k=TOP_K, grid_size=GRID_SIZE)


# ---------------------------------------------------------------------------
# Test 1 (direction a): aux losses ON -> selector MUST receive non-zero gradient
# ---------------------------------------------------------------------------

def test_attn_weights_grad_nonzero_with_aux_losses():
    """With sparsity/continuity weights > 0, attn_weights.grad is non-zero."""
    model = _make_model()
    patch_tokens = _make_batch()
    class_labels, concept_labels = _make_labels()

    logits, concept_scores, attn = model(patch_tokens)
    total, _parts = loss_fn(
        logits, concept_scores, class_labels, concept_labels,
        attn=attn,
        sparsity_weight=SPARSITY_W,
        continuity_weight=CONTINUITY_W,
        grid_size=GRID_SIZE,
    )
    total.backward()

    grad = model.attn_weights.grad
    assert grad is not None, (
        "attn_weights.grad is None even WITH aux losses — the selector is "
        "disconnected from its only supervision."
    )
    grad_sum = grad.abs().sum().item()
    assert grad_sum > 0, (
        f"attn_weights.grad sums to zero ({grad_sum}) with aux losses on — "
        "the sparsity/continuity path to the selector is broken."
    )


# ---------------------------------------------------------------------------
# Test 2 (direction b): aux losses OFF -> selector gets ZERO gradient
# ---------------------------------------------------------------------------

def test_attn_weights_grad_zero_without_aux_losses():
    """With BOTH aux weights 0.0, attn_weights.grad is None or sums to zero.

    This is the real proof: the classification path uses a hard top-k selection
    and leaks no gradient to the selector, so removing the aux losses freezes it.
    """
    model = _make_model()
    patch_tokens = _make_batch()
    class_labels, concept_labels = _make_labels()

    logits, concept_scores, attn = model(patch_tokens)
    total, _parts = loss_fn(
        logits, concept_scores, class_labels, concept_labels,
        attn=attn,
        sparsity_weight=0.0,
        continuity_weight=0.0,
        grid_size=GRID_SIZE,
    )
    total.backward()

    grad = model.attn_weights.grad
    is_zero = (grad is None) or (grad.abs().sum().item() == 0)
    assert is_zero, (
        "attn_weights received non-zero gradient with the aux losses OFF — "
        "gradient is leaking through the classification path, so the selector "
        "is NOT trained solely by the auxiliary losses."
    )


# ---------------------------------------------------------------------------
# Test 3: output shapes — forward returns (logits, concept_scores, attn)
# ---------------------------------------------------------------------------

def test_output_shapes():
    """logits (B, K), concept_scores (B, C), attn (B, C, N)."""
    model = _make_model()
    patch_tokens = _make_batch()
    logits, concept_scores, attn = model(patch_tokens)
    assert logits.shape == (B, NUM_CLASSES), f"bad logits shape: {logits.shape}"
    assert concept_scores.shape == (B, NUM_CONCEPTS), \
        f"bad concept_scores shape: {concept_scores.shape}"
    assert attn.shape == (B, NUM_CONCEPTS, N), f"bad attn shape: {attn.shape}"
