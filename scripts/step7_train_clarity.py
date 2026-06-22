"""Step 7 — train VisualCLARITY (x3 seeds).

Same structure as scripts/step6_train_baseline.py, with the VisualCLARITY-specific
differences:
  * model returns a 3-tuple: (logits, concept_scores, attn)
  * loss_fn is called WITH attn + sparsity_weight + continuity_weight from the
    config, plus grid_size. These auxiliary losses are the SOLE supervision for
    the selector (attn_weights). Both weights must be > 0, or the selector
    receives no gradient and never trains. gate.yaml documents this.

Training/evaluation boundary: this script reads ONLY cached features / labels /
concepts. It never reads part-location data and never imports
clarity_vision.evaluation. Pointing-game evaluation is step8's job.

Eval split: a deterministic validation set is carved from the cached TRAIN
split (clarity_vision.train_utils.make_loaders_with_val) and used for per-epoch
train/val monitoring. The cached TEST split is left untouched and is still the
final reported accuracy, so the GlobalCBM vs VisualCLARITY comparison stays
apples-to-apples — the only difference between the models is the selection
mechanism. step6 uses the identical split and monitoring.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

# Bootstrap repo root onto sys.path so `clarity_vision` imports when this script
# is run directly (python scripts/step7_train_clarity.py). Mirrors step6.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clarity_vision.models import VisualCLARITY, loss_fn
from clarity_vision.train_utils import (
    make_loaders_with_val,
    num_concepts_of,
    save_checkpoint,
    set_seed,
)


# ---------------------------------------------------------------------------
# Accuracy (top-1) — mirrors step6's _accuracy, unpacks the 4-tuple batch and
# the model's 3-tuple output (index [0] = logits).
# ---------------------------------------------------------------------------
@torch.no_grad()
def _accuracy(model: nn.Module, loader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for features, labels, _concepts, _image_ids in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)[0]            # VisualCLARITY -> (logits, scores, attn)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Validation pass — average loss components and top-1 accuracy over a loader.
# Mirrors the training-loop loss call (aux losses ON) so train and val losses
# are directly comparable.
# ---------------------------------------------------------------------------
@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader,
    device: str,
    concept_loss_weight: float,
    sparsity_weight: float,
    continuity_weight: float,
    grid_size: int,
    max_batches: int | None = None,
) -> tuple[dict, float]:
    """Return ({"total","cls","concept","sparsity","continuity"}, accuracy)."""
    model.eval()
    running = {"total": 0.0, "cls": 0.0, "concept": 0.0,
               "sparsity": 0.0, "continuity": 0.0}
    n_batches = 0
    correct = 0
    total = 0
    for b_idx, (features, labels, concepts, _image_ids) in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        features = features.to(device)
        labels = labels.to(device)
        concepts = concepts.to(device)
        logits, concept_scores, attn = model(features)
        _total, parts = loss_fn(
            logits, concept_scores, labels, concepts,
            attn=attn,
            concept_loss_weight=concept_loss_weight,
            sparsity_weight=sparsity_weight,
            continuity_weight=continuity_weight,
            grid_size=grid_size,
        )
        for k in running:
            running[k] += parts[k].item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        n_batches += 1
    n = max(n_batches, 1)
    avg = {k: v / n for k, v in running.items()}
    acc = correct / max(total, 1)
    return avg, acc


def train_one_seed(
    seed: int,
    cfg: dict,
    train_loader,
    val_loader,
    test_loader,
    num_concepts: int,
    device: str,
    epochs: int,
    max_batches: int | None = None,
) -> dict:
    """Train one VisualCLARITY seed; return a checkpoint dict."""
    set_seed(seed)

    # Config key is `top_k_patches`; the model parameter is `top_k`. Remap here.
    top_k = cfg.get("top_k_patches", 8)
    embed_dim = cfg.get("embed_dim", 768)
    grid_size = int(math.isqrt(cfg.get("num_patches", 256)))  # 256 -> 16

    model = VisualCLARITY(
        num_concepts=num_concepts,
        num_classes=cfg["num_classes"],
        embed_dim=embed_dim,
        top_k=top_k,
        grid_size=grid_size,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Auxiliary-loss weights — the sole supervision for the selector.
    sparsity_weight = cfg["sparsity_weight"]
    continuity_weight = cfg["continuity_weight"]
    concept_loss_weight = cfg["concept_loss_weight"]

    # Hard guard: both weights must be > 0 or the selector cannot learn. Fail loud
    # rather than silently training a selector that never leaves its random init.
    if sparsity_weight <= 0 or continuity_weight <= 0:
        raise ValueError(
            f"sparsity_weight ({sparsity_weight}) and continuity_weight "
            f"({continuity_weight}) must both be > 0, else the VisualCLARITY "
            f"selector (attn_weights) receives zero gradient and never trains."
        )

    history = []  # per-epoch train/val loss + acc, for curve regeneration

    for epoch in range(epochs):
        model.train()
        running = {"total": 0.0, "cls": 0.0, "concept": 0.0,
                   "sparsity": 0.0, "continuity": 0.0}
        n_batches = 0
        train_correct = 0
        train_total = 0

        for b_idx, (features, labels, concepts, _image_ids) in enumerate(train_loader):
            if max_batches is not None and b_idx >= max_batches:
                break

            features = features.to(device)
            labels = labels.to(device)
            concepts = concepts.to(device)

            logits, concept_scores, attn = model(features)
            total, parts = loss_fn(
                logits,
                concept_scores,
                labels,
                concepts,
                attn=attn,
                concept_loss_weight=concept_loss_weight,
                sparsity_weight=sparsity_weight,
                continuity_weight=continuity_weight,
                grid_size=grid_size,
            )

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()

            for k in running:
                running[k] += parts[k].item()
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)
            n_batches += 1

        scheduler.step()

        denom = max(n_batches, 1)
        avg = {k: v / denom for k, v in running.items()}
        train_acc = train_correct / max(train_total, 1)

        # Validation: same loss (aux ON) + accuracy on the held-out split.
        val_avg, val_acc = _evaluate(
            model, val_loader, device,
            concept_loss_weight=concept_loss_weight,
            sparsity_weight=sparsity_weight,
            continuity_weight=continuity_weight,
            grid_size=grid_size,
            max_batches=max_batches,
        )

        print(
            f"[seed {seed}] epoch {epoch + 1}/{epochs} "
            f"loss={avg['total']:.4f} (cls={avg['cls']:.4f} "
            f"concept={avg['concept']:.4f} sparsity={avg['sparsity']:.4f} "
            f"continuity={avg['continuity']:.4f}) "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_avg['total']:.4f} (val_cls={val_avg['cls']:.4f} "
            f"val_concept={val_avg['concept']:.4f} val_sparsity={val_avg['sparsity']:.4f} "
            f"val_continuity={val_avg['continuity']:.4f}) val_acc={val_acc:.4f}",
            flush=True,
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg["total"],
            "train_cls": avg["cls"],
            "train_concept": avg["concept"],
            "train_sparsity": avg["sparsity"],
            "train_continuity": avg["continuity"],
            "train_acc": train_acc,
            "val_loss": val_avg["total"],
            "val_cls": val_avg["cls"],
            "val_concept": val_avg["concept"],
            "val_sparsity": val_avg["sparsity"],
            "val_continuity": val_avg["continuity"],
            "val_acc": val_acc,
        })

    test_acc = _accuracy(model, test_loader, device)
    print(f"[seed {seed}] test_acc={test_acc:.4f}", flush=True)

    return {
        "model": "VisualCLARITY",
        "seed": seed,
        "num_concepts": num_concepts,
        "num_classes": cfg["num_classes"],
        "embed_dim": embed_dim,
        "top_k": top_k,
        "epochs": epochs,
        "sparsity_weight": sparsity_weight,
        "continuity_weight": continuity_weight,
        "test_acc": test_acc,
        "history": history,
        "model_state": model.state_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 7 — train VisualCLARITY.")
    parser.add_argument("--config", default="configs/gate.yaml")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs from config.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override seeds from config.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true",
                        help="1 seed, 1 epoch, 3 batches, 0 workers — fast wiring check.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cache_dir = cfg["cache_dir"]
    if not Path(cache_dir).exists():
        raise FileNotFoundError(f"Cache dir not found: {cache_dir}. Run step5 first.")

    # Smoke mode: fast, minimal, deterministic wiring check.
    if args.smoke:
        seeds = [cfg["seeds"][0]]
        epochs = 1
        num_workers = 0
        max_batches = 3
    else:
        seeds = args.seeds if args.seeds is not None else cfg["seeds"]
        epochs = args.epochs if args.epochs is not None else cfg["epochs"]
        num_workers = args.num_workers
        max_batches = None

    train_loader, val_loader, test_loader = make_loaders_with_val(
        cache_dir,
        batch_size=cfg["batch_size"],
        num_workers=num_workers,
    )

    # num_concepts is NOT in gate.yaml — infer from the cached concept targets,
    # exactly as step6 does.
    num_concepts = num_concepts_of(train_loader)
    print(f"Inferred num_concepts={num_concepts} from dataset.", flush=True)

    checkpoint_dir = cfg["checkpoint_dir"]
    for seed in seeds:
        ckpt = train_one_seed(
            seed=seed,
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_concepts=num_concepts,
            device=args.device,
            epochs=epochs,
            max_batches=max_batches,
        )
        out_path = f"{checkpoint_dir}/visualclarity_seed{seed}.pt"
        save_checkpoint(ckpt, out_path)
        print(f"[seed {seed}] saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
