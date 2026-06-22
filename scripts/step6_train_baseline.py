"""Step 6 — Train GlobalCBM (baseline) across 3 seeds.

Loads cached DINOv2 features from outputs/cache/, trains the GlobalCBM
model using mean-pooled patch tokens, saves one checkpoint per seed.

GlobalCBM is the CONTROL model for the pointing-game comparison: it has no
patch selector, so it calls loss_fn with attn=None and the selector auxiliary
losses (sparsity/continuity) are exactly zero. Only cls + concept loss apply.

Usage:
    python scripts/step6_train_baseline.py
    python scripts/step6_train_baseline.py --smoke   # fast local sanity run

NOTE: requires cached features from Step 5 (outputs/cache/).

Training/evaluation boundary: this script reads ONLY cached features/labels/
concepts. It never reads part-location data and never imports
clarity_vision.evaluation.
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clarity_vision.models import GlobalCBM, loss_fn
from clarity_vision.train_utils import (
    make_loaders_with_val,
    num_concepts_of,
    save_checkpoint,
    set_seed,
)


@torch.no_grad()
def _accuracy(model: GlobalCBM, loader, device: torch.device) -> float:
    """Top-1 classification accuracy over a loader.

    Computed inline (not via clarity_vision.evaluation) so this training script
    stays free of any part-location / pointing-game machinery.
    """
    model.eval()
    correct = 0
    total = 0
    for features, labels, _concepts, _image_ids in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits, _concept_scores = model(features)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.numel()
    return correct / total if total else float("nan")


@torch.no_grad()
def _evaluate(
    model: GlobalCBM,
    loader,
    device: torch.device,
    concept_loss_weight: float,
    max_batches: int | None = None,
) -> tuple[dict, float]:
    """Average loss components and top-1 accuracy over a loader (no grad).

    Returns ({"total","cls","concept"}, accuracy). Mirrors the training-loop
    loss call (attn=None for GlobalCBM) so train and val losses are comparable.
    """
    model.eval()
    running = {"total": 0.0, "cls": 0.0, "concept": 0.0}
    n_batches = 0
    correct = 0
    total = 0
    for batch_idx, (features, labels, concepts, _ids) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        features = features.to(device)
        labels = labels.to(device)
        concepts = concepts.to(device)
        logits, concept_scores = model(features)
        _total, parts = loss_fn(
            logits, concept_scores, labels, concepts,
            attn=None, concept_loss_weight=concept_loss_weight,
        )
        for k in running:
            running[k] += parts[k].item()
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.numel()
        n_batches += 1
    n = max(n_batches, 1)
    avg = {k: v / n for k, v in running.items()}
    acc = correct / total if total else float("nan")
    return avg, acc


def train_one_seed(
    seed: int,
    cfg: dict,
    train_loader,
    val_loader,
    test_loader,
    num_concepts: int,
    device: torch.device,
    epochs: int,
    max_batches: int | None,
) -> dict:
    """Train one GlobalCBM instance for `seed`; return a checkpoint dict."""
    set_seed(seed)

    model = GlobalCBM(
        num_concepts=num_concepts,
        num_classes=cfg["num_classes"],
        embed_dim=cfg["embed_dim"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    concept_loss_weight = cfg["concept_loss_weight"]
    history = []  # per-epoch train/val loss + acc, for curve regeneration

    for epoch in range(epochs):
        model.train()
        running = {"total": 0.0, "cls": 0.0, "concept": 0.0}
        n_batches = 0
        train_correct = 0
        train_total = 0
        for batch_idx, (features, labels, concepts, _ids) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            features = features.to(device)
            labels = labels.to(device)
            concepts = concepts.to(device)

            logits, concept_scores = model(features)
            # GlobalCBM has no selector -> attn=None -> aux losses are 0.
            total, parts = loss_fn(
                logits,
                concept_scores,
                labels,
                concepts,
                attn=None,
                concept_loss_weight=concept_loss_weight,
            )

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            running["total"] += parts["total"].item()
            running["cls"] += parts["cls"].item()
            running["concept"] += parts["concept"].item()
            train_correct += (logits.argmax(dim=-1) == labels).sum().item()
            train_total += labels.numel()
            n_batches += 1

        scheduler.step()
        n = max(n_batches, 1)
        train_avg = {k: v / n for k, v in running.items()}
        train_acc = train_correct / train_total if train_total else float("nan")

        # Validation: same loss + accuracy on the held-out split (no grad).
        val_avg, val_acc = _evaluate(
            model, val_loader, device, concept_loss_weight, max_batches=max_batches,
        )

        print(
            f"[step6][seed {seed}] epoch {epoch + 1}/{epochs}  "
            f"loss={train_avg['total']:.4f}  "
            f"cls={train_avg['cls']:.4f}  concept={train_avg['concept']:.4f}  "
            f"train_acc={train_acc:.4f}  "
            f"val_loss={val_avg['total']:.4f}  val_cls={val_avg['cls']:.4f}  "
            f"val_concept={val_avg['concept']:.4f}  val_acc={val_acc:.4f}",
            flush=True,
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_avg["total"],
            "train_cls": train_avg["cls"],
            "train_concept": train_avg["concept"],
            "train_acc": train_acc,
            "val_loss": val_avg["total"],
            "val_cls": val_avg["cls"],
            "val_concept": val_avg["concept"],
            "val_acc": val_acc,
        })

    test_acc = _accuracy(model, test_loader, device)
    print(f"[step6][seed {seed}] test top-1 accuracy = {test_acc:.4f}", flush=True)

    return {
        "model": "GlobalCBM",
        "seed": seed,
        "num_concepts": num_concepts,
        "num_classes": cfg["num_classes"],
        "embed_dim": cfg["embed_dim"],
        "epochs": epochs,
        "test_acc": test_acc,
        "history": history,
        "model_state": model.state_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GlobalCBM baseline.")
    parser.add_argument("--config", default="configs/gate.yaml")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override cfg['epochs'].")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override cfg['seeds'].")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast sanity run: 1 seed, 1 epoch, 3 batches, 0 workers.",
    )
    parser.add_argument("--device", default=None,
                        help="cpu / cuda. Default: cuda if available.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config))

    cache_dir = ROOT / cfg["cache_dir"]
    if not cache_dir.exists():
        print(f"[step6] ERROR: {cache_dir} not found. Run step5 first.")
        sys.exit(1)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    num_workers = args.num_workers
    max_batches = None

    if args.smoke:
        epochs = 1
        seeds = [seeds[0]]
        num_workers = 0
        max_batches = 3
        print("[step6] SMOKE MODE: 1 seed, 1 epoch, 3 batches.")

    print(f"[step6] device={device}  epochs={epochs}  seeds={seeds}")

    train_loader, val_loader, test_loader = make_loaders_with_val(
        str(cache_dir), batch_size=cfg["batch_size"], num_workers=num_workers
    )

    # Infer concept count from the cached arrays so it always matches the data.
    num_concepts = num_concepts_of(train_loader)
    print(f"[step6] num_concepts={num_concepts}  num_classes={cfg['num_classes']}")

    ckpt_dir = ROOT / cfg["checkpoint_dir"]
    for seed in seeds:
        ckpt = train_one_seed(
            seed=seed,
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_concepts=num_concepts,
            device=device,
            epochs=epochs,
            max_batches=max_batches,
        )
        ckpt_path = ckpt_dir / f"globalcbm_seed{seed}.pt"
        save_checkpoint(ckpt, str(ckpt_path))
        print(f"[step6][seed {seed}] saved checkpoint -> {ckpt_path}", flush=True)

    print("[step6] Done.")


if __name__ == "__main__":
    main()
