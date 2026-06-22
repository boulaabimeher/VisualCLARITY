"""Step 9 — metrics and figures.

Reads the REAL artifacts (training logs + checkpoints) and produces:
  outputs/metrics/metrics.json     — all numbers (mean ± std across seeds)
  outputs/metrics/*.png            — figures

The written report is maintained by hand in report/; this step produces only the
numbers and figures that report cites.

Design choices for honesty:
  * Accuracy + pointing game are RE-COMPUTED from the saved checkpoints here, not
    re-parsed from eval logs — the numbers come from the actual models, so they
    cannot drift from what the models do.
  * Loss trajectories (cls/concept/sparsity/continuity per epoch) are PARSED from
    the training logs, since that per-epoch history only exists there.
  * Pointing game is VisualCLARITY-vs-random (GlobalCBM has no spatial selection),
    matching step8's framing.

Run on the cluster after a full run:
    python scripts/step9_metrics.py --config configs/gate.yaml \
        --train-log-glob 'logs/train_*_*.log' --device cuda
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clarity_vision.models import GlobalCBM, VisualCLARITY
from clarity_vision.evaluation import pointing_game
from clarity_vision.data import cub_root, load_image_list, load_part_locs_EVAL_ONLY
from clarity_vision.train_utils import make_loaders, load_checkpoint
from PIL import Image


# ---------------------------------------------------------------------------
# 1. Parse training logs -> per-epoch loss curves, keyed by (model, seed)
# ---------------------------------------------------------------------------
# step6 line: [step6][seed 1] epoch 12/100  loss=1.00  cls=0.82  concept=1.77
# step7 line: [seed 1] epoch 12/100 loss=0.80 (cls=0.61 concept=1.68 sparsity=0.22 continuity=0.0158)
STEP6_RE = re.compile(
    r"\[step6\]\[seed (\d+)\] epoch (\d+)/\d+\s+loss=([\d.]+)\s+cls=([\d.]+)\s+concept=([\d.]+)"
)
STEP7_RE = re.compile(
    r"\[seed (\d+)\] epoch (\d+)/\d+ loss=([\d.]+) \(cls=([\d.]+) concept=([\d.]+) "
    r"sparsity=([\d.]+) continuity=([\d.]+)\)"
)


def parse_training_logs(log_glob: str) -> dict:
    """Return {model: {seed: {epoch: {term: value}}}}."""
    out = {"GlobalCBM": {}, "VisualCLARITY": {}}
    for path in glob.glob(log_glob):
        text = Path(path).read_text()
        for m in STEP6_RE.finditer(text):
            seed, ep, loss, cls, con = m.groups()
            out["GlobalCBM"].setdefault(int(seed), {})[int(ep)] = {
                "loss": float(loss), "cls": float(cls), "concept": float(con)}
        for m in STEP7_RE.finditer(text):
            seed, ep, loss, cls, con, sp, ct = m.groups()
            out["VisualCLARITY"].setdefault(int(seed), {})[int(ep)] = {
                "loss": float(loss), "cls": float(cls), "concept": float(con),
                "sparsity": float(sp), "continuity": float(ct)}
    return out


# ---------------------------------------------------------------------------
# 2. Re-evaluate checkpoints (accuracy + pointing game) from saved models
# ---------------------------------------------------------------------------
def rebuild(ckpt, device):
    if ckpt["model"] == "VisualCLARITY":
        m = VisualCLARITY(ckpt["num_concepts"], ckpt["num_classes"],
                          embed_dim=ckpt["embed_dim"], top_k=ckpt["top_k"])
    else:
        m = GlobalCBM(ckpt["num_concepts"], ckpt["num_classes"],
                      embed_dim=ckpt["embed_dim"])
    m.load_state_dict(ckpt["model_state"])
    return m.to(device).eval()


@torch.no_grad()
def top1(model, loader, device):
    correct = total = 0
    for f, y, _c, _i in loader:
        f, y = f.to(device), y.to(device)
        correct += (model(f)[0].argmax(-1) == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def random_mask_like(mask, top_k, seed=0):
    B, C, N = mask.shape
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(B, C, N)
    for b in range(B):
        for c in range(C):
            out[b, c, torch.randperm(N, generator=g)[:top_k]] = 1.0
    return out


@torch.no_grad()
def pointing_for_model(model, loader, cfg, device):
    dp = cfg["dataset_path"]
    img_list = load_image_list(dp)
    images_dir = cub_root(dp) / "images"
    cpm = json.load(open(ROOT / cfg["concept_part_map_json"]))
    concept_part_map = {e["concept_id"]: e["part_ids"] for e in cpm["concepts"]}
    part_locs = load_part_locs_EVAL_ONLY(dp)

    all_masks, all_ids, all_sizes = [], [], []
    for f, _y, _c, ids in loader:
        f = f.to(device)
        all_masks.append(model.concept_patch_map(f).cpu())
        ids = [int(x) for x in ids]
        all_ids.extend(ids)
        for iid in ids:
            with Image.open(images_dir / img_list[iid]) as im:
                all_sizes.append(im.size)
    masks = torch.cat(all_masks, 0)

    kw = dict(concept_part_map=concept_part_map, part_locs=part_locs,
              image_ids=all_ids, img_sizes=all_sizes,
              img_size=cfg.get("img_size", 224), patch_size=cfg.get("patch_size", 14),
              tolerance=cfg.get("pointing_game_tolerance", 1))
    overall, per_concept = pointing_game(concept_patch_masks=masks, **kw)
    rand = random_mask_like(masks, model.top_k, seed=0)
    rand_overall, _ = pointing_game(concept_patch_masks=rand, **kw)
    return overall, per_concept, rand_overall


# ---------------------------------------------------------------------------
# 3. Figures
# ---------------------------------------------------------------------------
def fig_accuracy(metrics, out):
    g = metrics["accuracy"]["GlobalCBM"]
    v = metrics["accuracy"]["VisualCLARITY"]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["GlobalCBM\n(baseline)", "VisualCLARITY\n(proposed)"],
                  [g["mean"], v["mean"]], yerr=[g["std"], v["std"]],
                  capsize=6, color=["#888780", "#534AB7"])
    ax.set_ylabel("Top-1 accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"Accuracy (3 seeds) — gap +{(v['mean']-g['mean'])*100:.1f} pts")
    for b, m in zip(bars, [g["mean"], v["mean"]]):
        ax.text(b.get_x() + b.get_width()/2, m + 0.02, f"{m:.3f}", ha="center")
    fig.tight_layout(); fig.savefig(out / "fig_accuracy.png", dpi=140); plt.close(fig)


def fig_concept_trajectory(traj, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, color in [("GlobalCBM", "#888780"), ("VisualCLARITY", "#534AB7")]:
        seeds = traj[model]
        if not seeds:
            continue
        # average concept-BCE across seeds per epoch
        epochs = sorted(next(iter(seeds.values())).keys())
        mean = [np.mean([seeds[s][e]["concept"] for s in seeds if e in seeds[s]])
                for e in epochs]
        ax.plot(epochs, mean, label=model, color=color)
    ax.set_xlabel("epoch"); ax.set_ylabel("concept BCE")
    ax.set_title("Concept loss over training (rises early, then falls)")
    ax.legend(); fig.tight_layout()
    fig.savefig(out / "fig_concept_trajectory.png", dpi=140); plt.close(fig)


def fig_sparsity_trajectory(traj, out):
    seeds = traj["VisualCLARITY"]
    if not seeds:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = sorted(next(iter(seeds.values())).keys())
    mean = [np.mean([seeds[s][e]["sparsity"] for s in seeds if e in seeds[s]])
            for e in epochs]
    ax.plot(epochs, mean, color="#1D9E75")
    ax.set_xlabel("epoch"); ax.set_ylabel("sparsity loss")
    ax.set_title("Selector sparsity over training (drops = selector concentrating)")
    fig.tight_layout(); fig.savefig(out / "fig_sparsity_trajectory.png", dpi=140); plt.close(fig)


def _epoch_mean(histories_for_model, field):
    """Mean of `field` per epoch across seeds, from checkpoint history lists.

    Returns (epochs, values) aligned on the epochs common to all seeds, or
    (None, None) if no usable history is present.
    """
    seeds = [s for s, h in histories_for_model.items() if h]
    if not seeds:
        return None, None
    # Map each seed to {epoch: value}; intersect epochs so the mean is well-defined.
    by_seed = {s: {e["epoch"]: e[field] for e in histories_for_model[s] if field in e}
               for s in seeds}
    common = sorted(set.intersection(*(set(d) for d in by_seed.values())))
    if not common:
        return None, None
    values = [float(np.mean([by_seed[s][e] for s in seeds])) for e in common]
    return common, values


def fig_trainval_loss(histories, out):
    """Train vs val total loss per epoch, both models (seed-averaged)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    styles = {"GlobalCBM": "#888780", "VisualCLARITY": "#534AB7"}
    drew = False
    for model, color in styles.items():
        ep, tr = _epoch_mean(histories[model], "train_loss")
        _, va = _epoch_mean(histories[model], "val_loss")
        if ep is None:
            continue
        ax.plot(ep, tr, color=color, label=f"{model} train")
        ax.plot(ep, va, color=color, ls="--", label=f"{model} val")
        drew = True
    if not drew:
        plt.close(fig)
        return False
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("Train vs validation loss")
    ax.legend(); fig.tight_layout()
    fig.savefig(out / "fig_trainval_loss.png", dpi=140); plt.close(fig)
    return True


def fig_trainval_accuracy(histories, out):
    """Train vs val top-1 accuracy per epoch, both models (seed-averaged)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    styles = {"GlobalCBM": "#888780", "VisualCLARITY": "#534AB7"}
    drew = False
    for model, color in styles.items():
        ep, tr = _epoch_mean(histories[model], "train_acc")
        _, va = _epoch_mean(histories[model], "val_acc")
        if ep is None:
            continue
        ax.plot(ep, tr, color=color, label=f"{model} train")
        ax.plot(ep, va, color=color, ls="--", label=f"{model} val")
        drew = True
    if not drew:
        plt.close(fig)
        return False
    ax.set_xlabel("epoch"); ax.set_ylabel("top-1 accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Train vs validation accuracy")
    ax.legend(); fig.tight_layout()
    fig.savefig(out / "fig_trainval_accuracy.png", dpi=140); plt.close(fig)
    return True


def fig_pointing(metrics, out):
    pg = metrics["pointing_game"]
    fig, ax = plt.subplots(figsize=(5, 4))
    seeds = sorted(pg["per_seed"].keys())
    vals = [pg["per_seed"][s]["visualclarity"] for s in seeds]
    rand = pg["random_baseline"]
    ax.bar([f"seed {s}" for s in seeds], vals, color="#534AB7", label="VisualCLARITY")
    ax.axhline(rand, color="#D85A30", ls="--", label=f"random ({rand:.3f})")
    ax.set_ylabel("pointing-game accuracy")
    ax.set_title(f"Localization vs random — ~{pg['mean']/rand:.1f}x above chance")
    ax.legend(); fig.tight_layout()
    fig.savefig(out / "fig_pointing_game.png", dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def mean_std(xs):
    return {"mean": float(np.mean(xs)), "std": float(np.std(xs)), "values": [float(x) for x in xs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gate.yaml")
    ap.add_argument("--train-log-glob", default="logs/train_*_*.log")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config))
    seeds = args.seeds or cfg["seeds"]
    ckpt_dir = ROOT / cfg["checkpoint_dir"]
    out = ROOT / cfg["output_dir"] / "metrics"
    out.mkdir(parents=True, exist_ok=True)

    _, test_loader = make_loaders(cfg["cache_dir"], batch_size=cfg["batch_size"], num_workers=0)

    print("Re-evaluating checkpoints (accuracy + pointing game)...")
    g_acc, v_acc, pg_per_seed = [], [], {}
    # Per-epoch train/val curves come from the history saved in each checkpoint.
    histories = {"GlobalCBM": {}, "VisualCLARITY": {}}
    rand_overall = None
    for s in seeds:
        gck = load_checkpoint(str(ckpt_dir / f"globalcbm_seed{s}.pt"), device=args.device)
        vck = load_checkpoint(str(ckpt_dir / f"visualclarity_seed{s}.pt"), device=args.device)
        histories["GlobalCBM"][s] = gck.get("history", [])
        histories["VisualCLARITY"][s] = vck.get("history", [])
        gm, vm = rebuild(gck, args.device), rebuild(vck, args.device)
        ga, va = top1(gm, test_loader, args.device), top1(vm, test_loader, args.device)
        pg, _per, rnd = pointing_for_model(vm, test_loader, cfg, args.device)
        g_acc.append(ga); v_acc.append(va)
        pg_per_seed[s] = {"visualclarity": pg}
        rand_overall = rnd
        print(f"  seed {s}: GlobalCBM={ga:.4f}  VisualCLARITY={va:.4f}  "
              f"pointing={pg:.4f}  random={rnd:.4f}")

    pg_vals = [pg_per_seed[s]["visualclarity"] for s in seeds]
    metrics = {
        "seeds": seeds,
        "accuracy": {"GlobalCBM": mean_std(g_acc), "VisualCLARITY": mean_std(v_acc),
                     "gap": float(np.mean(v_acc) - np.mean(g_acc))},
        "pointing_game": {**mean_std(pg_vals), "random_baseline": rand_overall,
                          "lift_over_random": float(np.mean(pg_vals) - rand_overall),
                          "x_above_chance": float(np.mean(pg_vals) / rand_overall),
                          "per_seed": pg_per_seed},
    }

    print("Parsing training logs for loss trajectories...")
    traj = parse_training_logs(str(ROOT / args.train_log_glob))

    print("Rendering figures...")
    fig_accuracy(metrics, out)
    fig_pointing(metrics, out)
    if any(traj[m] for m in traj):
        fig_concept_trajectory(traj, out)
        fig_sparsity_trajectory(traj, out)
    else:
        print("  WARNING: no training-log lines parsed — skipping trajectory figures. "
              "Check --train-log-glob.")

    # Train vs val curves come from the checkpoint history (no log dependency).
    if fig_trainval_loss(histories, out) and fig_trainval_accuracy(histories, out):
        print("  wrote fig_trainval_loss.png and fig_trainval_accuracy.png")
    else:
        print("  WARNING: no per-epoch history in checkpoints — skipping train/val "
              "curves. Re-run step6/step7 to populate ckpt['history'].")

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nDone. Wrote metrics.json and figures to {out}")
    print("The written report is maintained by hand in report/ — this step only "
          "produces metrics.json and the figures it embeds.")


if __name__ == "__main__":
    main()