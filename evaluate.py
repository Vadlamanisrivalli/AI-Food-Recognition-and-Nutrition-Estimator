"""Evaluate a trained checkpoint on the test split.

Outputs (in --output-dir): metrics.json, classification_report.txt,
confusion_matrix.png, top_confusions.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import DATASET_DIR, DEFAULT_CHECKPOINT, OUTPUT_DIR
from .dataset import make_loaders
from .model import load_checkpoint
from .train import pick_device


@torch.no_grad()
def collect_predictions(model, loader, device, k=5):
    preds, topk_hits, targets = [], [], []
    for images, y in tqdm(loader, desc="test"):
        logits = model(images.to(device))
        kk = min(k, logits.size(1))
        top = logits.topk(kk, dim=1).indices.cpu()
        preds.append(top[:, 0])
        topk_hits.append((top == y.unsqueeze(1)).any(dim=1))
        targets.append(y)
    return torch.cat(preds).numpy(), torch.cat(topk_hits).numpy(), torch.cat(targets).numpy()


def save_confusion_png(cm, classes, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Normalised confusion matrix (rows = true class)")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    if len(classes) <= 40:
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=90, fontsize=7)
        ax.set_yticklabels(classes, fontsize=7)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--dataset-root", type=Path, default=DATASET_DIR)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args(argv)

    from sklearn.metrics import classification_report, confusion_matrix

    device = pick_device()
    model, meta = load_checkpoint(args.checkpoint, device)
    classes = meta["classes"]
    _, _, test_loader, data_classes = make_loaders(
        batch_size=args.batch_size, workers=args.workers, img_size=meta["img_size"],
        dataset_root=args.dataset_root, data_dir=args.data_dir, max_samples=args.max_samples)
    if data_classes != classes:
        raise SystemExit("Dataset classes do not match the checkpoint's classes.")

    preds, top5_hits, targets = collect_predictions(model, test_loader, device)
    top1 = float((preds == targets).mean())
    top5 = float(top5_hits.mean())
    print(f"Test top-1: {top1:.4f} | top-5: {top5:.4f} | samples: {len(targets)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(classes)))
    report = classification_report(targets, preds, labels=labels, target_names=classes,
                                   digits=3, zero_division=0)
    (args.output_dir / "classification_report.txt").write_text(report)

    cm = confusion_matrix(targets, preds, labels=labels)
    save_confusion_png(cm, classes, args.output_dir / "confusion_matrix.png")

    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = [(int(off[i, j]), classes[i], classes[j]) for i in labels for j in labels if off[i, j] > 0]
    pairs.sort(reverse=True)
    with open(args.output_dir / "top_confusions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["count", "true_class", "predicted_as"])
        w.writerows(pairs[:30])
    print("Most confused pairs (count, true -> predicted):")
    for count, t, pr in pairs[:10]:
        print(f"  {count:3d}  {t} -> {pr}")

    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    order = np.argsort(per_class)
    metrics = {
        "top1_accuracy": top1, "top5_accuracy": top5, "num_samples": int(len(targets)),
        "arch": meta["arch"], "val_acc_at_checkpoint": meta.get("val_acc"),
        "worst_classes": {classes[i]: float(per_class[i]) for i in order[:10]},
        "best_classes": {classes[i]: float(per_class[i]) for i in order[::-1][:10]},
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved reports to {args.output_dir}")


if __name__ == "__main__":
    main()
