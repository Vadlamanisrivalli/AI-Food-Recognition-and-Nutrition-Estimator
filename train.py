"""Train a food classifier with two-phase transfer learning.

Phase 1: train only the new classification head (backbone frozen).
Phase 2: fine-tune the whole network with a small learning rate + cosine decay.

Examples
--------
    python -m foodnutri.train                                  # Food-101, EfficientNet-B0
    python -m foodnutri.train --arch resnet50 --batch-size 32
    python -m foodnutri.train --data-dir my_data               # custom ImageFolder dataset
    python -m foodnutri.train --max-samples 512 --head-epochs 1 --finetune-epochs 1   # quick test
"""
import argparse
import contextlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import DATASET_DIR, DEFAULT_ARCH, DEFAULT_IMG_SIZE, OUTPUT_DIR, SEED
from .dataset import make_loaders
from .model import SUPPORTED_ARCHS, build_model, head_module, set_backbone_trainable, split_parameters


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_train_mode(model: nn.Module, arch: str, backbone_frozen: bool) -> None:
    """Train mode, but keep a frozen backbone's BatchNorm statistics fixed."""
    model.train()
    if backbone_frozen:
        head = head_module(model, arch)
        for child in model.children():
            if child is not head:
                child.eval()


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False, desc=""):
    training = optimizer is not None
    total_loss, correct, seen = 0.0, 0, 0
    grad_ctx = torch.enable_grad() if training else torch.no_grad()
    with grad_ctx:
        for images, targets in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext()
            with amp_ctx:
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            seen += images.size(0)
    return total_loss / max(seen, 1), correct / max(seen, 1)


def plot_history(history, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(range(1, len(history) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(xs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(xs, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("Loss")
    axes[1].plot(xs, [h["train_acc"] for h in history], label="train")
    axes[1].plot(xs, [h["val_acc"] for h in history], label="val")
    axes[1].set_title("Accuracy")
    n_head = sum(1 for h in history if h["phase"] == "head")
    for ax in axes:
        ax.set_xlabel("epoch")
        if 0 < n_head < len(history):
            ax.axvline(n_head + 0.5, color="gray", linestyle="--", linewidth=1)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=None, help="custom ImageFolder dataset (train/ val/ [test/])")
    p.add_argument("--dataset-root", type=Path, default=DATASET_DIR, help="where Food-101 is downloaded")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--arch", choices=SUPPORTED_ARCHS, default=DEFAULT_ARCH)
    p.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--head-epochs", type=int, default=3, help="phase 1 epochs (frozen backbone)")
    p.add_argument("--finetune-epochs", type=int, default=12, help="phase 2 epochs (all layers)")
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-finetune", type=float, default=2e-4, help="backbone LR in phase 2 (head gets 5x)")
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--max-samples", type=int, default=None, help="limit each split (for quick tests)")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--no-pretrained", action="store_true", help="train from scratch (not recommended)")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = pick_device()
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | AMP: {use_amp} | arch: {args.arch}")

    train_loader, val_loader, _, classes = make_loaders(
        batch_size=args.batch_size, workers=args.workers, img_size=args.img_size,
        dataset_root=args.dataset_root, data_dir=args.data_dir, max_samples=args.max_samples)
    print(f"Classes: {len(classes)} | train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    model = build_model(args.arch, len(classes), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    history, best_val, global_epoch = [], 0.0, 0
    ckpt_path = args.output_dir / "best.pt"

    def run_phase(phase: str, epochs: int, optimizer, scheduler, backbone_frozen: bool):
        nonlocal best_val, global_epoch
        for epoch in range(1, epochs + 1):
            start = time.time()
            set_train_mode(model, args.arch, backbone_frozen)
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp,
                                        desc=f"{phase} {epoch}/{epochs} train")
            model.eval()
            va_loss, va_acc = run_epoch(model, val_loader, criterion, device, use_amp=use_amp,
                                        desc=f"{phase} {epoch}/{epochs} val")
            if scheduler is not None:
                scheduler.step()
            global_epoch += 1
            row = {"phase": phase, "epoch": global_epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                   "val_loss": va_loss, "val_acc": va_acc, "lr": optimizer.param_groups[0]["lr"],
                   "seconds": round(time.time() - start, 1)}
            history.append(row)
            flag = ""
            if va_acc > best_val:
                best_val = va_acc
                torch.save({"model_state": model.state_dict(), "arch": args.arch, "classes": classes,
                            "img_size": args.img_size, "val_acc": va_acc, "epoch": global_epoch}, ckpt_path)
                flag = "  * saved best"
            print(f"[{phase}] epoch {global_epoch}: train {tr_loss:.3f}/{tr_acc:.3f} | "
                  f"val {va_loss:.3f}/{va_acc:.3f} | {row['seconds']}s{flag}")

    # Phase 1: head only
    if args.head_epochs > 0:
        set_backbone_trainable(model, args.arch, False)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=args.lr_head, weight_decay=args.weight_decay)
        run_phase("head", args.head_epochs, opt, None, backbone_frozen=True)

    # Phase 2: fine-tune everything
    if args.finetune_epochs > 0:
        set_backbone_trainable(model, args.arch, True)
        backbone, head = split_parameters(model, args.arch)
        opt = torch.optim.AdamW(
            [{"params": backbone, "lr": args.lr_finetune},
             {"params": head, "lr": args.lr_finetune * 5}],
            weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.finetune_epochs)
        run_phase("finetune", args.finetune_epochs, opt, sched, backbone_frozen=False)

    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    if history:
        plot_history(history, args.output_dir / "training_curves.png")
    print(f"Done. Best validation accuracy: {best_val:.4f}")
    print(f"Checkpoint: {ckpt_path}\nNext: python -m foodnutri.evaluate")


if __name__ == "__main__":
    main()
