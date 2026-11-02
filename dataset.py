"""Datasets, transforms and dataloaders.

Default dataset is Food-101 (101 classes, 101k images) downloaded through
torchvision. A custom dataset in ImageFolder layout also works:

    my_data/
        train/<class_name>/*.jpg
        val/<class_name>/*.jpg
        test/<class_name>/*.jpg     (optional, falls back to val)
"""
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .config import DATASET_DIR, IMAGENET_MEAN, IMAGENET_STD, SEED


def eval_resize_size(img_size: int) -> int:
    return int(round(img_size * 256 / 224))


def train_transform(img_size: int):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(eval_resize_size(img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def display_transform(img_size: int):
    """Same crop the model sees, but still a PIL image (used for Grad-CAM overlays)."""
    return transforms.Compose([
        transforms.Resize(eval_resize_size(img_size)),
        transforms.CenterCrop(img_size),
    ])


def _split_indices(n: int, val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    n_val = int(n * val_fraction)
    return perm[n_val:], perm[:n_val]


def _limit(ds: Dataset, max_samples: Optional[int], seed: int) -> Dataset:
    if not max_samples or max_samples >= len(ds):
        return ds
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=gen)[:max_samples].tolist()
    return Subset(ds, idx)


def build_datasets(
    dataset_root: Path = DATASET_DIR,
    data_dir: Optional[Path] = None,
    img_size: int = 224,
    val_fraction: float = 0.1,
    download: bool = True,
):
    """Return (train_ds, val_ds, test_ds, class_names)."""
    if data_dir is not None:
        data_dir = Path(data_dir)
        train_ds = datasets.ImageFolder(data_dir / "train", transform=train_transform(img_size))
        val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_transform(img_size))
        test_dir = data_dir / "test" if (data_dir / "test").exists() else data_dir / "val"
        test_ds = datasets.ImageFolder(test_dir, transform=eval_transform(img_size))
        if val_ds.classes != train_ds.classes or test_ds.classes != train_ds.classes:
            raise ValueError("train/val/test folders must contain the same class folders")
        return train_ds, val_ds, test_ds, train_ds.classes

    dataset_root = Path(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    train_aug = datasets.Food101(dataset_root, split="train",
                                 transform=train_transform(img_size), download=download)
    train_eval = datasets.Food101(dataset_root, split="train",
                                  transform=eval_transform(img_size), download=False)
    test_ds = datasets.Food101(dataset_root, split="test",
                               transform=eval_transform(img_size), download=False)
    train_idx, val_idx = _split_indices(len(train_aug), val_fraction, SEED)
    return Subset(train_aug, train_idx), Subset(train_eval, val_idx), test_ds, list(train_aug.classes)


def make_loaders(
    batch_size: int = 64,
    workers: int = 4,
    img_size: int = 224,
    dataset_root: Path = DATASET_DIR,
    data_dir: Optional[Path] = None,
    max_samples: Optional[int] = None,
):
    """Return (train_loader, val_loader, test_loader, class_names).

    max_samples limits every split to that many images (handy for quick tests).
    """
    train_ds, val_ds, test_ds, classes = build_datasets(
        dataset_root=dataset_root, data_dir=data_dir, img_size=img_size)
    train_ds = _limit(train_ds, max_samples, SEED)
    val_ds = _limit(val_ds, max_samples, SEED)
    test_ds = _limit(test_ds, max_samples, SEED)

    common = dict(batch_size=batch_size, num_workers=workers,
                  pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader, classes
