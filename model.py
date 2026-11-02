"""Model construction (transfer learning) and checkpoint loading."""
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision import models

SUPPORTED_ARCHS = ("efficientnet_b0", "resnet50", "mobilenet_v3_large")


def build_model(arch: str, num_classes: int, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """ImageNet-pretrained backbone with a new classification head."""
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif arch == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[2] = nn.Dropout(dropout)
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported arch '{arch}'. Choose from {SUPPORTED_ARCHS}")
    return model


def head_module(model: nn.Module, arch: str) -> nn.Module:
    return model.fc if arch == "resnet50" else model.classifier


def split_parameters(model: nn.Module, arch: str) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Return (backbone_params, head_params)."""
    head_ids = {id(p) for p in head_module(model, arch).parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    head = [p for p in model.parameters() if id(p) in head_ids]
    return backbone, head


def set_backbone_trainable(model: nn.Module, arch: str, trainable: bool) -> None:
    backbone, head = split_parameters(model, arch)
    for p in backbone:
        p.requires_grad = trainable
    for p in head:
        p.requires_grad = True


def gradcam_target_layer(model: nn.Module, arch: str) -> nn.Module:
    """Last convolutional block, used by Grad-CAM."""
    if arch == "resnet50":
        return model.layer4[-1]
    return model.features[-1]


def load_checkpoint(path, device="cpu"):
    """Load a checkpoint written by train.py. Returns (model_in_eval_mode, meta_dict)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Train a model first (python -m foodnutri.train).")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != "model_state"}
    return model, meta
