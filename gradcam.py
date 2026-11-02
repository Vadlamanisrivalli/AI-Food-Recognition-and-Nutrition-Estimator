"""Minimal Grad-CAM: shows which image regions drove the model's prediction."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self._acts = None
        self._grads = None
        self._handle = target_layer.register_forward_hook(self._on_forward)

    def _on_forward(self, module, inputs, output):
        self._acts = output.detach()
        output.register_hook(lambda grad: setattr(self, "_grads", grad.detach()))

    def remove(self) -> None:
        self._handle.remove()

    def __call__(self, x: torch.Tensor, class_idx: int = None):
        """x: (1, 3, H, W) normalised tensor. Returns (cam in [0,1] of shape HxW, class_idx)."""
        self.model.eval()
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())
            self.model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), class_idx


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a [0,1] heatmap onto a PIL image of the same aspect."""
    from matplotlib import colormaps

    heat = (colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat).resize(image.size, Image.BILINEAR)
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    blended = (1 - alpha) * base + alpha * np.asarray(heat_img, dtype=np.float32)
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))
