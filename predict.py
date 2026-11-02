"""Inference: image -> food class(es) -> nutrition estimate.

CLI:
    python -m foodnutri.predict photo.jpg
    python -m foodnutri.predict photo.jpg --grams 250 --topk 5 --weighted --gradcam cam.png
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageOps

from .config import DEFAULT_CHECKPOINT
from .dataset import display_transform, eval_transform
from .gradcam import GradCAM, overlay_heatmap
from .model import gradcam_target_layer, load_checkpoint
from .nutrition import NutritionDB, pretty_name


def _auto_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FoodRecognizer:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device=None, nutrition: Optional[NutritionDB] = None):
        self.device = torch.device(device) if device else _auto_device()
        self.model, self.meta = load_checkpoint(checkpoint, self.device)
        self.classes: List[str] = self.meta["classes"]
        self.arch: str = self.meta["arch"]
        self.img_size: int = self.meta["img_size"]
        self.nutrition = nutrition or NutritionDB()
        self._transform = eval_transform(self.img_size)
        self._display = display_transform(self.img_size)

    @staticmethod
    def _prepare(image: Image.Image) -> Image.Image:
        return ImageOps.exif_transpose(image).convert("RGB")

    def _tensor(self, image: Image.Image) -> torch.Tensor:
        return self._transform(self._prepare(image)).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict_proba(self, image: Image.Image) -> np.ndarray:
        logits = self.model(self._tensor(image))
        return torch.softmax(logits, dim=1)[0].cpu().numpy()

    def predict(self, image: Image.Image, topk: int = 5) -> List[Dict]:
        probs = self.predict_proba(image)
        order = np.argsort(probs)[::-1][:topk]
        return [{"class": self.classes[i], "label": pretty_name(self.classes[i]), "prob": float(probs[i])}
                for i in order]

    def analyze(self, image: Image.Image, topk: int = 5, grams: Optional[float] = None,
                weighted: bool = False) -> Dict:
        """Top-k predictions plus a nutrition estimate.

        grams=None uses the typical serving size. weighted=True blends the
        nutrition of the top-k classes by their probabilities.
        """
        preds = self.predict(image, topk)
        known = {p["class"]: p["prob"] for p in preds if p["class"] in self.nutrition}
        if weighted and known:
            nutrition = self.nutrition.estimate_weighted(known, grams)
            nutrition["mode"] = "weighted"
        elif preds[0]["class"] in self.nutrition:
            nutrition = self.nutrition.estimate(preds[0]["class"], grams)
            nutrition["mode"] = "top-1"
        else:
            nutrition = None
        return {"predictions": preds, "nutrition": nutrition}

    def gradcam(self, image: Image.Image, class_idx: Optional[int] = None) -> Image.Image:
        """Heatmap overlay on the cropped view the model actually saw."""
        image = self._prepare(image)
        cam_engine = GradCAM(self.model, gradcam_target_layer(self.model, self.arch))
        try:
            cam, _ = cam_engine(self._transform(image).unsqueeze(0).to(self.device), class_idx)
        finally:
            cam_engine.remove()
        return overlay_heatmap(self._display(image), cam)


def format_report(result: Dict) -> str:
    lines = ["Top predictions:"]
    for p in result["predictions"]:
        lines.append(f"  {p['label']:<28s} {p['prob'] * 100:5.1f}%")
    n = result["nutrition"]
    if n:
        lines += ["", f"Nutrition estimate ({n['mode']}): {n['label']}, {n['grams']} g",
                  f"  Calories : {n['kcal']} kcal",
                  f"  Protein  : {n['protein_g']} g",
                  f"  Carbs    : {n['carbs_g']} g",
                  f"  Fat      : {n['fat_g']} g"]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", type=Path)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--grams", type=float, default=None, help="portion size in grams (default: typical serving)")
    p.add_argument("--weighted", action="store_true", help="blend nutrition across top-k predictions")
    p.add_argument("--gradcam", type=Path, default=None, help="save a Grad-CAM overlay to this path")
    args = p.parse_args(argv)

    recognizer = FoodRecognizer(args.checkpoint)
    img = Image.open(args.image)
    print(format_report(recognizer.analyze(img, args.topk, args.grams, args.weighted)))
    if args.gradcam:
        recognizer.gradcam(img).save(args.gradcam)
        print(f"\nGrad-CAM saved to {args.gradcam}")


if __name__ == "__main__":
    main()
