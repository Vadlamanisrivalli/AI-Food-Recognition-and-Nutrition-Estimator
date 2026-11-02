"""Nutrition lookup: per-100 g values for each food class, scaled by portion size.

The numbers in data/nutrition.csv are typical approximations compiled for a
class project. They are not clinical values; real dishes vary a lot.
"""
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from .config import NUTRITION_CSV


def normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def pretty_name(name: str) -> str:
    return normalize_name(name).replace("_", " ").title()


@dataclass(frozen=True)
class NutritionInfo:
    name: str
    kcal_per_100g: float
    protein_per_100g: float
    carbs_per_100g: float
    fat_per_100g: float
    serving_g: float

    @property
    def label(self) -> str:
        return pretty_name(self.name)

    def for_grams(self, grams: Optional[float] = None) -> Dict[str, float]:
        """Nutrition for a portion. grams=None uses the typical serving size."""
        g = self.serving_g if grams is None else float(grams)
        f = g / 100.0
        return {
            "grams": round(g, 1),
            "kcal": round(self.kcal_per_100g * f, 1),
            "protein_g": round(self.protein_per_100g * f, 1),
            "carbs_g": round(self.carbs_per_100g * f, 1),
            "fat_g": round(self.fat_per_100g * f, 1),
        }


class NutritionDB:
    def __init__(self, path: Path = NUTRITION_CSV):
        self._items: Dict[str, NutritionInfo] = {}
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                item = NutritionInfo(
                    name=normalize_name(row["class"]),
                    kcal_per_100g=float(row["kcal_per_100g"]),
                    protein_per_100g=float(row["protein_g"]),
                    carbs_per_100g=float(row["carbs_g"]),
                    fat_per_100g=float(row["fat_g"]),
                    serving_g=float(row["serving_g"]),
                )
                self._items[item.name] = item

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, name: str) -> bool:
        return normalize_name(name) in self._items

    def names(self) -> List[str]:
        return list(self._items)

    def get(self, name: str) -> NutritionInfo:
        key = normalize_name(name)
        if key not in self._items:
            raise KeyError(f"No nutrition data for '{name}'")
        return self._items[key]

    def estimate(self, name: str, grams: Optional[float] = None) -> Dict[str, float]:
        """Nutrition for one food class (typical serving unless grams is given)."""
        info = self.get(name)
        out = info.for_grams(grams)
        out["label"] = info.label
        return out

    def estimate_weighted(
        self, probs: Mapping[str, float], grams: Optional[float] = None
    ) -> Dict[str, float]:
        """Probability-weighted nutrition across several candidate classes.

        Useful when the model is unsure: instead of trusting only the top-1
        class, blend the per-100 g values by the (renormalised) probabilities.
        If grams is None, the blended typical serving size is used.
        """
        items = [(self.get(n), p) for n, p in probs.items() if n in self and p > 0]
        if not items:
            raise ValueError("No known classes with positive probability")
        total = sum(p for _, p in items)
        w = [(info, p / total) for info, p in items]

        def blend(attr: str) -> float:
            return sum(getattr(info, attr) * p for info, p in w)

        blended = NutritionInfo(
            name="blend",
            kcal_per_100g=blend("kcal_per_100g"),
            protein_per_100g=blend("protein_per_100g"),
            carbs_per_100g=blend("carbs_per_100g"),
            fat_per_100g=blend("fat_per_100g"),
            serving_g=blend("serving_g"),
        )
        out = blended.for_grams(grams)
        out["label"] = f"Blend of top {len(items)} predictions"
        return out
