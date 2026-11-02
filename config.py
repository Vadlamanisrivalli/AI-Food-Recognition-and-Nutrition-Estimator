"""Project-wide paths and defaults."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATASET_DIR = ROOT / "datasets"      # Food-101 is downloaded here
OUTPUT_DIR = ROOT / "outputs"        # checkpoints, plots, reports
NUTRITION_CSV = DATA_DIR / "nutrition.csv"
DEFAULT_CHECKPOINT = OUTPUT_DIR / "best.pt"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_ARCH = "efficientnet_b0"
DEFAULT_IMG_SIZE = 224
SEED = 42
