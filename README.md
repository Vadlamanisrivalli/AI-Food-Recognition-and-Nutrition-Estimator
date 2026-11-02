# AI Food Recognition & Nutrition Estimator

A deep-learning project that looks at a photo of a meal, **recognises the dish** (101 classes, Food-101) and
**estimates calories, protein, carbs and fat** for a chosen portion size.

- **Model:** ImageNet-pretrained CNN (EfficientNet-B0 by default; ResNet-50 and MobileNetV3 also supported)
  fine-tuned with two-phase transfer learning.
- **Nutrition:** per-100 g lookup table (`data/nutrition.csv`) scaled by portion size, with an optional
  probability-weighted estimate across the top-k predictions.
- **Explainability:** Grad-CAM heatmaps show which parts of the photo drove the prediction.
- **Demo:** Streamlit web app with upload and camera input.

## Project structure

```
food_nutrition_ai/
├── app.py                  # Streamlit demo
├── foodnutri/
│   ├── config.py           # paths and defaults
│   ├── classes.py          # the 101 Food-101 class names
│   ├── dataset.py          # Food-101 / custom ImageFolder loaders + augmentation
│   ├── model.py            # model builders, freezing helpers, checkpoint loading
│   ├── train.py            # two-phase training (head, then fine-tune)
│   ├── evaluate.py         # top-1/top-5, per-class report, confusion matrix
│   ├── nutrition.py        # nutrition lookup and portion scaling
│   ├── gradcam.py          # Grad-CAM implementation
│   └── predict.py          # inference class + command-line tool
├── data/nutrition.csv      # kcal / protein / carbs / fat per 100 g + typical serving
├── notebooks/train_colab.ipynb   # run training on a free Colab GPU
├── tests/                  # pytest suite
├── outputs/                # checkpoints, plots, reports are written here
└── requirements.txt
```

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.9+ is fine. A GPU is strongly recommended for training (a free Colab T4 works).

## 2. Train

```bash
python -m foodnutri.train
```

The first run downloads Food-101 (about 4.7 GB) into `datasets/`. Training uses 90% of the official train
split for training and 10% for validation; the official test split is kept for the final evaluation.

Useful options:

```bash
python -m foodnutri.train --arch resnet50 --batch-size 32
python -m foodnutri.train --head-epochs 3 --finetune-epochs 15 --img-size 260
python -m foodnutri.train --data-dir my_data            # your own ImageFolder dataset
python -m foodnutri.train --max-samples 512 --head-epochs 1 --finetune-epochs 1   # 2-minute sanity check
```

What happens during training:

1. **Phase 1 (head only):** the backbone is frozen and only the new classifier is trained (LR 1e-3).
2. **Phase 2 (fine-tune):** all layers are unfrozen, AdamW with a small backbone LR (2e-4), head LR 5x higher,
   cosine decay. Label smoothing 0.1, mixed precision on CUDA.
3. The best checkpoint by validation accuracy is saved to `outputs/best.pt`, plus `history.json` and
   `training_curves.png`.

## 3. Evaluate

```bash
python -m foodnutri.evaluate
```

Writes `metrics.json`, `classification_report.txt`, `confusion_matrix.png` and `top_confusions.csv` to `outputs/`.
Reports top-1 and top-5 accuracy on the official test set and lists the most confused dish pairs.

## 4. Predict from the command line

```bash
python -m foodnutri.predict photo.jpg
python -m foodnutri.predict photo.jpg --grams 250 --topk 5 --weighted --gradcam cam.png
```

## 5. Run the web app

```bash
streamlit run app.py
```

Upload a photo or use the camera, pick a portion size (typical serving or custom grams), and optionally
turn on the Grad-CAM view.

## 6. Tests

```bash
pytest
```

The nutrition tests check that all 101 classes have plausible, internally consistent values. The pipeline tests
build randomly initialised models, so they need no dataset or internet.

## Important notes and limitations

- **Nutrition values are approximations.** `data/nutrition.csv` holds typical per-100 g figures for each dish
  type. Real calories vary widely with recipe, oil, sauce and size. For a real product, replace the table with a
  source such as USDA FoodData Central. The app labels results as estimates and is not dietary advice.
- **Portion size is not measured from the image.** It comes from a typical serving or from the grams the user
  enters. Estimating portion size from a photo (depth, reference objects, segmentation) is a natural extension.
- **Closed vocabulary.** The model only knows the 101 Food-101 classes; anything else is forced into the
  nearest one. A confidence threshold or an "unknown food" class would help.
- Several classes are visually similar (steak vs. filet mignon vs. prime rib, tiramisu vs. chocolate mousse),
  so expect most errors there.

## Ideas to improve results

- Larger backbones (EfficientNet-B3/B4, ConvNeXt, ViT) and higher resolution (260-384 px).
- Longer fine-tuning, MixUp/CutMix, EMA of weights, test-time augmentation.
- Train on a food-specific dataset with nutrition labels (e.g. Nutrition5k) to predict calories directly.
- Add a food-segmentation step to estimate portion size per item on a mixed plate.

## Suggested report outline

1. Problem statement and motivation (diet tracking, health apps)
2. Dataset: Food-101 (101 classes, 1,000 images each; 750 train / 250 test)
3. Method: transfer learning, augmentation, two-phase training, loss and optimiser choices
4. Nutrition estimation: lookup table, portion scaling, probability-weighted variant
5. Results: training curves, top-1/top-5 accuracy, confusion matrix, hardest classes, Grad-CAM examples
6. Ablations (arch comparison, frozen vs. fine-tuned, image size)
7. Limitations and future work (see above)
