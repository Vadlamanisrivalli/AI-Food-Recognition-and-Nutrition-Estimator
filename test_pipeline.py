"""End-to-end smoke tests with randomly initialised weights (no dataset or internet needed)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from PIL import Image

from foodnutri.classes import FOOD101_CLASSES
from foodnutri.model import SUPPORTED_ARCHS, build_model, gradcam_target_layer, set_backbone_trainable
from foodnutri.predict import FoodRecognizer, format_report


@pytest.mark.parametrize("arch", SUPPORTED_ARCHS)
def test_forward_shape_and_freezing(arch):
    model = build_model(arch, 101, pretrained=False).eval()
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 101)

    set_backbone_trainable(model, arch, False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert 0 < trainable < total
    set_backbone_trainable(model, arch, True)
    assert all(p.requires_grad for p in model.parameters())


@pytest.fixture(scope="module")
def recognizer(tmp_path_factory):
    path = tmp_path_factory.mktemp("ckpt") / "best.pt"
    arch = "mobilenet_v3_large"
    model = build_model(arch, 101, pretrained=False)
    torch.save({"model_state": model.state_dict(), "arch": arch, "classes": FOOD101_CLASSES,
                "img_size": 224, "val_acc": 0.0, "epoch": 0}, path)
    return FoodRecognizer(path, device="cpu")


def _image():
    return Image.new("RGB", (320, 200), (200, 120, 60))


def test_probabilities_sum_to_one(recognizer):
    probs = recognizer.predict_proba(_image())
    assert probs.shape == (101,)
    assert probs.sum() == pytest.approx(1.0, abs=1e-4)


def test_analyze_top1_and_weighted(recognizer):
    r1 = recognizer.analyze(_image(), topk=3, grams=150)
    assert len(r1["predictions"]) == 3
    assert r1["nutrition"]["grams"] == 150 and r1["nutrition"]["mode"] == "top-1"
    r2 = recognizer.analyze(_image(), topk=3, weighted=True)
    assert r2["nutrition"]["mode"] == "weighted"
    assert "Top predictions" in format_report(r2)


def test_gradcam_overlay_matches_model_crop(recognizer):
    overlay = recognizer.gradcam(_image())
    assert overlay.size == (224, 224)


def test_gradcam_target_layers_exist():
    for arch in SUPPORTED_ARCHS:
        model = build_model(arch, 10, pretrained=False)
        assert gradcam_target_layer(model, arch) is not None
