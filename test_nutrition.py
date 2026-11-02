import pytest

from foodnutri.classes import FOOD101_CLASSES
from foodnutri.nutrition import NutritionDB


@pytest.fixture(scope="module")
def db():
    return NutritionDB()


def test_class_list_has_101_unique_sorted_names():
    assert len(FOOD101_CLASSES) == 101
    assert len(set(FOOD101_CLASSES)) == 101
    assert FOOD101_CLASSES == sorted(FOOD101_CLASSES)


def test_every_class_has_nutrition_and_nothing_extra(db):
    assert set(db.names()) == set(FOOD101_CLASSES)
    assert len(db) == 101


def test_values_are_plausible(db):
    for name in db.names():
        n = db.get(name)
        assert 20 <= n.kcal_per_100g <= 600, name
        assert n.protein_per_100g >= 0 and n.carbs_per_100g >= 0 and n.fat_per_100g >= 0, name
        assert n.protein_per_100g + n.carbs_per_100g + n.fat_per_100g <= 100, name
        assert 20 <= n.serving_g <= 600, name
        # Energy should roughly match the Atwater factors (4/4/9 kcal per g).
        atwater = 4 * n.protein_per_100g + 4 * n.carbs_per_100g + 9 * n.fat_per_100g
        assert abs(atwater - n.kcal_per_100g) / n.kcal_per_100g <= 0.15, (name, atwater, n.kcal_per_100g)


def test_estimate_scales_linearly(db):
    a = db.estimate("pizza", 100)
    b = db.estimate("pizza", 200)
    assert b["kcal"] == pytest.approx(2 * a["kcal"], abs=0.2)
    assert a["label"] == "Pizza"


def test_default_uses_typical_serving(db):
    assert db.estimate("pizza")["grams"] == db.get("pizza").serving_g


def test_name_normalisation(db):
    assert "Apple Pie" in db and db.get("apple-pie").name == "apple_pie"


def test_unknown_class_raises(db):
    with pytest.raises(KeyError):
        db.get("not_a_food")


def test_weighted_estimate_lies_between_candidates(db):
    pizza = db.estimate("pizza", 100)["kcal"]
    salad = db.estimate("caesar_salad", 100)["kcal"]
    mix = db.estimate_weighted({"pizza": 0.5, "caesar_salad": 0.5}, 100)["kcal"]
    assert min(pizza, salad) <= mix <= max(pizza, salad)
    only = db.estimate_weighted({"pizza": 0.9}, 100)["kcal"]
    assert only == pytest.approx(pizza, abs=0.2)
