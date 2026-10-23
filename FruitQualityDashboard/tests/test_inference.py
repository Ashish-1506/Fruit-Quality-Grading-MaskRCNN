import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import associate_defects_and_calculate_areas, filter_predictions


def test_synthetic_single_fruit_single_defect():
    """Exact defect percentage on a synthetic 100x100 grid."""
    h, w = 100, 100

    fruit_mask = np.zeros((h, w), dtype=bool)
    fruit_mask[20:60, 20:70] = True  # 40x50 = 2000 px

    rotten_mask = np.zeros((h, w), dtype=bool)
    rotten_mask[30:40, 30:40] = True  # 10x10 = 100 px inside fruit

    filtered = {
        "fruit_masks": np.array([fruit_mask]),
        "fruit_scores": np.array([0.95]),
        "fruit_boxes": np.array([[20, 20, 70, 60]]),
        "rotten_masks": np.array([rotten_mask]),
        "rotten_scores": np.array([0.90]),
        "rotten_boxes": np.array([[30, 30, 40, 40]]),
    }

    res = associate_defects_and_calculate_areas(filtered, image_shape=(h, w))
    f0 = res["fruits"][0]
    assert len(res["fruits"]) == 1
    assert f0["fruit_pixels"] == 2000
    assert f0["rotten_pixels"] == 100
    assert f0["defect_percentage"] == pytest.approx(5.0, rel=1e-5)
    assert f0["grading"]["grade"] == "Grade B"
    assert len(res["orphan_rotten_instances"]) == 0


def test_synthetic_defect_partly_outside_fruit():
    """Only defect pixels inside the fruit mask are counted."""
    h, w = 100, 100

    fruit_mask = np.zeros((h, w), dtype=bool)
    fruit_mask[0:50, 0:50] = True  # 2500 px

    rotten_mask = np.zeros((h, w), dtype=bool)
    rotten_mask[40:60, 40:60] = True  # 400 px total; 100 px inside fruit

    filtered = {
        "fruit_masks": np.array([fruit_mask]),
        "fruit_scores": np.array([0.90]),
        "fruit_boxes": np.array([[0, 0, 50, 50]]),
        "rotten_masks": np.array([rotten_mask]),
        "rotten_scores": np.array([0.80]),
        "rotten_boxes": np.array([[40, 40, 60, 60]]),
    }

    res = associate_defects_and_calculate_areas(filtered, image_shape=(h, w), association_threshold=0.20)
    f0 = res["fruits"][0]
    assert f0["rotten_pixels"] == 100
    assert f0["fruit_pixels"] == 2500
    assert f0["defect_percentage"] == pytest.approx(4.0, rel=1e-5)
    assert f0["grading"]["grade"] == "Grade A"


def test_overlapping_defects_union_merged():
    """Overlapping defect instances assigned to the same fruit are merged (no double counting)."""
    h, w = 100, 100
    fruit_mask = np.zeros((h, w), dtype=bool)
    fruit_mask[10:90, 10:90] = True  # 6400 px

    r1 = np.zeros((h, w), dtype=bool)
    r1[20:40, 20:40] = True  # 400 px

    r2 = np.zeros((h, w), dtype=bool)
    r2[30:50, 30:50] = True  # 400 px; overlaps r1 by 100 px -> union = 700 px

    filtered = {
        "fruit_masks": np.array([fruit_mask]),
        "fruit_scores": np.array([0.92]),
        "fruit_boxes": np.array([[10, 10, 90, 90]]),
        "rotten_masks": np.array([r1, r2]),
        "rotten_scores": np.array([0.85, 0.82]),
        "rotten_boxes": np.array([[20, 20, 40, 40], [30, 30, 50, 50]]),
    }

    res = associate_defects_and_calculate_areas(filtered, image_shape=(h, w))
    f0 = res["fruits"][0]
    assert f0["rotten_pixels"] == 700
    assert f0["fruit_pixels"] == 6400
    assert f0["defect_percentage"] == pytest.approx(100.0 * 700 / 6400, rel=1e-5)


def test_overlapping_fruits_exclusive_ownership():
    """Higher-confidence fruit claims overlapping pixels first."""
    h, w = 100, 100

    f1 = np.zeros((h, w), dtype=bool)
    f1[0:60, 0:60] = True  # 3600 px

    f2 = np.zeros((h, w), dtype=bool)
    f2[30:90, 30:90] = True  # 3600 px; overlap with f1: [30:60, 30:60] = 900 px

    filtered = {
        "fruit_masks": np.array([f1, f2]),
        "fruit_scores": np.array([0.95, 0.80]),
        "fruit_boxes": np.array([[0, 0, 60, 60], [30, 30, 90, 90]]),
        "rotten_masks": np.zeros((0, h, w), dtype=bool),
        "rotten_scores": np.zeros(0),
        "rotten_boxes": np.zeros((0, 4)),
    }

    res = associate_defects_and_calculate_areas(filtered, image_shape=(h, w))
    fruits = res["fruits"]
    assert fruits[0]["fruit_pixels"] == 3600
    assert fruits[1]["fruit_pixels"] == 2700
    assert fruits[0]["fruit_pixels"] + fruits[1]["fruit_pixels"] == int((f1 | f2).sum())


def test_orphan_defect_outside_fruit():
    """Defects outside any fruit mask are recorded as orphans."""
    h, w = 100, 100
    fruit_mask = np.zeros((h, w), dtype=bool)
    fruit_mask[0:30, 0:30] = True

    rotten_mask = np.zeros((h, w), dtype=bool)
    rotten_mask[70:90, 70:90] = True

    filtered = {
        "fruit_masks": np.array([fruit_mask]),
        "fruit_scores": np.array([0.90]),
        "fruit_boxes": np.array([[0, 0, 30, 30]]),
        "rotten_masks": np.array([rotten_mask]),
        "rotten_scores": np.array([0.75]),
        "rotten_boxes": np.array([[70, 70, 90, 90]]),
    }

    res = associate_defects_and_calculate_areas(filtered, image_shape=(h, w))
    assert len(res["orphan_rotten_instances"]) == 1
    assert res["orphan_rotten_instances"][0]["rotten_index"] == 0
    assert res["fruits"][0]["rotten_pixels"] == 0
    assert res["fruits"][0]["defect_percentage"] == 0.0


def test_small_fruit_mask_handling():
    """Fruit masks below min_fruit_pixels threshold are flagged as unavailable."""
    h, w = 100, 100
    fruit_mask = np.zeros((h, w), dtype=bool)
    fruit_mask[10:14, 10:14] = True  # 16 px

    filtered = {
        "fruit_masks": np.array([fruit_mask]),
        "fruit_scores": np.array([0.88]),
        "fruit_boxes": np.array([[10, 10, 14, 14]]),
        "rotten_masks": np.zeros((0, h, w), dtype=bool),
        "rotten_scores": np.zeros(0),
        "rotten_boxes": np.zeros((0, 4)),
    }

    res = associate_defects_and_calculate_areas(filtered, image_shape=(h, w), min_fruit_pixels=50)
    f0 = res["fruits"][0]
    assert f0["status"] == "invalid_small_fruit_mask"
    assert f0["defect_percentage"] is None
    assert f0["grading"]["grade"] == "UNAVAILABLE"
