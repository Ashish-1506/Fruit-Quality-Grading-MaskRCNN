import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grading import (
    grade_fruit,
    calculate_image_defect_aggregate,
    CONDITION_BY_PERCENTAGE,
    CONDITION_ROTTEN_FORCE_REJECT,
)


def test_grade_boundaries_disjoint_intervals():
    """Exact disjoint numerical grading intervals are applied correctly."""
    assert grade_fruit(0.0)["grade"] == "Grade A"
    assert grade_fruit(4.99)["grade"] == "Grade A"
    assert grade_fruit(4.9999)["grade"] == "Grade A"

    assert grade_fruit(5.0)["grade"] == "Grade B"
    assert not grade_fruit(5.0)["is_rejected"]
    assert grade_fruit(10.0)["grade"] == "Grade B"
    assert grade_fruit(14.99)["grade"] == "Grade B"

    assert grade_fruit(15.0)["grade"] == "Grade C"
    assert not grade_fruit(15.0)["is_rejected"]
    assert grade_fruit(20.0)["grade"] == "Grade C"
    assert grade_fruit(25.0)["grade"] == "Grade C"
    assert not grade_fruit(25.0)["is_rejected"]

    assert grade_fruit(25.0001)["grade"] == "REJECT"
    assert grade_fruit(25.0001)["is_rejected"]
    assert grade_fruit(30.0)["grade"] == "REJECT"
    assert grade_fruit(100.0)["grade"] == "REJECT"


def test_unrounded_precision():
    """Unrounded percentages are evaluated rather than prematurely rounded."""
    assert grade_fruit(4.999)["grade"] == "Grade A"
    assert grade_fruit(4.999)["defect_percentage"] == 4.999

    assert grade_fruit(14.999)["grade"] == "Grade B"
    assert grade_fruit(25.001)["grade"] == "REJECT"


def test_rotten_condition_override():
    """Rotten condition override forces REJECT regardless of percentage."""
    res = grade_fruit(0.0, condition=CONDITION_ROTTEN_FORCE_REJECT)
    assert res["grade"] == "REJECT"
    assert res["is_rejected"]
    assert res["applied_condition"] == CONDITION_ROTTEN_FORCE_REJECT

    assert grade_fruit(3.5, condition=CONDITION_ROTTEN_FORCE_REJECT)["grade"] == "REJECT"


def test_unavailable_measurements():
    """None defect percentage returns UNAVAILABLE grade."""
    res = grade_fruit(None, condition=CONDITION_BY_PERCENTAGE)
    assert res["grade"] == "UNAVAILABLE"
    assert res["defect_percentage"] is None
    assert not res["is_rejected"]


def test_image_aggregate_defect_percentage():
    """Area-weighted aggregate defect percentage is calculated correctly."""
    records = [
        {"fruit_pixels": 1000, "rotten_pixels": 50},
        {"fruit_pixels": 2000, "rotten_pixels": 250},
    ]
    assert calculate_image_defect_aggregate(records) == pytest.approx(10.0, rel=1e-5)

    records_with_none = [
        {"fruit_pixels": 1000, "rotten_pixels": 50},
        {"fruit_pixels": 20, "rotten_pixels": None},
    ]
    assert calculate_image_defect_aggregate(records_with_none) == pytest.approx(5.0, rel=1e-5)

    assert calculate_image_defect_aggregate([]) is None
