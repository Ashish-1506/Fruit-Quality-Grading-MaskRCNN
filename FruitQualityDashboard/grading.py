from typing import Any, Dict, List, Optional

CONDITION_BY_PERCENTAGE = "Grade by defect percentage"
CONDITION_ROTTEN_FORCE_REJECT = "Rotten - force reject"
VALID_CONDITIONS = [CONDITION_BY_PERCENTAGE, CONDITION_ROTTEN_FORCE_REJECT]

GRADE_COLORS = {
    "Grade A": "#059669",       # Emerald Green
    "Grade B": "#0284C7",       # Refined Sky Blue
    "Grade C": "#D97706",       # Warm Amber
    "REJECT": "#DC2626",        # Crisp Crimson Red
    "UNAVAILABLE": "#64748B",   # Slate Gray
}


def grade_fruit(
    defect_percentage: Optional[float],
    condition: str = CONDITION_BY_PERCENTAGE,
) -> Dict[str, Any]:
    """Grade an individual fruit instance according to the project grading rules.

    Rules:
    - 0 <= percentage < 5:   Grade A
    - 5 <= percentage < 15:  Grade B
    - 15 <= percentage <= 25: Grade C
    - percentage > 25:       REJECT
    - Rotten condition:      REJECT regardless of percentage
    - Unavailable percentage: UNAVAILABLE

    Args:
        defect_percentage: Unrounded percentage of defective pixels (0.0 to 100.0), or None.
        condition: CONDITION_BY_PERCENTAGE or CONDITION_ROTTEN_FORCE_REJECT.

    Returns:
        dict with keys: grade, defect_percentage, applied_condition, reason, color, is_rejected
    """
    if condition == CONDITION_ROTTEN_FORCE_REJECT:
        return {
            "grade": "REJECT",
            "defect_percentage": defect_percentage,
            "applied_condition": condition,
            "reason": "Overridden as Rotten - immediate REJECT independent of defect percentage.",
            "color": GRADE_COLORS["REJECT"],
            "is_rejected": True,
        }

    if defect_percentage is None:
        return {
            "grade": "UNAVAILABLE",
            "defect_percentage": None,
            "applied_condition": condition,
            "reason": "Fruit mask is empty or too small (< minimum threshold) for reliable surface quantification.",
            "color": GRADE_COLORS["UNAVAILABLE"],
            "is_rejected": False,
        }

    pct = max(0.0, min(100.0, float(defect_percentage)))

    if 0.0 <= pct < 5.0:
        grade = "Grade A"
        reason = f"Defect {pct:.2f}% (< 5%) - Premium Quality"
        is_rejected = False
    elif 5.0 <= pct < 15.0:
        grade = "Grade B"
        reason = f"Defect {pct:.2f}% (5% to < 15%) - Standard Quality"
        is_rejected = False
    elif 15.0 <= pct <= 25.0:
        grade = "Grade C"
        reason = f"Defect {pct:.2f}% (15% to 25%, inclusive) - Marginal Quality"
        is_rejected = False
    else:
        grade = "REJECT"
        reason = f"Defect {pct:.2f}% (> 25%) - Threshold exceeded"
        is_rejected = True

    return {
        "grade": grade,
        "defect_percentage": pct,
        "applied_condition": condition,
        "reason": reason,
        "color": GRADE_COLORS[grade],
        "is_rejected": is_rejected,
    }


def calculate_image_defect_aggregate(
    fruit_records: List[Dict[str, Any]],
) -> Optional[float]:
    """Calculate the image-level area-weighted defect percentage.

    Returns 100.0 * total defective pixels / total valid fruit-mask pixels,
    or None if no valid fruit pixels exist.
    """
    total_fruit_px = 0
    total_rotten_px = 0

    for item in fruit_records:
        f_px = item.get("fruit_pixels")
        r_px = item.get("rotten_pixels")
        if f_px is not None and f_px > 0 and r_px is not None:
            total_fruit_px += int(f_px)
            total_rotten_px += int(r_px)

    if total_fruit_px == 0:
        return None

    pct = 100.0 * float(total_rotten_px) / float(total_fruit_px)
    return max(0.0, min(100.0, pct))
