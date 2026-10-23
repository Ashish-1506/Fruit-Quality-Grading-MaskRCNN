from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from grading import grade_fruit, calculate_image_defect_aggregate, CONDITION_BY_PERCENTAGE


def load_image(image_input: Union[str, Path, bytes, BytesIO, Image.Image]) -> Tuple[Image.Image, Tuple[int, int]]:
    """Safely decode an image input into an RGB PIL Image and return (image, (width, height))."""
    if isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    elif isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found at {path}")
        image = Image.open(path).convert("RGB")
    elif isinstance(image_input, (bytes, bytearray)):
        image = Image.open(BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, BytesIO):
        image = Image.open(image_input).convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    width, height = image.size
    return image, (width, height)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert RGB PIL Image to a float tensor of shape [3, H, W] normalized to [0, 1]."""
    np_img = np.asarray(image, dtype=np.uint8)
    tensor = torch.from_numpy(np_img.copy()).permute(2, 0, 1).float().div(255.0)
    return tensor


@torch.no_grad()
def run_model_inference(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    """Execute forward inference through the Mask R-CNN model.

    Returns raw numpy arrays:
        boxes: [N, 4], scores: [N], labels: [N], masks: [N, H, W] (probabilities in [0, 1])
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    prediction = model([image_tensor.to(device)])[0]

    return {
        "boxes": prediction["boxes"].detach().cpu().numpy(),
        "scores": prediction["scores"].detach().cpu().numpy(),
        "labels": prediction["labels"].detach().cpu().numpy().astype(int),
        "masks": prediction["masks"][:, 0].detach().cpu().numpy(),
    }


def filter_predictions(
    raw_outputs: Dict[str, np.ndarray],
    fruit_score_thresh: float = 0.50,
    rotten_score_thresh: float = 0.50,
    mask_prob_thresh: float = 0.50,
) -> Dict[str, Any]:
    """Filter raw model predictions by class score thresholds and binarize masks.

    Class IDs: 1 = fruit, 2 = rotten_region
    """
    scores = raw_outputs["scores"]
    labels = raw_outputs["labels"]
    boxes = raw_outputs["boxes"]
    masks = raw_outputs["masks"]

    if len(scores) == 0:
        return {
            "fruit_boxes": np.zeros((0, 4), dtype=np.float32),
            "fruit_scores": np.zeros(0, dtype=np.float32),
            "fruit_masks": np.zeros((0, 0, 0), dtype=bool),
            "rotten_boxes": np.zeros((0, 4), dtype=np.float32),
            "rotten_scores": np.zeros(0, dtype=np.float32),
            "rotten_masks": np.zeros((0, 0, 0), dtype=bool),
        }

    bin_masks = masks >= mask_prob_thresh

    fruit_sel = (labels == 1) & (scores >= fruit_score_thresh)
    rotten_sel = (labels == 2) & (scores >= rotten_score_thresh)

    return {
        "fruit_boxes": boxes[fruit_sel],
        "fruit_scores": scores[fruit_sel],
        "fruit_masks": bin_masks[fruit_sel],
        "rotten_boxes": boxes[rotten_sel],
        "rotten_scores": scores[rotten_sel],
        "rotten_masks": bin_masks[rotten_sel],
    }


def associate_defects_and_calculate_areas(
    filtered_preds: Dict[str, Any],
    image_shape: Tuple[int, int],
    association_threshold: float = 0.30,
    min_fruit_pixels: int = 50,
) -> Dict[str, Any]:
    """Associate rotten-region instances with fruit instances and calculate defect percentages.

    Rules:
    1. Highest-confidence fruit claims overlapping pixels first (exclusive ownership).
    2. Rotten regions are assigned to the fruit with highest overlap if >= association_threshold.
    3. Unassigned rotten regions are recorded as orphan defects.
    4. Overlapping defect masks on the same fruit are merged (union) before counting pixels.
    5. Defect pixels outside the fruit mask are excluded.
    6. Fruit masks smaller than min_fruit_pixels are flagged as unavailable.
    """
    fruit_masks = filtered_preds["fruit_masks"]
    fruit_scores = filtered_preds["fruit_scores"]
    fruit_boxes = filtered_preds["fruit_boxes"]
    rotten_masks = filtered_preds["rotten_masks"]
    rotten_scores = filtered_preds["rotten_scores"]
    rotten_boxes = filtered_preds["rotten_boxes"]

    h, w = image_shape
    num_fruits = len(fruit_scores)
    num_rotten = len(rotten_scores)

    if num_fruits == 0:
        return {
            "fruits": [],
            "orphan_rotten_instances": [
                {
                    "rotten_index": i,
                    "score": float(rotten_scores[i]),
                    "box": rotten_boxes[i].tolist(),
                    "mask_pixels": int(rotten_masks[i].sum()),
                    "reason": "No fruit detected to associate with this defect.",
                }
                for i in range(num_rotten)
            ],
            "image_rotten_percent": None,
        }

    order = np.argsort(-np.asarray(fruit_scores), kind="stable")
    taken = np.zeros((h, w), dtype=bool)
    exclusive_fruits: Dict[int, np.ndarray] = {}

    for idx in order:
        idx_int = int(idx)
        raw_mask = np.asarray(fruit_masks[idx_int], dtype=bool)
        exclusive_mask = raw_mask & ~taken
        exclusive_fruits[idx_int] = exclusive_mask
        taken |= exclusive_mask

    assigned: Dict[int, List[int]] = defaultdict(list)
    orphan_instances: List[Dict[str, Any]] = []

    for r_idx in range(num_rotten):
        r_mask = np.asarray(rotten_masks[r_idx], dtype=bool)
        r_area = int(r_mask.sum())
        if r_area == 0:
            continue

        overlaps: Dict[int, float] = {}
        for f_idx, f_mask in exclusive_fruits.items():
            intersection = int((r_mask & f_mask).sum())
            if intersection > 0:
                overlaps[f_idx] = float(intersection) / float(r_area)

        if overlaps:
            best_f_idx = max(overlaps, key=overlaps.get)
            best_overlap = overlaps[best_f_idx]
            if best_overlap >= association_threshold:
                assigned[best_f_idx].append(r_idx)
            else:
                orphan_instances.append({
                    "rotten_index": r_idx,
                    "score": float(rotten_scores[r_idx]),
                    "box": rotten_boxes[r_idx].tolist(),
                    "mask_pixels": r_area,
                    "best_fruit_overlap": round(best_overlap, 3),
                    "reason": f"Maximum overlap ({best_overlap:.1%}) with Fruit #{best_f_idx + 1} below association threshold ({association_threshold:.0%}).",
                })
        else:
            orphan_instances.append({
                "rotten_index": r_idx,
                "score": float(rotten_scores[r_idx]),
                "box": rotten_boxes[r_idx].tolist(),
                "mask_pixels": r_area,
                "best_fruit_overlap": 0.0,
                "reason": "Defect does not intersect any detected fruit mask.",
            })

    fruit_results: List[Dict[str, Any]] = []

    for f_idx in range(num_fruits):
        f_mask = exclusive_fruits[f_idx]
        fruit_area = int(f_mask.sum())
        assigned_r_indices = assigned[f_idx]

        if fruit_area < min_fruit_pixels:
            fruit_results.append({
                "fruit_id": f_idx + 1,
                "confidence": float(fruit_scores[f_idx]),
                "box": fruit_boxes[f_idx].tolist(),
                "fruit_pixels": fruit_area,
                "raw_fruit_pixels": int(fruit_masks[f_idx].sum()),
                "assigned_rotten_indices": assigned_r_indices,
                "assigned_rotten_count": len(assigned_r_indices),
                "rotten_pixels": None,
                "defect_percentage": None,
                "status": "invalid_small_fruit_mask",
                "warning": f"Fruit mask area ({fruit_area} px) is below reliability threshold ({min_fruit_pixels} px).",
                "exclusive_mask": f_mask,
                "assigned_rotten_mask": np.zeros((h, w), dtype=bool),
                "grading": grade_fruit(None, CONDITION_BY_PERCENTAGE),
            })
            continue

        if assigned_r_indices:
            rotten_stack = np.stack([rotten_masks[j] for j in assigned_r_indices], axis=0)
            rotten_union = np.any(rotten_stack, axis=0) & f_mask
            rotten_area = int(rotten_union.sum())
        else:
            rotten_union = np.zeros((h, w), dtype=bool)
            rotten_area = 0

        defect_pct = min(100.0, 100.0 * float(rotten_area) / float(fruit_area))

        fruit_results.append({
            "fruit_id": f_idx + 1,
            "confidence": float(fruit_scores[f_idx]),
            "box": fruit_boxes[f_idx].tolist(),
            "fruit_pixels": fruit_area,
            "raw_fruit_pixels": int(fruit_masks[f_idx].sum()),
            "assigned_rotten_indices": assigned_r_indices,
            "assigned_rotten_count": len(assigned_r_indices),
            "rotten_pixels": rotten_area,
            "defect_percentage": defect_pct,
            "status": "ok",
            "warning": None,
            "exclusive_mask": f_mask,
            "assigned_rotten_mask": rotten_union,
            "grading": grade_fruit(defect_pct, CONDITION_BY_PERCENTAGE),
        })

    image_defect_pct = calculate_image_defect_aggregate(fruit_results)

    return {
        "fruits": fruit_results,
        "orphan_rotten_instances": orphan_instances,
        "image_rotten_percent": image_defect_pct,
    }


def analyze_fruit_image(
    model: nn.Module,
    image_input: Union[str, Path, bytes, BytesIO, Image.Image],
    fruit_score_thresh: float = 0.50,
    rotten_score_thresh: float = 0.50,
    mask_prob_thresh: float = 0.50,
    association_thresh: float = 0.30,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Complete analysis pipeline for a single uploaded photograph."""
    pil_image, (width, height) = load_image(image_input)
    tensor = image_to_tensor(pil_image)
    raw_preds = run_model_inference(model, tensor, device=device)
    filtered_preds = filter_predictions(
        raw_preds,
        fruit_score_thresh=fruit_score_thresh,
        rotten_score_thresh=rotten_score_thresh,
        mask_prob_thresh=mask_prob_thresh,
    )
    quant_results = associate_defects_and_calculate_areas(
        filtered_preds,
        image_shape=(height, width),
        association_threshold=association_thresh,
    )

    return {
        "image": pil_image,
        "image_size": (width, height),
        "raw_predictions": raw_preds,
        "filtered_predictions": filtered_preds,
        "quantification": quant_results,
    }
