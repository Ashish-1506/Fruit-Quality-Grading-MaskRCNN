from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

FRUIT_COLOR_RGB = (16, 185, 129)    # Fresh Emerald Green
FRUIT_BOX_RGB = (5, 150, 105)        # Crisp Emerald-600
ROTTEN_COLOR_RGB = (239, 68, 68)     # High-contrast Red
ROTTEN_BOX_RGB = (220, 38, 38)       # Deep Crimson Red
CONTOUR_COLOR_RGB = (245, 158, 11)   # Vibrant Amber


def get_mask_contour(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Extract a boolean boundary contour of a 2D binary mask using numpy morphology."""
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    pad = np.pad(mask, thickness, mode="constant", constant_values=False)
    eroded = pad.copy()
    for dy in range(-thickness, thickness + 1):
        for dx in range(-thickness, thickness + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = pad[
                thickness + dy : pad.shape[0] - thickness + dy,
                thickness + dx : pad.shape[1] - thickness + dx,
            ]
            eroded[thickness : -thickness, thickness : -thickness] &= shifted

    inner = eroded[thickness : -thickness, thickness : -thickness]
    return mask & ~inner


def render_segmentation_overlay(
    image: Image.Image,
    fruit_masks: List[np.ndarray],
    fruit_scores: List[float],
    fruit_boxes: List[List[float]],
    rotten_masks: List[np.ndarray],
    rotten_scores: List[float],
    rotten_boxes: List[List[float]],
    opacity: float = 0.45,
    show_boxes: bool = True,
    show_fruit_masks: bool = True,
    show_rotten_masks: bool = True,
    show_labels: bool = True,
) -> Image.Image:
    """Render the full segmentation overlay on the original photograph.

    Maintains exact original dimensions with zero distortion.
    """
    img_rgba = image.convert("RGBA")
    w, h = image.size
    overlay_arr = np.zeros((h, w, 4), dtype=np.uint8)

    if show_fruit_masks and len(fruit_masks) > 0:
        alpha_val = int(255 * opacity)
        for f_mask in fruit_masks:
            m = np.asarray(f_mask, dtype=bool)
            if m.shape != (h, w):
                continue
            overlay_arr[m, 0] = FRUIT_COLOR_RGB[0]
            overlay_arr[m, 1] = FRUIT_COLOR_RGB[1]
            overlay_arr[m, 2] = FRUIT_COLOR_RGB[2]
            overlay_arr[m, 3] = alpha_val

    if show_rotten_masks and len(rotten_masks) > 0:
        alpha_val = int(255 * min(1.0, opacity + 0.20))
        for r_mask in rotten_masks:
            m = np.asarray(r_mask, dtype=bool)
            if m.shape != (h, w):
                continue
            overlay_arr[m, 0] = ROTTEN_COLOR_RGB[0]
            overlay_arr[m, 1] = ROTTEN_COLOR_RGB[1]
            overlay_arr[m, 2] = ROTTEN_COLOR_RGB[2]
            overlay_arr[m, 3] = alpha_val

    mask_img = Image.fromarray(overlay_arr, mode="RGBA")
    composite = Image.alpha_composite(img_rgba, mask_img)

    draw = ImageDraw.Draw(composite)
    line_w = max(2, int(min(w, h) / 300))
    font_size = max(12, int(min(w, h) / 40))

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    if show_boxes and len(fruit_boxes) > 0:
        for idx, (box, score) in enumerate(zip(fruit_boxes, fruit_scores)):
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=FRUIT_BOX_RGB, width=line_w)
            if show_labels:
                draw_label_badge(draw, (x1, y1), f"Fruit #{idx + 1}: {score:.2f}", font, bg_color=(11, 19, 43, 220), text_color=FRUIT_BOX_RGB)

    if show_boxes and len(rotten_boxes) > 0:
        for idx, (box, score) in enumerate(zip(rotten_boxes, rotten_scores)):
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=ROTTEN_BOX_RGB, width=line_w)
            if show_labels:
                draw_label_badge(draw, (x1, y1), f"Rotten: {score:.2f}", font, bg_color=(40, 10, 10, 220), text_color=ROTTEN_BOX_RGB)

    return composite.convert("RGB")


def render_defect_detail_view(
    image: Image.Image,
    rotten_masks: List[np.ndarray],
    isolate_defects: bool = False,
    show_contours: bool = True,
    show_defect_overlay: bool = True,
    opacity: float = 0.55,
) -> Image.Image:
    """Render the defect detail view with optional contours and isolation."""
    w, h = image.size
    base_arr = np.asarray(image, dtype=np.float32)

    if isolate_defects:
        base_arr = base_arr * 0.20

    base_img = Image.fromarray(base_arr.astype(np.uint8), mode="RGB").convert("RGBA")
    overlay_arr = np.zeros((h, w, 4), dtype=np.uint8)

    if len(rotten_masks) > 0:
        combined_rotten = np.zeros((h, w), dtype=bool)
        for r_mask in rotten_masks:
            m = np.asarray(r_mask, dtype=bool)
            if m.shape == (h, w):
                combined_rotten |= m

        if show_defect_overlay and np.any(combined_rotten):
            alpha = int(255 * opacity)
            overlay_arr[combined_rotten, 0] = ROTTEN_COLOR_RGB[0]
            overlay_arr[combined_rotten, 1] = ROTTEN_COLOR_RGB[1]
            overlay_arr[combined_rotten, 2] = ROTTEN_COLOR_RGB[2]
            overlay_arr[combined_rotten, 3] = alpha

        if show_contours and np.any(combined_rotten):
            contour_thickness = max(2, int(min(w, h) / 350))
            contours = get_mask_contour(combined_rotten, thickness=contour_thickness)
            overlay_arr[contours, 0] = CONTOUR_COLOR_RGB[0]
            overlay_arr[contours, 1] = CONTOUR_COLOR_RGB[1]
            overlay_arr[contours, 2] = CONTOUR_COLOR_RGB[2]
            overlay_arr[contours, 3] = 255

    overlay_img = Image.fromarray(overlay_arr, mode="RGBA")
    return Image.alpha_composite(base_img, overlay_img).convert("RGB")


def draw_label_badge(
    draw: ImageDraw.ImageDraw,
    pos: Tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    bg_color: Tuple[int, int, int, int] = (15, 23, 42, 220),
    text_color: Tuple[int, int, int] = (0, 240, 255),
):
    """Draw a label badge with background padding."""
    x, y = pos
    try:
        bbox = draw.textbbox((x, y), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except Exception:
        tw, th = 80, 16

    pad = 4
    rect_box = [x, max(0, y - th - 2 * pad), x + tw + 2 * pad, y]
    draw.rectangle(rect_box, fill=bg_color)
    draw.text((x + pad, max(0, y - th - pad)), text, fill=text_color, font=font)


def create_summary_card(
    analysis_results: Dict[str, Any],
    image_name: str = "Fruit Photograph",
) -> Image.Image:
    """Generate a downloadable visual summary card (PNG) with metrics and grades."""
    quant = analysis_results["quantification"]
    fruits = quant["fruits"]
    orphans = quant["orphan_rotten_instances"]
    img_size = analysis_results["image_size"]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#FFFFFF")
    ax.set_facecolor("#F8FAFC")
    ax.axis("off")

    # Outer border
    rect = plt.Rectangle((0.01, 0.01), 0.98, 0.98, fill=False, edgecolor="#E2E8F0", linewidth=1.5, transform=ax.transAxes)
    ax.add_patch(rect)

    # Title & Metadata
    ax.text(0.05, 0.89, "Fruit Quality Grading & Surface Defect Summary", color="#0F172A", fontsize=16, weight="bold")
    ax.text(0.05, 0.82, f"Image: {image_name}  |  Dimensions: {img_size[0]} x {img_size[1]} px", color="#64748B", fontsize=11)
    ax.plot([0.05, 0.95], [0.77, 0.77], color="#E2E8F0", linewidth=1.5)

    img_defect_pct = quant.get("image_rotten_percent")
    pct_text = f"{img_defect_pct:.2f}%" if img_defect_pct is not None else "N/A"

    metrics = [
        ("Fruits Detected", str(len(fruits))),
        ("Rotten Instances", str(len(quant.get("fruits", [])) + len(orphans))),
        ("Aggregate Defect %", pct_text),
        ("Orphan Defects", str(len(orphans))),
    ]

    for i, (label, val) in enumerate(metrics):
        bx = 0.05 + i * 0.23
        ax.text(bx, 0.67, label, color="#64748B", fontsize=10, weight="semibold")
        ax.text(bx, 0.58, val, color="#0F172A", fontsize=16, weight="bold")

    ax.text(0.05, 0.47, "Per-Fruit Quality Decisions", color="#0F172A", fontsize=13, weight="bold")

    y_pos = 0.37
    if not fruits:
        ax.text(0.05, y_pos, "No fruit instances detected in image.", color="#64748B", fontsize=11, style="italic")
    else:
        for fruit in fruits[:4]:
            fid = fruit["fruit_id"]
            gr = fruit.get("grading", {})
            grade_name = gr.get("grade", "UNAVAILABLE")
            defect_p = fruit.get("defect_percentage")
            p_str = f"{defect_p:.2f}%" if defect_p is not None else "N/A"
            color = gr.get("color", "#64748B")
            line = f"Fruit #{fid}   |   Score: {fruit['confidence']:.2f}   |   Defect: {p_str}   |   Decision: {grade_name}"
            ax.text(0.05, y_pos, line, color="#334155", fontsize=11)
            ax.scatter([0.92], [y_pos + 0.015], color=color, s=130, edgecolor="#CBD5E1", linewidth=1)
            y_pos -= 0.08

    ax.text(
        0.05, 0.05,
        "* Visible surface defect quantification based on deep segmentation. Does not measure internal decay.",
        color="#94A3B8", fontsize=9, style="italic",
    )

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
