"""
app.py - Fruit Quality Grading and Surface Defect Quantification Dashboard

Professional Light Theme Dashboard for local evaluation of custom Mask R-CNN + CBAM.
Features:
- Clean, modern executive light theme with high-contrast typography
- Real-time hardware device detection (CUDA GPU / CPU fallback)
- Safe checkpoint loading with cached model instances
- Interactive confidence and mask threshold controls
- Three-panel inspection (Original Image, Segmentation Result, Defect Detail)
- Per-fruit surface defect quantification and quality grading
- Manual condition override (Rotten — force reject)
- Structured results table and multi-format exports (PNG, CSV, JSON, Summary Card)
"""

from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

from grading import (
    CONDITION_BY_PERCENTAGE,
    CONDITION_ROTTEN_FORCE_REJECT,
    GRADE_COLORS,
    VALID_CONDITIONS,
    calculate_image_defect_aggregate,
    grade_fruit,
)
from inference import (
    associate_defects_and_calculate_areas,
    filter_predictions,
    image_to_tensor,
    load_image,
    run_model_inference,
)
from model import CLASS_NAMES, get_inference_device, load_checkpoint
from visualization import (
    create_summary_card,
    render_defect_detail_view,
    render_segmentation_overlay,
)

# Page Configuration
st.set_page_config(
    page_title="Fruit Quality Grading & Surface Defect Quantification",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Application Paths
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = BASE_DIR / "models" / "best_maskrcnn_cbam_fpn.pth"
SAMPLE_DIR = BASE_DIR / "samples"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Executive Light Theme Styling
st.markdown(
    """
    <style>
    /* Global Base */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Headers & Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    p, span, label {
        color: #334155;
    }
    
    /* Top Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #FFFFFF 0%, #F1F5F9 100%);
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .header-title {
        font-size: 26px;
        font-weight: 800;
        color: #0F172A;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .header-subtitle {
        font-size: 14px;
        color: #64748B;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #0F172A !important;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 14px;
        text-align: center;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px -1px rgba(0, 0, 0, 0.02);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.07);
    }
    .metric-card-title {
        font-size: 11px;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .metric-card-value {
        font-size: 26px;
        font-weight: 800;
        line-height: 1.1;
    }
    
    /* Fruit Inspection Card */
    .fruit-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #2563EB;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .fruit-card-header {
        font-size: 15px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 8px;
    }
    .fruit-card-stats {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        font-size: 13px;
        color: #475569;
    }
    .fruit-card-stats span b {
        color: #0F172A;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 12px;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .badge-grade-a {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    .badge-grade-b {
        background-color: #EFF6FF;
        color: #1E40AF;
        border: 1px solid #BFDBFE;
    }
    .badge-grade-c {
        background-color: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    .badge-reject {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
    }
    .badge-unavailable {
        background-color: #F1F5F9;
        color: #475569;
        border: 1px solid #CBD5E1;
    }
    
    /* Info / Disclaimer Box */
    .callout-box {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-left: 4px solid #2563EB;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 13px;
        color: #1E3A8A;
        margin-top: 24px;
        line-height: 1.5;
    }
    
    /* Panel Cards */
    .panel-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_cached_model(checkpoint_path_str: str, device_str: str):
    """Load model once and cache in memory."""
    device = torch.device(device_str)
    model, metadata = load_checkpoint(checkpoint_path_str, device=device)
    return model, metadata


def format_grade_badge(grade: str) -> str:
    """Return a styled HTML badge for grade display."""
    grade_class_map = {
        "Grade A": "badge-grade-a",
        "Grade B": "badge-grade-b",
        "Grade C": "badge-grade-c",
        "REJECT": "badge-reject",
        "UNAVAILABLE": "badge-unavailable",
    }
    cls_name = grade_class_map.get(grade, "badge-unavailable")
    return f'<span class="badge {cls_name}">{grade}</span>'


def main():
    # Top Header
    st.markdown(
        """
        <div class="header-banner">
            <h1 class="header-title">🍎 Fruit Quality Grading & Defect Quantification</h1>
            <p class="header-subtitle">
                Automated Surface Defect Quantification & Quality Grading Using Mask R-CNN (ResNet-50 FPN v2) + CBAM Attention
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hardware Detection
    device = get_inference_device()
    is_cuda = device.type == "cuda"
    device_info = f"CUDA GPU ({torch.cuda.get_device_name(0)})" if is_cuda else "CPU (Local Execution)"

    # Sidebar Controls
    with st.sidebar:
        st.subheader("⚙️ System & Model")
        st.markdown(f"**Compute Device:** `{device_info}`")

        checkpoint_path = DEFAULT_CHECKPOINT_PATH
        custom_ckpt = st.text_input(
            "Checkpoint Path",
            value=str(DEFAULT_CHECKPOINT_PATH),
            help="Default path to models/best_maskrcnn_cbam_fpn.pth",
        )
        if custom_ckpt:
            checkpoint_path = Path(custom_ckpt)

        model = None
        model_metadata = None

        if checkpoint_path.is_file():
            try:
                with st.spinner("Loading custom CBAM Mask R-CNN weights..."):
                    model, model_metadata = get_cached_model(str(checkpoint_path), str(device))
                epoch_val = model_metadata.get("epoch", "N/A")
                score_val = model_metadata.get("score")
                score_str = f"{score_val:.4f}" if score_val is not None else "N/A"
                st.success(
                    f"**Checkpoint Loaded Successfully**\n\n"
                    f"• Epoch: **{epoch_val}**\n\n"
                    f"• Best Val Score: **{score_str}**\n\n"
                    f"• Parameters: **{model_metadata.get('total_parameters', 0)/1e6:.1f}M**"
                )
            except Exception as exc:
                st.error(f"Failed to load checkpoint: {exc}")
        else:
            st.warning(
                f"⚠️ **Trained Checkpoint Not Found**\n\n"
                f"Expected location:\n`{checkpoint_path.resolve()}`\n\n"
                "Please place `best_maskrcnn_cbam_fpn.pth` in the `FruitQualityDashboard/models/` folder."
            )

        st.markdown("---")
        st.subheader("🔍 Detection Controls")
        fruit_score_thr = st.slider("Fruit Score Threshold", 0.05, 1.00, 0.50, 0.05, help="Confidence threshold to detect fruit instances.")
        rotten_score_thr = st.slider("Rotten Region Score Threshold", 0.05, 1.00, 0.40, 0.05, help="Confidence threshold to detect rotten defect regions.")
        mask_prob_thr = st.slider("Mask Probability Threshold", 0.10, 0.90, 0.50, 0.05, help="Sigmoid probability threshold to binarize instance masks.")

        st.markdown("---")
        st.subheader("🎨 Overlay Controls")
        opacity = st.slider("Mask Opacity", 0.10, 1.00, 0.45, 0.05)
        show_boxes = st.checkbox("Show Bounding Boxes", value=True)
        show_fruit_masks = st.checkbox("Show Fruit Masks", value=True)
        show_rotten_masks = st.checkbox("Show Defect Masks", value=True)
        show_labels = st.checkbox("Show Instance Labels", value=True)

        st.markdown("---")
        st.markdown(
            """
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:12px; font-size:12px; color:#475569;">
            <b style="color:#0F172A;">Grading Rules:</b><br>
            • <b>Grade A:</b> Defect &lt; 5%<br>
            • <b>Grade B:</b> 5% ≤ Defect &lt; 15%<br>
            • <b>Grade C:</b> 15% ≤ Defect ≤ 25%<br>
            • <b>REJECT:</b> Defect &gt; 25% or Rotten
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Image Upload & Sample Selection
    st.subheader("📤 Image Upload & Analysis")
    col_up, col_samples = st.columns([2, 1])

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload Fruit Photograph",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a single fruit photograph (.jpg, .jpeg, .png, .webp) for segmentation and quality grading.",
        )

    with col_samples:
        st.write("**Or select a sample image:**")
        sample_options = ["None"]
        if SAMPLE_DIR.is_dir():
            sample_files = list(SAMPLE_DIR.glob("*.jpg")) + list(SAMPLE_DIR.glob("*.png"))
            sample_options += [f.name for f in sample_files]

        selected_sample = st.selectbox("Sample Library", sample_options)

    # Active Image Selection
    active_image = None
    image_name = "Uploaded Image"

    if uploaded_file is not None:
        try:
            active_image = Image.open(uploaded_file).convert("RGB")
            image_name = uploaded_file.name
        except Exception as exc:
            st.error(f"Error opening uploaded image: {exc}")
    elif selected_sample != "None":
        sample_path = SAMPLE_DIR / selected_sample
        if sample_path.is_file():
            active_image = Image.open(sample_path).convert("RGB")
            image_name = selected_sample

    if active_image is None:
        st.info("👆 Please upload a fruit photograph or pick a sample image above to begin.")
        if model is None:
            st.warning("Notice: Checkpoint file is required for analysis. Place `best_maskrcnn_cbam_fpn.pth` in `models/`.")
        return

    # Image Dimensions
    orig_w, orig_h = active_image.size
    st.markdown(f"**Loaded Image:** `{image_name}` &nbsp;|&nbsp; **Resolution:** `{orig_w} × {orig_h} px`")

    analyze_clicked = st.button("🔍 Analyze Fruit Quality", type="primary", use_container_width=True)

    # Session State
    if "analysis_results" not in st.session_state:
        st.session_state["analysis_results"] = None
        st.session_state["image_name"] = None
    if "condition_overrides" not in st.session_state:
        st.session_state["condition_overrides"] = {}

    if analyze_clicked:
        if model is None:
            st.error("Cannot run inference: Model checkpoint is missing. Place `best_maskrcnn_cbam_fpn.pth` in `FruitQualityDashboard/models/`.")
            return

        with st.spinner("Running Mask R-CNN with CBAM feature pyramid attention..."):
            try:
                tensor = image_to_tensor(active_image)
                raw_preds = run_model_inference(model, tensor, device=device)
                filtered = filter_predictions(
                    raw_preds,
                    fruit_score_thresh=fruit_score_thr,
                    rotten_score_thresh=rotten_score_thr,
                    mask_prob_thresh=mask_prob_thr,
                )
                quant = associate_defects_and_calculate_areas(
                    filtered,
                    image_shape=(orig_h, orig_w),
                    association_threshold=0.30,
                )

                st.session_state["analysis_results"] = {
                    "image": active_image,
                    "image_size": (orig_w, orig_h),
                    "raw_predictions": raw_preds,
                    "filtered_predictions": filtered,
                    "quantification": quant,
                }
                st.session_state["image_name"] = image_name
                st.session_state["condition_overrides"] = {}
            except torch.cuda.OutOfMemoryError:
                st.error("CUDA Out of Memory. Please switch to CPU inference in the sidebar.")
                return
            except Exception as exc:
                st.error(f"Inference error: {type(exc).__name__}: {exc}")
                return

    results = st.session_state.get("analysis_results")
    if results is None:
        st.image(active_image, caption=f"Original Photograph ({orig_w} × {orig_h} px)", use_container_width=True)
        return

    # Visualization Rendering
    st.markdown("---")
    st.subheader("🖼️ Visual Inspection Panels")

    filtered = results["filtered_predictions"]
    quant = results["quantification"]
    fruits = quant["fruits"]
    orphans = quant["orphan_rotten_instances"]

    fruit_masks = list(filtered["fruit_masks"])
    fruit_scores = list(filtered["fruit_scores"])
    fruit_boxes = [box.tolist() for box in filtered["fruit_boxes"]]

    rotten_masks = list(filtered["rotten_masks"])
    rotten_scores = list(filtered["rotten_scores"])
    rotten_boxes = [box.tolist() for box in filtered["rotten_boxes"]]

    seg_overlay = render_segmentation_overlay(
        image=results["image"],
        fruit_masks=fruit_masks,
        fruit_scores=fruit_scores,
        fruit_boxes=fruit_boxes,
        rotten_masks=rotten_masks,
        rotten_scores=rotten_scores,
        rotten_boxes=rotten_boxes,
        opacity=opacity,
        show_boxes=show_boxes,
        show_fruit_masks=show_fruit_masks,
        show_rotten_masks=show_rotten_masks,
        show_labels=show_labels,
    )

    defect_detail = render_defect_detail_view(
        image=results["image"],
        rotten_masks=rotten_masks,
        isolate_defects=False,
        show_contours=True,
        show_defect_overlay=True,
        opacity=opacity,
    )

    panel_a, panel_b, panel_c = st.columns(3)

    with panel_a:
        st.markdown("**Panel A: Original Photograph**")
        st.image(results["image"], use_container_width=True)
        st.caption(f"Unmodified Input • {orig_w} × {orig_h} px")

    with panel_b:
        st.markdown("**Panel B: Segmentation Result**")
        st.image(seg_overlay, use_container_width=True)
        st.caption("Green: Fruit Instances • Red: Rotten Regions • BBoxes & Scores")

    with panel_c:
        st.markdown("**Panel C: Defect Detail View**")
        st.image(defect_detail, use_container_width=True)
        st.caption("Yellow Contours: Defect Boundaries • Isolated Defect Regions")

    # Per-Fruit Grading Section
    st.markdown("---")
    st.subheader("⚖️ Per-Fruit Quality Grading & Condition Decision")

    if not fruits:
        st.warning("⚠️ **No fruit instances were detected** with the current confidence threshold. Try lowering the Fruit Score Threshold in the sidebar.")
    else:
        for fruit in fruits:
            f_id = fruit["fruit_id"]
            override_key = f"cond_{f_id}"
            current_condition = st.session_state["condition_overrides"].get(
                override_key, CONDITION_BY_PERCENTAGE
            )

            updated_grade = grade_fruit(fruit["defect_percentage"], condition=current_condition)
            fruit["grading"] = updated_grade

            pct_val = fruit["defect_percentage"]
            pct_display = f"{pct_val:.2f}%" if pct_val is not None else "N/A"
            status_note = fruit.get("warning") or ""

            # Card Display
            st.markdown(
                f"""
                <div class="fruit-card">
                    <div class="fruit-card-header">Fruit #{f_id}</div>
                    <div class="fruit-card-stats">
                        <span>Confidence: <b>{fruit["confidence"]:.3f}</b></span>
                        <span>Fruit Area: <b>{fruit["fruit_pixels"]:,} px</b></span>
                        <span>Defect Area: <b>{fruit["rotten_pixels"] if fruit["rotten_pixels"] is not None else "N/A"} px</b></span>
                        <span>Defect: <b style="color:#2563EB;">{pct_display}</b></span>
                        <span>Assigned Grade: {format_grade_badge(updated_grade['grade'])}</span>
                    </div>
                    {f'<div style="color:#D97706; font-size:12px; margin-top:8px;">⚠ {status_note}</div>' if status_note else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_sel, col_desc = st.columns([1, 2])
            with col_sel:
                new_condition = st.selectbox(
                    f"Decision Override (Fruit #{f_id})",
                    options=VALID_CONDITIONS,
                    index=VALID_CONDITIONS.index(current_condition),
                    key=f"sel_fruit_{f_id}",
                    help="Default grades objectively by surface defect percentage. Choose 'Rotten - force reject' to manually force rejection.",
                )
                if new_condition != current_condition:
                    st.session_state["condition_overrides"][override_key] = new_condition
                    st.rerun()

            with col_desc:
                st.caption(f"**Rationale:** {updated_grade['reason']}")

    # Summary Metrics
    st.markdown("---")
    st.subheader("📊 Executive Summary Metrics")

    grade_counts = {"Grade A": 0, "Grade B": 0, "Grade C": 0, "REJECT": 0, "UNAVAILABLE": 0}
    for f in fruits:
        g = f["grading"]["grade"]
        grade_counts[g] = grade_counts.get(g, 0) + 1

    img_defect_pct = quant.get("image_rotten_percent")
    img_pct_str = f"{img_defect_pct:.2f}%" if img_defect_pct is not None else "N/A"

    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-card-title">Fruits</div><div class="metric-card-value" style="color:#0F172A;">{len(fruits)}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-card-title">Rotten Regions</div><div class="metric-card-value" style="color:#0F172A;">{len(rotten_scores)}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-card-title">Aggregate Defect</div><div class="metric-card-value" style="color:#2563EB;">{img_pct_str}</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card" style="border-top: 3px solid #059669;"><div class="metric-card-title">Grade A</div><div class="metric-card-value" style="color:#059669;">{grade_counts["Grade A"]}</div></div>', unsafe_allow_html=True)
    with m5:
        st.markdown(f'<div class="metric-card" style="border-top: 3px solid #0284C7;"><div class="metric-card-title">Grade B</div><div class="metric-card-value" style="color:#0284C7;">{grade_counts["Grade B"]}</div></div>', unsafe_allow_html=True)
    with m6:
        st.markdown(f'<div class="metric-card" style="border-top: 3px solid #D97706;"><div class="metric-card-title">Grade C</div><div class="metric-card-value" style="color:#D97706;">{grade_counts["Grade C"]}</div></div>', unsafe_allow_html=True)
    with m7:
        st.markdown(f'<div class="metric-card" style="border-top: 3px solid #DC2626;"><div class="metric-card-title">Rejected</div><div class="metric-card-value" style="color:#DC2626;">{grade_counts["REJECT"]}</div></div>', unsafe_allow_html=True)

    # Structured Results Table
    st.markdown("---")
    st.subheader("📋 Structured Results Table")

    table_rows = []
    for fruit in fruits:
        f_id = fruit["fruit_id"]
        gr = fruit["grading"]
        pct_val = fruit["defect_percentage"]
        pct_display = round(pct_val, 2) if pct_val is not None else "N/A"
        warning = fruit.get("warning") or "None"

        table_rows.append({
            "Fruit Instance ID": f"Fruit #{f_id}",
            "Detection Confidence": f"{fruit['confidence']:.3f}",
            "Fruit Area (px)": f"{fruit['fruit_pixels']:,}",
            "Assigned Defects": fruit["assigned_rotten_count"],
            "Defect Area (px)": f"{fruit['rotten_pixels']:,}" if fruit["rotten_pixels"] is not None else "N/A",
            "Defect %": f"{pct_display}%" if pct_display != "N/A" else "N/A",
            "Applied Condition": gr["applied_condition"],
            "Final Grade": gr["grade"],
            "Quality Flags": warning,
        })

    if table_rows:
        df_display = pd.DataFrame(table_rows)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.caption("No fruit detections to display in results table.")

    if orphans:
        st.markdown("##### ⚠️ Unassociated Defect Regions (Outside Detected Fruits)")
        st.caption(
            f"Detected {len(orphans)} defect instance(s) that did not reliably intersect any detected fruit mask. "
            "These regions were excluded from fruit percentages to prevent false penalty."
        )
        orphan_df = pd.DataFrame([
            {
                "Orphan Defect ID": item["rotten_index"] + 1,
                "Confidence": f"{item['score']:.3f}",
                "Pixel Area": f"{item['mask_pixels']:,}",
                "Reason": item["reason"],
            }
            for item in orphans
        ])
        st.dataframe(orphan_df, use_container_width=True, hide_index=True)

    # Downloadable Outputs
    st.markdown("---")
    st.subheader("💾 Export & Download Results")

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        buf_img = BytesIO()
        seg_overlay.save(buf_img, format="PNG")
        st.download_button(
            label="🖼️ Download Annotated PNG",
            data=buf_img.getvalue(),
            file_name=f"annotated_{Path(image_name).stem}.png",
            mime="image/png",
            use_container_width=True,
        )

    with d2:
        csv_data = pd.DataFrame(table_rows).to_csv(index=False).encode("utf-8") if table_rows else b"No fruit detections\n"
        st.download_button(
            label="📊 Download Results CSV",
            data=csv_data,
            file_name=f"grading_results_{Path(image_name).stem}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d3:
        json_payload = {
            "image": {"filename": image_name, "width": orig_w, "height": orig_h},
            "model": {
                "checkpoint": str(checkpoint_path),
                "device": str(device),
                "metadata": {
                    "epoch": model_metadata.get("epoch") if model_metadata else None,
                    "validation_score": model_metadata.get("score") if model_metadata else None,
                },
            },
            "parameters": {
                "fruit_score_threshold": fruit_score_thr,
                "rotten_score_threshold": rotten_score_thr,
                "mask_probability_threshold": mask_prob_thr,
            },
            "summary": {
                "fruit_count": len(fruits),
                "rotten_count": len(rotten_scores),
                "image_aggregate_defect_percentage": img_defect_pct,
                "grade_counts": grade_counts,
            },
            "fruit_instances": [
                {
                    "fruit_id": f["fruit_id"],
                    "confidence": f["confidence"],
                    "bounding_box": f["box"],
                    "fruit_mask_pixels": f["fruit_pixels"],
                    "rotten_pixels": f["rotten_pixels"],
                    "defect_percentage": f["defect_percentage"],
                    "applied_condition": f["grading"]["applied_condition"],
                    "final_grade": f["grading"]["grade"],
                    "status": f["status"],
                    "warning": f.get("warning"),
                }
                for f in fruits
            ],
            "orphan_defects": orphans,
        }
        json_data = json.dumps(json_payload, indent=2).encode("utf-8")
        st.download_button(
            label="📋 Download Full JSON Report",
            data=json_data,
            file_name=f"report_{Path(image_name).stem}.json",
            mime="application/json",
            use_container_width=True,
        )

    with d4:
        summary_card = create_summary_card(results, image_name=image_name)
        buf_card = BytesIO()
        summary_card.save(buf_card, format="PNG")
        st.download_button(
            label="📈 Download Summary Card PNG",
            data=buf_card.getvalue(),
            file_name=f"summary_card_{Path(image_name).stem}.png",
            mime="image/png",
            use_container_width=True,
        )

    # Disclaimer Callout
    st.markdown(
        """
        <div class="callout-box">
        <b>Academic & Engineering Notice:</b> Surface defect quantification and quality grading decisions are computed
        strictly from visible 2D segmentation masks. This project measures external visible surface blemish areas and does not assess
        internal flesh decomposition, core mold, fruit mass, firmness, sweetness (Brix), or microbiological safety.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
