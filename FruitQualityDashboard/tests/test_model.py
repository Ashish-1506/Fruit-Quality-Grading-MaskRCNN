import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import CBAM, CBAMFPNBackbone, build_model, load_checkpoint, NUM_CLASSES


def test_cbam_attention_forward():
    """CBAM forward pass preserves tensor shape and applies gating."""
    channels = 256
    cbam = CBAM(channels, reduction=16)
    cbam.eval()

    x = torch.randn(2, channels, 16, 16)
    with torch.no_grad():
        out = cbam(x)

    assert out.shape == x.shape
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_build_model_architecture():
    """Mask R-CNN architecture is constructed correctly without pretrained weights."""
    model = build_model(num_classes=NUM_CLASSES, img_size=640, use_attention=True)
    model.eval()

    box_predictor = model.roi_heads.box_predictor
    assert box_predictor.cls_score.out_features == 3
    assert box_predictor.bbox_pred.out_features == 3 * 4

    mask_predictor = model.roi_heads.mask_predictor
    assert mask_predictor.mask_fcn_logits.out_channels == 3

    assert isinstance(model.backbone, CBAMFPNBackbone)
    for key in ("0", "1", "2", "3", "pool"):
        assert key in model.backbone.attention
        assert isinstance(model.backbone.attention[key], CBAM)


def test_load_checkpoint_missing_file():
    """FileNotFoundError with actionable instructions when checkpoint is missing."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_checkpoint("non_existent_checkpoint_file.pth")
    assert "best_maskrcnn_cbam_fpn.pth" in str(exc_info.value)


def test_load_checkpoint_empty_file():
    """ValueError when checkpoint file is 0 bytes."""
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with pytest.raises(ValueError) as exc_info:
            load_checkpoint(tmp_path)
        assert "empty (0 bytes)" in str(exc_info.value)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_load_checkpoint_synthetic_valid():
    """Valid synthetic checkpoint loads without error."""
    model_ref = build_model(num_classes=NUM_CLASSES, img_size=640, use_attention=True)
    state_dict = model_ref.state_dict()

    checkpoint = {
        "model": state_dict,
        "epoch": 42,
        "score": 0.981,
        "cfg": {"img_size": 640, "use_attention": True},
    }

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        torch.save(checkpoint, tmp_path)
        loaded_model, metadata = load_checkpoint(tmp_path)

        assert metadata["epoch"] == 42
        assert metadata["score"] == pytest.approx(0.981)
        assert loaded_model is not None
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
