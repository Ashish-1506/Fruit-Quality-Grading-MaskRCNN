from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

CLASS_NAMES = {0: "background", 1: "fruit", 2: "rotten_region"}
NUM_CLASSES = 3


class CBAM(nn.Module):
    """Convolutional Block Attention Module applied to FPN feature maps."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)
        nn.init.zeros_(self.channel_mlp[-1].weight)
        nn.init.zeros_(self.channel_mlp[-1].bias)
        nn.init.zeros_(self.spatial_conv.weight)
        nn.init.zeros_(self.spatial_conv.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        channel_gate = torch.sigmoid(
            self.channel_mlp(self.pool_avg(features)) + self.channel_mlp(self.pool_max(features))
        ) * 2.0
        features = features * channel_gate
        spatial_avg = torch.mean(features, dim=1, keepdim=True)
        spatial_max = torch.max(features, dim=1, keepdim=True).values
        spatial_gate = torch.sigmoid(
            self.spatial_conv(torch.cat([spatial_avg, spatial_max], dim=1))
        ) * 2.0
        return features * spatial_gate


class CBAMFPNBackbone(nn.Module):
    """Backbone wrapper applying CBAM attention to FPN pyramid levels ('0','1','2','3','pool')."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.body = backbone
        self.out_channels = backbone.out_channels
        self.attention = nn.ModuleDict({
            key: CBAM(self.out_channels) for key in ("0", "1", "2", "3", "pool")
        })

    def forward(self, images: torch.Tensor) -> OrderedDict:
        features = self.body(images)
        return OrderedDict(
            (key, self.attention[key](value) if key in self.attention else value)
            for key, value in features.items()
        )


def build_model(
    num_classes: int = NUM_CLASSES,
    img_size: int = 640,
    min_size: int = 640,
    max_size: int = 640,
    trainable_backbone_layers: int = 3,
    use_attention: bool = True,
    box_detections_per_img: int = 100,
    box_score_thresh: float = 0.001,
) -> nn.Module:
    """Instantiate Mask R-CNN ResNet-50 FPN v2 with optional CBAM attention.

    Constructs the architecture without COCO pretrained weights,
    ready to load weights from the trained checkpoint.
    """
    model = maskrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=min_size,
        max_size=max_size,
        box_detections_per_img=box_detections_per_img,
        box_score_thresh=box_score_thresh,
    )
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, num_classes)

    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, num_classes)

    if use_attention:
        model.backbone = CBAMFPNBackbone(model.backbone)

    return model


def get_inference_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_checkpoint(
    checkpoint_path: str | Path,
    device: Optional[torch.device | str] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Validate and safely load a trained checkpoint into the reproduced architecture.

    Args:
        checkpoint_path: Path to the .pth file.
        device: Target torch device (defaults to CUDA if available, else CPU).

    Returns:
        (model, metadata_dict)
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at: {path.resolve()}\n"
            "Please download your trained checkpoint from Kaggle and place it at:\n"
            "  FruitQualityDashboard/models/best_maskrcnn_cbam_fpn.pth"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"Checkpoint file at {path.resolve()} is empty (0 bytes). "
            "Please ensure the file was completely downloaded from Kaggle."
        )

    # Detect Git LFS pointer files and attempt automatic pull
    if path.stat().st_size < 1024:
        try:
            header = path.read_text(encoding="utf-8", errors="ignore")
            if "git-lfs.github.com" in header:
                import subprocess
                # Try pulling LFS objects
                repo_root = path.resolve().parent.parent.parent
                subprocess.run(["git", "lfs", "pull"], cwd=str(repo_root), timeout=180, check=False)
        except Exception:
            pass
        if path.stat().st_size < 1024:
            raise RuntimeError(
                f"Checkpoint at {path.resolve()} is a Git LFS pointer file, not the binary weights.\n"
                "To resolve on Streamlit Cloud, ensure 'git-lfs' is in packages.txt, or download the full .pth file."
            )

    if device is None:
        device = get_inference_device()
    elif isinstance(device, str):
        device = torch.device(device)

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read checkpoint file at {path.resolve()}.\n"
            f"Underlying error: {type(exc).__name__}: {exc}\n"
            "The checkpoint may be corrupted or not a valid PyTorch file."
        ) from exc

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected checkpoint at {path.resolve()} to be a dictionary, "
            f"got {type(checkpoint).__name__} instead."
        )

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        sample_keys = list(checkpoint.keys())[:5]
        if any(k.startswith("backbone.") or k.startswith("roi_heads.") for k in sample_keys):
            state_dict = checkpoint
        else:
            raise KeyError(
                f"Checkpoint at {path.resolve()} does not contain a 'model' state dictionary. "
                f"Keys found: {list(checkpoint.keys())[:10]}"
            )

    cfg = checkpoint.get("cfg", {}) if isinstance(checkpoint, dict) else {}
    img_size = cfg.get("img_size", 640)
    use_attention = cfg.get("use_attention", True)

    try:
        model = build_model(
            num_classes=NUM_CLASSES,
            img_size=img_size,
            min_size=img_size,
            max_size=img_size,
            use_attention=use_attention,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to instantiate model architecture: {exc}") from exc

    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        model_keys = set(model.state_dict().keys())
        loaded_keys = set(state_dict.keys())
        missing = model_keys - loaded_keys
        unexpected = loaded_keys - model_keys
        msg = [
            f"Architecture state-dict mismatch while loading {path.resolve()}:",
            f"Underlying error: {exc}",
        ]
        if missing:
            msg.append(f"Missing keys ({len(missing)}): {list(missing)[:5]}...")
        if unexpected:
            msg.append(f"Unexpected keys ({len(unexpected)}): {list(unexpected)[:5]}...")
        raise RuntimeError("\n".join(msg)) from exc

    model.to(device)
    model.eval()

    metadata = {
        "epoch": checkpoint.get("epoch", None),
        "score": checkpoint.get("score", None),
        "cfg": cfg,
        "device": str(device),
        "checkpoint_path": str(path.resolve()),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }

    return model, metadata
