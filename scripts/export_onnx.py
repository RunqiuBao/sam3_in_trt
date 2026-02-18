#!/usr/bin/env python3
"""Export SAM 2 image encoder from PyTorch to ONNX."""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

# SAM 2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class ImageEncoderWrapper(torch.nn.Module):
    """Wraps SAM 2 image encoder for clean ONNX export."""

    def __init__(self, sam_model):
        super().__init__()
        self.image_encoder = sam_model.image_encoder
        self.no_mem_embed = sam_model.no_mem_embed

    def forward(self, image: torch.Tensor):
        # image: [B, 3, 1024, 1024] normalized
        backbone_out = self.image_encoder(image)

        # Extract multi-scale features
        # backbone_out contains feature maps at different resolutions
        # Return the high-res features used by mask decoder
        _, vision_feats, _, _ = self.image_encoder._prepare_backbone_features(backbone_out)

        # Add no-memory embedding for image-only mode
        feats = [
            feat + self.no_mem_embed.unsqueeze(0)
            if feat.shape[-1] == self.no_mem_embed.shape[-1]
            else feat
            for feat in vision_feats
        ]
        return feats


def export_image_encoder(checkpoint: str, config_name: str, output: str,
                         image_size: int = 1024, opset: int = 17):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model
    print(f"Loading SAM 2 from {checkpoint} (config: {config_name})")
    sam2_model = build_sam2(config_name, checkpoint, device=device)
    sam2_model.eval()

    # Wrap encoder
    encoder = ImageEncoderWrapper(sam2_model).to(device)
    encoder.eval()

    # Dummy input
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    # Export
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting ONNX (opset {opset}) → {output_path}")
    with torch.no_grad():
        torch.onnx.export(
            encoder,
            dummy,
            str(output_path),
            opset_version=opset,
            input_names=["image"],
            output_names=["image_embeddings"],
            dynamic_axes={
                "image": {0: "batch_size"},
                "image_embeddings": {0: "batch_size"},
            },
        )

    size_mb = output_path.stat().st_size / 1e6
    print(f"Exported: {output_path} ({size_mb:.1f} MB)")

    # Validate
    import onnx
    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)
    print("ONNX validation passed ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SAM 2 image encoder to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to SAM 2 .pt checkpoint")
    parser.add_argument("--config", default="sam2.1_hiera_l.yaml", help="SAM 2 model config name")
    parser.add_argument("--output", default="models/sam2_image_encoder.onnx")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_image_encoder(args.checkpoint, args.config, args.output, args.image_size, args.opset)
