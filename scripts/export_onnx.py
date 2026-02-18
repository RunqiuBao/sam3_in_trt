#!/usr/bin/env python3
"""Export SAM 3 components from PyTorch to ONNX.

SAM 3 architecture:
  - Vision encoder (shared backbone) — heaviest component
  - Detector (DETR-based, text/geometry/exemplar conditioned)
  - Tracker (SAM 2-style, stateful — not exported here)

Requires HuggingFace auth: `huggingface-cli login`
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from sam3.model_builder import build_sam3_image_model


class VisionEncoderWrapper(torch.nn.Module):
    """Wraps SAM 3 vision encoder for clean ONNX export."""

    def __init__(self, sam3_model):
        super().__init__()
        self.image_encoder = sam3_model.image_encoder

    def forward(self, image: torch.Tensor):
        """
        Args:
            image: [B, 3, 1024, 1024] normalized tensor
        Returns:
            Vision features for detector/tracker consumption
        """
        return self.image_encoder(image)


class DetectorWrapper(torch.nn.Module):
    """Wraps SAM 3 DETR-based detector for ONNX export.

    Note: text conditioning requires pre-computed text embeddings.
    Export the detector with dummy text embeddings; at runtime,
    compute text embeddings separately and feed them in.
    """

    def __init__(self, sam3_model):
        super().__init__()
        self.detector = sam3_model.detector

    def forward(self, vision_features: torch.Tensor, text_embeddings: torch.Tensor):
        """
        Args:
            vision_features: output from vision encoder
            text_embeddings: pre-computed text prompt embeddings
        Returns:
            boxes, masks, scores, presence logits
        """
        return self.detector(vision_features, text_embeddings)


def export_component(component: str, output: str, image_size: int = 1024, opset: int = 17):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build SAM 3 model (downloads checkpoint from HF if needed)
    print("Loading SAM 3 model...")
    sam3_model = build_sam3_image_model()
    sam3_model = sam3_model.to(device)
    sam3_model.eval()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if component == "vision_encoder":
        wrapper = VisionEncoderWrapper(sam3_model).to(device)
        wrapper.eval()

        dummy_image = torch.randn(1, 3, image_size, image_size, device=device)

        print(f"Exporting vision encoder (opset {opset}) → {output_path}")
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                dummy_image,
                str(output_path),
                opset_version=opset,
                input_names=["image"],
                output_names=["vision_features"],
                dynamic_axes={
                    "image": {0: "batch_size"},
                    "vision_features": {0: "batch_size"},
                },
            )

    elif component == "detector":
        wrapper = DetectorWrapper(sam3_model).to(device)
        wrapper.eval()

        # Get vision feature shape by running a dummy forward pass
        with torch.no_grad():
            dummy_image = torch.randn(1, 3, image_size, image_size, device=device)
            enc = VisionEncoderWrapper(sam3_model).to(device)
            enc.eval()
            vision_feats = enc(dummy_image)

        # Determine text embedding dims from model config
        # Use a dummy text embedding (actual dims depend on text encoder)
        dummy_text = torch.randn(1, 256, device=device)  # adjust dim as needed

        print(f"Exporting detector (opset {opset}) → {output_path}")
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (vision_feats, dummy_text),
                str(output_path),
                opset_version=opset,
                input_names=["vision_features", "text_embeddings"],
                output_names=["boxes", "masks", "scores"],
                dynamic_axes={
                    "vision_features": {0: "batch_size"},
                    "text_embeddings": {0: "batch_size"},
                    "boxes": {0: "batch_size"},
                    "masks": {0: "batch_size"},
                    "scores": {0: "batch_size"},
                },
            )
    else:
        raise ValueError(f"Unknown component: {component}. Use 'vision_encoder' or 'detector'.")

    size_mb = output_path.stat().st_size / 1e6
    print(f"Exported: {output_path} ({size_mb:.1f} MB)")

    # Validate
    import onnx
    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)
    print("ONNX validation passed ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SAM 3 components to ONNX")
    parser.add_argument("--component", required=True, choices=["vision_encoder", "detector"],
                        help="Which component to export")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_component(args.component, args.output, args.image_size, args.opset)
