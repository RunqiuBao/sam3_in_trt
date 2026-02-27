#!/usr/bin/env python3
"""Export SAM 3 components from PyTorch to ONNX.

SAM 3 architecture:
  - Vision encoder (shared backbone) — heaviest component
  - Detector (DETR-based, text/geometry/exemplar conditioned)
  - Tracker (SAM 2-style, stateful — not exported here)

Requires HuggingFace auth: `huggingface-cli login`
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import argparse
import torch
import os
import time
import numpy as np
from pathlib import Path
import logging
from torchvision.transforms import v2

from sam3 import build_sam3_image_model
from sam3.model.tokenizer_ve import SimpleTokenizer
import sam3
if TYPE_CHECKING:
    from sam3.model.sam3_image import Sam3Image
    from sam3.model.vl_combiner import SAM3VLBackbone


def ConfigLogging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True,
    )


class BackboneImageEncoderWrapper(torch.nn.Module):
    """Wraps SAM 3 backbone image encoder for clean ONNX export."""

    def __init__(self, sam3_backbone: SAM3VLBackbone):
        super().__init__()
        self.sam3_backbone = sam3_backbone

    def forward(self, image: torch.Tensor):
        """
        Args:
            image: [B, 3, 1024, 1024] normalized tensor
        Returns:
            Vision features for detector/tracker consumption
        """
        return self.sam3_backbone.forward_image(image)


class BackboneTextEncoderWrapper(torch.nn.Module):
    """Wraps SAM 3 backbone text encoder for clean ONNX export."""

    def __init__(self, sam3_backbone: SAM3VLBackbone):
        super().__init__()
        self.sam3_backbone = sam3_backbone

    def forward(self, text_tokens: torch.Tensor):
        """
        Args:
            text_tokens: [B, 32] CLIP tokens tensor
        Returns:
            language features for detector/tracker consumption
        """
        output = {}
        text_attention_mask, text_memory, text_embeds = self.sam3_backbone.language_backbone.forward_onnx(
            text_tokens
        )
        text_memory = text_memory[:, :1]  # Note: sam3 default config
        text_attention_mask = text_attention_mask[:1]
        text_embeds = text_embeds[:, :1]
        output["language_features"] = text_memory
        output["language_mask"] = text_attention_mask
        output["language_embeds"] = (
            text_embeds  # Text embeddings before forward to the encoder
        )
        return output


class GeometryEncoderWrapper(torch.nn.Module):
    """
    Wrap SAM3 geometry encoder for clean ONNX export.
    """
    def __init__(self, sam3: Sam3Image):
        super().__init__()
        self.sam3_geometry_encoder = sam3.geometry_encoder

    def forward(
        self,
        points: torch.Tensor,  # (xp, b, 2)
        points_mask: torch.Tensor,  # (b, xp)
        points_labels: torch.Tensor,  # (xp, b)
        boxes: torch.Tensor,  # (xb, b, 4)
        boxes_mask: torch.Tensor,  # (b, xb)
        boxes_labels: torch.Tensor,  # (xb, b)
        seq_first_img_feats: torch.Tensor,  # (res, res, b, space)
        seq_first_img_pos_embeds: torch.Tensor,  # (res, res, b, space)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ...
        Returns:
            geo_feats: (xp + xb + 1, b, space) shape.
            geo_masks: (b, xp + xb + 1) shape.
        """
        img_size = seq_first_img_feats.shape[:2]
        batch_size, space_size = seq_first_img_feats.shape[2:]
        return self.sam3_geometry_encoder.forward_onnx(
            points,
            points_mask,
            points_labels,
            boxes,
            boxes_mask,
            boxes_labels,
            seq_first_img_feats.view(-1, batch_size, space_size),
            seq_first_img_pos_embeds.view(-1, batch_size, space_size),
            img_size,
        )


class TransformerDetectorWrapper(torch.nn.Module):
    """
    Wrap transformer detector network for clean ONNX export.
    """
    def __init__(self, sam3: Sam3Image):
        super().__init__()
        self.sam3_model = sam3

    def forward(
        self,
        backbone_fpn_0: torch.Tensor,
        backbone_fpn_1:  torch.Tensor,
        backbone_fpn_2: torch.Tensor,  # (b, space, res, res)
        vision_pos_enc_2: torch.Tensor,  # (b, space, res, res)
        prompt: torch.Tensor,  # (xp, b, space)
        prompt_mask: torch.Tensor,  # (b, xp)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ...
        Returns:
            ...
        """
        return self.sam3_model.forward_grounding_transformerDetector_onnx(
            backbone_fpn_0,
            backbone_fpn_1,
            backbone_fpn_2,  # (b, space, res, res)
            vision_pos_enc_2,  # (b, space, res, res)
            prompt,  # (xp, b, space)
            prompt_mask,  # (b, xp)
        )


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


def export_transformerDetector(
    sam3_model: Sam3Image,
    output_path: Path,
    opset: int,
):
    device = next(sam3_model.parameters()).device
    transformer = TransformerDetectorWrapper(sam3_model).to(device)
    transformer.eval()
    onnx_model_path = str(output_path / "transformer.onnx")
    with torch.no_grad():
        torch.onnx.export(
            transformer,
            (
                torch.rand(1, 256, 288, 288).to(device),
                torch.rand(1, 256, 144, 144).to(device),
                torch.rand(1, 256, 72, 72).to(device),
                torch.rand(1, 256, 72, 72).to(device),
                torch.rand(36, 1, 256).to(device),
                torch.randint(0, 2, (1, 36)).bool().to(device)
            ),
            onnx_model_path,
            opset_version=opset,
            dynamo=False,  # Note: dynamo gives an error about aten_split
            input_names=[
                "backbone_fpn_0",
                "backbone_fpn_1",
                "backbone_fpn_2",
                "vision_pos_enc_2",
                "prompt",
                "prompt_mask",
            ],
            output_names=[
                "pred_boxes",
                "pred_logits",
                "pred_masks",
                "presence_logit_dec",
            ],
            dynamic_axes={
                "backbone_fpn_0": {0: "batch_size"},
                "backbone_fpn_1": {0: "batch_size"},
                "backbone_fpn_2": {0: "batch_size"},
                "vision_pos_enc_2": {0: "batch_size"},
                "prompt": {0: "xp", 1: "batch_size"},
                "prompt_mask": {0: "batch_size", 1: "xp"},
                "pred_boxes": {0: "batch_size"},
                "pred_logits": {0: "batch_size"},
                "pred_masks": {0: "batch_size"},
                "presence_logit_dec": {0: "batch_size"},
            },
        )


def export_geometryEncoder(
    sam3_model: Sam3Image,
    output_path: Path,
    opset: int,
):
    device = next(sam3_model.parameters()).device
    geometryEncoder = GeometryEncoderWrapper(sam3_model).to(device)
    geometryEncoder.eval()
    onnx_model_path = str(output_path / "geometryEncoder.onnx")
    with torch.no_grad():
        torch.onnx.export(
            geometryEncoder,
            (
                torch.rand(1, 1, 2).to(device),
                torch.rand(1, 1).to(device),
                torch.rand(1, 1).to(device),
                torch.rand(1, 1, 4).to(device),
                torch.rand(1, 1).to(device),
                torch.rand(1, 1).to(device),
                torch.rand(72, 72, 1, 256).to(device),  # Note: sam3 default params
                torch.rand(72, 72, 1, 256).to(device),
            ),
            onnx_model_path,
            opset_version=opset,
            dynamo=False,  # force legacy exporter because roi_align ops in geometry encoder.
            input_names=[
                "points",
                "points_mask",
                "points_label",
                "boxes",
                "boxes_mask",
                "boxes_labels",
                "seq_first_img_feats",
                "seq_first_img_pos_embeds",
            ],
            output_names=[
                "geo_feats",
                "geo_masks",
            ],
            dynamic_axes={
                "points": {0: "xp", 1: "batch_size"},
                "points_mask": {0: "batch_size", 1: "xp"},
                "points_label": {0: "xp", 1: "batch_size"},
                "boxes": {0: "xb", 1: "batch_size"},
                "boxes_mask": {0: "batch_size", 1: "xb"},
                "boxes_labels": {0: "xb", 1: "batch_size"},
                "seq_first_img_feats": {2: "batch_size"},
                "seq_first_img_pos_embeds": {2: "batch_size"},
                "geo_feats": {0: "xbpxp", 1: "batch_size"},
                "geo_masks": {0: "batch_size", 1: "xbpxp"},
            }
        )
    return


def export_backboneTextEncoder(
    sam3_model_backbone: SAM3VLBackbone,
    caption: str,
    tokenizer: SimpleTokenizer,
    output_path: Path,
    opset: int,
):
    device = next(sam3_model_backbone.parameters()).device
    context_length = 32  # Note: as in CLIP.
    backboneTextWrapper = BackboneTextEncoderWrapper(sam3_model_backbone).to(device)
    backboneTextWrapper.eval()
    onnx_model_path = str(output_path / "backboneTextEncoder.onnx")

    # Encode the text
    tokenized = tokenizer(
        [caption],
        context_length=context_length,
    ).to(device)  # [b, seq_len]
    with torch.no_grad():
        torch.onnx.export(
            backboneTextWrapper,
            tokenized,
            onnx_model_path,
            opset_version=opset,
            input_names=["text_tokens"],
            output_names=[
                "text_attention_mask",
                "text_memory_resized",
                "inputs_embeds",
            ],
            dynamic_axes={
                "text_tokens": {0: "batch_size"},
                "text_attention_mask": {0: "batch_size"},
                "text_memory_resized": {1: "batch_size"},
                "inputs_embeds": {1: "batch_size"},
            }
        )
    return


def export_backboneImageEncoder(
    sam3_model_backbone: SAM3VLBackbone,
    dummy_image: torch.Tensor,
    output_path: Path,
    opset: int,
):
    device = next(sam3_model_backbone.parameters()).device
    backboneImageWrapper = BackboneImageEncoderWrapper(sam3_model_backbone).to(device)
    backboneImageWrapper.eval()
    onnx_model_path = str(output_path / "backboneImageEncoder.onnx")
    with torch.no_grad():
        torch.onnx.export(
            backboneImageWrapper,
            dummy_image,
            onnx_model_path,
            opset_version=opset,
            input_names=["image"],
            output_names=[
                "vision_features",
                "vision_pos_enc0",
                "vision_pos_enc1",
                "vision_pos_enc2",
                "backbone_fpn0",
                "backbone_fpn1",
                "backbone_fpn2",
            ],
            dynamic_axes={
                "image": {0: "batch_size"},
                "vision_features": {0: "batch_size"},
                "vision_pos_enc0": {0: "batch_size"},
                "vision_pos_enc1": {0: "batch_size"},
                "vision_pos_enc2": {0: "batch_size"},
                "backbone_fpn0": {0: "batch_size"},
                "backbone_fpn1": {0: "batch_size"},
                "backbone_fpn2": {0: "batch_size"},
            },
        )
    return


def export_component(
    output: str,
    image_size: tuple[int] = (720, 1280),
    opset: int = 17,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # construct the model
    model_image_resolution: int = 1008
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = f"{sam3_root}/sam3/assets/bpe_simple_vocab_16e6.txt.gz"  # text tokenizer for CLIP
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)

    checkpoint_path = f"{sam3_root}/sam3/assets/sam3.pt"
    starttime = time.time()
    sam3_model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path=checkpoint_path)
    logging.info(f"Model loaded in {time.time() - starttime} sec.")

    transform = v2.Compose(
        [
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(model_image_resolution, model_image_resolution)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    sam3_model = sam3_model.to(device)
    sam3_model.eval()

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    dummy_image = torch.randn(3, image_size[0], image_size[1], device=device)
    dummy_image = transform(dummy_image).unsqueeze(0)

    logging.info(f"Exporting backbone image encoder (opset {opset}) → {output_path}")
    export_backboneImageEncoder(
        sam3_model.backbone,
        dummy_image,
        output_path,
        opset,
    )

    logging.info(f"Exporting backbone text prompt encoder (opset {opset}) → {output_path}")
    export_backboneTextEncoder(
        sam3_model.backbone,
        "visual features",
        tokenizer,
        output_path,
        opset,
    )

    logging.info(f"Exporting geometry encoder (opset {opset}) -> {output_path}")
    export_geometryEncoder(
        sam3_model,
        output_path,
        opset,
    )

    logging.info(f"Exporting export_transformerDetector (opset {opset}) > {output_path}")
    export_transformerDetector(
        sam3_model,
        output_path,
        opset,
    )

    # # Validate
    # import onnx
    # model = onnx.load(onnx_model_path)
    # onnx.checker.check_model(model)
    # logging.info("ONNX validation passed ✓")

    logging.info("finished exporting 'backboneImageEncoder', 'backboneTextEncoder', 'geometryEncoder', 'transformerDetector'.")


if __name__ == "__main__":
    ConfigLogging()

    parser = argparse.ArgumentParser(description="Export SAM 3 to ONNX")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--imagesize", type=int, nargs=2, default=[720, 1280])
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_component(args.output, args.imagesize, args.opset)
