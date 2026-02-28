#!/usr/bin/env python3
"""Run SAM 3 inference with TensorRT engines.

Pipeline (4 engines):
  1. backboneImageEncoder  — image → vision features + FPN + pos encodings
  2. backboneTextEncoder   — CLIP tokens → text features + mask + embeds
  3. geometryEncoder       — points/boxes + image features → geometry features
  4. transformer           — FPN + pos enc + prompt → boxes, logits, masks

Usage:
  python sam3_trt_infer.py \
    --engine-dir ./engines/ \
    --image test.jpg \
    --text "a person" \
    --points 320,240 640,480 \
    --point-labels 1 1

Note: Maximum 32 point prompts and 32 box prompts supported.
"""

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from pathlib import Path

import tensorrt as trt

# Register TensorRT plugins (e.g. ROIAlign_TRT used by geometry encoder)
trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), "")
import pycuda.driver as cuda
import pycuda.autoinit  # auto-initializes CUDA context


def ConfigLogging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        force=True,
    )


# ─── TRT Engine Wrapper ─────────────────────────────────────────

class TRTEngine:
    """Generic TensorRT engine wrapper with proper memory management."""

    def __init__(self, engine_path: str, engine_bytes: bytes | None = None):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.name = Path(engine_path).stem
        logging.info(f"Loading TRT engine: {engine_path}")

        if engine_bytes is None:
            with open(engine_path, "rb") as f:
                engine_bytes = f.read()

        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        assert self.engine is not None, (
            f"Failed to deserialize engine: {engine_path}. "
            f"Ensure it was built with the same GPU and TensorRT version."
        )

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Discover I/O tensors
        self.input_names = []
        self.output_names = []
        self.dtypes = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            self.dtypes[name] = trt.nptype(self.engine.get_tensor_dtype(name))
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        logging.info(f"  Inputs:  {self.input_names}")
        logging.info(f"  Outputs: {self.output_names}")

    def __call__(self, **inputs) -> dict:
        d_buffers = {}

        try:
            # Set input shapes, allocate and copy to device
            for name in self.input_names:
                arr = np.ascontiguousarray(inputs[name])
                self.context.set_input_shape(name, arr.shape)

                d_mem = cuda.mem_alloc(arr.nbytes)
                d_buffers[name] = d_mem
                cuda.memcpy_htod_async(d_mem, arr, self.stream)
                self.context.set_tensor_address(name, int(d_mem))

            # Allocate output buffers
            h_outputs = {}
            for name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                dtype = self.dtypes[name]
                h_buf = np.empty(shape, dtype=dtype)
                d_mem = cuda.mem_alloc(h_buf.nbytes)
                d_buffers[name] = d_mem
                h_outputs[name] = h_buf
                self.context.set_tensor_address(name, int(d_mem))

            # Execute
            ok = self.context.execute_async_v3(self.stream.handle)
            assert ok, f"TRT execution failed for {self.name}"

            # Copy outputs back to host
            for name in self.output_names:
                h_buf = h_outputs[name]
                cuda.memcpy_dtoh_async(h_buf, d_buffers[name], self.stream)

            self.stream.synchronize()
            return h_outputs

        finally:
            # Always free GPU memory
            for d_mem in d_buffers.values():
                d_mem.free()


# ─── SAM3 TRT Pipeline ──────────────────────────────────────────

class SAM3TRTPipeline:
    """Full SAM3 inference pipeline using 4 TensorRT engines."""

    def __init__(self, engine_dir: str):
        p = Path(engine_dir)
        ctx = cuda.Context.get_current()
        names = [
            "backboneImageEncoder.engine",
            "backboneTextEncoder.engine",
            "geometryEncoder.engine",
            "transformer.engine",
        ]

        # Loading tensorrt engines in parallel: Phase1 + Phase2
        starttime = time.time()

        # Phase 1: read all files in parallel (pure I/O, no CUDA)
        def read_bytes(name):
            with open(str(p / name), "rb") as f:
                return f.read()

        with ThreadPoolExecutor(max_workers=4) as ex:
            all_bytes = list(ex.map(read_bytes, names))
        logging.info(f"  File I/O: {(time.time() - starttime) * 1000:.1f} ms")

        # Phase 2: deserialize all engines in parallel (CUDA)
        t1 = time.time()
        def deserialize(args):
            name, data = args
            ctx.push()
            try:
                return TRTEngine(name, engine_bytes=data)
            finally:
                ctx.pop()

        with ThreadPoolExecutor(max_workers=4) as ex:
            engines = list(ex.map(deserialize, zip(names, all_bytes)))
        logging.info(f"  Deserialize: {(time.time() - t1) * 1000:.1f} ms")

        self.image_encoder, self.text_encoder, self.geometry_encoder, self.transformer = engines
        logging.info(f"Engine loading total: {(time.time() - starttime) * 1000:.1f} ms")

        # Load CLIP tokenizer
        self.tokenizer = self._load_tokenizer()

    def _load_tokenizer(self):
        """Load the CLIP BPE tokenizer."""
        from text_tokenizer import SimpleTokenizer
        import os
        bpe_path = os.path.join(os.path.dirname(__file__), "..", "sam3", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
        return SimpleTokenizer(bpe_path=bpe_path)

    # ── Preprocessing ────────────────────────────────────────────

    def preprocess_image(self, image_path: str, model_size: int = 1008) -> tuple:
        """Load and preprocess image to match SAM3 training transform.

        Matches:
            v2.ToDtype(torch.uint8, scale=True)
            v2.Resize(size=(1008, 1008))
            v2.ToDtype(torch.float32, scale=True)   # uint8 [0,255] -> float32 [0,1]
            v2.Normalize(mean=[0.5]*3, std=[0.5]*3)  # [0,1] -> [-1,1]

        Returns:
            image_tensor: [1, 3, 1008, 1008] float32 normalized to [-1, 1]
            original_size: (H, W) of original image
        """
        img = cv2.imread(image_path)
        assert img is not None, f"Failed to load image: {image_path}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_size = img.shape[:2]

        # Clamp to uint8 (matching v2.ToDtype(torch.uint8, scale=True))
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Resize — direct squash, no aspect ratio preservation
        # (matching v2.Resize(size=(1008, 1008)))
        img = cv2.resize(img, (model_size, model_size), interpolation=cv2.INTER_LINEAR)

        # uint8 [0,255] -> float32 [0,1] (matching v2.ToDtype(torch.float32, scale=True))
        img = img.astype(np.float32) / 255.0

        # Normalize [0,1] -> [-1,1] (matching v2.Normalize(mean=[0.5]*3, std=[0.5]*3))
        img = (img - 0.5) / 0.5

        # HWC -> CHW -> BCHW
        tensor = img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        return tensor, original_size

    def tokenize_text(self, text: str, context_length: int = 32) -> np.ndarray:
        """Tokenize text using CLIP tokenizer.

        Returns:
            tokens: [1, 32] int64/int32 array
        """
        tokens = self.tokenizer([text], context_length=context_length)
        return tokens

    def prepare_geometry_inputs(
        self,
        points: np.ndarray | None = None,       # (N, 2) pixel coords
        point_labels: np.ndarray | None = None,  # (N,) 1=foreground, 0=background
        boxes: np.ndarray | None = None,         # (M, 4) pixel [x1,y1,x2,y2]
        box_labels: np.ndarray | None = None,    # (M,)
        original_size: tuple = None,             # (H, W) of original image
        model_size: int = 1008,
        max_points: int = 32,
        max_boxes: int = 32,
    ) -> dict:
        """Pad point/box inputs to fixed size for TRT engine.

        Coordinates are in pixel space of the original image and will be
        normalized to [0, 1] relative to model_size.
        """
        assert original_size is not None, "original_size required to normalize coordinates"
        assert max_points <= 32, "Maximum 32 point prompts supported"
        assert max_boxes <= 32, "Maximum 32 box prompts supported"
        if points is not None:
            assert len(points) <= max_points, f"Got {len(points)} points, max is {max_points}"
        if boxes is not None:
            assert len(boxes) <= max_boxes, f"Got {len(boxes)} boxes, max is {max_boxes}"
        orig_h, orig_w = original_size
        scale_x = model_size / orig_w
        scale_y = model_size / orig_h
        # Points: (max_points, 1, 2), mask: (1, max_points), label: (max_points, 1)
        # Mask convention everywhere: 0=real prompt, 1=padded
        pts = np.zeros((max_points, 1, 2), dtype=np.float32)
        pts_mask = np.ones((1, max_points), dtype=bool)  # all padded by default
        pts_label = np.zeros((max_points, 1), dtype=bool)

        if points is not None and len(points) > 0:
            n = min(len(points), max_points)
            # Normalize pixel coords to model space [0, model_size]
            # then to [0, 1] relative to model_size
            scaled = points[:n].copy()
            scaled[:, 0] = scaled[:, 0] * scale_x / model_size  # x
            scaled[:, 1] = scaled[:, 1] * scale_y / model_size  # y
            pts[:n, 0, :] = scaled
            pts_mask[0, :n] = False  # False = real prompt
            if point_labels is not None:
                pts_label[:n, 0] = point_labels[:n].astype(bool)
            else:
                pts_label[:n, 0] = True  # default: foreground

        # Boxes: (max_boxes, 1, 4), mask: (1, max_boxes), label: (max_boxes, 1)
        # Mask convention everywhere: 0=real prompt, 1=padded
        bxs = np.zeros((max_boxes, 1, 4), dtype=np.float32)
        bxs_mask = np.ones((1, max_boxes), dtype=bool)  # all padded by default
        bxs_label = np.zeros((max_boxes, 1), dtype=bool)

        if boxes is not None and len(boxes) > 0:
            m = min(len(boxes), max_boxes)
            # Normalize pixel coords to model space [0, model_size]
            # then to [0, 1] relative to model_size
            scaled = boxes[:m].copy()
            scaled[:, 0] = scaled[:, 0] * scale_x / model_size  # x1
            scaled[:, 1] = scaled[:, 1] * scale_y / model_size  # y1
            scaled[:, 2] = scaled[:, 2] * scale_x / model_size  # x2
            scaled[:, 3] = scaled[:, 3] * scale_y / model_size  # y2
            bxs[:m, 0, :] = scaled
            bxs_mask[0, :m] = False  # False = real prompt
            if box_labels is not None:
                bxs_label[:m, 0] = box_labels[:m].astype(bool)
            else:
                bxs_label[:m, 0] = True

        return {
            "points": pts,
            "points_mask": pts_mask,
            "points_label": pts_label,
            "boxes": bxs,
            "boxes_mask": bxs_mask,
            "boxes_labels": bxs_label,
        }

    # ── Engine calls ─────────────────────────────────────────────

    def encode_image(self, image: np.ndarray) -> dict:
        """Run backbone image encoder.

        Args:
            image: [1, 3, 1008, 1008] float32

        Returns dict with keys:
            vision_features, vision_pos_enc0/1/2, backbone_fpn0/1/2
        """
        return self.image_encoder(image=image)

    def encode_text(self, tokens: np.ndarray) -> dict:
        """Run backbone text encoder.

        Args:
            tokens: [1, 32] CLIP tokens

        Returns dict with keys:
            language_features: (xt, 1, 256) text memory
            language_mask: (1, xt) text attention mask
            language_embeds: (xt, 1, 256) text embeddings
        """
        return self.text_encoder(text_tokens=tokens)

    def encode_geometry(
        self,
        geo_inputs: dict,
        img_feats: np.ndarray,
        img_pos: np.ndarray,
    ) -> dict:
        """Run geometry encoder.

        Args:
            geo_inputs: dict from prepare_geometry_inputs()
            img_feats: backbone_fpn2 reshaped to (72, 72, 1, 256)
            img_pos: vision_pos_enc2 reshaped to (72, 72, 1, 256)

        Returns dict with keys:
            geo_feats: (xp+xb+1, 1, 256)
            geo_masks: (1, xp+xb+1)
        """
        return self.geometry_encoder(
            **geo_inputs,
            seq_first_img_feats=img_feats,
            seq_first_img_pos_embeds=img_pos,
        )

    def detect(
        self,
        backbone_fpn_0: np.ndarray,
        backbone_fpn_1: np.ndarray,
        backbone_fpn_2: np.ndarray,
        vision_pos_enc_2: np.ndarray,
        prompt: np.ndarray,
        prompt_mask: np.ndarray,
    ) -> dict:
        """Run transformer detector.

        Returns dict with keys:
            pred_boxes, pred_logits, pred_masks, presence_logit_dec
        """
        return self.transformer(
            backbone_fpn_0=backbone_fpn_0,
            backbone_fpn_1=backbone_fpn_1,
            backbone_fpn_2=backbone_fpn_2,
            vision_pos_enc_2=vision_pos_enc_2,
            prompt=prompt,
            prompt_mask=prompt_mask,
        )

    # ── Helper: reshape FPN features for geometry encoder ────────

    @staticmethod
    def fpn_to_seq_first(fpn: np.ndarray, pos: np.ndarray):
        """Convert (B, C, H, W) FPN features to (H, W, B, C) for geometry encoder.

        Args:
            fpn: (1, 256, 72, 72) backbone_fpn2
            pos: (1, 256, 72, 72) vision_pos_enc2

        Returns:
            img_feats: (72, 72, 1, 256)
            img_pos:   (72, 72, 1, 256)
        """
        # (B, C, H, W) -> (H, W, B, C)
        img_feats = fpn.transpose(2, 3, 0, 1).astype(np.float32)
        img_pos = pos.transpose(2, 3, 0, 1).astype(np.float32)
        return img_feats, img_pos

    # ── Full pipeline ────────────────────────────────────────────

    def run(
        self,
        image_path: str,
        text: str | None = None,
        points: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        boxes: np.ndarray | None = None,
        box_labels: np.ndarray | None = None,
    ) -> dict:
        """Run the full SAM3 detection pipeline.

        Args:
            image_path: path to input image
            text: text prompt (e.g., "a person")
            points: (N, 2) point coordinates in pixel space (max 32)
            point_labels: (N,) point labels (1=fg, 0=bg)
            boxes: (M, 4) box coordinates in pixel space [x1,y1,x2,y2] (max 32)
            box_labels: (M,) box labels

        Returns:
            dict with pred_boxes, pred_logits, pred_masks, presence_logit_dec
        """
        timings = {}

        # 1. Encode image
        t0 = time.perf_counter()
        image, original_size = self.preprocess_image(image_path)
        vision_out = self.encode_image(image)
        t1 = time.perf_counter()
        timings["image_encoder"] = (t1 - t0) * 1000
        logging.info(f"  Image encoder: {timings['image_encoder']:.1f} ms")

        prompt_parts = []
        mask_parts = []

        # 2. Encode text (always runs — defaults to "visual" if no text prompt given)
        t2 = time.perf_counter()
        text_prompt = text if text is not None else "visual"
        tokens = self.tokenize_text(text_prompt)
        text_out = self.encode_text(tokens)
        t3 = time.perf_counter()
        timings["text_encoder"] = (t3 - t2) * 1000
        logging.info(f"  Text encoder:  {timings['text_encoder']:.1f} ms (prompt: '{text_prompt}')")

        # language_features: (xt, 1, 256), language_mask: (1, xt)
        prompt_parts.append(text_out["language_features"])
        mask_parts.append(text_out["language_mask"])

        # 3. Encode geometry (always runs — empty prompts use all-padded tensors)
        t4 = time.perf_counter()

        geo_inputs = self.prepare_geometry_inputs(
            points, point_labels, boxes, box_labels,
            original_size=original_size,
        )

        # Reshape FPN features for geometry encoder
        img_feats, img_pos = self.fpn_to_seq_first(
            vision_out["backbone_fpn2"],
            vision_out["vision_pos_enc2"],
        )

        geo_out = self.encode_geometry(geo_inputs, img_feats, img_pos)
        t5 = time.perf_counter()
        timings["geometry_encoder"] = (t5 - t4) * 1000
        logging.info(f"  Geo encoder:   {timings['geometry_encoder']:.1f} ms")

        # geo_feats: (xp+xb+1, 1, 256), geo_masks: (1, xp+xb+1)
        prompt_parts.append(geo_out["geo_feats"])
        mask_parts.append(geo_out["geo_masks"])

        # 4. Combine prompts
        prompt = np.concatenate(prompt_parts, axis=0).astype(np.float32)
        prompt_mask = np.concatenate(mask_parts, axis=1)  # keep original dtype (0=real, 1=padded)

        logging.info(f"  Prompt shape: {prompt.shape}, mask shape: {prompt_mask.shape}")

        # 5. Run transformer detector
        t6 = time.perf_counter()
        results = self.detect(
            backbone_fpn_0=vision_out["backbone_fpn0"],
            backbone_fpn_1=vision_out["backbone_fpn1"],
            backbone_fpn_2=vision_out["backbone_fpn2"],
            vision_pos_enc_2=vision_out["vision_pos_enc2"],
            prompt=prompt,
            prompt_mask=prompt_mask,
        )
        t7 = time.perf_counter()
        timings["transformer"] = (t7 - t6) * 1000
        logging.info(f"  Transformer:   {timings['transformer']:.1f} ms")

        total = sum(timings.values())
        logging.info(f"  Total:         {total:.1f} ms")

        results["timings"] = timings
        results["original_size"] = original_size
        return results


# ─── Visualization ───────────────────────────────────────────────

def postprocess(results: dict, original_size: tuple, confidence_threshold: float = 0.5) -> dict:
    """Post-process transformer detector outputs following SAM3's logic.

    Args:
        results: dict with pred_boxes, pred_logits, pred_masks, presence_logit_dec
        original_size: (H, W) of original image
        confidence_threshold: minimum confidence to keep a detection

    Returns:
        dict with boxes, masks, scores (filtered and scaled to original image)
    """
    pred_boxes = results["pred_boxes"]         # (1, num_queries, 4) cxcywh normalized
    pred_logits = results["pred_logits"]       # (1, num_queries, num_classes)
    pred_masks = results["pred_masks"]         # (1, num_queries, H, W)
    presence = results["presence_logit_dec"]   # (1, num_queries)

    # sigmoid
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    out_probs = sigmoid(pred_logits[0])                    # (num_queries, num_classes)
    presence_score = sigmoid(presence[0])[:, np.newaxis]   # (num_queries, 1)
    out_probs = (out_probs * presence_score).squeeze(-1)   # (num_queries,)

    # Filter by confidence
    keep = out_probs > confidence_threshold
    out_probs = out_probs[keep]
    out_boxes = pred_boxes[0][keep]     # (N, 4) cxcywh
    out_masks = pred_masks[0][keep]     # (N, H, W)

    # Convert cxcywh -> xyxy
    cx, cy, w, h = out_boxes[:, 0], out_boxes[:, 1], out_boxes[:, 2], out_boxes[:, 3]
    x0 = cx - w / 2
    y0 = cy - h / 2
    x1 = cx + w / 2
    y1 = cy + h / 2
    boxes_xyxy = np.stack([x0, y0, x1, y1], axis=-1)

    # Scale boxes to original image size
    img_h, img_w = original_size
    scale = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    boxes_xyxy = boxes_xyxy * scale[np.newaxis, :]

    # Resize and sigmoid masks to original image size
    masks = []
    for m in out_masks:
        m_resized = cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        masks.append(sigmoid(m_resized))
    masks = np.array(masks) if len(masks) > 0 else np.zeros((0, img_h, img_w))

    return {
        "boxes": boxes_xyxy,          # (N, 4) xyxy in pixel coords
        "masks": masks > 0.5,         # (N, H, W) binary masks
        "masks_logits": masks,         # (N, H, W) soft masks
        "scores": out_probs,           # (N,)
    }


def visualize(image_path: str, post: dict, output_path: str):
    """Draw detected masks and boxes on the image."""
    img = cv2.imread(image_path)

    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    ]

    for i, score in enumerate(post["scores"]):
        color = colors[i % len(colors)]

        # Draw box
        x1, y1, x2, y2 = post["boxes"][i].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label background + border + text: "id=0, prob=0.93"
        label = f"id={i}, prob={score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y2), (x1 + tw, y2 + th + baseline + 4), (255, 255, 255), -1)
        cv2.rectangle(img, (x1, y2), (x1 + tw, y2 + th + baseline + 4), color, 1)
        cv2.putText(img, label, (x1, y2 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 139), 1, cv2.LINE_AA)

        # Overlay mask
        mask = post["masks"][i]
        overlay = img.copy()
        overlay[mask] = (
            overlay[mask] * 0.5 +
            np.array(color, dtype=np.float32) * 0.5
        ).astype(np.uint8)
        img = overlay

    cv2.imwrite(output_path, img)
    logging.info(f"Saved: {output_path}")


# ─── CLI ─────────────────────────────────────────────────────────

def parse_points(points_list: list[str] | None) -> np.ndarray | None:
    """Parse ['320,240', '640,480'] -> (N, 2) array in pixel coordinates."""
    if not points_list:
        return None
    pts = []
    for p in points_list:
        x, y = p.strip().split(",")
        pts.append([float(x), float(y)])
    return np.array(pts, dtype=np.float32)


def parse_labels(labels_list: list[str] | None) -> np.ndarray | None:
    """Parse ['1', '0', '1'] -> (N,) array."""
    if not labels_list:
        return None
    return np.array([int(x) for x in labels_list], dtype=np.float32)


def parse_boxes(boxes_list: list[str] | None) -> np.ndarray | None:
    """Parse ['100,200,500,600', '300,400,700,800'] -> (M, 4) array in pixel coordinates."""
    if not boxes_list:
        return None
    bxs = []
    for b in boxes_list:
        coords = [float(x) for x in b.strip().split(",")]
        assert len(coords) == 4, f"Box must have 4 coords, got {len(coords)}"
        bxs.append(coords)
    return np.array(bxs, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="SAM 3 TensorRT Inference")
    parser.add_argument("--engine-dir", required=True,
                        help="Directory containing .engine files")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--text", default=None,
                        help="Text prompt (e.g., 'a person')")
    parser.add_argument("--points", nargs="+", default=None,
                        help="Point prompts, each as 'x,y' (pixel coords, max 32 total). "
                             "Example: --points 320,240 640,480 100,300")
    parser.add_argument("--point-labels", nargs="+", default=None,
                        help="Point labels, each as '0' or '1' (1=fg, 0=bg). "
                             "Example: --point-labels 1 1 0")
    parser.add_argument("--boxes", nargs="+", default=None,
                        help="Box prompts, each as 'x1,y1,x2,y2' (pixel coords, max 32 total). "
                             "Example: --boxes 100,200,500,600 300,400,700,800")
    parser.add_argument("--box-labels", nargs="+", default=None,
                        help="Box labels, each as '0' or '1' (1=fg, 0=bg). "
                             "Example: --box-labels 1 0")
    parser.add_argument("--output", default=None,
                        help="Output image path (default: <input>_sam3_output.jpg)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Detection score threshold")

    args = parser.parse_args()

    # Parse inputs
    points = parse_points(args.points)
    point_labels = parse_labels(args.point_labels)
    boxes = parse_boxes(args.boxes)
    box_labels = parse_labels(args.box_labels)

    # Build pipeline
    pipeline = SAM3TRTPipeline(args.engine_dir)

    # Run
    logging.info(f"Running SAM3 on: {args.image}")
    results = pipeline.run(
        image_path=args.image,
        text=args.text,
        points=points,
        point_labels=point_labels,
        boxes=boxes,
        box_labels=box_labels,
    )

    # Print output shapes
    logging.info("Raw outputs:")
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            logging.info(f"  {k}: {v.shape} ({v.dtype})")

    # Post-process
    post = postprocess(results, results["original_size"], args.threshold)
    logging.info(f"Detections: {len(post['scores'])} (threshold={args.threshold})")
    for i, score in enumerate(post["scores"]):
        logging.info(f"  [{i}] score={score:.3f} box={post['boxes'][i].astype(int).tolist()}")

    # Visualize
    output_path = args.output or str(Path(args.image).stem) + "_sam3_output.jpg"
    visualize(args.image, post, output_path)


if __name__ == "__main__":
    ConfigLogging()
    main()
