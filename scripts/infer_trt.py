#!/usr/bin/env python3
"""Run SAM 3 inference with TensorRT vision encoder + detector.

Pipeline:
  1. Preprocess image
  2. TRT vision encoder → vision features
  3. Compute text embeddings (PyTorch, lightweight)
  4. TRT detector → boxes, masks, scores
  5. Postprocess and visualize
"""

import argparse
import time
import numpy as np
import cv2
import torch
from pathlib import Path

import tensorrt as trt
from cuda import cudart


class TRTEngine:
    """Generic TensorRT engine wrapper."""

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        print(f"Loading TRT engine: {engine_path}")

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cudart.cudaStreamCreate()[1]

        # Discover I/O tensors
        self.io_info = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            self.io_info[name] = {
                "dtype": dtype,
                "is_input": mode == trt.TensorIOMode.INPUT,
                "shape": shape,
                "device": None,
            }

    def _alloc(self, name, shape):
        dtype = self.io_info[name]["dtype"]
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        _, d_mem = cudart.cudaMalloc(size)
        self.io_info[name]["device"] = d_mem
        self.io_info[name]["alloc_size"] = size
        return d_mem

    def __call__(self, **inputs) -> dict:
        # Set input shapes and copy data
        for name, arr in inputs.items():
            self.context.set_input_shape(name, arr.shape)
            d_mem = self._alloc(name, arr.shape)
            cudart.cudaMemcpyAsync(
                d_mem, arr.ctypes.data, arr.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream,
            )

        # Allocate outputs
        for name, info in self.io_info.items():
            if not info["is_input"]:
                shape = self.context.get_tensor_shape(name)
                self._alloc(name, shape)

        # Set tensor addresses
        for name, info in self.io_info.items():
            self.context.set_tensor_address(name, info["device"])

        # Execute
        self.context.execute_async_v3(self.stream)

        # Copy outputs back
        results = {}
        for name, info in self.io_info.items():
            if not info["is_input"]:
                shape = self.context.get_tensor_shape(name)
                host = np.empty(shape, dtype=info["dtype"])
                cudart.cudaMemcpyAsync(
                    host.ctypes.data, info["device"], host.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream,
                )
                results[name] = host

        cudart.cudaStreamSynchronize(self.stream)

        # Free allocations
        for info in self.io_info.values():
            if info["device"] is not None:
                cudart.cudaFree(info["device"])
                info["device"] = None

        return results

    def __del__(self):
        for info in self.io_info.values():
            if info.get("device"):
                cudart.cudaFree(info["device"])
        cudart.cudaStreamDestroy(self.stream)


def preprocess_image(image_path: str, size: int = 1024):
    """Load and preprocess image for SAM 3."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = img.shape[:2]

    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h))

    padded = np.zeros((size, size, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = img

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (padded.astype(np.float32) / 255.0 - mean) / std
    tensor = normalized.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    return tensor, original_size, (new_h, new_w)


def get_text_embeddings(text: str, model=None):
    """Compute text embeddings using SAM 3's text encoder.

    The text encoder is lightweight — runs in PyTorch, not worth TRT conversion.
    """
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if model is None:
        model = build_sam3_image_model()

    processor = Sam3Processor(model)
    # Extract text embeddings from the processor
    # (implementation depends on SAM 3's internal API)
    text_emb = processor.encode_text(text)
    return text_emb.cpu().numpy()


def visualize(image_path: str, masks, boxes, scores, output_path: str, threshold: float = 0.5):
    """Draw detected masks and boxes on the image."""
    img = cv2.imread(image_path)

    for i, score in enumerate(scores):
        if score < threshold:
            continue

        # Draw box
        if boxes is not None and i < len(boxes):
            x1, y1, x2, y2 = boxes[i].astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Overlay mask
        if masks is not None and i < len(masks):
            mask = masks[i] > 0.5
            overlay = img.copy()
            overlay[mask] = overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5
            img = overlay

    cv2.imwrite(output_path, img)
    print(f"Saved visualization: {output_path}")


def main(encoder_engine: str, detector_engine: str, image_path: str, text: str):
    # Load TRT engines
    encoder = TRTEngine(encoder_engine)
    detector = TRTEngine(detector_engine)

    # Preprocess image
    image, orig_size, resized = preprocess_image(image_path)
    print(f"Image: {image_path} ({orig_size[1]}x{orig_size[0]}) → {resized[1]}x{resized[0]}")

    # Vision encoder (TRT)
    t0 = time.perf_counter()
    enc_out = encoder(image=image)
    t1 = time.perf_counter()
    print(f"TRT vision encoder: {(t1-t0)*1000:.1f} ms")

    # Text embeddings (PyTorch — lightweight)
    t2 = time.perf_counter()
    text_emb = get_text_embeddings(text)
    t3 = time.perf_counter()
    print(f"Text encoding: {(t3-t2)*1000:.1f} ms")

    # Detector (TRT)
    vision_features = list(enc_out.values())[0]  # primary feature output
    t4 = time.perf_counter()
    det_out = detector(vision_features=vision_features, text_embeddings=text_emb)
    t5 = time.perf_counter()
    print(f"TRT detector: {(t5-t4)*1000:.1f} ms")

    print(f"\nTotal: {(t5-t0)*1000:.1f} ms")
    for name, arr in det_out.items():
        print(f"  {name}: {arr.shape}")

    # Visualize
    output_path = str(Path(image_path).stem) + "_sam3_output.jpg"
    if "masks" in det_out and "boxes" in det_out and "scores" in det_out:
        visualize(image_path, det_out["masks"], det_out["boxes"],
                  det_out["scores"], output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM 3 TensorRT inference")
    parser.add_argument("--encoder-engine", required=True, help="Vision encoder TRT engine")
    parser.add_argument("--detector-engine", required=True, help="Detector TRT engine")
    parser.add_argument("--image", required=True, help="Input image")
    parser.add_argument("--text", required=True, help="Text prompt (e.g. 'dog', 'person in red')")
    args = parser.parse_args()

    main(args.encoder_engine, args.detector_engine, args.image, args.text)
