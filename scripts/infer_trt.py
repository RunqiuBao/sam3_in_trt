#!/usr/bin/env python3
"""Run SAM 2 inference with TensorRT image encoder + PyTorch prompt/mask decoder."""

import argparse
import time
import numpy as np
import cv2
import torch
from pathlib import Path

import tensorrt as trt
from cuda import cudart


class TRTImageEncoder:
    """TensorRT-accelerated SAM 2 image encoder."""

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        print(f"Loading TRT engine: {engine_path}")

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate I/O buffers
        self.inputs = {}
        self.outputs = {}
        self.bindings = []
        self.stream = cudart.cudaStreamCreate()[1]

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            # Replace dynamic dims with max
            shape = tuple(max(1, s) for s in shape)
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize

            # Allocate device memory
            _, d_mem = cudart.cudaMalloc(size)

            binding = {"name": name, "dtype": dtype, "shape": shape, "device": d_mem, "size": size}

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = binding
            else:
                self.outputs[name] = binding

    def __call__(self, image: np.ndarray) -> dict:
        """
        Run image encoder.
        image: [B, 3, 1024, 1024] float32 normalized
        """
        # Set input shape
        self.context.set_input_shape("image", image.shape)

        # Copy input to device
        inp = self.inputs["image"]
        cudart.cudaMemcpyAsync(
            inp["device"], image.ctypes.data,
            image.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream,
        )

        # Set tensor addresses
        for name, buf in {**self.inputs, **self.outputs}.items():
            self.context.set_tensor_address(name, buf["device"])

        # Execute
        self.context.execute_async_v3(self.stream)

        # Copy outputs
        results = {}
        for name, buf in self.outputs.items():
            shape = self.context.get_tensor_shape(name)
            host = np.empty(shape, dtype=buf["dtype"])
            cudart.cudaMemcpyAsync(
                host.ctypes.data, buf["device"],
                host.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            )
            results[name] = host

        cudart.cudaStreamSynchronize(self.stream)
        return results

    def __del__(self):
        for buf in list(self.inputs.values()) + list(self.outputs.values()):
            cudart.cudaFree(buf["device"])
        cudart.cudaStreamDestroy(self.stream)


def preprocess_image(image_path: str, size: int = 1024):
    """Load and preprocess image for SAM 2."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = img.shape[:2]

    # Resize longest side to 1024
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h))

    # Pad to 1024x1024
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = img

    # Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (padded.astype(np.float32) / 255.0 - mean) / std

    # CHW, add batch dim
    tensor = normalized.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    return tensor, original_size, (new_h, new_w)


def main(engine_path: str, image_path: str, point: tuple, label: int = 1):
    # Init TRT encoder
    encoder = TRTImageEncoder(engine_path)

    # Preprocess
    image, orig_size, resized = preprocess_image(image_path)
    print(f"Image: {image_path} ({orig_size[1]}x{orig_size[0]}) → {resized[1]}x{resized[0]}")

    # Encode with TRT
    t0 = time.perf_counter()
    embeddings = encoder(image)
    t1 = time.perf_counter()
    print(f"TRT encoder: {(t1-t0)*1000:.1f} ms")

    for name, arr in embeddings.items():
        print(f"  {name}: {arr.shape} ({arr.dtype})")

    # Prompt encoding + mask decoding stays in PyTorch
    # (lightweight, not worth TRT conversion)
    print(f"\nImage embeddings ready. Point prompt: {point}, label: {label}")
    print("To get masks, pass embeddings through PyTorch prompt encoder + mask decoder.")
    print("See SAM 2 SAM2ImagePredictor for reference.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM 2 TensorRT inference")
    parser.add_argument("--engine", required=True, help="TRT engine path")
    parser.add_argument("--image", required=True, help="Input image")
    parser.add_argument("--point", default="500,375", help="Point prompt x,y")
    parser.add_argument("--label", type=int, default=1, help="Point label (1=fg, 0=bg)")
    args = parser.parse_args()

    px, py = map(int, args.point.split(","))
    main(args.engine, args.image, (px, py), args.label)
