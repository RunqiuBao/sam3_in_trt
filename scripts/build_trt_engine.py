#!/usr/bin/env python3
"""Convert ONNX model to TensorRT engine."""

import argparse
import os
from pathlib import Path

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine(onnx_path: str, engine_path: str, precision: str = "fp16",
                 workspace_mb: int = 4096, min_batch: int = 1,
                 opt_batch: int = 1, max_batch: int = 4):

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print(f"Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parse failed")

    print(f"Network: {network.num_inputs} inputs, {network.num_outputs} outputs, {network.num_layers} layers")

    # Build config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("WARNING: FP16 not natively supported on this GPU, proceeding anyway")
        config.set_flag(trt.BuilderFlag.FP16)
        print("Enabled FP16 precision")
    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            print("WARNING: INT8 not natively supported on this GPU")
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # fallback layers
        print("Enabled INT8 precision (with FP16 fallback)")

    # Dynamic shapes via optimization profile
    profile = builder.create_optimization_profile()
    # image input: [batch, 3, 1024, 1024]
    profile.set_shape(
        "image",
        min=(min_batch, 3, 1024, 1024),
        opt=(opt_batch, 3, 1024, 1024),
        max=(max_batch, 3, 1024, 1024),
    )
    config.add_optimization_profile(profile)

    # Build
    print(f"Building TensorRT engine (this may take several minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    # Save
    out = Path(engine_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(serialized)

    size_mb = out.stat().st_size / 1e6
    print(f"Engine saved: {out} ({size_mb:.1f} MB)")
    print(f"TensorRT version: {trt.__version__}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--output", default="engines/sam2_image_encoder.engine")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    parser.add_argument("--workspace-mb", type=int, default=4096)
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=4)
    args = parser.parse_args()

    build_engine(args.onnx, args.output, args.precision, args.workspace_mb,
                 args.min_batch, args.opt_batch, args.max_batch)
