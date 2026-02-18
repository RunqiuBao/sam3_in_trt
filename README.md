# SAM2 → TensorRT Conversion Pipeline

Convert Meta's [Segment Anything Model 2 (SAM 2)](https://github.com/facebookresearch/sam2) to TensorRT for optimized inference on NVIDIA GPUs.

## Pipeline

1. **Download** SAM 2 checkpoints from Meta
2. **Export** PyTorch model → ONNX
3. **Convert** ONNX → TensorRT engine
4. **Run** inference with TensorRT

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.x
- TensorRT 10.x
- PyTorch 2.x with CUDA support

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download SAM 2 checkpoint
python scripts/download_checkpoint.py --model sam2.1_hiera_large

# 3. Export to ONNX
python scripts/export_onnx.py \
    --checkpoint checkpoints/sam2.1_hiera_large.pt \
    --output models/sam2_image_encoder.onnx

# 4. Build TensorRT engine
python scripts/build_trt_engine.py \
    --onnx models/sam2_image_encoder.onnx \
    --output engines/sam2_image_encoder.engine \
    --fp16

# 5. Run inference
python scripts/infer_trt.py \
    --engine engines/sam2_image_encoder.engine \
    --image test_images/example.jpg \
    --point 500,375
```

## Project Structure

```
sam3_in_trt/
├── README.md
├── requirements.txt
├── configs/
│   └── sam2_trt.yaml          # Model & engine config
├── scripts/
│   ├── download_checkpoint.py # Download SAM 2 weights
│   ├── export_onnx.py         # PyTorch → ONNX export
│   ├── build_trt_engine.py    # ONNX → TensorRT conversion
│   └── infer_trt.py           # TensorRT inference
├── checkpoints/               # SAM 2 .pt files (gitignored)
├── models/                    # ONNX models (gitignored)
├── engines/                   # TensorRT engines (gitignored)
└── test_images/               # Sample images
```

## Notes

- The image encoder is the main bottleneck — that's what we convert to TensorRT.
- Prompt encoder and mask decoder are lightweight and can stay in PyTorch.
- FP16 mode gives ~2x speedup with minimal accuracy loss.
