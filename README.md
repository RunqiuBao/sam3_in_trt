# SAM 3 → TensorRT Conversion Pipeline

Convert Meta's [Segment Anything Model 3 (SAM 3)](https://github.com/facebookresearch/sam3) to TensorRT for optimized inference on NVIDIA GPUs.

SAM 3 is a unified foundation model for promptable segmentation using text or visual prompts. It has 848M parameters with a DETR-based detector and SAM 2-style tracker sharing a vision encoder.

## Pipeline

1. **Download** SAM 3 checkpoints from HuggingFace (requires auth)
2. **Export** PyTorch model → ONNX (vision encoder + detector)
3. **Convert** ONNX → TensorRT engine
4. **Run** inference with TensorRT

## Requirements

- Python 3.12+
- PyTorch 2.7+ with CUDA support
- CUDA 12.6+
- TensorRT 10.x
- HuggingFace account with SAM 3 checkpoint access

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Authenticate with HuggingFace
huggingface-cli login

# 3. Export vision encoder to ONNX
python scripts/export_onnx.py \
    --component vision_encoder \
    --output models/sam3_vision_encoder.onnx

# 4. Export detector to ONNX
python scripts/export_onnx.py \
    --component detector \
    --output models/sam3_detector.onnx

# 5. Build TensorRT engines
python scripts/build_trt_engine.py \
    --onnx models/sam3_vision_encoder.onnx \
    --output engines/sam3_vision_encoder.engine \
    --fp16

python scripts/build_trt_engine.py \
    --onnx models/sam3_detector.onnx \
    --output engines/sam3_detector.engine \
    --fp16

# 6. Run inference
python scripts/infer_trt.py \
    --encoder-engine engines/sam3_vision_encoder.engine \
    --detector-engine engines/sam3_detector.engine \
    --image test_images/example.jpg \
    --text "dog"
```

## Project Structure

```
sam3_in_trt/
├── README.md
├── requirements.txt
├── configs/
│   └── sam3_trt.yaml          # Model & engine config
├── scripts/
│   ├── export_onnx.py         # PyTorch → ONNX export
│   ├── build_trt_engine.py    # ONNX → TensorRT conversion
│   └── infer_trt.py           # TensorRT inference
├── models/                    # ONNX models (gitignored)
├── engines/                   # TensorRT engines (gitignored)
└── test_images/               # Sample images
```

## Architecture Notes

SAM 3 has three main components:
- **Vision encoder** — shared backbone (heavy, best TRT target)
- **Detector** — DETR-based, conditioned on text/geometry/exemplars (good TRT target)
- **Tracker** — SAM 2-style encoder-decoder for video (stateful, harder to convert)

Strategy: convert vision encoder + detector to TRT. Keep tracker in PyTorch for video use cases (stateful memory makes TRT tricky).

## Notes

- Checkpoints require HuggingFace access: https://huggingface.co/facebook/sam3
- FP16 gives ~2x speedup with minimal accuracy loss on the vision encoder
- The presence token (new in SAM 3) helps discriminate similar text prompts
