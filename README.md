# SAM3 in TensorRT format

Convert Meta's [Segment Anything Model 3 (SAM 3)](https://github.com/facebookresearch/sam3) to TensorRT for optimized inference on NVIDIA GPUs.
- Image predictor inference constructs and finishes in 2 seconds.
```
2026-03-01 00:28:24,471 - INFO - infer_trt.py:153 -   File I/O: 519.8 ms
...
2026-03-01 00:28:26,674 - INFO - infer_trt.py:591 - Saved: IMG_2146_sam3_output.jpg
```

## Installation for inference
(make sure CUDA and TensorRT are installed in the os.)
- `git clone git@github.com:RunqiuBao/sam3_in_trt.git && cd sam3_in_trt/`
- `make installdeps[infer]`
- `make env[infer]`

## Quick Start

```bash
# Download onnx model
cd sam3_in_trt/ && mkdir onnx_models/ && mkdir trt_engines/
make download[onnx]

# Convert onnx model to tensorrt engines
make export[trt]

# inference examples
python3 scripts/infer_trt.py --engine-dir ./trt_engines/ --image ./assets/images/IMG_4456.jpg --text "bunny"

python3 scripts/infer_trt.py --engine-dir ./trt_engines/ --image ./assets/images/IMG_4713.jpg --text "robot"

python3 scripts/infer_trt.py --engine-dir ./trt_engines/ --image ./assets/images/IMG_5078.jpg --text "face"

python3 scripts/infer_trt.py --engine-dir ./trt_engines/ --image ./assets/images/IMG_2146.jpeg --text "female human" --points 622,966 330,780 328,864 700,790 684,826 --point-labels 1 0 0 0 0
```
<table>
<tr>
    <td><img src="assets/readme/IMG_2146_sam3_output.jpg" width="200"/></td>
    <td><img src="assets/readme/IMG_4713_sam3_output.jpg" width="200"/></td>
    <td><img src="assets/readme/IMG_4456_sam3_output.jpg" width="200"/></td>                                                        
    <td><img src="assets/readme/IMG_5078_sam3_output.jpg" width="200"/></td>
</tr>
</table>



## Set up dev environment
- `make builddocker`
- `make rundocker`
- `make execdocker`