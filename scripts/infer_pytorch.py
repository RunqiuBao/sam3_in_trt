import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import logging
import time
import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results


def ConfigLogging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True,
    )


def CheckIfAmpereGPU():
    if not torch.cuda.is_available():
        return False
    # Get the compute capability (major, minor)
    major = torch.cuda.get_device_capability()[0]
    return major == 8


def _main(
    image_path: Path,
    text_prompt: str = "",
    box_prompt: list | None = None,
):
    logging.info("Starting SAM3 PyTorch model inference...")
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    if CheckIfAmpereGPU():
        # turn on tfloat32 for Ampere GPUs, this accelerate float32 operation by 8x.
        # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # use bfloat16 for the entire process
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    
    # construct the model
    bpe_path = f"{sam3_root}/sam3/assets/bpe_simple_vocab_16e6.txt.gz"  # text tokenizer for CLIP
    checkpoint_path = f"{sam3_root}/sam3/assets/sam3.pt"
    starttime = time.time()
    model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path=checkpoint_path)
    logging.info(f"Model loaded in {time.time() - starttime} sec.")

    # load image
    image = Image.open(image_path)
    width, height = image.size[:2]
    logging.info(f"Image loaded: {image_path}, size: {width}x{height}")

    processor = Sam3Processor(model, confidence_threshold=0.5)
    starttime = time.time()
    inference_state = processor.set_image(image)
    logging.info(f"image backbone finished in {time.time() - starttime} sec.")
    processor.reset_all_prompts(inference_state)
    if text_prompt != "":
        starttime = time.time()
        inference_state = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        logging.info(f"text prompt processed in {time.time() - starttime} sec.")
    if box_prompt is not None:
        width, height = image.size
        box_input_xywh = torch.tensor(box_prompt).view(-1, 4)
        box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)
        norm_box_cxcywh = normalize_bbox(box_input_cxcywh, width, height).flatten().tolist()
        starttime = time.time()
        inference_state = processor.add_geometric_prompt(
            state=inference_state,
            box=norm_box_cxcywh,
            label=True,
        )
        logging.info(f"box prompt processed in {time.time() - starttime} sec.")

    img0 = Image.open(image_path)
    render_image = plot_results(img0, inference_state, return_image=True)
    render_image.save("output.png")
    logging.info("Result saved to output.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM3 pytorch model.")
    parser.add_argument(
        "-i", "--image_path", 
        type=Path, 
        help="Path to the input image file"
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        default="",
        help="text prompt for the detection target",
    )
    parser.add_argument(
        "--box-prompt",
        type=float,
        nargs=4,
        default=None,
        help="box prompt in [topleft_x, topleft_y, width, height] format",
    )

    args = parser.parse_args()
    ConfigLogging()
    _main(
        args.image_path,
        text_prompt=args.text_prompt,
        box_prompt=args.box_prompt,
    )
