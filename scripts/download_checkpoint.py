#!/usr/bin/env python3
"""Download SAM 2 model checkpoints from Meta."""

import argparse
import os
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm

CHECKPOINTS = {
    "sam2.1_hiera_tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_small": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    "sam2.1_hiera_base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_large": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}


class DownloadProgress(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(model: str, output_dir: str):
    if model not in CHECKPOINTS:
        print(f"Unknown model: {model}")
        print(f"Available: {', '.join(CHECKPOINTS.keys())}")
        return

    url = CHECKPOINTS[model]
    out_path = Path(output_dir) / f"{model}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"Already exists: {out_path}")
        return

    print(f"Downloading {model} → {out_path}")
    with DownloadProgress(unit="B", unit_scale=True, miniters=1, desc=model) as t:
        urlretrieve(url, out_path, reporthook=t.update_to)

    print(f"Done: {out_path} ({out_path.stat().st_size / 1e9:.1f} GB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SAM 2 checkpoints")
    parser.add_argument("--model", default="sam2.1_hiera_large", choices=CHECKPOINTS.keys())
    parser.add_argument("--output-dir", default="checkpoints")
    args = parser.parse_args()
    download(args.model, args.output_dir)
