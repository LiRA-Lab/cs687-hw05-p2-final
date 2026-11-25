"""
The MNIST training digits, and the course's trained noise predictor of the digits.

The 60,000 training digits of MNIST, 28 by 28 grey pixels each, ship in
data/train-images-idx3-ubyte.gz, the original file of the collection. The Lecture 3
notes scale the numbers of an image to lie between -1 and 1.

weights/digit_noise_predictor.pt holds the parameters of the course's small
U-Net trained on these digits by the loss of DDPM, the run of section 6 of
Homework 4: 10,000 steps in batches of 128 with Adam at the learning rate
0.0002. This module is supplied complete.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import torch

DEFAULT_PATH = Path("data") / "train-images-idx3-ubyte.gz"
NOISE_PREDICTOR_PATH = Path("weights") / "digit_noise_predictor.pt"


def load_training_digits(path: str | Path = DEFAULT_PATH) -> torch.Tensor:
    """The training digits as a float tensor of shape (60000, 1, 28, 28), scaled to [-1, 1]."""
    with gzip.open(path, "rb") as file:
        raw = file.read()
    magic, count, rows, columns = np.frombuffer(raw[:16], dtype=">i4")
    if magic != 2051:
        raise ValueError(f"{path} is not an MNIST image file.")
    pixels = np.frombuffer(raw[16:], dtype=np.uint8).reshape(count, 1, rows, columns)
    return torch.from_numpy(pixels.astype(np.float32) / 127.5 - 1.0)


def load_course_digit_network(device: torch.device | str = "cpu", path: str | Path = NOISE_PREDICTOR_PATH):
    """The course's trained noise predictor of the digits, a SmallUNet on the given device, ready to sample."""
    from cs687.networks import SmallUNet

    model = SmallUNet().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
