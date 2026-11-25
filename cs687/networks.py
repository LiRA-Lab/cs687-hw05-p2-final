"""
The networks of the two clusters and of the digits, as noise predictors and as velocities.

The Lecture 3 notes treat the network as a function that takes a noisy image
and a level and returns a predicted noise of the same shape as the image. One
network serves every level, so the level enters it as a vector: the sinusoidal
position vector of Lecture 1, with the level in the place of the slot.

For the two clusters, a small network of a few linear layers that reads the
point and the vector of the level side by side is enough. For the digits the
network is the small U-Net that made the figures of the Lecture 3 notes.

The Lecture 5 notes train the same two networks by flow matching. The network
then takes a point and a time between 0 and 1 and returns a velocity, and the
time enters as the sinusoidal vector of 1000 t, which gives it the range of
frequencies it had for the levels. ToyVelocity and VelocityUNet are the two
networks with that one change. This module is supplied complete.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as nnf


def level_vector(t: torch.Tensor, width: int) -> torch.Tensor:
    """The sinusoidal vector of each level: sines in the first half, cosines in the second.

    Pair i holds the sine and the cosine of t times 10000 to the power -i / (width / 2),
    so the frequencies fall geometrically from one pair to the next.

    Args:
        t: shape (B,), the levels.
        width: the length of the vector, an even number.

    Returns:
        A float tensor of shape (B, width).
    """
    half = width // 2
    frequencies = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    angles = t.float()[:, None] * frequencies[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class ToyNoisePredictor(nn.Module):
    """The network of the two clusters: the point and its level vector side by side,
    then three linear layers with SiLU between them."""

    def __init__(self, level_width: int = 64, hidden: int = 128):
        super().__init__()
        self.level_width = level_width
        self.net = nn.Sequential(
            nn.Linear(2 + level_width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: shape (B, 2), the noisy points; t: shape (B,), their levels. Returns shape (B, 2)."""
        return self.net(torch.cat([x, level_vector(t, self.level_width)], dim=1))


class _Block(nn.Module):
    """Two convolutions with a residual connection, told the noise level."""

    def __init__(self, c_in: int, c_out: int, t_width: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.level = nn.Linear(t_width, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, t_vec: torch.Tensor) -> torch.Tensor:
        h = self.conv1(nnf.silu(self.norm1(x)))
        h = h + self.level(t_vec)[:, :, None, None]
        h = self.conv2(nnf.silu(self.norm2(h)))
        return h + self.skip(x)


class SmallUNet(nn.Module):
    """The small U-Net of the Lecture 3 figures, for 28 by 28 digits.

    It reduces the image to grids of 14 by 14 and 7 by 7 and builds it back up,
    with connections that carry the detail at each resolution from the way down
    to the way up, so its output has the shape of its input.
    """

    def __init__(self, c: int = 64, t_width: int = 256):
        super().__init__()
        self.t_mlp = nn.Sequential(nn.Linear(128, t_width), nn.SiLU(), nn.Linear(t_width, t_width))
        self.inp = nn.Conv2d(1, c, 3, padding=1)
        self.enc1 = _Block(c, c, t_width)
        self.down1 = nn.Conv2d(c, c, 3, stride=2, padding=1)                 # 28 -> 14
        self.enc2 = _Block(c, 2 * c, t_width)
        self.down2 = nn.Conv2d(2 * c, 2 * c, 3, stride=2, padding=1)         # 14 -> 7
        self.mid1 = _Block(2 * c, 2 * c, t_width)
        self.mid2 = _Block(2 * c, 2 * c, t_width)
        self.up2 = nn.ConvTranspose2d(2 * c, 2 * c, 4, stride=2, padding=1)  # 7 -> 14
        self.dec2 = _Block(4 * c, 2 * c, t_width)
        self.up1 = nn.ConvTranspose2d(2 * c, c, 4, stride=2, padding=1)      # 14 -> 28
        self.dec1 = _Block(2 * c, c, t_width)
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: shape (B, 1, 28, 28), the noisy digits; t: shape (B,), their levels."""
        tv = self.t_mlp(level_vector(t, 128))
        h1 = self.enc1(self.inp(x), tv)
        h2 = self.enc2(self.down1(h1), tv)
        m = self.mid2(self.mid1(self.down2(h2), tv), tv)
        d2 = self.dec2(torch.cat([self.up2(m), h2], dim=1), tv)
        d1 = self.dec1(torch.cat([self.up1(d2), h1], dim=1), tv)
        return self.out(d1)


class ToyVelocity(ToyNoisePredictor):
    """The network of the two clusters as a velocity: the same layers, with the time t between 0 and 1
    entering as the sinusoidal vector of 1000 t."""

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: shape (B, 2), the points; t: shape (B,), their times, floats from 0 to 1. Returns shape (B, 2)."""
        return super().forward(x, 1000.0 * t)


class VelocityUNet(SmallUNet):
    """The small U-Net as a velocity: the same layers, with the time t between 0 and 1 entering as the
    sinusoidal vector of 1000 t."""

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: shape (B, 1, 28, 28), the points; t: shape (B,), their times, floats from 0 to 1."""
        return super().forward(x, 1000.0 * t)
