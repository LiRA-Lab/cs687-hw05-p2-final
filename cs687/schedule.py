"""
The noise schedule of DDPM.

The Lecture 3 notes add noise in T = 1000 steps. Step t scales the current
image down by the square root of alpha_t = 1 - beta_t and adds Gaussian noise of
variance beta_t, and beta_t rises in equal increments from 0.0001 at the first
step to 0.02 at the last. The product alpha_bar_t = alpha_1 alpha_2 ... alpha_t
is the share of the variance of the clean image that survives to level t.

This module is supplied complete. Its lists are indexed by the level, as the
notes count levels, so `schedule.alpha_bar[t]` is alpha_bar_t for t from 1 to
T. Index 0 holds the values of the clean image: alpha_bar_0 = 1, the product
of no factors, and beta_0 = 0, which no step uses.
"""

from __future__ import annotations

import torch


class Schedule:
    """beta_t, alpha_t and alpha_bar_t of the linear schedule, for t from 0 to T.

    Attributes:
        T: the number of noise levels.
        beta, alpha, alpha_bar: float32 tensors of shape (T + 1,), indexed by
            the level, on the device the schedule was moved to.
    """

    def __init__(self, T: int = 1000, beta_first: float = 1e-4, beta_last: float = 0.02):
        self.T = T
        # The products are formed in double precision and stored in single,
        # the precision of the networks; double() gives them back in double.
        beta = torch.cat([
            torch.zeros(1, dtype=torch.float64),
            torch.linspace(beta_first, beta_last, T, dtype=torch.float64),
        ])
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self._double = (beta, alpha, alpha_bar)
        self.beta = beta.float()
        self.alpha = alpha.float()
        self.alpha_bar = alpha_bar.float()

    @property
    def device(self) -> torch.device:
        return self.beta.device

    def to(self, device: torch.device | str) -> "Schedule":
        """A copy of the schedule whose tensors live on the given device."""
        moved = Schedule.__new__(Schedule)
        moved.T = self.T
        moved._double = self._double
        moved.beta = self.beta.to(device)
        moved.alpha = self.alpha.to(device)
        moved.alpha_bar = self.alpha_bar.to(device)
        return moved

    def double(self) -> "Schedule":
        """A copy of the schedule in double precision, on the processor.

        Single precision keeps about seven significant digits, so a difference
        of two numbers close to one, such as 1 - alpha_bar_1 = 0.0001, keeps
        only three or four of them. Calculations by hand with the schedule use
        this copy; the networks and the tasks use the schedule itself.
        """
        copy = Schedule.__new__(Schedule)
        copy.T = self.T
        copy._double = self._double
        copy.beta, copy.alpha, copy.alpha_bar = self._double
        return copy
