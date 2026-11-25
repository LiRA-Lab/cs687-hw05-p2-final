"""
DDIM: the noise predictor of DDPM, sampled in a few large deterministic steps.

This module follows the Lecture 4 notes' subsection "Large steps from the
estimate of the clean image". One function is left for you to write, the DDIM
step. The notebook attaches your version to this module in place of the stub,
so the supplied loop sample_ddim in cs687/samplers.py calls your code.
"""

from __future__ import annotations

from typing import Callable

import torch

from cs687.schedule import Schedule


# ----------------------------------------------------------------------
# Task 1: one DDIM step
# ----------------------------------------------------------------------
def ddim_step(
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_t: torch.Tensor,
    t: int,
    s: int,
    schedule: Schedule,
) -> torch.Tensor:
    """One step of deterministic DDIM, from level t to a lower level s.

    The notes' algorithm "sample with deterministic DDIM", step 2: ask the
    network for the predicted noise at the current level, form the estimate of
    the clean image from it, and combine the estimate and the same predicted
    noise with the numbers of the schedule at the lower level. The two levels
    need not be neighbours.

    Two details are fixed:

    1. Call the model once, as model(x_t, levels), with levels a long tensor
       of B copies of t on the device of x_t.
    2. Draw no noise. The step is deterministic.

    Leave the tensor x_t unchanged.

    The schedule stores alpha_bar_0 = 1 at index 0, so the step to level 0
    needs no special case.

    Args:
        model: a noise predictor called as model(x_t, levels).
        x_t: shape (B, ...), a batch at level t, every row at the same level.
        t: a Python integer, the current level, from 1 to T.
        s: a Python integer, the lower level, from 0 to t - 1.
        schedule: the schedule, on the device of x_t.

    Returns:
        The batch at level s, of the shape of x_t.

    The supplied loop sample_ddim runs under torch.no_grad(), so the step
    needs none of its own.
    """
    raise NotImplementedError("Task 1: implement ddim_step")
