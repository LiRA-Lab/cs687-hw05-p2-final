"""
Flow matching: the loss, the Euler step, and DDIM seen on the straight line.

This module follows the Lecture 5 notes. Time runs as in the notes: t = 0 is
the data and t = 1 is the noise, and the straight line of a pair is
x_t = (1 - t) x0 + t eps. Four functions are left for you to write: the loss
of flow matching, one Euler step, the estimates of the clean image and of the
noise from a velocity, and the time of the line that has the noise scale of a
level of the schedule. The notebook attaches your versions to this module in
place of the stubs, so the supplied training loop and sampling loops call
your code.
"""

from __future__ import annotations

from typing import Callable

import torch

from cs687.schedule import Schedule


# ----------------------------------------------------------------------
# Task 2: the loss of flow matching
# ----------------------------------------------------------------------
def flow_matching_loss(
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """The squared error between the velocity of each line and the network's velocity, on one batch.

    The notes' algorithm "train by flow matching", steps 2 to 5. The draws are
    made in this order, so that a seeded run gives the same numbers as the
    course's:

    1. One time per row, uniformly between 0 and 1, with torch.rand,
       generator=generator and device=x0.device.
    2. Then one standard Gaussian vector of the shape of x0, with torch.randn,
       the same generator and device=x0.device.

    Form each point on its line, call the model as model(x_t, t) with the
    times as a float tensor of shape (B,), and compare its output with the
    velocity of the line.

    Args:
        model: a velocity network called as model(x_t, t), returning a tensor of x_t's shape.
        x0: shape (B, ...), a batch of clean data.
        generator: the random generator, on the device of x0.

    Returns:
        The mean of the squared differences over every number of the batch, a
        tensor with one value that keeps its gradient. A model that returns
        zeros scores about 3.25 on the two clusters.

    The times have shape (B,). Give them one trailing axis of size one for
    every axis of x0 after the first, as Task 1 of Homework 4 did with the
    numbers of the schedule, so that each multiplies its own row.
    """
    raise NotImplementedError("Task 2: implement flow_matching_loss")


# ----------------------------------------------------------------------
# Task 3: one Euler step
# ----------------------------------------------------------------------
def euler_step(
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_t: torch.Tensor,
    t: float,
    s: float,
) -> torch.Tensor:
    """One Euler step of the sampler of flow matching, from time t down to the earlier time s.

    The notes' section "Sampling with Euler steps": the step keeps the
    network's velocity at the start of the step for the whole step, and
    sampling runs from the noise at time 1 towards the data at time 0.

    Call the model once, as model(x_t, times), with times a float tensor of
    B copies of t on the device of x_t.

    Args:
        model: a velocity network called as model(x_t, times).
        x_t: shape (B, ...), a batch at time t, every row at the same time.
        t: a Python float, the current time, at most 1.
        s: a Python float, the next time, from 0 to less than t.

    Returns:
        The batch at time s, of the shape of x_t.
    """
    raise NotImplementedError("Task 3: implement euler_step")


# ----------------------------------------------------------------------
# Task 4: the clean image and the noise from a velocity
# ----------------------------------------------------------------------
def estimates_from_velocity(
    v: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The estimates of the clean image and of the noise that a velocity gives at a point on the line.

    The notes' subsection "The update of the previous lecture on a straight
    line": the point on a line can be written from either end, and putting the
    network's velocity in the place of the velocity of the line gives the two
    estimates.

    Args:
        v: shape (B, ...), the velocities at the points.
        x_t: shape (B, ...), the points.
        t: shape (B,), a float tensor of their times.

    Returns:
        The tuple (x0_hat, eps_hat), in that order, each of the shape of x_t. Leave the
        tensor x_t unchanged.

    Give the times one trailing axis of size one for every axis of x_t after
    the first, so that each multiplies its own row.
    """
    raise NotImplementedError("Task 4: implement estimates_from_velocity")


# ----------------------------------------------------------------------
# Task 5: the time of the line with the noise scale of a level
# ----------------------------------------------------------------------
def line_time(levels: torch.Tensor, schedule: Schedule) -> torch.Tensor:
    """The time of the straight line at which the noise scale equals that of each level of the schedule.

    The notes' subsection "Every noise schedule as a path": every path
    a x0 + b eps, divided by the weight a of the clean image, is the rescaled
    image x0 + sigma eps with the noise scale sigma = b / a. Two paths carry the
    same amount of noise where their noise scales are equal. For each level,
    find the noise scale of the noising process at that level, and return the
    time of the line with the same noise scale.

    Args:
        levels: a long tensor of levels from 0 to T, of any shape.
        schedule: the schedule.

    Returns:
        A float tensor of times between 0 and 1, of the shape of levels. Level 0
        has the noise scale 0 and the time 0.
    """
    raise NotImplementedError("Task 5: implement line_time")
