"""
The three samplers of Project 2, as loops over a list of levels or times, and their evenly spaced lists.

Each loop starts from the points it is given, so that several samplers can
start from the same draws of the noise, runs under torch.no_grad(), calls the
module's version of its step, and counts the calls of the network. A list of
n + 1 levels or times gives n steps and n calls. This module is supplied
complete.

- sample_ancestral: the larger ancestral steps of the Lecture 4 notes' "The
  cost of a thousand steps", over a falling list of levels; with every level
  from T down to 0 it is the ancestral sampler of Homework 4.
- sample_ddim: the Lecture 4 notes' algorithm "sample with deterministic
  DDIM", over a falling list of levels, with your Task 1.
- sample_euler: the Lecture 5 notes' algorithm "sample by flow matching",
  over a falling list of times, with your Task 3.
"""

from __future__ import annotations

from typing import Callable

import torch

from cs687.schedule import Schedule


class CountedCalls:
    """A network wrapped so that its calls are counted."""

    def __init__(self, model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        self.model = model
        self.calls = 0

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.model(x, t)


def even_levels(calls: int, T: int = 1000) -> list[int]:
    """calls + 1 levels evenly spaced from T down to 0, each rounded to the nearest whole level."""
    return [int(level) for level in torch.linspace(T, 0, calls + 1, dtype=torch.float64).round().tolist()]


def even_times(calls: int) -> list[float]:
    """calls + 1 times evenly spaced from 1 down to 0."""
    return [1 - i / calls for i in range(calls + 1)]


def _falling(values: list) -> None:
    if len(values) < 2 or any(b >= a for a, b in zip(values, values[1:])):
        raise ValueError("A sampler needs a falling list of at least two levels or times.")


@torch.no_grad()
def sample_ancestral(
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    schedule: Schedule,
    x: torch.Tensor,
    levels: list[int],
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    """Larger ancestral steps from the points x at levels[0] down the list. Returns the samples and the calls."""
    from cs687 import diffusion

    _falling(levels)
    counted = CountedCalls(model)
    for t, s in zip(levels[:-1], levels[1:]):
        x = diffusion.ancestral_jump(counted, x, int(t), int(s), schedule, generator)
    return x, counted.calls


@torch.no_grad()
def sample_ddim(
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    schedule: Schedule,
    x: torch.Tensor,
    levels: list[int],
) -> tuple[torch.Tensor, int]:
    """DDIM steps from the points x at levels[0] down the list. Returns the samples and the calls."""
    from cs687 import ddim

    _falling(levels)
    counted = CountedCalls(model)
    for t, s in zip(levels[:-1], levels[1:]):
        x = ddim.ddim_step(counted, x, int(t), int(s), schedule)
    return x, counted.calls


@torch.no_grad()
def sample_euler(
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    times: list[float],
) -> tuple[torch.Tensor, int]:
    """Euler steps from the points x at times[0] down the list. Returns the samples and the calls."""
    from cs687 import flow

    _falling(times)
    counted = CountedCalls(model)
    for t, s in zip(times[:-1], times[1:]):
        x = flow.euler_step(counted, x, float(t), float(s))
    return x, counted.calls
