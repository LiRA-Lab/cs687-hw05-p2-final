"""
Denoising diffusion as Homework 4 built it, the larger ancestral step, and the training loop.

This module is supplied complete. The first two functions are the course's
answers to Tasks 1 and 2 of Homework 4: the noisy image at any level and the
loss of DDPM. The third is the step of the Lecture 4 notes' "The cost of a
thousand steps", which jumps from one level to any lower level; between two
neighbouring levels it is the step of Task 3 of Homework 4. The training loop trains a noise predictor with the loss of
DDPM or a velocity network with the loss of flow matching, your Task 2.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

import torch

from cs687.schedule import Schedule


def _per_row(values: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Give one number per row trailing axes of size one, so that it multiplies its own row of `like`."""
    return values.reshape(-1, *([1] * (like.dim() - 1)))


def noisy_at_level(x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, schedule: Schedule) -> torch.Tensor:
    """The noisy version of each clean row at its own level: sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps."""
    alpha_bar = _per_row(schedule.alpha_bar[t], x0)
    return alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * eps


def noise_prediction_loss(
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    schedule: Schedule,
    generator: torch.Generator,
) -> torch.Tensor:
    """The loss of DDPM on one batch: a level per row from 1 to T, then the noise, then the squared error."""
    t = torch.randint(1, schedule.T + 1, (x0.shape[0],), generator=generator, device=x0.device)
    eps = torch.randn(x0.shape, generator=generator, device=x0.device)
    return ((model(noisy_at_level(x0, t, eps, schedule), t) - eps) ** 2).mean()


def ancestral_jump(
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_t: torch.Tensor,
    t: int,
    s: int,
    schedule: Schedule,
    generator: torch.Generator,
) -> torch.Tensor:
    """A step of ancestral sampling from level t to any lower level s, with the estimate of the clean image.

    The Lecture 4 notes' "The cost of a thousand steps": the step back given the
    clean image is a Gaussian between any two levels, because it is a product of
    two Gaussian densities, and the sampler puts the estimate of the clean image
    in the place of the clean image. With r = alpha_bar_t / alpha_bar_s, the
    Gaussian has the mean

        sqrt(alpha_bar_s) (1 - r) / (1 - alpha_bar_t) x0_hat + sqrt(r) (1 - alpha_bar_s) / (1 - alpha_bar_t) x_t

    and the variance (1 - alpha_bar_s)(1 - r) / (1 - alpha_bar_t). With s = t - 1
    these are the weights gamma_t and delta_t and the variance beta_tilde_t of
    the Lecture 3 notes, and the step is the step of Homework 4. The step to
    level 0 adds no noise and returns the estimate of the clean image.

    The network is called once, with B copies of t; if s > 0, one standard
    Gaussian vector of the shape of x_t is drawn from the generator.
    """
    levels = torch.full((x_t.shape[0],), t, dtype=torch.long, device=x_t.device)
    eps_hat = model(x_t, levels)
    alpha_bar_t, alpha_bar_s = schedule.alpha_bar[t], schedule.alpha_bar[s]
    x0_hat = (x_t - (1 - alpha_bar_t).sqrt() * eps_hat) / alpha_bar_t.sqrt()
    r = alpha_bar_t / alpha_bar_s
    mean = (alpha_bar_s.sqrt() * (1 - r) / (1 - alpha_bar_t) * x0_hat
            + r.sqrt() * (1 - alpha_bar_s) / (1 - alpha_bar_t) * x_t)
    if s == 0:
        return mean
    variance = (1 - alpha_bar_s) * (1 - r) / (1 - alpha_bar_t)
    return mean + variance.sqrt() * torch.randn(x_t.shape, generator=generator, device=x_t.device)


def train_run(
    model: torch.nn.Module,
    draw_batch: Callable[[int, torch.Generator], torch.Tensor],
    objective: str,
    steps: int,
    batch_size: int,
    learning_rate: float,
    generator: torch.Generator,
    schedule: Schedule | None = None,
    checkpoint_path: str | Path | None = None,
    print_every: int = 500,
    shortened_by_grader: bool = False,
) -> list[float]:
    """Train a network with Adam, by the loss of DDPM or by the loss of flow matching.

    Each step draws a batch with draw_batch(batch_size, generator), computes the
    loss, and takes one step of the optimizer. With objective "noise" the loss
    is noise_prediction_loss of this module, which needs the schedule; with
    objective "velocity" it is flow_matching_loss of cs687.flow, your Task 2,
    the notes' algorithm "train by flow matching". It returns the loss of every
    step.

    With a checkpoint path, the trained parameters and the curve are saved at
    the end, and a later call with the same objective and number of steps
    restores them in place of training again, so that a rerun of the notebook
    does not repeat a long run. When shortened_by_grader is true, the
    environment variable CS687_TRAINING_STEPS, which the grader sets, replaces
    the number of steps.
    """
    if objective not in ("noise", "velocity"):
        raise ValueError('objective must be "noise" or "velocity".')
    if objective == "noise" and schedule is None:
        raise ValueError("The loss of DDPM needs the schedule.")
    if shortened_by_grader and "CS687_TRAINING_STEPS" in os.environ:
        steps = int(os.environ["CS687_TRAINING_STEPS"])
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_file():
            saved = torch.load(checkpoint_path, map_location=next(model.parameters()).device)
            if saved["steps"] == steps and saved.get("objective") == objective:
                model.load_state_dict(saved["model"])
                model.eval()
                print(f"restored the finished run of {steps} steps from {checkpoint_path}, "
                      f"which took {saved['seconds']:.0f} s")
                return saved["curve"]

    from cs687 import diffusion, flow

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    curve = []
    started = time.time()
    for step in range(1, steps + 1):
        x0 = draw_batch(batch_size, generator)
        if objective == "noise":
            loss = diffusion.noise_prediction_loss(model, x0, schedule, generator)
        else:
            loss = flow.flow_matching_loss(model, x0, generator)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        curve.append(loss.item())
        if step % print_every == 0 or step == steps:
            recent = sum(curve[-print_every:]) / len(curve[-print_every:])
            print(f"step {step:6d}   mean loss of the last {min(print_every, step)} steps {recent:.4f}   "
                  f"{time.time() - started:6.0f} s")
    model.eval()
    seconds = time.time() - started
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "curve": curve, "steps": steps, "objective": objective,
                    "seconds": seconds}, checkpoint_path)
    return curve
