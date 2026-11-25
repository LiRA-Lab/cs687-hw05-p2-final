"""
The two clusters: the running example of the diffusion lectures.

Points in two dimensions, half from each of two Gaussians with a standard
deviation of 0.5 in both coordinates, centred at (-2, 0) and (2, 0). Their
density can be written down at every noise level and at every time of the
straight line, so the two things a trained network only estimates can be
computed exactly here: the best possible noise prediction of Lecture 3, the
conditional mean of the noise given the noisy point, and the mean velocity of
the straight lines of Lecture 5, the conditional mean of the velocity
eps - x0 given the point on the line. This module is supplied complete.

The three statistics at the end are the measure of quality of Project 2. Both
checkpoints compute them with this function, unchanged.
"""

from __future__ import annotations

import torch

from cs687.schedule import Schedule

CENTRES = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])
SD = 0.5


def sample_clusters(n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """n points of the two clusters, shape (n, 2), each cluster chosen with probability one half."""
    which = torch.randint(0, 2, (n,), generator=generator)
    return CENTRES[which] + SD * torch.randn(n, 2, generator=generator)


def _levels(t: torch.Tensor | int, n: int) -> torch.Tensor:
    if isinstance(t, int):
        return torch.full((n,), t, dtype=torch.long)
    return t.to("cpu").long()


def _clean_estimate(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """E[x0 | a x0 + b eps = x] for the two clusters, in double precision; a and b have shape (B, 1)."""
    variance = a ** 2 * SD ** 2 + b ** 2
    centres = CENTRES.double()
    # The probability that each cluster produced the point, from the two densities of the noisy point.
    logits = torch.stack([-((x - a * c) ** 2).sum(1) / (2 * variance[:, 0]) for c in centres], dim=1)
    weights = torch.softmax(logits, dim=1)
    # The conditional mean of the clean point, within each cluster and then averaged.
    gain = a * SD ** 2 / variance
    return sum(weights[:, k:k + 1] * (centres[k] + gain * (x - a * centres[k])) for k in range(2))


class ExactNoisePredictor:
    """The best possible noise prediction for the two clusters, used as a network.

    Called as model(x, t) like a trained noise predictor, with t the levels,
    it returns the conditional mean of the noise given the noisy point,
    E[eps | x_t], computed from the density of the noisy clusters. A
    regression on the noise can at best learn this function.
    """

    def __init__(self, schedule: Schedule):
        self.schedule = schedule

    def __call__(self, x: torch.Tensor, t: torch.Tensor | int) -> torch.Tensor:
        device, dtype = x.device, x.dtype
        x = x.detach().to("cpu").double()
        alpha_bar = self.schedule.alpha_bar.to("cpu").double()[_levels(t, len(x))][:, None]
        clean = _clean_estimate(x, alpha_bar.sqrt(), (1 - alpha_bar).sqrt())
        noise = (x - alpha_bar.sqrt() * clean) / (1 - alpha_bar).sqrt()
        return noise.to(device=device, dtype=dtype)


class ExactVelocity:
    """The mean velocity of the straight lines for the two clusters, used as a network.

    Called as model(x, t) like a trained velocity network, with t the times
    between 0 and 1, it returns the conditional mean of the velocity of the
    lines through the point, E[eps - x0 | x_t = x], which the Lecture 5 notes
    write as (x - E[x0 | x_t = x]) / t. A regression on the velocity can at
    best learn this function. At time 1 the point says nothing about the clean
    point, whose estimate is the mean of the data, the origin.
    """

    def __call__(self, x: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
        device, dtype = x.device, x.dtype
        x = x.detach().to("cpu").double()
        if not torch.is_tensor(t):
            t = torch.full((len(x),), float(t))
        t = t.detach().to("cpu").double().reshape(-1, 1).clamp_min(1e-9)
        clean = _clean_estimate(x, 1 - t, t)
        return ((x - clean) / t).to(device=device, dtype=dtype)


def cluster_statistics(x: torch.Tensor) -> dict[str, float]:
    """The three statistics of a set of samples of the two clusters.

    share_right: the share of the samples whose first coordinate is positive, 0.5 for the data.
    mean_distance: the average distance of the first coordinate from zero, about 2 for the data.
    spread_within: the standard deviation of the first coordinate around the mean of its own
        half, pooled over the two halves, about 0.5 for the data.
    """
    first = x[:, 0].detach().to("cpu").double()
    right = first > 0
    halves = [half for half in (first[right], first[~right]) if len(half)]
    deviations = torch.cat([half - half.mean() for half in halves])
    return {
        "share_right": right.double().mean().item(),
        "mean_distance": first.abs().mean().item(),
        "spread_within": deviations.pow(2).mean().sqrt().item(),
    }
