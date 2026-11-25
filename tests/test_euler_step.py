"""Tests for Task 3: euler_step.

The notebook runs these for you. Each test reaches the function through the
module, so that the version the notebook attached is the one tested. The last
test runs the supplied Euler loop with the exact mean field of the two
clusters, which cannot be wrong, so a failure there is a failure of the step.
"""

import pytest
import torch

from cs687 import flow, samplers
from cs687.toy import ExactVelocity, cluster_statistics


def test_the_model_is_called_once_with_the_time():
    calls = []

    def recording(x, times):
        calls.append(times)
        return torch.zeros_like(x)

    out = flow.euler_step(recording, torch.zeros(3, 2), 0.7, 0.4)
    assert out.shape == (3, 2), f"The result must have the shape of x_t, (3, 2), not {tuple(out.shape)}."
    assert len(calls) == 1, f"The model should be called once per step, not {len(calls)} times."
    times = calls[0]
    assert torch.is_tensor(times) and times.is_floating_point() and times.shape == (3,), (
        "The model should receive a float tensor with one time per row."
    )
    assert torch.allclose(times, torch.full((3,), 0.7)), (
        f"Every row is at the current time 0.7, but the model received {times.tolist()}."
    )


def test_the_step_moves_against_the_velocity():
    out = flow.euler_step(lambda x, t: torch.full_like(x, 2.0), torch.full((1, 2), 1.0), 0.7, 0.3)
    assert torch.allclose(out, torch.full((1, 2), 0.2), atol=1e-6), (
        f"From the value 1.0 at time 0.7 with the velocity 2.0, a step to time 0.3 should give 0.2, not "
        f"{out[0, 0].item():.4f}. The velocity points from the data towards the noise, and sampling runs from "
        "the noise at time 1 towards the data at time 0."
    )


def test_one_step_follows_one_line_exactly():
    generator = torch.Generator().manual_seed(0)
    x0, eps = torch.randn(4, 2, generator=generator), torch.randn(4, 2, generator=generator)
    out = flow.euler_step(lambda x, t: eps - x0, eps, 1.0, 0.0)
    assert torch.allclose(out, x0, atol=1e-6), (
        "With the velocity of each pair's line, one step from its noise at time 1 to time 0 should land on its "
        "clean point."
    )


def test_images_keep_their_shape():
    out = flow.euler_step(lambda x, t: torch.zeros_like(x), torch.zeros(2, 1, 28, 28), 0.5, 0.25)
    assert out.shape == (2, 1, 28, 28), f"A batch of images must keep its shape, not become {tuple(out.shape)}."


def test_the_exact_field_reproduces_the_two_clusters_in_fifty_calls():
    x_1 = torch.randn(4000, 2, generator=torch.Generator().manual_seed(3))
    x, calls = samplers.sample_euler(ExactVelocity(), x_1, samplers.even_times(50))
    stats = cluster_statistics(x)
    assert calls == 50, f"Fifty steps should cost fifty calls of the network, not {calls}."
    assert 0.46 <= stats["share_right"] <= 0.54, (
        f"Half of the samples should land in each cluster; {stats['share_right']:.3f} landed on the right."
    )
    assert stats["mean_distance"] == pytest.approx(1.97, abs=0.03), (
        f"With fifty calls the samples should sit near the centres at -2 and 2, at a mean distance from zero "
        f"of about 1.97; theirs is {stats['mean_distance']:.3f}."
    )
    assert stats["spread_within"] == pytest.approx(0.50, abs=0.03), (
        f"Each cluster should have a spread of about 0.5; the samples have {stats['spread_within']:.3f}."
    )
