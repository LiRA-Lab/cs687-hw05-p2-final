"""Tests for Task 1: ddim_step.

The notebook runs these for you. Each test reaches the function through the
module, so that the version the notebook attached is the one tested. The last
test runs the supplied DDIM loop with the best possible noise prediction of the
two clusters, which cannot be wrong, so a failure there is a failure of the
step.
"""

import pytest
import torch

from cs687 import ddim, samplers
from cs687.schedule import Schedule
from cs687.toy import ExactNoisePredictor, cluster_statistics

SCHEDULE = Schedule()


def constant(value):
    return lambda x, levels: torch.full_like(x, value)


def test_the_model_is_called_once_with_the_level():
    calls = []

    def recording(x, levels):
        calls.append(levels)
        return torch.zeros_like(x)

    out = ddim.ddim_step(recording, torch.zeros(3, 2), 700, 300, SCHEDULE)
    assert out.shape == (3, 2), f"The result must have the shape of x_t, (3, 2), not {tuple(out.shape)}."
    assert len(calls) == 1, f"The model should be called once per step, not {len(calls)} times."
    levels = calls[0]
    assert torch.is_tensor(levels) and levels.dtype == torch.long and levels.shape == (3,), (
        "The model should receive a long tensor with one level per row."
    )
    assert torch.all(levels == 700), (
        f"Every row is at the current level 700, but the model received {levels.tolist()}."
    )


def test_the_step_to_level_zero_returns_the_estimate_of_the_clean_image():
    out = ddim.ddim_step(constant(0.3), torch.full((1, 2), 0.5), 400, 0, SCHEDULE)
    assert torch.allclose(out, torch.full((1, 2), 0.522597), atol=1e-4), (
        f"From the value 0.5 at level 400 with the predicted noise 0.3, the step to level 0 should give "
        f"0.522597, the estimate of the clean image, not {out[0, 0].item():.6f}."
    )


def test_a_strided_step_matches_the_course_numbers():
    out = ddim.ddim_step(constant(0.2), torch.tensor([[0.4, -1.0]]), 500, 150, SCHEDULE)
    expected = torch.tensor([[0.750857, -3.683224]])
    assert torch.allclose(out, expected, atol=1e-4), (
        f"The step from level 500 to level 150 with the predicted noise 0.2 should give {expected.tolist()}, "
        f"not {[round(v, 6) for v in out[0].tolist()]}."
    )


def test_one_step_is_exact_when_the_data_is_one_point():
    """The notes' "The error of few steps": with one clean point the predicted noise is constant along each path."""
    point = torch.tensor([1.5, -0.5])

    def one_point(x, levels):
        alpha_bar = SCHEDULE.alpha_bar[levels][:, None]
        return (x - alpha_bar.sqrt() * point) / (1 - alpha_bar).sqrt()

    start = torch.randn(5, 2, generator=torch.Generator().manual_seed(0))
    out = ddim.ddim_step(one_point, start, 1000, 0, SCHEDULE)
    assert torch.allclose(out, point.expand(5, 2), atol=1e-3), (
        "When the data is the single point (1.5, -0.5) and the prediction is exact, one DDIM step from "
        f"level 1000 to level 0 should land on that point, not on {out[0].tolist()}."
    )
    x = start
    for t, s in ((1000, 600), (600, 250), (250, 0)):
        x = ddim.ddim_step(one_point, x, t, s, SCHEDULE)
    assert torch.allclose(x, point.expand(5, 2), atol=1e-3), (
        "With a single clean point, the steps 1000 to 600 to 250 to 0 should also land exactly on it."
    )


def test_the_step_draws_no_noise_and_leaves_x_t_unchanged():
    before = torch.get_rng_state()
    x = torch.full((4, 2), 0.7)
    first = ddim.ddim_step(constant(0.1), x, 300, 100, SCHEDULE)
    second = ddim.ddim_step(constant(0.1), x, 300, 100, SCHEDULE)
    assert torch.equal(torch.get_rng_state(), before), "The DDIM step is deterministic and must draw no random numbers."
    assert torch.equal(first, second), "The same input must give the same output."
    assert torch.equal(x, torch.full((4, 2), 0.7)), "The step must leave the tensor x_t unchanged."


def test_images_keep_their_shape():
    out = ddim.ddim_step(constant(0.0), torch.zeros(2, 1, 28, 28), 500, 250, SCHEDULE)
    assert out.shape == (2, 1, 28, 28), f"A batch of images must keep its shape, not become {tuple(out.shape)}."


def test_the_exact_prediction_reproduces_the_two_clusters_in_fifty_calls():
    x_T = torch.randn(4000, 2, generator=torch.Generator().manual_seed(3))
    x, calls = samplers.sample_ddim(ExactNoisePredictor(SCHEDULE), SCHEDULE, x_T, samplers.even_levels(50))
    stats = cluster_statistics(x)
    assert calls == 50, f"Fifty steps should cost fifty calls of the network, not {calls}."
    assert 0.46 <= stats["share_right"] <= 0.54, (
        f"Half of the samples should land in each cluster; {stats['share_right']:.3f} landed on the right."
    )
    assert stats["mean_distance"] == pytest.approx(1.97, abs=0.03), (
        f"With fifty calls the samples should sit near the centres at -2 and 2, at a mean distance from zero "
        f"of about 1.97; theirs is {stats['mean_distance']:.3f}."
    )
    assert stats["spread_within"] == pytest.approx(0.49, abs=0.03), (
        f"Each cluster should have a spread of about 0.49; the samples have {stats['spread_within']:.3f}."
    )
