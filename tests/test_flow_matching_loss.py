"""Tests for Task 2: flow_matching_loss.

The notebook runs these for you. Each test reaches the function through the
module, so that the version the notebook attached is the one tested.
"""

import pytest
import torch

from cs687 import flow
from cs687.toy import sample_clusters


def course_draws(x0, seed):
    """The draws of the task, in the stated order: the times first, then the noise."""
    generator = torch.Generator().manual_seed(seed)
    t = torch.rand((x0.shape[0],), generator=generator)
    eps = torch.randn(x0.shape, generator=generator)
    return t, eps


def test_the_model_sees_the_point_on_the_line_and_its_time():
    x0 = sample_clusters(6, torch.Generator().manual_seed(0))
    seen = []

    def recording(x, t):
        seen.append((x.clone(), t.clone()))
        return torch.zeros_like(x)

    flow.flow_matching_loss(recording, x0, torch.Generator().manual_seed(5))
    assert len(seen) == 1, f"The model should be called once, not {len(seen)} times."
    x_t, t = seen[0]
    t_expected, eps = course_draws(x0, 5)
    assert t.shape == (6,) and t.is_floating_point(), (
        f"The model should receive the times as a float tensor of shape (6,), not {t.dtype} of shape {tuple(t.shape)}."
    )
    assert torch.allclose(t, t_expected), (
        "The times should be the first draw from the generator, one per row, with torch.rand, uniform between 0 and 1."
    )
    expected = (1 - t_expected[:, None]) * x0 + t_expected[:, None] * eps
    assert torch.allclose(x_t, expected, atol=1e-6), (
        "The model should receive each point on its line at its time, (1 - t) x0 + t eps, with the noise drawn "
        "second, after the times."
    )


def test_the_target_is_the_velocity_of_the_line():
    x0 = sample_clusters(8, torch.Generator().manual_seed(1))
    t, eps = course_draws(x0, 6)
    loss = flow.flow_matching_loss(lambda x, times: eps - x0, x0, torch.Generator().manual_seed(6))
    assert loss.item() == pytest.approx(0.0, abs=1e-10), (
        f"A model that returns the velocity of each line exactly should have the loss 0, not {loss.item():.4f}."
    )


def test_a_model_of_zeros_scores_about_three_and_a_quarter():
    x0 = sample_clusters(50_000, torch.Generator().manual_seed(1))
    loss = flow.flow_matching_loss(lambda x, t: torch.zeros_like(x), x0, torch.Generator().manual_seed(2))
    assert loss.item() == pytest.approx(3.255, abs=0.005), (
        f"On the two clusters a model that returns zeros should score about 3.25 per number, the mean over "
        f"every number of the batch, not {loss.item():.4f}."
    )


def test_the_loss_matches_the_course_number():
    x0 = sample_clusters(64, torch.Generator().manual_seed(21))
    loss = flow.flow_matching_loss(lambda x, t: 0.5 * x, x0, torch.Generator().manual_seed(22))
    assert loss.item() == pytest.approx(4.861648, abs=1e-4), (
        f"With the seed of this test and the model 0.5 x_t, the loss should be 4.861648, not {loss.item():.6f}."
    )


def test_images_and_the_gradient():
    weight = torch.tensor(0.5, requires_grad=True)
    x0 = torch.rand(3, 1, 28, 28, generator=torch.Generator().manual_seed(0)) * 2 - 1
    loss = flow.flow_matching_loss(lambda x, t: weight * x, x0, torch.Generator().manual_seed(1))
    assert loss.dim() == 0, f"The loss should be a tensor with one value, not of shape {tuple(loss.shape)}."
    loss.backward()
    assert weight.grad is not None and weight.grad.item() != 0, (
        "The loss must keep its gradient with respect to the model's parameters."
    )
