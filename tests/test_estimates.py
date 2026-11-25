"""Tests for Task 4: estimates_from_velocity.

The notebook runs these for you. Each test reaches the function through the
module, so that the version the notebook attached is the one tested.
"""

import torch

from cs687 import flow


def pairs(n, shape=(2,), seed=0):
    generator = torch.Generator().manual_seed(seed)
    x0 = torch.randn(n, *shape, generator=generator)
    eps = torch.randn(n, *shape, generator=generator)
    t = torch.rand(n, generator=generator)
    t_rows = t.reshape(-1, *([1] * len(shape)))
    return x0, eps, t, (1 - t_rows) * x0 + t_rows * eps


def test_a_point_on_a_line_gives_back_its_pair():
    x0, eps, t, x_t = pairs(6)
    result = flow.estimates_from_velocity(eps - x0, x_t, t)
    assert isinstance(result, tuple) and len(result) == 2, "The function should return the tuple (x0_hat, eps_hat)."
    x0_hat, eps_hat = result
    assert torch.allclose(x0_hat, x0, atol=1e-5), (
        "With the velocity of its own line, a point should give back its clean point as the first estimate."
    )
    assert torch.allclose(eps_hat, eps, atol=1e-5), (
        "With the velocity of its own line, a point should give back its noise as the second estimate."
    )


def test_each_row_uses_its_own_time_for_images():
    x0, eps, t, x_t = pairs(3, shape=(1, 28, 28), seed=1)
    x0_hat, eps_hat = flow.estimates_from_velocity(eps - x0, x_t, t)
    assert x0_hat.shape == (3, 1, 28, 28) and eps_hat.shape == (3, 1, 28, 28), (
        "Both estimates must have the shape of x_t."
    )
    assert torch.allclose(x0_hat, x0, atol=1e-5) and torch.allclose(eps_hat, eps, atol=1e-5), (
        "Each row must use its own time."
    )


def test_mixing_the_estimates_at_a_lower_time_is_the_euler_step():
    generator = torch.Generator().manual_seed(2)
    x_t, v = torch.randn(5, 2, generator=generator), torch.randn(5, 2, generator=generator)
    t, s = 0.8, 0.3
    x0_hat, eps_hat = flow.estimates_from_velocity(v, x_t, torch.full((5,), t))
    assert torch.allclose((1 - s) * x0_hat + s * eps_hat, x_t + (s - t) * v, atol=1e-5), (
        "For any velocity, the estimates mixed with the weights of the line at a lower time s should give the "
        "point that one Euler step from t to s gives."
    )


def test_the_inputs_are_left_unchanged():
    x0, eps, t, x_t = pairs(4, seed=3)
    v, before = eps - x0, x_t.clone()
    flow.estimates_from_velocity(v, x_t, t)
    assert torch.equal(x_t, before), "The function must leave the tensor x_t unchanged."
