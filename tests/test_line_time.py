"""Tests for Task 5: line_time.

The notebook runs these for you. Each test reaches the function through the
module, so that the version the notebook attached is the one tested.
"""

import pytest
import torch

from cs687 import flow
from cs687.schedule import Schedule

SCHEDULE = Schedule()


def test_the_ends_of_the_schedule():
    times = flow.line_time(torch.tensor([0, 1000]), SCHEDULE)
    assert times[0].item() == pytest.approx(0.0, abs=1e-7), (
        f"Level 0 has the noise scale 0, so its time on the line is 0, not {times[0].item():.4f}."
    )
    assert times[1].item() == pytest.approx(0.9937, abs=1e-4), (
        f"Level 1000 has the noise scale of about 157, so its time on the line is about 0.9937, "
        f"not {times[1].item():.4f}."
    )


def test_the_notes_example_at_level_800():
    time = flow.line_time(torch.tensor([800]), SCHEDULE)[0].item()
    assert time == pytest.approx(0.9623, abs=1e-3), (
        f"The notes' example: level 800 has a noise scale of about 26, so its time on the line is about 26/27, "
        f"0.963, not {time:.4f}."
    )


def test_the_noise_scale_of_the_line_matches_every_level():
    levels = torch.arange(1, 1001)
    times = flow.line_time(levels, SCHEDULE).double()
    alpha_bar = SCHEDULE.alpha_bar.double()[levels]
    assert torch.allclose(times / (1 - times), ((1 - alpha_bar) / alpha_bar).sqrt(), rtol=1e-3), (
        "At each returned time, the noise scale of the line, t / (1 - t), should equal the noise scale of the "
        "level, the standard deviation of the noise once the image is divided by the weight of the clean image."
    )


def test_the_shape_and_type():
    times = flow.line_time(torch.tensor([[0, 250, 500], [750, 900, 1000]]), SCHEDULE)
    assert times.shape == (2, 3) and times.is_floating_point(), (
        f"The result should be a float tensor of the shape of the levels, (2, 3), not {times.dtype} of shape "
        f"{tuple(times.shape)}."
    )
    assert torch.all(times[:, 1:] > times[:, :-1]), "Higher levels carry more noise, so their times are later."
