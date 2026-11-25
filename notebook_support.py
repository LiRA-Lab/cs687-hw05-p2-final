"""Operational helpers used by the Homework 5 student notebook.

The notebook keeps the assessed code and the demonstrations visible. This
module holds only the setup, the checks of the guided calculations, the
plots, and the running of the public tests, none of which students are asked
to write.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable


def prepare_notebook(repo_root: Path) -> Path:
    """Enter the repository, install its requirements, and make it importable."""
    repo_root = repo_root.resolve()
    required = (repo_root / "cs687", repo_root / "tests", repo_root / "requirements.txt")
    if not required[0].is_dir() or not required[1].is_dir() or not required[2].is_file():
        raise RuntimeError(
            "The notebook setup did not locate a Homework 5 repository containing "
            "cs687/, tests/, and requirements.txt."
        )

    os.chdir(repo_root)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        check=True,
    )
    root_text = str(repo_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    print("ready")
    return repo_root


def pick_device() -> "torch.device":
    """A CUDA device when the runtime has one, then a Mac's GPU, otherwise the processor."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _is_incomplete(function: Callable[..., object]) -> bool:
    """Return whether an answer still contains its supplied placeholder."""
    return "NotImplementedError" in function.__code__.co_names


def _close(value: object, expected: float, tolerance: float) -> bool:
    try:
        return abs(float(value) - expected) <= tolerance
    except (TypeError, ValueError):
        return False


def _numbers(value: object, count: int) -> list[float] | None:
    """The value as a list of `count` plain numbers, or None if it is not one."""
    try:
        numbers = [float(v) for v in value]  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return None
    return numbers if len(numbers) == count else None


# ----------------------------------------------------------------------
# Checks of the four guided calculations
# ----------------------------------------------------------------------

def check_ddim_by_hand(
    schedule, x0_hat_600: object, x_300: object, sigma_600: object, sigma_300: object, x_300_rescaled: object
) -> None:
    """Check one DDIM step from level 600 to level 300, from 0.6 with the predicted noise 0.3, both ways."""
    exact = schedule.double()
    a6, a3 = exact.alpha_bar[600].item(), exact.alpha_bar[300].item()
    x, eps = 0.6, 0.3
    estimate = (x - math.sqrt(1 - a6) * eps) / math.sqrt(a6)
    if not _close(x0_hat_600, estimate, 1e-3):
        raise AssertionError(
            "x0_hat_600 is the estimate of the clean image that the predicted noise gives at level 600, the first "
            "stage of the DDIM step, with the numbers of exact_schedule."
        )
    if not _close(x_300, math.sqrt(a3) * estimate + math.sqrt(1 - a3) * eps, 1e-3):
        raise AssertionError(
            "x_300 combines the estimate of the clean image and the same predicted noise with the numbers of the "
            "schedule at level 300, the second stage of the DDIM step."
        )
    s6, s3 = math.sqrt((1 - a6) / a6), math.sqrt((1 - a3) / a3)
    if not _close(sigma_600, s6, 1e-3) or not _close(sigma_300, s3, 1e-3):
        raise AssertionError(
            "sigma_600 and sigma_300 are the noise scales of the Lecture 4 notes at the two levels, the standard "
            "deviation of the noise in the rescaled image."
        )
    if not _close(x_300_rescaled, math.sqrt(a3) * (x / math.sqrt(a6) + (s3 - s6) * eps), 1e-3):
        raise AssertionError(
            "x_300_rescaled is the rescaled point at level 600 moved by the change of the noise scale times the "
            "predicted noise, then multiplied back by the weight of the clean image at level 300."
        )
    print("The DDIM step by hand matches, and the two ways agree.")


def check_line_ends(zero_loss: object, floor_at_one: object, floor_near_zero: object) -> None:
    """Check the loss of the zero velocity and the floor at the two ends of the line, coordinate by coordinate."""
    expected = {
        "zero_loss": (zero_loss, (5.25, 1.25),
                      "the average of (eps - x0)^2 in each coordinate, with x0 a point of the two clusters and eps "
                      "independent standard Gaussian noise"),
        "floor_at_one": (floor_at_one, (4.25, 0.25),
                         "the conditional variance of the velocity eps - x0 given the point at time 1, the average "
                         "squared deviation from its conditional mean, where the point is the noise itself"),
        "floor_near_zero": (floor_near_zero, (1.0, 1.0),
                            "the conditional variance of the velocity eps - x0 given the point, as the time approaches "
                            "0 and the point approaches its clean point"),
    }
    for name, (given, target, meaning) in expected.items():
        values = _numbers(given, 2)
        if values is None:
            raise AssertionError(f"{name} must be two numbers, (first coordinate, second coordinate).")
        if any(abs(v - e) > 1e-3 for v, e in zip(values, target)):
            raise AssertionError(f"{name} is {meaning}, for the first coordinate and for the second.")
    print("The losses at the two ends of the line match.")


def check_line_step(x0_hat_line: object, eps_hat_line: object, x_s_mixed: object, x_s_euler: object) -> None:
    """Check the two estimates and the two ways of stepping on the line, at t = 0.75 to s = 0.25."""
    if not _close(x0_hat_line, -0.1, 1e-6) or not _close(eps_hat_line, 1.1, 1e-6):
        raise AssertionError(
            "x0_hat_line and eps_hat_line are the clean point and the noise that the velocity 1.2 gives at the point "
            "0.8 and the time 0.75, from the point written from either end of its line."
        )
    if not _close(x_s_mixed, 0.2, 1e-6):
        raise AssertionError("x_s_mixed mixes the two estimates with the weights of the line at the time 0.25.")
    if not _close(x_s_euler, 0.2, 1e-6):
        raise AssertionError("x_s_euler is one Euler step of the velocity 1.2 from the time 0.75 to the time 0.25.")
    print("The step on the line matches, and the two ways agree.")


def check_four_calls(schedule, ddim_sigmas: object, ddim_times: object, line_sigmas: object) -> None:
    """Check where four evenly spaced calls of each sampler fall on the noise scale and on the time of the line."""
    exact = schedule.double()
    levels = (1000, 750, 500, 250)
    sigmas = [math.sqrt((1 - exact.alpha_bar[l].item()) / exact.alpha_bar[l].item()) for l in levels]
    given = _numbers(ddim_sigmas, 4)
    if given is None or any(abs(g - s) > 1e-3 * s for g, s in zip(given, sigmas)):
        raise AssertionError(
            "ddim_sigmas are the four noise scales of the levels 1000, 750, 500 and 250, from exact_schedule."
        )
    given = _numbers(ddim_times, 4)
    if given is None or any(abs(g - s / (1 + s)) > 1e-3 for g, s in zip(given, sigmas)):
        raise AssertionError(
            "ddim_times are the times of the straight line that have the same noise scales as the four levels."
        )
    given = _numbers(line_sigmas, 4)
    if given is None or not math.isinf(given[0]) or any(abs(g - e) > 1e-3 for g, e in zip(given[1:], (3.0, 1.0, 1 / 3))):
        raise AssertionError(
            "line_sigmas are the noise scales of the straight line at the times 1, 0.75, 0.5 and 0.25; write "
            "math.inf for the first."
        )
    print("The noise scales and the times of the four calls match.")


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

LEFT_COLOUR = "#2a6f97"
RIGHT_COLOUR = "#c8553d"
SAMPLER_COLOURS = {"ancestral": "#7b6d8d", "DDIM": "#c8553d", "Euler": "#2a6f97"}


def plot_points(panels: list[tuple], limit: float = 4.0) -> None:
    """Draw sets of points of the two clusters side by side, each panel a pair (title, points)."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.0), squeeze=False)
    for axis, (title, points) in zip(axes[0], panels):
        points = points.detach().cpu()
        right = points[:, 0] > 0
        axis.scatter(points[~right, 0], points[~right, 1], s=3, alpha=0.5, color=LEFT_COLOUR)
        axis.scatter(points[right, 0], points[right, 1], s=3, alpha=0.5, color=RIGHT_COLOUR)
        axis.set_title(title, fontsize=9)
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_aspect("equal")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    plt.show()


def plot_loss_against_floor(curve: list[float], floor: float, zero: float, window: int = 100) -> None:
    """Draw the loss of every step, its running mean, the floor, and the loss of the zero velocity."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 3.6))
    axis.plot(range(1, len(curve) + 1), curve, color="#9db4c0", linewidth=0.6, label="loss of each step")
    if len(curve) >= window:
        running = [sum(curve[i - window:i]) / window for i in range(window, len(curve) + 1)]
        axis.plot(range(window, len(curve) + 1), running, color="#1f3b73", label=f"mean of the last {window} steps")
    axis.axhline(floor, color=RIGHT_COLOUR, linestyle="--", linewidth=1.2, label=f"floor, the exact mean field: {floor:.3f}")
    axis.axhline(zero, color="#555555", linestyle=":", linewidth=1.2, label=f"the zero velocity: {zero:.3f}")
    axis.set_xlabel("step")
    axis.set_ylabel("loss per number")
    axis.set_ylim(0, zero * 1.2)
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    plt.show()


def plot_quality_against_calls(calls: list[int], table: dict, data: dict) -> None:
    """Draw the mean distance and the spread within a half against the number of calls.

    table maps (sampler, source) to a list of statistics, one per entry of
    calls, with sampler "ancestral", "DDIM" or "Euler" and source "exact" or
    "network"; data holds the statistics of the data.
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for axis, key, name in ((axes[0], "mean_distance", "mean distance from the middle"),
                            (axes[1], "spread_within", "spread within a half")):
        for (sampler, source), rows in table.items():
            axis.plot(calls, [row[key] for row in rows], marker="o", markersize=3.5,
                      color=SAMPLER_COLOURS[sampler], linestyle="-" if source == "exact" else "--",
                      label=f"{sampler}, {'exact field' if source == 'exact' else 'your network'}")
        axis.axhline(data[key], color="#333333", linewidth=0.8, linestyle=":", label="the data")
        axis.set_xscale("log")
        axis.set_xlabel("calls of the network per sample")
        axis.set_title(name, fontsize=10)
        axis.grid(alpha=0.3)
    axes[0].set_ylim(0, 2.3)
    axes[1].set_ylim(0, 1.0)
    axes[1].legend(fontsize=7, loc="upper right")
    figure.tight_layout()
    plt.show()


def plot_digit_rows(rows: list[tuple[str, "torch.Tensor"]], title: str) -> None:
    """Draw one row of digits per entry of rows, a pair (label, images of shape (N, 1, 28, 28))."""
    import matplotlib.pyplot as plt

    count = len(rows[0][1])
    figure, axes = plt.subplots(len(rows), count, figsize=(1.0 * count + 1.2, 1.05 * len(rows)), squeeze=False)
    for r, (label, images) in enumerate(rows):
        images = images[:, 0].detach().float().cpu().clamp(-1, 1)
        for c in range(count):
            axis = axes[r][c]
            axis.imshow(images[c], cmap="gray", vmin=-1, vmax=1)
            axis.set_xticks([])
            axis.set_yticks([])
            if c == 0:
                axis.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Attaching the answers and running the public tests
# ----------------------------------------------------------------------

def _run_pytest(path: str, failure_message: str) -> None:
    """Run one public-test target in the current notebook kernel."""
    import pytest

    # A test module imported by an earlier run would keep the earlier answer,
    # so the test modules are reloaded on every run.
    for name in list(sys.modules):
        if name.startswith("test_"):
            del sys.modules[name]

    result = pytest.main(["-q", "--tb=short", "-p", "no:cacheprovider", path])
    if result != pytest.ExitCode.OK:
        raise AssertionError(failure_message)


def _attach(module_name: str, function: Callable[..., object]) -> None:
    """Make the notebook's function the one the package and its loops use."""
    import importlib
    import cs687

    module = importlib.import_module(f"cs687.{module_name}")
    setattr(module, function.__name__, function)
    setattr(cs687, function.__name__, function)


def _require_complete(pairs: list[tuple[str, Callable[..., object]]]) -> None:
    incomplete = [name for name, function in pairs if _is_incomplete(function)]
    if incomplete:
        raise AssertionError(
            "Complete and rerun these answer cells first: " + ", ".join(incomplete)
        )


_TASKS = {
    1: ("ddim", "tests/test_ddim_step.py"),
    2: ("flow", "tests/test_flow_matching_loss.py"),
    3: ("flow", "tests/test_euler_step.py"),
    4: ("flow", "tests/test_estimates.py"),
    5: ("flow", "tests/test_line_time.py"),
}


def _run_task_tests(number: int, function: Callable[..., object]) -> None:
    if _is_incomplete(function):
        raise AssertionError(
            f"Task {number} is still incomplete. Replace the NotImplementedError in the "
            f"answer-task-{number} cell, run that cell again, and rerun this check."
        )
    module_name, path = _TASKS[number]
    _attach(module_name, function)
    _run_pytest(path, f"Task {number} has not passed all public tests. Review the failures above.")
    print(f"Task {number} passed all public tests.")


def run_task1_tests(ddim_step: Callable[..., object]) -> None:
    """Attach the Task 1 answer and run its public tests."""
    _run_task_tests(1, ddim_step)


def run_task2_tests(flow_matching_loss: Callable[..., object]) -> None:
    """Attach the Task 2 answer and run its public tests."""
    _run_task_tests(2, flow_matching_loss)


def run_task3_tests(euler_step: Callable[..., object]) -> None:
    """Attach the Task 3 answer and run its public tests."""
    _run_task_tests(3, euler_step)


def run_task4_tests(estimates_from_velocity: Callable[..., object]) -> None:
    """Attach the Task 4 answer and run its public tests."""
    _run_task_tests(4, estimates_from_velocity)


def run_task5_tests(line_time: Callable[..., object]) -> None:
    """Attach the Task 5 answer and run its public tests."""
    _run_task_tests(5, line_time)


def run_all_public_tests(
    ddim_step: Callable[..., object],
    flow_matching_loss: Callable[..., object],
    euler_step: Callable[..., object],
    estimates_from_velocity: Callable[..., object],
    line_time: Callable[..., object],
) -> None:
    """Attach all five answers and run the complete public test suite."""
    pairs = [
        ("Task 1", ddim_step),
        ("Task 2", flow_matching_loss),
        ("Task 3", euler_step),
        ("Task 4", estimates_from_velocity),
        ("Task 5", line_time),
    ]
    _require_complete(pairs)
    for number, (_, function) in enumerate(pairs, start=1):
        _attach(_TASKS[number][0], function)
    _run_pytest("tests", "One or more public tests failed. Review the failures above.")
    print("All 25 public tests passed.")
