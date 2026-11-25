"""CS 687 Homework 5: DDIM, flow matching, and sample quality against network calls."""

# Check the documented minimum before importing the course modules.
import sys as _sys

if _sys.version_info < (3, 10):
    raise RuntimeError(
        "This repository requires Python 3.10 or later. "
        "The current interpreter is "
        + ".".join(str(_part) for _part in _sys.version_info[:3])
        + f" at {_sys.executable}."
    )

from cs687.schedule import Schedule
from cs687.networks import level_vector, ToyNoisePredictor, SmallUNet, ToyVelocity, VelocityUNet
from cs687.toy import (
    CENTRES,
    SD,
    sample_clusters,
    ExactNoisePredictor,
    ExactVelocity,
    cluster_statistics,
)
from cs687.digits import load_training_digits, load_course_digit_network
from cs687.diffusion import noisy_at_level, noise_prediction_loss, ancestral_jump, train_run
from cs687.ddim import ddim_step
from cs687.flow import flow_matching_loss, euler_step, estimates_from_velocity, line_time
from cs687.samplers import even_levels, even_times, sample_ancestral, sample_ddim, sample_euler

__all__ = [
    "Schedule",
    "level_vector",
    "ToyNoisePredictor",
    "SmallUNet",
    "ToyVelocity",
    "VelocityUNet",
    "CENTRES",
    "SD",
    "sample_clusters",
    "ExactNoisePredictor",
    "ExactVelocity",
    "cluster_statistics",
    "load_training_digits",
    "load_course_digit_network",
    "noisy_at_level",
    "noise_prediction_loss",
    "ancestral_jump",
    "train_run",
    "ddim_step",
    "flow_matching_loss",
    "euler_step",
    "estimates_from_velocity",
    "line_time",
    "even_levels",
    "even_times",
    "sample_ancestral",
    "sample_ddim",
    "sample_euler",
]
