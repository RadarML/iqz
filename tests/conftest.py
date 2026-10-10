"""Shared fixtures."""

from typing import Literal

import numpy as np
import pytest

import iqz


@pytest.fixture(params=["numpy", "native"])
def backend(request) -> Literal["native", "numpy"]:
    """Run the test against each backend."""
    return request.param


def chirps(n: int, shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    """Radar-like int16 chirps: a static component plus noise."""
    rng = np.random.default_rng(seed)
    static = rng.integers(-2000, 2000, size=shape)
    noise = rng.normal(0, 20, size=(n, *shape))
    return np.clip(static + noise, -32768, 32767).astype(np.int16)


def extremes(n: int, shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    """Adversarial int16 chirps, concentrated at the ends of the range."""
    rng = np.random.default_rng(seed)
    choices = np.array([-32768, -32767, -1, 0, 1, 32766, 32767], np.int16)
    return rng.choice(choices, size=(n, *shape))
