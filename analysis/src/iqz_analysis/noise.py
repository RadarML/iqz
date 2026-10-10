"""Thermal noise floor, estimated from range-Doppler maps.

Thermal noise is white along slow time: after a (unitary) Doppler FFT, it
spreads evenly over all Doppler bins. Scene content does not: static clutter
falls into Doppler bin 0 (which the iqz mean removes anyway), and targets and
moving clutter occupy only part of the Doppler axis. Each range bin's noise
level is therefore estimated from a low quantile of the power in its Doppler
cells, using the fact that the power of complex Gaussian noise is
exponentially distributed.

Noise levels differ across virtual channels (by up to 2-3x), and across
range bins, since the receive chain filters the noise (high-pass at short
range, anti-alias low-pass at the far end). Levels are therefore estimated
per channel and per group of adjacent range bins. For such coloured
Gaussian noise, the entropy rate (Szego's theorem) is
`0.5 * log2(2 pi e) + mean of log2(sigma)` over all channels and range bins,
in bits per real value: that of white noise at the geometric-mean level.
"""

import numpy as np

from .transforms import deinterleave

GAUSSIAN_BITS = 0.5 * np.log2(2 * np.pi * np.e)
"""Differential entropy of a unit-variance Gaussian, in bits (~2.05)."""


def noise_profile(
    x: np.ndarray, quantile: float = 0.1, pool: int = 4
) -> np.ndarray:
    """Estimate the noise level per virtual channel and range bin group.

    Args:
        x: int16 chunk, `(chirps, tx, rx, fast)` in TI's IIQQ layout.
        quantile: quantile of cell power used; lower is more robust to scene
            content, at the cost of variance.
        pool: number of adjacent range bins pooled per estimate.

    Returns:
        Noise standard deviation per I/Q component (LSB), with shape
            `(tx * rx, range bins / pool)`.
    """
    c = deinterleave(x.astype(np.float64))  # (S, tx, rx, 2, F); [Q, I]
    z = c[..., 1, :] + 1j * c[..., 0, :]
    S, T, R, F = z.shape
    z = z.reshape(S, T * R, F)
    for axis in (0, 2):  # Doppler, range; both unitary
        z = np.fft.fft(z, axis=axis, norm="ortho")
    power = np.abs(z[1:]) ** 2  # Doppler bin 0 is static clutter
    power = power.reshape(S - 1, T * R, F // pool, pool)
    power = power.transpose(1, 2, 0, 3).reshape(T * R, F // pool, -1)
    q = np.quantile(power, quantile, axis=-1)
    # Exponential with mean 2 sigma^2: q = -ln(1 - p) * 2 sigma^2.
    return np.sqrt(q / (-np.log1p(-quantile) * 2))


def noise_floor(x: np.ndarray) -> dict[str, float]:
    """Noise summary for one chunk.

    Returns:
        `sigma`: geometric-mean noise level over channels and range bins
            (LSB, per I/Q component); `sigma_p50`: the same from the median
            instead of the 10th percentile, as a diagnostic (close to
            `sigma` when the estimate is not contaminated by scene content);
            `sigma_ch_min`, `sigma_ch_max`: range of per-channel levels.
    """
    log = np.log(noise_profile(x, 0.1))
    per_channel = np.exp(log.mean(axis=1))
    return {
        "sigma": float(np.exp(log.mean())),
        "sigma_p50": float(np.exp(np.log(noise_profile(x, 0.5)).mean())),
        "sigma_ch_min": float(per_channel.min()),
        "sigma_ch_max": float(per_channel.max()),
    }


def bits(sigma: np.ndarray | float) -> np.ndarray | float:
    """Minimum bits per real value for Gaussian noise of level `sigma`."""
    return np.log2(sigma) + GAUSSIAN_BITS
