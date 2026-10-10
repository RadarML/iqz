"""Measured variants: per-chunk compressed sizes, grouped by experiment.

Each group maps one int16 chunk `(64, tx, rx, fast)` to a dict of sizes in
bytes. Unless noted, residuals are wrapped to int16, zigzagged, byte
shuffled, and compressed with zstd level 3 (exactly as in iqz), and sizes
of mean-based variants include the iqz header and compressed mean chirp.
"""

import math
from collections.abc import Callable

import iqz
import numpy as np

from .coders import CODERS, ESTIMATES
from .transforms import MeanResidual, coded, deinterleave, wrap, zigzag, zsize


def _core(x: np.ndarray) -> dict[str, int]:
    """Raw size, zstd alone, previous-chirp delta, and iqz."""
    flat = x.reshape(x.shape[0], -1)
    d = flat.copy()
    d[1:] -= flat[:-1]
    return {
        "raw": x.nbytes,
        "zstd": zsize(flat),
        "delta": coded(zigzag(d)),
        "iqz": len(iqz.encode(x, backend="native")),
    }


def _fixed(e: np.ndarray, order: int) -> np.ndarray:
    """Residual of a fixed polynomial predictor along the last axis."""
    pred = np.zeros_like(e)
    if order >= 1:
        pred[..., 1:] = e[..., :-1]
    if order >= 2:
        pred[..., 2:] = 2 * e[..., 1:-1] - e[..., :-2]
    if order >= 3:
        pred[..., 3:] = 3 * e[..., 2:-1] - 3 * e[..., 1:-2] + e[..., :-3]
    return e - pred


def _lpc(
    e: np.ndarray, order: int = 4, shift: int = 12
) -> tuple[np.ndarray, int]:
    """Least-squares linear prediction along the last axis.

    One set of quantized coefficients (Q12) is fit per series (antenna and
    I/Q component), over all chirps in the chunk.

    Returns:
        The residual, and the number of coefficient bytes to store.
    """
    out = e.copy()
    series = e.reshape(e.shape[0], -1, e.shape[-1])  # (chirps, series, F)
    for s in range(series.shape[1]):
        y = series[:, s].astype(np.float64)
        lags = np.stack(
            [y[:, order - i - 1:-i - 1] for i in range(order)], axis=-1)
        a, *_ = np.linalg.lstsq(
            lags.reshape(-1, order), y[:, order:].reshape(-1), rcond=None)
        q = np.round(a * (1 << shift)).astype(np.int64)
        z = series[:, s].astype(np.int64)
        pred = np.zeros_like(z)
        for i in range(order):
            pred[:, order:] += q[i] * z[:, order - i - 1:-i - 1]
        pred = (pred + (1 << (shift - 1))) >> shift
        pred[:, :order] = 0
        idx = np.unravel_index(s, e.shape[1:-1])
        out[(slice(None), *idx)] = z - pred
    return out, 2 * order * series.shape[1]


def _predictors(x: np.ndarray) -> dict[str, int]:
    """Alternative predictors, alone or on top of the mean residual."""
    res = MeanResidual(x)
    r = res.residual.astype(np.int32)
    dr = deinterleave(r)  # (chirps, tx, rx, iq, fast)
    lpc, lpc_bytes = _lpc(dr)

    rx = r.copy()
    rx[:, :, 1:] -= r[:, :, :-1]
    tx = r.copy()
    tx[:, 1:] -= r[:, :-1]
    return {
        "none": coded(zigzag(x)),
        "mean_fast_delta": res.overhead + coded(zigzag(wrap(_fixed(dr, 1)))),
        "mean_fast_fixed2": res.overhead + coded(zigzag(wrap(_fixed(dr, 2)))),
        "mean_fast_fixed3": res.overhead + coded(zigzag(wrap(_fixed(dr, 3)))),
        "mean_fast_lpc4":
            res.overhead + lpc_bytes + coded(zigzag(wrap(lpc))),
        "mean_rx": res.overhead + coded(zigzag(wrap(rx))),
        "mean_tx": res.overhead + coded(zigzag(wrap(tx))),
    }


ORDERS = [
    "slow-tx-rx-iq-fast", "slow-tx-rx-fast-iq", "slow-fast-tx-rx-iq",
    "fast-slow-tx-rx-iq", "tx-rx-iq-slow-fast", "tx-rx-iq-fast-slow",
    "iq-tx-rx-slow-fast",
]
"""Axis orders of the (I/Q-separated) residual, outermost first."""

_AXES = {"slow": 0, "tx": 1, "rx": 2, "iq": 3, "fast": 4}


def _layouts(x: np.ndarray) -> dict[str, int]:
    """Reordered axes, and separate zstd frames per antenna or I/Q."""
    res = MeanResidual(x)
    ud = deinterleave(res.u)  # (slow, tx, rx, iq, fast)
    out = {}
    for order in ORDERS:
        perm = [_AXES[a] for a in order.split("-")]
        out[f"order_{order}"] = res.overhead + coded(ud.transpose(perm))
    out["frame_per_antenna"] = res.overhead + sum(
        coded(res.u[:, t, r])
        for t in range(x.shape[1]) for r in range(x.shape[2]))
    out["frame_per_iq"] = res.overhead + sum(
        coded(ud[:, :, :, c]) for c in range(2))
    return out


def _entropy(x: np.ndarray) -> dict[str, int]:
    """Entropy coders in place of zstd, on the same mean residual."""
    res = MeanResidual(x)
    out = {}
    for name, coder in CODERS.items():
        out[name] = res.overhead + sum(len(b) for b in coder.encode(res))
    for name, (_, bits) in ESTIMATES.items():
        out[name] = res.overhead + math.ceil(bits(res) / 8)
    return out


LEVELS = list(range(1, 10))
"""zstd levels compared."""


def _levels(x: np.ndarray) -> dict[str, int]:
    """Compressed size with iqz, at each zstd level."""
    return {f"level{lv}": len(iqz.encode(x, level=lv)) for lv in LEVELS}


GROUPS: dict[str, Callable[[np.ndarray], dict[str, int]]] = {
    "core": _core,
    "predictors": _predictors,
    "layouts": _layouts,
    "entropy": _entropy,
    "levels": _levels,
}
"""Variant groups, each run over the full 10k-chunk sample."""


SWEEP_SIZES = [16, 32, 64, 128, 256]
"""Chunk sizes compared in the sweep."""


def sweep(x: np.ndarray) -> dict[str, int]:
    """Mean (iqz) and delta sizes for each chunk size, on the same chirps.

    Args:
        x: 256 consecutive chirps.
    """
    out: dict[str, int] = {"raw": x.nbytes}
    for n in SWEEP_SIZES:
        chunks = x.reshape(-1, n, *x.shape[1:])
        out[f"mean{n}"] = sum(len(iqz.encode(c)) for c in chunks)
        size = 0
        for c in chunks:
            flat = c.reshape(n, -1)
            d = flat.copy()
            d[1:] -= flat[:-1]
            size += coded(zigzag(d))
        out[f"delta{n}"] = size
    return out
