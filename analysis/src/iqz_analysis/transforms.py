"""Building blocks shared by the measured variants."""

from dataclasses import dataclass
from functools import cache, cached_property

import numpy as np
import zstandard as zstd
from jaxtyping import Int16, UInt8, UInt16


@cache
def _cctx(level: int) -> zstd.ZstdCompressor:
    return zstd.ZstdCompressor(level=level)


def zsize(buf: bytes | np.ndarray, level: int = 3) -> int:
    """Size of `buf` after zstd compression at `level`."""
    data = buf if isinstance(buf, bytes) else np.ascontiguousarray(buf).data
    return len(_cctx(level).compress(data))


def zigzag(r: Int16[np.ndarray, "..."]) -> UInt16[np.ndarray, "..."]:
    """Map 0, -1, 1, -2, ... to 0, 1, 2, 3, ... (wrapping int16)."""
    r = r.astype(np.int16, copy=False)
    return ((r << 1) ^ (r >> 15)).view(np.uint16)


def shuffle(u: UInt16[np.ndarray, "..."]) -> UInt8[np.ndarray, "bytes"]:
    """Byte shuffle: all high bytes (in C order), then all low bytes."""
    b = np.ascontiguousarray(u).reshape(-1).view(np.uint8).reshape(-1, 2)
    return np.concatenate([b[:, 1], b[:, 0]])


def coded(u: UInt16[np.ndarray, "..."], level: int = 3) -> int:
    """Size of zigzagged values after byte shuffle and zstd."""
    return zsize(shuffle(u), level)


def wrap(e: np.ndarray) -> Int16[np.ndarray, "..."]:
    """Wrap a wider integer residual to int16 (exactly invertible)."""
    return e.astype(np.int16)


def deinterleave(x: np.ndarray) -> np.ndarray:
    """Split TI's IIQQ interleave along the last axis.

    The stored fast-time axis is `[c0[0], c0[1], c1[0], c1[1], c0[2], ...]`
    for the two I/Q components `c0, c1`; returns `(..., 2, fast // 2)`, with
    each component in time order.
    """
    *lead, f = x.shape
    y = x.reshape(*lead, f // 4, 2, 2)
    y = np.moveaxis(y, -2, -3)
    return y.reshape(*lead, 2, f // 2)


@dataclass
class MeanResidual:
    """A chunk with its mean chirp subtracted, as in the iqz codec.

    Args:
        x: int16 chunk, `(chirps, tx, rx, fast)`.
    """

    x: Int16[np.ndarray, "chirps tx rx fast"]

    @cached_property
    def mean(self) -> Int16[np.ndarray, "tx rx fast"]:
        """Rounded (half to even) mean chirp."""
        return np.round(
            self.x.mean(axis=0, dtype=np.float64)).astype(np.int16)

    @cached_property
    def residual(self) -> Int16[np.ndarray, "chirps tx rx fast"]:
        """Wrapping int16 residual `x - mean`."""
        return self.x - self.mean

    @cached_property
    def u(self) -> UInt16[np.ndarray, "chirps tx rx fast"]:
        """Zigzagged residual."""
        return zigzag(self.residual)

    @cached_property
    def overhead(self) -> int:
        """Bytes for the length header and the compressed mean chirp."""
        return 4 + zsize(self.mean)
