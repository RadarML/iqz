"""Pure numpy backend (reference implementation)."""

import threading

import numpy as np
import zstandard as zstd
from jaxtyping import Int16, UInt8

from . import _format

_local = threading.local()


def _cctx(level: int) -> zstd.ZstdCompressor:
    cache: dict[int, zstd.ZstdCompressor] = getattr(_local, "cctx", None) or {}
    _local.cctx = cache
    if level not in cache:
        cache[level] = zstd.ZstdCompressor(level=level)
    return cache[level]


def _dctx() -> zstd.ZstdDecompressor:
    if not hasattr(_local, "dctx"):
        _local.dctx = zstd.ZstdDecompressor()
    return _local.dctx


def encode_chunk(x: Int16[np.ndarray, "n M"], level: int) -> bytes:
    """Encode one C-contiguous `(n, M)` int16 chunk."""
    m = np.round(x.mean(axis=0, dtype=np.float64)).astype(np.int16)
    r = x - m                                      # int16, wraps on overflow
    u = (r << 1) ^ (r >> 15)                       # zigzag (arithmetic shift)
    b = u.view(np.uint8).reshape(-1, 2)            # little-endian: [lo, hi]
    residuals = np.concatenate([b[:, 1], b[:, 0]]) # all high bytes, then low
    cctx = _cctx(level)
    mz = cctx.compress(m.data)
    return len(mz).to_bytes(4, "little") + mz + cctx.compress(residuals.data)


def decode_into(
    blob: UInt8[np.ndarray, "nbytes"], out: Int16[np.ndarray, "n M"]
) -> None:
    """Decode `blob` into a preallocated C-contiguous `(n, M)` int16 array."""
    parts = _format.IQZBlob.from_buffer(blob)
    n, M = parts.chirps, parts.samples
    if out.shape != (n, M):
        raise ValueError(f"blob does not match output shape {out.shape}")
    dctx = _dctx()
    try:
        m = np.frombuffer(dctx.decompress(parts.mean), np.int16)
        p = np.frombuffer(dctx.decompress(parts.residuals), np.uint8)
    except zstd.ZstdError as e:
        raise ValueError(f"zstd error: {e}") from e
    h = n * M
    u = (p[:h].astype(np.uint16) << 8) | p[h:]
    r = (u >> 1).view(np.int16) ^ -(u & 1).view(np.int16)
    np.add(r.reshape(n, M), m, out=out)            # int16, wraps back
