"""Entropy coders compared against zstd, applied to the iqz mean residual.

Each coder has an `encode(res) -> list[bytes]` and a matching
`decode(blobs, res)`, used for sizes (sum of blob lengths) and decode speed;
estimates give a size in bits with no implementation behind them.
"""

import io
import lzma
from collections.abc import Callable
from dataclasses import dataclass

import imagecodecs
import lz4.frame
import numpy as np
import soundfile
import zstandard as zstd
from pcodec import ChunkConfig, standalone

from .transforms import MeanResidual, deinterleave, shuffle


@dataclass(frozen=True)
class Coder:
    """A lossless coder for the (zigzagged) mean residual.

    Args:
        label: display name.
        encode: returns one or more compressed blobs.
        decode: inverts `encode` (output is discarded; used for timing).
    """

    label: str
    encode: Callable[[MeanResidual], list[bytes]]
    decode: Callable[[list[bytes], MeanResidual], object]


def _residuals(res: MeanResidual) -> bytes:
    return shuffle(res.u).tobytes()


def _series(res: MeanResidual) -> np.ndarray:
    """Residual as I/Q-separated fast-time series, `(chirps, tx, rx, 2, F)`."""
    return np.ascontiguousarray(deinterleave(res.residual))


def _tiles(res: MeanResidual) -> list[np.ndarray]:
    """Zigzagged residual as `(chirps, fast)` images per antenna and I/Q."""
    ud = deinterleave(res.u)
    return [np.ascontiguousarray(ud[:, t, r, c])
            for t in range(ud.shape[1]) for r in range(ud.shape[2])
            for c in range(2)]


def _zstd(level: int) -> Coder:
    cctx, dctx = zstd.ZstdCompressor(level=level), zstd.ZstdDecompressor()
    return Coder(
        f"zstd level {level}",
        lambda res: [cctx.compress(_residuals(res))],
        lambda blobs, res: dctx.decompress(blobs[0]))


def _flac_encode(res: MeanResidual) -> list[bytes]:
    buf = io.BytesIO()
    soundfile.write(
        buf, _series(res).reshape(-1), 48000, format="FLAC",
        subtype="PCM_16", compression_level=1.0)  # = flac -8
    return [buf.getvalue()]


def _flac_decode(blobs: list[bytes], res: MeanResidual) -> object:
    return soundfile.read(io.BytesIO(blobs[0]), dtype="int16")


def _image(
    label: str, encode: Callable, decode: Callable, **kwargs
) -> Coder:
    return Coder(
        label,
        lambda res: [encode(t, **kwargs) for t in _tiles(res)],
        lambda blobs, res: [decode(b) for b in blobs])


CODERS: dict[str, Coder] = {
    "zstd3": _zstd(3),
    "zstd_m1": _zstd(-1),
    "lz4": Coder(
        "lz4",
        lambda res: [lz4.frame.compress(_residuals(res))],
        lambda blobs, res: lz4.frame.decompress(blobs[0])),
    "lzma": Coder(
        "lzma",
        lambda res: [lzma.compress(_residuals(res))],
        lambda blobs, res: lzma.decompress(blobs[0])),
    "pcodec": Coder(
        "pcodec",
        lambda res: [standalone.simple_compress(
            _series(res).reshape(-1), ChunkConfig())],
        lambda blobs, res: standalone.simple_decompress(blobs[0])),
    "flac": Coder("FLAC, level 8", _flac_encode, _flac_decode),
    "jpegls": _image(
        "JPEG-LS", imagecodecs.jpegls_encode, imagecodecs.jpegls_decode),
    "jpeg2k": _image(
        "JPEG 2000", imagecodecs.jpeg2k_encode, imagecodecs.jpeg2k_decode,
        level=0, reversible=True, numthreads=1),
    "jpegxl": _image(
        "JPEG-XL", imagecodecs.jpegxl_encode, imagecodecs.jpegxl_decode,
        lossless=True, numthreads=1),
    "htj2k": _image(
        "HTJ2K", imagecodecs.htj2k_encode, imagecodecs.htj2k_decode,
        reversible=True),
}
"""Implemented coders, keyed by result column name."""


def _rice_bits(res: MeanResidual, block: int = 512) -> float:
    """Rice coding with the best parameter `k` per block of values."""
    u = res.u.reshape(-1, block).astype(np.int64)
    best = np.full(u.shape[0], np.inf)
    for k in range(16):
        bits = (u >> k).sum(axis=1) + u.shape[1] * (k + 1)
        best = np.minimum(best, bits)
    return float(best.sum() + 4 * u.shape[0])


def _bitpack_bits(res: MeanResidual, block: int = 128) -> float:
    """Bit-packing with the minimum width per block of values."""
    u = res.u.reshape(-1, block)
    width = np.ceil(np.log2(u.max(axis=1).astype(np.float64) + 1))
    return float((width * block).sum() + 5 * u.shape[0])


def _entropy_bits(res: MeanResidual) -> float:
    """Order-0 entropy of the 16-bit residual symbols."""
    counts = np.bincount(res.u.reshape(-1), minlength=65536)
    p = counts[counts > 0] / res.u.size
    return float(-(p * np.log2(p)).sum() * res.u.size)


ESTIMATES: dict[str, tuple[str, Callable[[MeanResidual], float]]] = {
    "rice": ("Rice coding, best parameter per 512 values (estimate)",
             _rice_bits),
    "bitpack": ("Bit-packing, width per 128 values (estimate)",
                _bitpack_bits),
    "entropy0": ("Order-0 entropy: best symbol-by-symbol coder (bound)",
                 _entropy_bits),
}
"""Size estimates (bits), keyed by result column name."""
