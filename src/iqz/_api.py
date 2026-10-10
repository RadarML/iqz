"""Functional API: one chunk in, one blob out (and back)."""

import math
from types import ModuleType
from typing import Literal, overload

import numpy as np
from jaxtyping import Int16, UInt8

from . import _format, _numpy
from ._format import Buffer

try:
    from . import _native
except ImportError:  # pragma: no cover
    _native = None


def _impl(backend: Literal["native", "numpy"] | None) -> ModuleType:
    if backend is None:
        return _numpy if _native is None else _native
    if backend == "native":
        if _native is None:
            raise ImportError("The native iqz backend is not available.")
        return _native
    if backend == "numpy":
        return _numpy
    raise ValueError(f"Unknown backend: {backend!r}")


def _as_chunk(
    x: Int16[np.ndarray, "chirps *sample"] | Buffer,
    shape: tuple[int, ...] | None,
) -> tuple[Int16[np.ndarray, "chirps M"], tuple[int, ...]]:
    """Coerce encoder input to an aligned, C-contiguous `(n, M)` view.

    Returns:
        The flattened chunk, and the per-chirp sample shape.
    """
    if isinstance(x, np.ndarray):
        if x.dtype != np.int16:
            raise TypeError(f"Expected an int16 array; got {x.dtype}.")
        if x.ndim < 1:
            raise ValueError("Input must have a leading chirps axis.")
        if shape is not None and tuple(shape) != x.shape[1:]:
            raise ValueError(
                f"Input shape {x.shape[1:]} (excluding chirps) does not "
                f"match the specified shape {tuple(shape)}.")
        shape = x.shape[1:]
    else:
        if shape is None:
            raise ValueError("`shape` is required for buffer input.")
        shape = tuple(shape)
        if math.prod(shape) == 0:
            raise ValueError(f"Chirps must be non-empty; got shape {shape}.")
        mv = memoryview(x)
        if not mv.c_contiguous:
            raise ValueError("Buffer input must be C-contiguous.")
        mv = mv.cast("B")
        chirp_bytes = 2 * math.prod(shape)
        if mv.nbytes % chirp_bytes != 0:
            raise ValueError(
                f"Buffer size ({mv.nbytes} B) is not a multiple of the "
                f"chirp size ({chirp_bytes} B for shape {shape}).")
        x = np.frombuffer(mv, dtype=np.int16).reshape(-1, *shape)

    M = math.prod(shape)
    if M == 0:
        raise ValueError(f"Chirps must be non-empty; got shape {shape}.")
    if x.shape[0] == 0:
        raise ValueError("Input must contain at least one chirp.")
    flat = np.require(x.reshape(x.shape[0], M), requirements=["C", "A"])
    return flat, shape


def _blob_array(blob: Buffer) -> UInt8[np.ndarray, "nbytes"]:
    return np.frombuffer(memoryview(blob).cast("B"), dtype=np.uint8)


def _sample_shape(
    shape: tuple[int, ...] | None, samples: int
) -> tuple[int, ...]:
    if shape is None:
        return (samples,)
    shape = tuple(shape)
    if math.prod(shape) != samples:
        raise ValueError(
            f"Blob has {samples} samples per chirp, which does not match the "
            f"specified shape {shape}.")
    return shape


def _decode(
    blob: Buffer, out: Int16[np.ndarray, "n M"],
    backend: Literal["native", "numpy"] | None
) -> None:
    impl = _impl(backend)
    if out.flags.aligned:
        impl.decode_into(_blob_array(blob), out)
    else:
        tmp = np.empty(out.shape, dtype=np.int16)
        impl.decode_into(_blob_array(blob), tmp)
        out[...] = tmp


def info(blob: Buffer) -> tuple[int, int]:
    """Get the shape of an encoded chunk.

    The shape is recovered from the content sizes recorded in the blob's two
    zstd frame headers; nothing is decompressed.

    Args:
        blob: encoded chunk.

    Returns:
        `(n, M)`: the number of chirps, and the number of samples per chirp.

    Raises:
        ValueError: if the blob is malformed.
    """
    parts = _format.IQZBlob.from_buffer(blob)
    return parts.chirps, parts.samples


def encode(
    x: Int16[np.ndarray, "chirps *sample"] | Buffer,
    shape: tuple[int, ...] | None = None, *,
    level: int = 3, backend: Literal["native", "numpy"] | None = None
) -> bytes:
    """Encode one chunk of chirps.

    !!! info

        The codec is designed for chunks of 64 chirps; any `n >= 1` works,
        though shorter chunks compress worse.

    Args:
        x: int16 chirps, either as an array with a leading chirps axis, or as
            any buffer of little-endian int16 samples (with `shape`).
        shape: per-chirp sample shape, excluding the leading chirps axis. If
            `x` is an array, this is optional, and is checked against
            `x.shape[1:]` when provided. Required if `x` is a buffer.
        level: zstd compression level.
        backend: implementation to use; if `None`, uses `native` when the
            extension is available, and `numpy` otherwise.

    Returns:
        Encoded blob.

    Raises:
        TypeError: if `x` is an array, but not int16.
        ValueError: if `x` is empty, its shape does not match `shape`, or
            `shape` is missing for buffer input.
    """
    flat, _ = _as_chunk(x, shape)
    return _impl(backend).encode_chunk(flat, level)


@overload
def decode(
    blob: Buffer, shape: tuple[int, ...] | None = None, *,
    format: Literal["numpy"] = "numpy",
    backend: Literal["native", "numpy"] | None = None
) -> Int16[np.ndarray, "chirps *sample"]: ...

@overload
def decode(
    blob: Buffer, shape: tuple[int, ...] | None = None, *,
    format: Literal["bytes"],
    backend: Literal["native", "numpy"] | None = None
) -> bytes: ...

def decode(
    blob: Buffer, shape: tuple[int, ...] | None = None, *,
    format: Literal["numpy", "bytes"] = "numpy",
    backend: Literal["native", "numpy"] | None = None
) -> Int16[np.ndarray, "chirps *sample"] | bytes:
    """Decode one chunk.

    Args:
        blob: encoded chunk; any buffer, e.g. `bytes`, `memoryview`, or a
            slice of an `mmap`.
        shape: per-chirp sample shape, excluding the chirps axis. If `None`,
            chirps are returned flattened, with shape `(M,)`.
        format: `numpy` to return an int16 array of shape `(n, *shape)`, or
            `bytes` to return the raw little-endian int16 samples.
        backend: implementation to use; if `None`, uses `native` when the
            extension is available, and `numpy` otherwise.

    Returns:
        Decoded chirps.

    Raises:
        ValueError: if the blob is malformed, or does not match `shape`.
    """
    if format not in ("numpy", "bytes"):
        raise ValueError(f"Unknown format: {format!r}")
    n, M = info(blob)
    out = np.empty((n, *_sample_shape(shape, M)), dtype=np.int16)
    _decode(blob, out.reshape(n, M), backend)
    return out if format == "numpy" else out.tobytes()


@overload
def decode_into(  # pyright: ignore[reportOverlappingOverload]
    blob: Buffer, out: Int16[np.ndarray, "chirps *sample"], *,
    backend: Literal["native", "numpy"] | None = None
) -> Int16[np.ndarray, "chirps *sample"]: ...

@overload
def decode_into(
    blob: Buffer, out: Buffer, *,
    backend: Literal["native", "numpy"] | None = None
) -> memoryview: ...

def decode_into(
    blob: Buffer, out: Int16[np.ndarray, "chirps *sample"] | Buffer, *,
    backend: Literal["native", "numpy"] | None = None
) -> Int16[np.ndarray, "chirps *sample"] | memoryview:
    """Decode one chunk into a buffer provided by the caller.

    Args:
        blob: encoded chunk; any buffer, e.g. `bytes`, `memoryview`, or a
            slice of an `mmap`.
        out: destination; either a writable, C-contiguous int16 array with
            exactly `n * M` elements (of any shape), or any writable,
            C-contiguous buffer of exactly `2 * n * M` bytes, which receives
            little-endian int16 samples.
        backend: implementation to use; if `None`, uses `native` when the
            extension is available, and `numpy` otherwise.

    Returns:
        `out` itself if it is an array; otherwise, a byte `memoryview` of
            `out`.

    Raises:
        TypeError: if `out` is an array, but not int16.
        ValueError: if the blob is malformed, or `out` has the wrong size,
            is read-only, or is not C-contiguous.
    """
    n, M = info(blob)
    if isinstance(out, np.ndarray):
        if out.dtype != np.int16:
            raise TypeError(f"Expected an int16 array; got {out.dtype}.")
        if out.size != n * M:
            raise ValueError(
                f"Output has {out.size} elements; the blob decodes to "
                f"{n} x {M} = {n * M}.")
        if not out.flags.c_contiguous or not out.flags.writeable:
            raise ValueError("Output must be writable and C-contiguous.")
        _decode(blob, out.reshape(n, M), backend)
        return out
    else:
        mv = memoryview(out)
        if mv.readonly or not mv.c_contiguous:
            raise ValueError("Output must be writable and C-contiguous.")
        mv = mv.cast("B")
        if mv.nbytes != 2 * n * M:
            raise ValueError(
                f"Output has {mv.nbytes} bytes; the blob decodes to "
                f"{n} x {M} x 2 = {2 * n * M}.")
        _decode(blob, np.frombuffer(mv, np.int16).reshape(n, M), backend)
        return mv
