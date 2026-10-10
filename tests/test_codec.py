"""Functional API: encode, decode, decode_into, info."""

import mmap
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import zstandard
from conftest import chirps, extremes

import iqz
from iqz import _format


@pytest.mark.parametrize("n", [1, 3, 37, 63, 64, 128])
@pytest.mark.parametrize("shape", [(), (6144,), (3, 4, 64)])
@pytest.mark.parametrize("gen", [chirps, extremes])
def test_roundtrip(backend, n, shape, gen):
    """Encode -> decode is exact, for any chunk length and sample shape."""
    x = gen(n, shape)
    blob = iqz.encode(x, backend=backend)
    y = iqz.decode(blob, shape, backend=backend)
    assert y.dtype == np.int16
    assert y.shape == x.shape
    assert np.array_equal(x, y)


@pytest.mark.parametrize("n", [1, 6, 37, 64])
def test_cross_backend(n):
    """Backends produce identical transforms, and decode each other's blobs."""
    x = extremes(n, (3, 4, 32), seed=n)
    native = iqz.encode(x, backend="native")
    numpy = iqz.encode(x, backend="numpy")

    dctx = zstandard.ZstdDecompressor()
    parts_n = _format.IQZBlob.from_buffer(native)
    parts_p = _format.IQZBlob.from_buffer(numpy)
    assert dctx.decompress(parts_n.mean) == dctx.decompress(parts_p.mean)
    assert dctx.decompress(parts_n.residuals) == dctx.decompress(parts_p.residuals)

    assert np.array_equal(iqz.decode(native, backend="numpy").reshape(x.shape), x)
    assert np.array_equal(iqz.decode(numpy, backend="native").reshape(x.shape), x)


def test_rounding_half_even(backend):
    """Mean rounding matches `np.round` (ties to even), including negatives."""
    x = np.array([[1, -1, 3, -3, 5], [2, -2, 4, -4, 6]], dtype=np.int16)
    blob = iqz.encode(x, backend=backend)
    mean = _format.IQZBlob.from_buffer(blob).mean
    m = np.frombuffer(zstandard.ZstdDecompressor().decompress(mean), np.int16)
    assert np.array_equal(m, np.round(x.mean(axis=0)).astype(np.int16))


def test_info(backend):
    x = chirps(37, (3, 4, 16))
    assert iqz.info(iqz.encode(x, backend=backend)) == (37, 3 * 4 * 16)


def test_level(backend):
    x = chirps(64, (1024,))
    blob = iqz.encode(x, level=1, backend=backend)
    assert np.array_equal(iqz.decode(blob, (1024,), backend=backend), x)


def test_encode_buffer(backend):
    """Buffers are accepted with a shape; the shape excludes chirps."""
    x = chirps(10, (4, 8))
    ref = iqz.encode(x, backend=backend)
    for buf in [x.tobytes(), memoryview(x.tobytes()), bytearray(x.tobytes())]:
        assert iqz.encode(buf, (4, 8), backend=backend) == ref

    with mmap.mmap(-1, x.nbytes) as mm:
        mm[:] = x.tobytes()
        assert iqz.encode(mm, (4, 8), backend=backend) == ref
        # Unaligned slice of a buffer
        unaligned = bytearray(1) + x.tobytes()
        assert iqz.encode(memoryview(unaligned)[1:], (4, 8), backend=backend) == ref


def test_encode_noncontiguous(backend):
    x = chirps(8, (16, 4))
    view = x[:, ::2]
    blob = iqz.encode(view, backend=backend)
    assert np.array_equal(iqz.decode(blob, (8, 4), backend=backend), view)


def test_encode_errors(backend):
    x = chirps(4, (8,))
    with pytest.raises(TypeError):
        iqz.encode(x.astype(np.int32), backend=backend)
    with pytest.raises(ValueError, match="does not match"):
        iqz.encode(x, (4,), backend=backend)
    with pytest.raises(ValueError, match="required"):
        iqz.encode(x.tobytes(), backend=backend)
    with pytest.raises(ValueError, match="multiple"):
        iqz.encode(x.tobytes()[:-2], (8,), backend=backend)
    with pytest.raises(ValueError, match="at least one"):
        iqz.encode(x[:0], backend=backend)
    with pytest.raises(ValueError, match="non-empty"):
        iqz.encode(np.zeros((4, 0), np.int16), backend=backend)
    with pytest.raises(ValueError, match="leading"):
        iqz.encode(np.zeros((), np.int16), backend=backend)
    with pytest.raises(ValueError, match="backend"):
        iqz.encode(x, backend="other")  # type: ignore


def test_decode_formats(backend):
    x = chirps(5, (2, 3))
    blob = iqz.encode(x, backend=backend)
    flat = iqz.decode(blob, backend=backend)
    assert flat.shape == (5, 6)
    raw = iqz.decode(blob, format="bytes", backend=backend)
    assert isinstance(raw, bytes) and raw == x.tobytes()
    with pytest.raises(ValueError, match="format"):
        iqz.decode(blob, format="other", backend=backend)  # type: ignore
    with pytest.raises(ValueError, match="does not match"):
        iqz.decode(blob, (4,), backend=backend)


def test_decode_buffer_input(backend):
    x = chirps(7, (16,))
    blob = iqz.encode(x, backend=backend)
    padded = b"xyz" + blob + b"abc"
    with mmap.mmap(-1, len(padded)) as mm:
        mm[:] = padded
        view = memoryview(mm)[3:3 + len(blob)]
        assert np.array_equal(iqz.decode(view, (16,), backend=backend), x)
        assert np.array_equal(
            iqz.decode(np.frombuffer(blob, np.uint8), (16,), backend=backend), x)
        del view


def test_decode_into(backend):
    x = chirps(6, (3, 4))
    blob = iqz.encode(x, backend=backend)

    out = np.empty((6, 3, 4), np.int16)
    assert iqz.decode_into(blob, out, backend=backend) is out
    assert np.array_equal(out, x)

    out_flat = np.empty(72, np.int16)
    iqz.decode_into(blob, out_flat, backend=backend)
    assert np.array_equal(out_flat, x.ravel())

    buf = bytearray(x.nbytes)
    mv = iqz.decode_into(blob, buf, backend=backend)
    assert isinstance(mv, memoryview)
    assert bytes(buf) == x.tobytes()

    unaligned = bytearray(x.nbytes + 1)
    iqz.decode_into(blob, memoryview(unaligned)[1:], backend=backend)
    assert bytes(unaligned[1:]) == x.tobytes()


def test_decode_into_errors(backend):
    x = chirps(6, (3, 4))
    blob = iqz.encode(x, backend=backend)
    with pytest.raises(TypeError):
        iqz.decode_into(blob, np.empty((6, 12), np.int32), backend=backend)
    with pytest.raises(ValueError, match="elements"):
        iqz.decode_into(blob, np.empty((5, 12), np.int16), backend=backend)
    with pytest.raises(ValueError, match="C-contiguous"):
        iqz.decode_into(blob, np.empty((12, 6), np.int16).T, backend=backend)
    readonly = np.empty((6, 12), np.int16)
    readonly.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        iqz.decode_into(blob, readonly, backend=backend)
    with pytest.raises(ValueError, match="writable"):
        iqz.decode_into(blob, bytes(x.nbytes), backend=backend)
    with pytest.raises(ValueError, match="bytes"):
        iqz.decode_into(blob, bytearray(x.nbytes + 2), backend=backend)


def test_malformed(backend):
    x = chirps(4, (8,))
    blob = iqz.encode(x, backend=backend)
    k = int.from_bytes(blob[:4], "little")
    cases = {
        "truncated header": blob[:3],
        "truncated mean": blob[:4 + k - 1],
        "bad magic": blob[:4] + b"\0" * 4 + blob[8:],
        "truncated residuals": blob[:-5],
        "swapped frames": (len(blob) - 4 - k).to_bytes(4, "little")
            + blob[4 + k:] + blob[4:4 + k],
    }
    for bad in cases.values():
        with pytest.raises(ValueError):
            iqz.decode(bad, backend=backend)


def test_threads(backend):
    """Concurrent decodes (GIL released in the native backend) are correct."""
    xs = [chirps(64, (3, 4, 128), seed=i) for i in range(16)]
    blobs = [iqz.encode(x, backend=backend) for x in xs]
    with ThreadPoolExecutor(8) as pool:
        ys = list(pool.map(
            lambda b: iqz.decode(b, (3, 4, 128), backend=backend), blobs * 4))
        zs = list(pool.map(lambda x: iqz.encode(x, backend=backend), xs * 4))
    assert all(np.array_equal(x, y) for x, y in zip(xs * 4, ys))
    assert zs == blobs * 4
