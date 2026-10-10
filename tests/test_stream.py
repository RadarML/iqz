"""Streaming API: Encoder and Decoder."""

import numpy as np
import pytest
from conftest import chirps

import iqz

SHAPE = (3, 4, 16)


def _encode_stream(x, pushes, window, backend):
    encoder = iqz.Encoder(window, backend=backend)
    blobs, i = [], 0
    for k in pushes:
        blobs.extend(encoder.push(x[i:i + k]))
        i += k
        assert encoder.pending == i % window
    return blobs + encoder.flush()


@pytest.mark.parametrize("pushes", [[1] * 70, [7] * 30, [64] * 3, [200], [3, 64, 61, 200, 1]])
def test_encoder_rechunks(backend, pushes):
    """Any push pattern gives the same blobs as one `encode` per window."""
    x = chirps(sum(pushes), SHAPE)
    blobs = _encode_stream(x, pushes, 64, backend)
    expected = [iqz.encode(x[i:i + 64], backend=backend) for i in range(0, len(x), 64)]
    assert blobs == expected


def test_encoder_flush(backend):
    encoder = iqz.Encoder(16, SHAPE, backend=backend)
    x = chirps(20, SHAPE)
    assert len(encoder.push(x)) == 1
    tail = encoder.flush()
    assert len(tail) == 1 and iqz.info(tail[0]) == (4, 3 * 4 * 16)
    assert encoder.flush() == []
    assert encoder.pending == 0


def test_encoder_input_reuse(backend):
    """Buffered chirps are copied, so callers can reuse their arrays."""
    encoder = iqz.Encoder(4, backend=backend)
    x = chirps(4, SHAPE)
    buf = x[:2].copy()
    encoder.push(buf)
    buf[:] = x[2:]
    (blob,) = encoder.push(buf)
    assert np.array_equal(iqz.decode(blob, SHAPE, backend=backend), x)


def test_encoder_buffer(backend):
    x = chirps(10, SHAPE)
    encoder = iqz.Encoder(4, SHAPE, backend=backend)
    blobs = encoder.push(x[:5].tobytes()) + encoder.push(x[5:]) + encoder.flush()
    assert blobs == [iqz.encode(x[i:i + 4], backend=backend) for i in (0, 4, 8)]


def test_encoder_errors(backend):
    x = chirps(4, SHAPE)
    with pytest.raises(ValueError, match="window"):
        iqz.Encoder(0, backend=backend)
    with pytest.raises(ValueError, match="required"):
        iqz.Encoder(backend=backend).push(x.tobytes())
    encoder = iqz.Encoder(backend=backend)
    encoder.push(x)
    assert encoder.shape == SHAPE
    with pytest.raises(ValueError, match="does not match"):
        encoder.push(chirps(4, (3, 4, 8)))
    with pytest.raises(TypeError):
        encoder.push(x.astype(np.float32))


@pytest.mark.parametrize("reads", [[1] * 50, [16] * 10, [100, 60], [5, 64, 1, 90]])
def test_decoder_reads(backend, reads):
    """Reads of any size cross chunk boundaries correctly."""
    x = chirps(160, SHAPE)
    decoder = iqz.Decoder(SHAPE, backend=backend)
    for i in range(0, 160, 64):
        decoder.push(iqz.encode(x[i:i + 64], backend=backend))
    assert decoder.available == 160

    i = 0
    for k in reads:
        k = min(k, 160 - i)
        y = decoder.read(k)
        assert y.shape == (k, *SHAPE)
        assert np.array_equal(y, x[i:i + k])
        i += k
    assert np.array_equal(decoder.read(), x[i:])
    assert decoder.available == 0


def test_decoder_bytes(backend):
    x = chirps(40, SHAPE)
    decoder = iqz.Decoder(backend=backend)
    decoder.push(iqz.encode(x[:13], backend=backend))
    decoder.push(iqz.encode(x[13:], backend=backend))
    assert decoder.shape == (3 * 4 * 16,)
    a = decoder.read(20, format="bytes")
    assert isinstance(a, bytes) and a == x[:20].tobytes()
    assert np.array_equal(decoder.read(5).reshape(5, *SHAPE), x[20:25])
    assert decoder.read(format="bytes") == x[25:].tobytes()
    assert decoder.read(format="bytes") == b""


def test_decoder_errors(backend):
    decoder = iqz.Decoder(backend=backend)
    decoder.push(iqz.encode(chirps(4, SHAPE), backend=backend))
    with pytest.raises(ValueError, match="format"):
        decoder.read(format="other")  # type: ignore
    assert decoder.available == 4
    with pytest.raises(ValueError, match="available"):
        decoder.read(5)
    with pytest.raises(ValueError, match="does not match"):
        decoder.push(iqz.encode(chirps(4, (8,)), backend=backend))


def test_stream_roundtrip(backend):
    """Encoder -> Decoder round-trips a long stream with mismatched batching."""
    rng = np.random.default_rng(0)
    x = chirps(1000, SHAPE)
    encoder = iqz.Encoder(64, backend=backend)
    decoder = iqz.Decoder(SHAPE, backend=backend)
    out, i = [], 0
    while i < len(x):
        k = int(rng.integers(1, 100))
        for blob in encoder.push(x[i:i + k]):
            decoder.push(blob)
        i += k
        while decoder.available >= 33:
            out.append(decoder.read(33))
    for blob in encoder.flush():
        decoder.push(blob)
    out.append(decoder.read())
    assert np.array_equal(np.concatenate(out), x)


def _blobs(x, n, backend):
    return [iqz.encode(x[i:i + n], backend=backend)
            for i in range(0, len(x), n)]


@pytest.mark.parametrize("k", [1, 7, 64, 100, 128])
def test_decoder_read_into(backend, k):
    """`read_into` refills one buffer, across and within chunk boundaries."""
    x = chirps(64 * 5 + 30, SHAPE)
    decoder = iqz.Decoder(SHAPE, backend=backend)
    for blob in _blobs(x, 64, backend):
        decoder.push(blob)
    out = np.empty((k, *SHAPE), np.int16)
    for i in range(0, len(x) - k + 1, k):
        assert decoder.read_into(out) is out
        assert np.array_equal(out, x[i:i + k])
    assert decoder.available == len(x) % k


def test_decoder_read_into_errors(backend):
    x = chirps(64, SHAPE)
    decoder = iqz.Decoder(SHAPE, backend=backend)
    decoder.push(iqz.encode(x, backend=backend))
    with pytest.raises(ValueError, match="available"):
        decoder.read_into(np.empty((65, *SHAPE), np.int16))
    with pytest.raises(ValueError, match="does not match"):
        decoder.read_into(np.empty((4, 3, 4, 8), np.int16))
    with pytest.raises(TypeError):
        decoder.read_into(np.empty((4, *SHAPE), np.int32))
    with pytest.raises(ValueError, match="C-contiguous"):
        decoder.read_into(np.empty((8, *SHAPE), np.int16)[::2])
    assert decoder.available == 64
    assert np.array_equal(decoder.read(), x)


def test_decoder_views_not_overwritten(backend):
    """Views returned by `read` (or derived from them) are never reused."""
    x = chirps(64 * 6, SHAPE)
    blobs = _blobs(x, 64, backend)
    decoder = iqz.Decoder(SHAPE, backend=backend)
    decoder.push(blobs[0])
    whole = decoder.read(64)
    decoder.push(blobs[1])
    derived = decoder.read(64)[5:9, 1]  # parent view is dropped immediately
    for blob in blobs[2:]:
        decoder.push(blob)
        decoder.read(64)
    assert np.array_equal(whole, x[:64])
    assert np.array_equal(derived, x[64 + 5:64 + 9, 1])


def test_decoder_reuse_mixed_sizes(backend):
    """Buffers are reused across chunks of different lengths."""
    x = chirps(400, SHAPE)
    decoder = iqz.Decoder(SHAPE, backend=backend)
    out, i = [], 0
    for n in [64, 10, 64, 64, 3, 128, 64, 3]:
        decoder.push(iqz.encode(x[i:i + n], backend=backend))
        i += n
        out.append(decoder.read(7 if decoder.available >= 7 else None).copy())
    out.append(decoder.read())
    assert np.array_equal(np.concatenate(out), x[:i])


def test_decoder_corrupt_blob(backend):
    """Header errors raise on push; corrupt data raises when read."""
    x = chirps(64, SHAPE)
    blob = iqz.encode(x, backend=backend)
    decoder = iqz.Decoder(SHAPE, backend=backend)
    with pytest.raises(ValueError):
        decoder.push(blob[:3])
    assert decoder.available == 0
    decoder.push(blob[:-5])
    with pytest.raises(ValueError):
        decoder.read()
