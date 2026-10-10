"""Reference test vectors: blobs produced by the original prototype.

Each vector is a reference encoding (`tests/data/<name>.iqz`) of one chunk;
`manifest.json` records the SHA-256 of the original int16 data, so exact
decoding is checked without storing the (larger) original.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import zstandard

import iqz
from iqz import _api

DATA = Path(__file__).parent / "data"
MANIFEST = json.loads((DATA / "manifest.json").read_text())
VECTORS = {v["name"]: v for v in MANIFEST["vectors"]}


def _zstd_version(backend: str) -> str:
    if backend == "native":
        return _api._native.ZSTD_VERSION  # type: ignore
    return ".".join(map(str, zstandard.ZSTD_VERSION))


@pytest.fixture(params=sorted(VECTORS))
def vector(request) -> tuple[dict, bytes]:
    """Each reference vector: its manifest entry, and the encoded blob."""
    v = VECTORS[request.param]
    return v, (DATA / f"{v['name']}.iqz").read_bytes()


def test_decode_reference(backend, vector):
    """Reference blobs decode exactly to the original data."""
    v, blob = vector
    assert hashlib.sha256(blob).hexdigest() == v["encoded_sha256"]
    x = iqz.decode(blob, tuple(v["shape"][1:]), backend=backend)
    assert x.shape == tuple(v["shape"])
    assert hashlib.sha256(x.tobytes()).hexdigest() == v["raw_sha256"]


def test_encode_reference(backend, vector):
    """Re-encoding round trips, and reproduces the reference blob exactly
    when using the same libzstd version."""
    v, blob = vector
    x = iqz.decode(blob, tuple(v["shape"][1:]), backend=backend)
    encoded = iqz.encode(x, backend=backend)
    assert np.array_equal(iqz.decode(encoded, x.shape[1:], backend=backend), x)
    if _zstd_version(backend) == MANIFEST["libzstd"]:
        assert encoded == blob
    else:
        assert abs(len(encoded) / len(blob) - 1) < 0.01
