"""A simple lossless compression codec for I/Q radar data.

Each chunk of `n` chirps (int16, any per-chirp shape) is encoded by
subtracting the chunk's rounded mean chirp, zigzag-encoding and
byte-shuffling the residuals, and compressing with zstd; see the
[format specification](index.md#format) for details.

- Functional API: [`encode`][.], [`decode`][.], [`decode_into`][.],
  [`info`][.].
- Streaming API: [`Encoder`][.] and [`Decoder`][.], which re-chunk chirps
  pushed or read in batches of any size.

Two implementations are provided, and can be selected with the `backend`
argument of each function or class: `native` (Rust) and `numpy`. Both
produce and accept identical blobs.
"""

from ._api import decode, decode_into, encode, info
from ._stream import Decoder, Encoder

__all__ = [
    "encode", "decode", "decode_into", "info", "Encoder", "Decoder",
]
