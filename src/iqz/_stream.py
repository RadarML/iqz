"""Streaming API: chirps in arbitrary batches, re-chunked into windows."""

import math
from collections import deque
from typing import Literal, overload

import numpy as np
from jaxtyping import Int16

from . import _api
from ._format import Buffer


class Encoder:
    """Streaming encoder, which re-chunks chirps into fixed-size windows.

    Chirps can be pushed in batches of any size; each completed window of
    `window` chirps is emitted as one blob. Call [`flush`][.] at the end of
    the stream to encode any remaining chirps as a final, shorter chunk.

    ```python
    encoder = iqz.Encoder(window=64)
    blobs = []
    for frame in frames:  # e.g. (1, 3, 4, 1024) int16 each
        blobs.extend(encoder.push(frame))
    blobs.extend(encoder.flush())
    ```

    Args:
        window: number of chirps per encoded chunk.
        shape: per-chirp sample shape, excluding the chirps axis. If `None`,
            this is taken from the first array pushed; buffer input requires
            the shape to be known.
        level: zstd compression level.
        backend: implementation to use; if `None`, uses `native` when the
            extension is available, and `numpy` otherwise.
    """

    def __init__(
        self, window: int = 64, shape: tuple[int, ...] | None = None, *,
        level: int = 3, backend: Literal["native", "numpy"] | None = None
    ) -> None:
        if window < 1:
            raise ValueError(f"`window` must be positive; got {window}.")
        self._impl = _api._impl(backend)
        self.window = window
        self.level = level
        self._shape = None if shape is None else tuple(shape)
        self._stage: Int16[np.ndarray, "window M"] | None = None
        self._fill = 0

    @property
    def shape(self) -> tuple[int, ...] | None:
        """Per-chirp sample shape, or `None` if not yet known."""
        return self._shape

    @property
    def pending(self) -> int:
        """Number of chirps buffered, but not yet encoded."""
        return self._fill

    def _encode(self, x: Int16[np.ndarray, "n M"]) -> bytes:
        return self._impl.encode_chunk(x, self.level)

    def push(
        self, x: Int16[np.ndarray, "chirps *sample"] | Buffer
    ) -> list[bytes]:
        """Add chirps to the stream.

        Args:
            x: one or more int16 chirps, either as an array with a leading
                chirps axis, or as a buffer of little-endian int16 samples.
                The data is copied (or encoded) before returning, so the
                caller may reuse it.

        Returns:
            Blobs for each window completed by this push (possibly none).

        Raises:
            TypeError: if `x` is an array, but not int16.
            ValueError: if `x` does not match the stream's shape, or is a
                buffer pushed before the shape is known.
        """
        flat, shape = _api._as_chunk(x, self._shape)
        if self._stage is None:
            self._shape = shape
            self._stage = np.empty(
                (self.window, math.prod(shape)), dtype=np.int16)
        stage = self._stage

        blobs = []
        i, k = 0, flat.shape[0]
        if self._fill > 0:
            take = min(self.window - self._fill, k)
            stage[self._fill:self._fill + take] = flat[:take]
            self._fill += take
            i = take
            if self._fill == self.window:
                blobs.append(self._encode(stage))
                self._fill = 0

        while k - i >= self.window:
            blobs.append(self._encode(flat[i:i + self.window]))
            i += self.window

        if i < k:
            stage[:k - i] = flat[i:]
            self._fill = k - i
        return blobs

    def flush(self) -> list[bytes]:
        """Encode any buffered chirps as a final (shorter) chunk.

        The encoder can be reused for a new stream (with the same shape)
        afterwards.

        Returns:
            One blob if any chirps were pending; otherwise, an empty list.
        """
        if self._fill == 0 or self._stage is None:
            return []
        blob = self._encode(self._stage[:self._fill])
        self._fill = 0
        return [blob]


class Decoder:
    """Streaming decoder, which re-chunks decoded chirps into any size.

    Push blobs with [`push`][.], each containing any number of chirps, and
    read chirps back out with [`read`][.] in batches of any size, regardless
    of where the chunk boundaries fall.

    ```python
    decoder = iqz.Decoder(shape=(3, 4, 1024))
    for blob in blobs:
        decoder.push(blob)
        while decoder.available >= 16:
            x = decoder.read(16)  # (16, 3, 4, 1024) int16
    ```

    Args:
        shape: per-chirp sample shape, excluding the chirps axis. If `None`,
            chirps are flattened to `(M,)`, where `M` is taken from the first
            blob.
        backend: implementation to use; if `None`, uses `native` when the
            extension is available, and `numpy` otherwise.
    """

    def __init__(
        self, shape: tuple[int, ...] | None = None, *,
        backend: Literal["native", "numpy"] | None = None
    ) -> None:
        _api._impl(backend)
        self.backend: Literal["native", "numpy"] | None = backend
        self._shape = None if shape is None else tuple(shape)
        self._chunks: deque[Int16[np.ndarray, "n *sample"]] = deque()
        self._offset = 0
        self._available = 0

    @property
    def shape(self) -> tuple[int, ...] | None:
        """Per-chirp sample shape, or `None` if not yet known."""
        return self._shape

    @property
    def available(self) -> int:
        """Number of decoded chirps which have not yet been read."""
        return self._available

    def push(self, blob: Buffer) -> None:
        """Decode a blob, and buffer its chirps.

        Args:
            blob: encoded chunk with any number of chirps.

        Raises:
            ValueError: if the blob is malformed, or does not match the
                stream's shape.
        """
        chunk = _api.decode(blob, self._shape, backend=self.backend)
        if self._shape is None:
            self._shape = chunk.shape[1:]
        self._chunks.append(chunk)
        self._available += chunk.shape[0]

    @overload
    def read(
        self, k: int | None = None, *, format: Literal["numpy"] = "numpy"
    ) -> Int16[np.ndarray, "chirps *sample"]: ...

    @overload
    def read(
        self, k: int | None = None, *, format: Literal["bytes"]
    ) -> bytes: ...

    def read(
        self, k: int | None = None, *,
        format: Literal["numpy", "bytes"] = "numpy"
    ) -> Int16[np.ndarray, "chirps *sample"] | bytes:
        """Read decoded chirps.

        Args:
            k: number of chirps to read; if `None`, read all available.
            format: `numpy` to return an int16 array of shape `(k, *shape)`,
                or `bytes` to return the raw little-endian int16 samples.

        Returns:
            `k` chirps. Array reads which fall within a single chunk are
                zero-copy views.

        Raises:
            ValueError: if fewer than `k` chirps are available.
        """
        if format not in ("numpy", "bytes"):
            raise ValueError(f"Unknown format: {format!r}")
        count = self._available if k is None else k
        if count < 0 or count > self._available:
            raise ValueError(
                f"Cannot read {count} chirps; {self._available} available.")

        pieces = []
        need = count
        while need > 0:
            head = self._chunks[0]
            take = min(head.shape[0] - self._offset, need)
            pieces.append(head[self._offset:self._offset + take])
            self._offset += take
            need -= take
            if self._offset == head.shape[0]:
                self._chunks.popleft()
                self._offset = 0
        self._available -= count

        if len(pieces) == 1:
            out = pieces[0]
        elif len(pieces) > 1:
            out = np.concatenate(pieces)
        else:
            out = np.empty((0, *(self._shape or (0,))), dtype=np.int16)
        return out if format == "numpy" else out.tobytes()
