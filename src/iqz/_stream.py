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
    read chirps back out with [`read`][.] or [`read_into`][.] in batches of
    any size, regardless of where the chunk boundaries fall.

    ```python
    decoder = iqz.Decoder(shape=(3, 4, 1024))
    out = np.empty((16, 3, 4, 1024), dtype=np.int16)
    for blob in blobs:
        decoder.push(blob)
        while decoder.available >= 16:
            decoder.read_into(out)  # reuses `out`; no allocation
    ```

    Blobs are only decoded when read: chunks which fit entirely in a read
    are decoded directly into the output, and only chunks which are split
    between reads go through an internal buffer.

    !!! warning

        Pushed blobs are referenced, not copied, until they have been read;
        they must not be modified in the meantime.

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
        self._blobs: deque[tuple[Buffer, int]] = deque()
        self._available = 0
        # Decoded head chunk, if it was split by a previous read.
        self._scratch = np.empty(0, dtype=np.int16)
        self._head: Int16[np.ndarray, "n M"] | None = None
        self._offset = 0

    @property
    def shape(self) -> tuple[int, ...] | None:
        """Per-chirp sample shape, or `None` if not yet known."""
        return self._shape

    @property
    def available(self) -> int:
        """Number of chirps pushed which have not yet been read."""
        return self._available

    def push(self, blob: Buffer) -> None:
        """Queue a blob for reading.

        Only the blob's header is checked here; it is decoded when read.

        Args:
            blob: encoded chunk with any number of chirps.

        Raises:
            ValueError: if the blob's header is malformed, or does not match
                the stream's shape.
        """
        n, M = _api.info(blob)
        self._shape = _api._sample_shape(self._shape, M)
        self._blobs.append((blob, n))
        self._available += n

    def read_into(
        self, out: Int16[np.ndarray, "chirps *sample"]
    ) -> Int16[np.ndarray, "chirps *sample"]:
        """Read the next `len(out)` chirps into a buffer provided by the caller.

        Args:
            out: writable, C-contiguous int16 array of shape `(k, *shape)`.

        Returns:
            `out`.

        Raises:
            TypeError: if `out` is not an int16 array.
            ValueError: if fewer than `k` chirps are available, `out` has
                the wrong shape, or a blob fails to decode.
        """
        if not isinstance(out, np.ndarray) or out.dtype != np.int16:
            raise TypeError("Output must be an int16 array.")
        if out.ndim < 1 or out.shape[0] > self._available:
            raise ValueError(
                f"Cannot read {out.shape[0] if out.ndim else '?'} chirps; "
                f"{self._available} available.")
        if self._shape is not None and out.shape[1:] != self._shape:
            raise ValueError(
                f"Output shape {out.shape[1:]} (excluding chirps) does not "
                f"match the stream's shape {self._shape}.")
        if not out.flags.c_contiguous or not out.flags.writeable:
            raise ValueError("Output must be writable and C-contiguous.")

        k = out.shape[0]
        flat = out.reshape(k, -1)
        pos = 0
        while pos < k:
            blob, n = self._blobs[0]
            if self._head is None and n <= k - pos:
                # Whole chunk: decode straight into the output.
                _api.decode_into(
                    blob, flat[pos:pos + n], backend=self.backend)
                pos += n
                self._blobs.popleft()
                continue
            if self._head is None:
                M = flat.shape[1]
                if self._scratch.size < n * M:
                    self._scratch = np.empty(n * M, dtype=np.int16)
                head = self._scratch[:n * M].reshape(n, M)
                _api.decode_into(blob, head, backend=self.backend)
                self._head = head
            take = min(n - self._offset, k - pos)
            flat[pos:pos + take] = self._head[self._offset:self._offset + take]
            pos += take
            self._offset += take
            if self._offset == n:
                self._blobs.popleft()
                self._head = None
                self._offset = 0
        self._available -= k
        return out

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
        """Read decoded chirps into a new array.

        Allocates a new output for every read; use [`read_into`][^.] to
        reuse a buffer instead.

        Args:
            k: number of chirps to read; if `None`, read all available.
            format: `numpy` to return an int16 array of shape `(k, *shape)`,
                or `bytes` to return the raw little-endian int16 samples.

        Returns:
            `k` chirps.

        Raises:
            ValueError: if fewer than `k` chirps are available, or a blob
                fails to decode.
        """
        if format not in ("numpy", "bytes"):
            raise ValueError(f"Unknown format: {format!r}")
        count = self._available if k is None else k
        if count < 0 or count > self._available:
            raise ValueError(
                f"Cannot read {count} chirps; {self._available} available.")
        out = np.empty((count, *(self._shape or (0,))), dtype=np.int16)
        if count > 0:
            self.read_into(out)
        return out if format == "numpy" else out.tobytes()
