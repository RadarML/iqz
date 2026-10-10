"""Blob layout: `u32le len(A) | A = zstd(mean) | B = zstd(residuals)`."""

import sys
from dataclasses import dataclass

import zstandard as zstd

if sys.version_info >= (3, 12):
    from collections.abc import Buffer
else:
    from typing_extensions import Buffer


@dataclass(frozen=True, eq=False)
class IQZBlob:
    """Zero-copy view of an encoded chunk's parts.

    Attributes:
        chirps: number of chirps `n` in the chunk.
        samples: number of samples `M` per chirp.
        mean: zstd frame containing the mean chirp.
        residuals: zstd frame containing the residuals, zigzag-encoded and
            byte-shuffled (all high bytes, then all low bytes).
    """

    chirps: int
    samples: int
    mean: memoryview
    residuals: memoryview

    @staticmethod
    def _content_size(frame: memoryview, what: str) -> int:
        try:
            size = zstd.frame_content_size(frame)
        except zstd.ZstdError as e:
            raise ValueError(f"{what} frame is not a valid zstd frame") from e
        if size < 0:
            raise ValueError(f"{what} frame has no content size")
        return size

    @classmethod
    def from_buffer(cls, blob: Buffer) -> "IQZBlob":
        """Split a blob into its two zstd frames, and recover its shape.

        The shape is read from the frames' content sizes (`2M` and `2nM`
        bytes); nothing is decompressed.

        Args:
            blob: encoded chunk.

        Raises:
            ValueError: if the blob is truncated or malformed.
        """
        mv = memoryview(blob).cast("B")
        if len(mv) < 4:
            raise ValueError("blob is truncated")
        k = int.from_bytes(mv[:4], "little")
        if len(mv) < 4 + k:
            raise ValueError("blob is truncated")
        mean, residuals = mv[4:4 + k], mv[4 + k:]

        size_m = cls._content_size(mean, "mean")
        size_r = cls._content_size(residuals, "residual")
        if size_m == 0 or size_m % 2 != 0 or size_r == 0 or size_r % size_m:
            raise ValueError(
                f"inconsistent frame sizes: mean {size_m} B, "
                f"residuals {size_r} B")
        return cls(
            chirps=size_r // size_m, samples=size_m // 2,
            mean=mean, residuals=residuals)
