"""Measure groups of variants over the shared sample."""

import os
from functools import partial
from typing import Literal

from ..datasets import Dataset, get_datasets
from ..sampling import map_chunks
from ..variants import GROUPS
from ._common import load_sample

Group = Literal["core", "predictors", "layouts", "entropy", "levels"]


def _measure(group: str, dataset: Dataset, path: str, k: int) -> dict:
    return GROUPS[group](dataset.chunk(path, k))


def cli_measure(
    tartanstar: str | None = None, iq1m: str | None = None,
    out: str = "results",
    group: tuple[Group, ...] = ("core", "predictors", "layouts", "entropy",
                                "levels"),
    workers: int = 32,
) -> None:
    """Measure compressed sizes of variant groups on every sampled chunk.

    Each group writes `measure_<group>_<dataset>.csv`, with one column of
    compressed sizes (bytes) per variant:

    - `core`: raw size, zstd alone, previous-chirp delta, and iqz.
    - `predictors`: no prediction, and fast-time, antenna, and LPC
      predictors on top of the mean residual.
    - `layouts`: axis orders, and separate zstd frames per antenna / I/Q.
    - `entropy`: other entropy coders in place of zstd (and estimates).
    - `levels`: iqz at zstd levels 1-9.

    Args:
        tartanstar: Tartan* root (extracted traces); skipped if omitted.
        iq1m: I/Q-1M root; skipped if omitted.
        out: results directory (must contain the sample).
        group: variant groups to measure.
        workers: worker processes.
    """
    for ds in get_datasets(tartanstar, iq1m):
        sample = load_sample(ds, out)
        for g in group:
            result = map_chunks(
                ds, sample, partial(_measure, g), workers, desc=g)
            result.to_csv(
                os.path.join(out, f"measure_{g}_{ds.name}.csv"), index=False)
