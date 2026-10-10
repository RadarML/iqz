"""Chunk-size sweep on paired windows of 256 chirps."""

import os

import numpy as np

from ..datasets import Dataset, get_datasets
from ..sampling import map_chunks
from ..variants import SWEEP_SIZES, sweep
from ._common import load_sample, load_traces

SPAN = max(SWEEP_SIZES) // 64


def _sweep(dataset: Dataset, path: str, k: int) -> dict:
    return sweep(dataset.span(path, k, SPAN))


def cli_sweep(
    tartanstar: str | None = None, iq1m: str | None = None,
    out: str = "results", windows: int = 1000, seed: int = 7,
    workers: int = 32,
) -> None:
    """Compare chunk sizes on the same 256 chirps.

    For each of `windows` sampled chunks, takes the 256 chirps starting there
    (moved back if needed to stay inside the trace), and measures the mean
    (iqz) and delta codecs at chunk sizes 16, 32, 64, 128, and 256. Writes
    `sweep_<dataset>.csv`.

    Args:
        tartanstar: Tartan* root (extracted traces); skipped if omitted.
        iq1m: I/Q-1M root; skipped if omitted.
        out: results directory (must contain the sample).
        windows: number of windows.
        seed: random seed for choosing windows from the sample.
        workers: worker processes.
    """
    for ds in get_datasets(tartanstar, iq1m):
        sample = load_sample(ds, out).sample(windows, random_state=seed)
        sample = sample.sort_values(["trace", "chunk"], ignore_index=True)
        n = load_traces(ds, out).set_index("trace").n_chunks
        sample["chunk"] = np.minimum(
            sample.chunk, n[sample.trace].to_numpy() - SPAN)
        result = map_chunks(ds, sample, _sweep, workers, desc="sweep")
        result.to_csv(os.path.join(out, f"sweep_{ds.name}.csv"), index=False)
