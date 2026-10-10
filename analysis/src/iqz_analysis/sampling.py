"""Shared sampling infrastructure: survey, 10k-chunk sample, parallel map."""

import os
import sys
from collections.abc import Callable
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .datasets import Dataset


def draw(survey: pd.DataFrame, n: int = 10_000, seed: int = 0) -> pd.DataFrame:
    """Draw `n` chunks uniformly at random (without replacement).

    Every chunk in the dataset is equally likely, so traces are represented
    in proportion to their length. With the same survey (traces sorted by
    name), this reproduces the original design study's sample exactly.

    Args:
        survey: output of [`Dataset.survey`][iqz_analysis.datasets.].
        n: number of chunks.
        seed: random seed.

    Returns:
        Sampled `trace`, `path`, and `chunk` index, sorted.
    """
    counts = survey["n_chunks"].to_numpy()
    rng = np.random.default_rng(seed)
    g = np.sort(rng.choice(counts.sum(), size=n, replace=False))
    edges = np.concatenate([[0], np.cumsum(counts)])
    ti = np.searchsorted(edges, g, side="right") - 1
    return pd.DataFrame({
        "trace": survey["trace"].to_numpy()[ti],
        "path": survey["path"].to_numpy()[ti],
        "chunk": g - edges[ti],
    })


def _init_worker() -> None:
    # Measurements are parallelized across processes; keep each single
    # threaded so that the pool does not oversubscribe the machine.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def _apply(
    fn: Callable[[Dataset, str, int], dict], dataset: Dataset,
    job: tuple[str, int]
) -> dict:
    return fn(dataset, *job)


def map_chunks(
    dataset: Dataset, sample: pd.DataFrame,
    fn: Callable[[Dataset, str, int], dict], workers: int,
    desc: str = ""
) -> pd.DataFrame:
    """Apply `fn(dataset, path, chunk)` to every sampled chunk, in parallel.

    Args:
        dataset: dataset to read chunks from.
        sample: rows with `trace`, `path`, and `chunk`.
        fn: picklable (top-level) function returning a flat dict of results.
        workers: number of worker processes.
        desc: progress label.

    Returns:
        `trace`, `chunk`, and the columns returned by `fn`, in sample order.
    """
    jobs = list(zip(sample["path"], sample["chunk"].astype(int)))
    results = []
    with Pool(workers, initializer=_init_worker) as pool:
        it = pool.imap(partial(_apply, fn, dataset), jobs, chunksize=2)
        for i, r in enumerate(it):
            results.append(r)
            if (i + 1) % 500 == 0 or i + 1 == len(jobs):
                print(f"  {desc} {dataset.label}: {i + 1}/{len(jobs)}",
                      file=sys.stderr, flush=True)
    out = pd.DataFrame(results)
    out.insert(0, "chunk", sample["chunk"].to_numpy())
    out.insert(0, "trace", sample["trace"].to_numpy())
    return out
