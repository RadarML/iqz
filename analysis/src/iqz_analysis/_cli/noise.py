"""Noise floor and the resulting limit on lossless compression."""

import os

from ..datasets import Dataset, get_datasets
from ..noise import noise_floor
from ..sampling import map_chunks
from ._common import load_sample


def _noise(dataset: Dataset, path: str, k: int) -> dict:
    return noise_floor(dataset.chunk(path, k))


def cli_noise(
    tartanstar: str | None = None, iq1m: str | None = None,
    out: str = "results", workers: int = 32,
) -> None:
    """Estimate the thermal noise floor of every sampled chunk.

    Noise is estimated per virtual channel and range bin from a low quantile
    of the power across Doppler bins, where scene content is sparse (see
    `iqz_analysis.noise`); this does not require a static scene.
    Writes `noise_<dataset>.csv`.

    Args:
        tartanstar: Tartan* root (extracted traces); skipped if omitted.
        iq1m: I/Q-1M root; skipped if omitted.
        out: results directory (must contain the sample).
        workers: worker processes.
    """
    for ds in get_datasets(tartanstar, iq1m):
        result = map_chunks(ds, load_sample(ds, out), _noise, workers,
                            desc="noise")
        result.to_csv(os.path.join(out, f"noise_{ds.name}.csv"), index=False)
