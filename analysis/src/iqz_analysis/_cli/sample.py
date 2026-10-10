"""Survey both datasets and draw the shared 10k-chunk sample."""

import os

from ..datasets import Dataset, get_datasets
from ..sampling import draw, map_chunks


def _valid(dataset: Dataset, path: str, k: int) -> dict:
    return {"valid": dataset.valid(path, k)}


def cli_sample(
    tartanstar: str | None = None, iq1m: str | None = None,
    out: str = "results", n: int = 10_000, seed: int = 0,
    workers: int = 32,
) -> None:
    r"""Survey the datasets and draw the uniform random chunk sample.

    Writes `traces_<dataset>.csv` (one row per trace: chunk count and labels)
    and `sample_<dataset>.csv` (sampled trace, chunk index, and whether all
    of its chirps are complete). Every later measurement runs on this sample.

    ```sh
    uv run iqz-analysis sample --tartanstar /data1/tartanstar/extracted \
        --iq1m /data0/iq1m --workers 64
    ```

    Args:
        tartanstar: Tartan* root (extracted traces); skipped if omitted.
        iq1m: I/Q-1M root; skipped if omitted.
        out: results directory.
        n: chunks to sample per dataset.
        seed: random seed.
        workers: worker processes.
    """
    os.makedirs(out, exist_ok=True)
    for ds in get_datasets(tartanstar, iq1m):
        survey = ds.survey()
        survey.drop(columns="path").to_csv(
            os.path.join(out, f"traces_{ds.name}.csv"), index=False)
        print(f"{ds.label}: {len(survey)} traces, "
              f"{survey.n_chunks.sum()} chunks")
        sample = draw(survey, n=n, seed=seed)
        valid = map_chunks(ds, sample, _valid, workers, desc="valid")
        valid.to_csv(os.path.join(out, f"sample_{ds.name}.csv"), index=False)
