"""Helpers shared by subcommands."""

import os

import pandas as pd

from ..datasets import Dataset


def load_sample(dataset: Dataset, results: str) -> pd.DataFrame:
    """Load a dataset's sample, with trace paths resolved under its root."""
    sample = pd.read_csv(os.path.join(results, f"sample_{dataset.name}.csv"))
    sample["path"] = [os.path.join(dataset.root, t) for t in sample.trace]
    return sample


def load_traces(dataset: Dataset, results: str) -> pd.DataFrame:
    """Load a dataset's trace survey."""
    return pd.read_csv(
        os.path.join(results, f"traces_{dataset.name}.csv"),
        keep_default_na=False)
