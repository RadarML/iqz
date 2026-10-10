"""Summary statistics over all measurements."""

import json
import os

from ..datasets import IQ1M, TartanStar
from ..stats import summarize


def cli_stats(out: str = "results") -> None:
    """Compute ratios and bootstrap confidence intervals.

    Reads the per-chunk measurement CSVs, and writes `summary.json`: ratios
    per variant and group with 95% trace-level bootstrap intervals, gains,
    win rates, per-chunk histograms, dataset totals, and the chunk-size
    sweep. Does not need access to the datasets.

    Args:
        out: results directory.
    """
    summary = {}
    for ds in (TartanStar, IQ1M):
        if os.path.exists(os.path.join(out, f"sample_{ds.name}.csv")):
            summary[ds.name] = summarize(out, ds.name, ds.label)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
