"""Compression ratios with trace-level bootstrap confidence intervals."""

import glob
import os

import numpy as np
import pandas as pd

from .noise import bits

B = 5000
"""Bootstrap resamples."""

GROUPS = {
    "tartanstar": [
        ("Scene type", "group",
         ["outdoor", "park", "offroad", "night", "mixed", "indoor"]),
        ("Motion", "motion", ["forward", "backward", "lateral", "mixed"]),
    ],
    "iq1m": [
        ("Category", "group", ["indoor", "outdoor", "bike"]),
        ("Indoor motion", "motion", ["fwd", "lat"]),
    ],
}
"""Breakdowns reported per dataset: (heading, column, values)."""

LABELS = {"fwd": "Forward", "lat": "Lateral"}

LOW_TAIL = {"tartanstar": 2.0, "iq1m": 1.6}
"""Per-chunk ratio below which the share of indoor chunks is reported."""

HIST_BINS = np.round(np.arange(1.0, 3.5 + 1e-9, 0.05), 2)


def load(results: str, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the survey and the merged per-chunk measurements.

    Returns:
        The trace survey, and one row per sampled chunk with labels and all
            measured sizes (plus `best_of`, the smaller of delta and iqz).
    """
    traces = pd.read_csv(
        os.path.join(results, f"traces_{dataset}.csv"), keep_default_na=False)
    df = pd.read_csv(os.path.join(results, f"sample_{dataset}.csv"))
    for path in sorted(glob.glob(
            os.path.join(results, f"measure_*_{dataset}.csv"))):
        m = pd.read_csv(path)
        m = m.drop(columns=[c for c in m.columns
                            if c in df.columns and c not in ("trace", "chunk")])
        df = df.merge(m, on=["trace", "chunk"], how="left")
    df = df.merge(traces[["trace", "group", "motion"]], on="trace")
    if "delta" in df and "iqz" in df:
        df["best_of"] = np.minimum(df["delta"], df["iqz"])
    return traces, df


class Bootstrap:
    """Trace-level bootstrap of ratios of sums.

    Whole traces are resampled with replacement, since chunks from the same
    trace are correlated.

    Args:
        df: per-chunk rows with a `trace` column.
        seed: random seed.
    """

    def __init__(self, df: pd.DataFrame, seed: int = 2) -> None:
        self.sums = df.groupby("trace").sum(numeric_only=True)
        rng = np.random.default_rng(seed)
        self.idx = rng.integers(0, len(self.sums), (B, len(self.sums)))

    def ci(self, num: str, den: str) -> list[float]:
        """95% interval of `sum(num) / sum(den)`."""
        a = self.sums[num].to_numpy()[self.idx].sum(axis=1)
        b = self.sums[den].to_numpy()[self.idx].sum(axis=1)
        return np.percentile(a / b, [2.5, 97.5]).round(4).tolist()


def chunk_ci(
    df: pd.DataFrame, num: str, den: str, seed: int = 2
) -> list[float]:
    """95% chunk-level bootstrap interval of `sum(num) / sum(den)`."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(df), (B, len(df)))
    a = df[num].to_numpy()[idx].sum(axis=1)
    b = df[den].to_numpy()[idx].sum(axis=1)
    return np.percentile(a / b, [2.5, 97.5]).round(4).tolist()


def _ratio(df: pd.DataFrame, num: str, den: str) -> float:
    return round(float(df[num].sum() / df[den].sum()), 4)


def _group_row(
    df: pd.DataFrame, label: str, group: str
) -> dict:
    bs = Bootstrap(df)
    row: dict = dict(
        label=label, group=group,
        traces=int(df.trace.nunique()), chunks=len(df))
    for col in ("zstd", "delta", "iqz"):
        row[col] = dict(ratio=_ratio(df, "raw", col), ci=bs.ci("raw", col))
    row["iqz"]["chunk_ci"] = chunk_ci(df, "raw", "iqz")
    row["gain"] = dict(
        ratio=_ratio(df, "delta", "iqz"), ci=bs.ci("delta", "iqz"))
    row["iqz_wins"] = round(float((df.iqz < df.delta).mean()), 4)
    return row


def _noise_row(df: pd.DataFrame, label: str, group: str) -> dict:
    bs = Bootstrap(df)
    return dict(
        label=label, group=group, traces=int(df.trace.nunique()),
        sigma=round(float(df.sigma.median()), 2),
        contamination=round(float((df.sigma_p50 / df.sigma).median()), 3),
        bound_bits=round(float(8 * df.bound.sum() / df.n_values.sum()), 3),
        iqz_bits=round(float(8 * df.iqz.sum() / df.n_values.sum()), 3),
        cap=dict(ratio=_ratio(df, "raw", "bound"), ci=bs.ci("raw", "bound")),
        iqz=_ratio(df, "raw", "iqz"))


def _noise(df: pd.DataFrame, dataset: str) -> dict:
    """Noise floor, and the compression limit it implies, by group.

    The limit for each chunk is the size of its thermal noise alone, at
    `bits(sigma)` per value; the dataset limit is the ratio of total sizes.
    """
    df = df.assign(n_values=df.raw / 2)
    df = df.assign(bound=df.n_values * bits(df.sigma) / 8)
    rows = [_noise_row(df, "All traces", "all")]
    for heading, col, values in GROUPS[dataset]:
        for v in values:
            s = df[df[col] == v]
            if s.trace.nunique() >= 3:
                rows.append(_noise_row(
                    s, LABELS.get(v, v.capitalize()), heading))
    ch = df.sigma_ch_max / df.sigma_ch_min
    return dict(
        groups=rows,
        channel_ratio=np.percentile(ch, [50, 95]).round(2).tolist(),
        min_margin_bits=round(float(
            (8 * df.iqz / df.n_values - bits(df.sigma)).min()), 3))


def summarize(results: str, dataset: str, label: str) -> dict:
    """Compute all statistics for one dataset.

    Args:
        results: directory with the measurement CSVs.
        dataset: dataset name (file suffix).
        label: display name.
    """
    traces, df = load(results, dataset)
    raw_per_chunk = int(df.raw.iloc[0])
    raw_tb = float(traces.n_chunks.sum() * raw_per_chunk / 1e12)
    out: dict = dict(
        label=label, traces=len(traces),
        chunks_total=int(traces.n_chunks.sum()),
        raw_tb=round(raw_tb, 3), sample=len(df),
        sampled_traces=int(df.trace.nunique()))

    rows = [_group_row(df, "All traces", "all")]
    for heading, col, values in GROUPS[dataset]:
        for v in values:
            s = df[df[col] == v]
            if s.trace.nunique() >= 3:
                rows.append(_group_row(
                    s, LABELS.get(v, v.capitalize()), heading))
    out["groups"] = rows
    out["tb"] = {c: round(raw_tb / rows[0][c]["ratio"], 3)
                 for c in ("zstd", "delta", "iqz")}

    ratio = df.raw / df.iqz
    counts, _ = np.histogram(ratio, bins=HIST_BINS)
    low = ratio < LOW_TAIL[dataset]
    out["hist"] = dict(
        bins=HIST_BINS.tolist(), counts=counts.tolist(),
        pct=np.percentile(ratio, [2.5, 50, 97.5]).round(3).tolist(),
        low_tail=dict(
            threshold=LOW_TAIL[dataset], chunks=int(low.sum()),
            indoor_share=round(float((df.group[low] == "indoor").mean()), 3)
            if low.any() else None))

    bs = Bootstrap(df)
    variants = {}
    skip = {"trace", "chunk", "valid", "raw", "group", "motion"}
    for col in df.columns:
        if col in skip or not np.issubdtype(df[col].dtype, np.number):
            continue
        variants[col] = dict(
            ratio=_ratio(df, "raw", col), ci=bs.ci("raw", col),
            rel=round(_ratio(df, "iqz", col) - 1, 4),
            rel_ci=[round(x - 1, 4) for x in bs.ci("iqz", col)])
    out["variants"] = variants

    noise_path = os.path.join(results, f"noise_{dataset}.csv")
    if os.path.exists(noise_path):
        out["noise"] = _noise(df.merge(pd.read_csv(noise_path)), dataset)

    sweep_path = os.path.join(results, f"sweep_{dataset}.csv")
    if os.path.exists(sweep_path):
        sw = pd.read_csv(sweep_path)
        sizes = sorted(int(c[4:]) for c in sw.columns if c.startswith("mean"))
        out["sweep"] = {
            "windows": len(sw),
            "sizes": sizes,
            **{kind: {
                str(n): dict(
                    ratio=_ratio(sw, "raw", f"{kind}{n}"),
                    rel=round(_ratio(sw, f"{kind}64", f"{kind}{n}") - 1, 4),
                    rel_ci=[round(x - 1, 4) for x in chunk_ci(
                        sw, f"{kind}64", f"{kind}{n}")])
                for n in sizes} for kind in ("mean", "delta")}}
    return out
