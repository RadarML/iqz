"""Dataset adapters over the official Tartan* and I/Q-1M APIs.

Both datasets are read only through their own loaders (`tartanstar` and
`roverd`), so local copies in other formats (e.g. compressed channels) work
as long as the loader supports them. Everything downstream sees the same
interface: a trace survey, and int16 chunks of 64 chirps.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache
from typing import ClassVar

import numpy as np
import pandas as pd
import yaml
from jaxtyping import Int16

CHUNK = 64
"""Chirps per chunk."""


class Dataset(ABC):
    """A radar dataset, viewed as a sequence of 64-chirp chunks per trace."""

    name: ClassVar[str]
    """Short identifier, used in file names."""
    label: ClassVar[str]
    """Display name."""

    @abstractmethod
    def survey(self) -> pd.DataFrame:
        """List all traces.

        Returns:
            One row per trace, sorted by `trace`, with columns `trace`
                (display name), `path`, `n_chunks`, `group` (scene type or
                category), `motion`, and `location`.
        """
        ...

    @abstractmethod
    def chunk(self, path: str, k: int) -> Int16[np.ndarray, "64 tx rx fast"]:
        """Read chunk `k` of the trace at `path`."""
        ...

    @abstractmethod
    def valid(self, path: str, k: int) -> bool:
        """Whether every chirp in chunk `k` was received completely."""
        ...

    def span(
        self, path: str, k: int, n: int
    ) -> Int16[np.ndarray, "chirps tx rx fast"]:
        """Read `n` consecutive chunks starting at chunk `k`."""
        return np.concatenate([self.chunk(path, k + i) for i in range(n)])


@cache
def _tartan_radar(path: str):
    from tartanstar.sensors import Radar
    return Radar(os.path.join(path, "radar"), window=CHUNK, stride=CHUNK)


@dataclass(frozen=True)
class TartanStar(Dataset):
    """Tartan* (`tartanstar` API).

    Tartan* records one chirp loop per frame, continuously; a chunk is 64
    consecutive frames (`Radar(window=64, stride=64)`). Scene type, motion,
    and location labels are read from each trace's `meta.yaml`.

    Args:
        root: directory containing the extracted traces.
    """

    root: str
    name: ClassVar[str] = "tartanstar"
    label: ClassVar[str] = "Tartan*"

    def survey(self) -> pd.DataFrame:
        from tartanstar import Trace

        rows = []
        for path in sorted(Trace.find(self.root)):
            if not os.path.isdir(os.path.join(path, "radar")):
                continue
            with open(os.path.join(path, "meta.yaml")) as f:
                meta = (yaml.safe_load(f) or {}).get("meta", {})
            rows.append(dict(
                trace=os.path.basename(path), path=path,
                n_chunks=len(_tartan_radar(path)),
                group=meta.get("type", ""), motion=meta.get("motion", ""),
                location=meta.get("location", "")))
        return pd.DataFrame(rows).sort_values("trace", ignore_index=True)

    def chunk(self, path: str, k: int) -> Int16[np.ndarray, "64 tx rx fast"]:
        return np.asarray(_tartan_radar(path)[k].iq[0, 0])

    def valid(self, path: str, k: int) -> bool:
        return bool(_tartan_radar(path).metadata.complete[k])


@cache
def _iq1m_radar(path: str):
    from roverd.sensors import XWRRadar
    return XWRRadar(os.path.join(path, "radar"))


@cache
def _iq1m_valid(path: str) -> np.ndarray:
    return _iq1m_radar(path)["valid"].read()


@dataclass(frozen=True)
class IQ1M(Dataset):
    """I/Q-1M (`roverd` API).

    Each native frame is already a chunk of 64 chirps. Labels come from the
    trace path: the category (first path component), and for indoor traces,
    the motion (`.fwd` / `.lat` suffix of the trace name).

    Args:
        root: directory containing the dataset's categories.
        categories: categories to include; traces in any other top-level
            directory under `root` are ignored.
    """

    root: str
    categories: tuple[str, ...] = ("bike", "indoor", "outdoor")
    name: ClassVar[str] = "iq1m"
    label: ClassVar[str] = "I/Q-1M"

    def survey(self) -> pd.DataFrame:
        from roverd import Dataset as RoverDataset

        rows = []
        for path in sorted(RoverDataset.find_traces(self.root)):
            trace = os.path.relpath(path, self.root)
            group = trace.split(os.sep)[0]
            if group not in self.categories:
                continue
            # Some traces have more frames in `iq` than in their metadata;
            # only frames covered by both `ts` and `valid` are used.
            n = min(len(_iq1m_radar(path)), len(_iq1m_valid(path)))
            rows.append(dict(
                trace=trace, path=path, n_chunks=n, group=group,
                motion=trace.rsplit(".", 1)[-1] if group == "indoor" else "",
                location=""))
        return pd.DataFrame(rows).sort_values("trace", ignore_index=True)

    def chunk(self, path: str, k: int) -> Int16[np.ndarray, "64 tx rx fast"]:
        return _iq1m_radar(path)["iq"].read(start=k, samples=1)[0]

    def valid(self, path: str, k: int) -> bool:
        return bool(_iq1m_valid(path)[k])


def get_datasets(
    tartanstar: str | None, iq1m: str | None
) -> list[Dataset]:
    """Create adapters for the datasets whose roots are given."""
    out: list[Dataset] = []
    if tartanstar is not None:
        out.append(TartanStar(tartanstar))
    if iq1m is not None:
        out.append(IQ1M(iq1m))
    return out
