"""Single-threaded encode / decode throughput, per backend.

Usage: `uv run python scripts/bench.py [testvectors-dir]`

Uses the `.npy` chunks in the test vector directory (default
`iqz-handoff/testvectors`); reports MB/s of raw (decoded) data.
"""

import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np

import iqz

BACKENDS: list[Literal["native", "numpy"]] = ["numpy", "native"]


def _rate(fn, nbytes: int, min_time: float = 1.0) -> float:
    fn()
    reps, start = 0, time.perf_counter()
    while (elapsed := time.perf_counter() - start) < min_time:
        fn()
        reps += 1
    return nbytes * reps / elapsed / 1e6


def _bench(name: str, x: np.ndarray, backend: Literal["native", "numpy"]) -> None:
    blob = iqz.encode(x, backend=backend)
    out = np.empty_like(x)
    enc = _rate(lambda: iqz.encode(x, backend=backend), x.nbytes)
    dec = _rate(lambda: iqz.decode(blob, backend=backend), x.nbytes)
    into = _rate(
        lambda: iqz.decode_into(blob, out, backend=backend), x.nbytes)
    print(f"{name:28s} {backend:7s} {x.nbytes / len(blob):6.3f} "
          f"{enc:7.0f} MB/s {dec:7.0f} MB/s {into:7.0f} MB/s")


def main(root: Path) -> None:
    """Benchmark each `.npy` chunk in `root`."""
    print(f"{'chunk':28s} {'backend':7s} {'ratio':>6s} "
          f"{'encode':>12s} {'decode':>12s} {'decode_into':>12s}")
    for path in sorted(root.glob("*.npy")):
        x = np.load(path)
        for backend in BACKENDS:
            _bench(path.stem, x, backend)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "iqz-handoff/testvectors"))
