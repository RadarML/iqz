"""Single-threaded encode / decode throughput."""

import json
import os
import platform
import time
from collections.abc import Callable

import iqz
import numpy as np
import pandas as pd
import zstandard as zstd

from ..coders import CODERS
from ..datasets import get_datasets
from ..transforms import MeanResidual
from ..variants import LEVELS
from ._common import load_sample


def _best(fn: Callable[[], object], repeats: int) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def _cpu() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def cli_speed(
    tartanstar: str | None = None, iq1m: str | None = None,
    out: str = "results", chunks: int = 200, seed: int = 0,
    repeats: int = 3,
) -> None:
    """Measure single-threaded encode and decode throughput.

    Loads `chunks` chunks from the sample into memory, then times (best of
    `repeats`, single thread, sequential) iqz with each backend, iqz at each
    zstd level, zstd on the raw bytes, and each entropy coder applied to the
    mean residual. iqz decodes into one reused output buffer, so that
    timings measure the codec rather than memory allocation. Throughput is
    in MB/s of raw data. Writes
    `speed_<dataset>.csv` and `speed_env.json` (CPU and library versions).

    !!! warning

        Run this on an otherwise idle machine.

    Args:
        tartanstar: Tartan* root (extracted traces); skipped if omitted.
        iq1m: I/Q-1M root; skipped if omitted.
        out: results directory (must contain the sample).
        chunks: number of chunks to time.
        seed: random seed for choosing chunks from the sample.
        repeats: timing repeats (the fastest is kept).
    """
    for ds in get_datasets(tartanstar, iq1m):
        sample = load_sample(ds, out).sample(chunks, random_state=seed)
        xs = [ds.chunk(p, int(k)) for p, k in zip(sample.path, sample.chunk)]
        raw = sum(x.nbytes for x in xs) / 1e6
        rows = []

        def add(kind: str, name: str, encode: Callable, decode: Callable):
            rows.append(dict(
                kind=kind, name=name,
                encode=round(raw / _best(encode, repeats), 1),
                decode=round(raw / _best(decode, repeats), 1)))
            print(f"  {ds.label} {kind} {name}: {rows[-1]['encode']:.0f} / "
                  f"{rows[-1]['decode']:.0f} MB/s")

        # Decode into one reused output buffer, so that timings measure the
        # codec rather than memory allocation (page faults).
        out_buf = np.empty_like(xs[0])

        for backend in ("native", "numpy"):
            blobs = [iqz.encode(x, backend=backend) for x in xs]
            add("iqz", backend,
                lambda b=backend: [iqz.encode(x, backend=b) for x in xs],
                lambda b=backend, bl=blobs: [
                    iqz.decode_into(y, out_buf, backend=b) for y in bl])

        for level in LEVELS:
            blobs = [iqz.encode(x, level=level) for x in xs]
            add("level", f"level{level}",
                lambda lv=level: [iqz.encode(x, level=lv) for x in xs],
                lambda bl=blobs: [iqz.decode_into(y, out_buf) for y in bl])

        cctx, dctx = zstd.ZstdCompressor(level=3), zstd.ZstdDecompressor()
        blobs = [cctx.compress(x.data) for x in xs]
        add("raw", "zstd",
            lambda: [cctx.compress(x.data) for x in xs],
            lambda: [dctx.decompress(y) for y in blobs])

        res = [MeanResidual(x) for x in xs]
        for r in res:
            r.u  # noqa: B018 -- precompute the transform; time coders only
        for name, coder in CODERS.items():
            blobs = [coder.encode(r) for r in res]
            add("coder", name,
                lambda c=coder: [c.encode(r) for r in res],
                lambda c=coder, bl=blobs: [
                    c.decode(y, r) for y, r in zip(bl, res)])

        pd.DataFrame(rows).to_csv(
            os.path.join(out, f"speed_{ds.name}.csv"), index=False)

    with open(os.path.join(out, "speed_env.json"), "w") as f:
        json.dump(dict(
            cpu=_cpu(), python=platform.python_version(),
            iqz_zstd=getattr(iqz._api._native, "ZSTD_VERSION", None),
            zstandard=zstd.__version__,
            libzstd=".".join(map(str, zstd.ZSTD_VERSION))), f, indent=1)
