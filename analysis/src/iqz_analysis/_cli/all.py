"""Run the full pipeline."""

from .measure import cli_measure
from .noise import cli_noise
from .render import cli_render
from .sample import cli_sample
from .speed import cli_speed
from .stats import cli_stats
from .sweep import cli_sweep


def cli_all(
    tartanstar: str | None = None, iq1m: str | None = None,
    out: str = "results", docs: str = "../docs/analysis", workers: int = 32,
) -> None:
    r"""Run every step: sample, measure, sweep, speed, noise, stats, render.

    ```sh
    uv run iqz-analysis all --tartanstar /data1/tartanstar/extracted \
        --iq1m /data0/iq1m --workers 64
    ```

    Args:
        tartanstar: Tartan* root (extracted traces); skipped if omitted.
        iq1m: I/Q-1M root; skipped if omitted.
        out: results directory.
        docs: output directory for rendered figures and tables.
        workers: worker processes.
    """
    cli_sample(tartanstar, iq1m, out=out, workers=workers)
    cli_measure(tartanstar, iq1m, out=out, workers=workers)
    cli_sweep(tartanstar, iq1m, out=out, workers=workers)
    cli_speed(tartanstar, iq1m, out=out)
    cli_noise(tartanstar, iq1m, out=out, workers=workers)
    cli_stats(out=out)
    cli_render(out=out, docs=docs)
