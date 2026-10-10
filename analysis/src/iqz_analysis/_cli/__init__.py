"""iqz-analysis CLI entry point."""

from typing import Annotated, Any, Union

import tyro

from .all import cli_all
from .measure import cli_measure
from .noise import cli_noise
from .render import cli_render
from .sample import cli_sample
from .speed import cli_speed
from .stats import cli_stats
from .sweep import cli_sweep

_DESC = "Reproducible measurements behind the iqz codec design."


def make_annotation(name, func):
    return Annotated[
        Any,
        tyro.conf.subcommand(
            name, description=func.__doc__.split("\n")[0],
            constructor=func
        )
    ]


def cli_main() -> None:
    commands = {
        "all": cli_all,
        "measure": cli_measure,
        "noise": cli_noise,
        "render": cli_render,
        "sample": cli_sample,
        "speed": cli_speed,
        "stats": cli_stats,
        "sweep": cli_sweep,
    }

    return tyro.cli(Union[  # type: ignore
        tuple(make_annotation(k, commands[k]) for k in sorted(commands.keys()))
    ], description=_DESC)
