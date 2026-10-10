"""Render figures and tables."""


def cli_render(out: str = "results", docs: str = "../docs/analysis") -> None:
    """Render figures and markdown tables from the results.

    Args:
        out: results directory.
        docs: output directory for figures and tables.
    """
    from ..render import render
    render(out, docs)
