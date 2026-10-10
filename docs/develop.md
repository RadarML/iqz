# Development Environment

!!! info "`iqz` uses `uv` and `maturin`"

    `iqz` is a mixed Python / Rust project, built with [maturin](https://www.maturin.rs/). You will need [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and a [Rust toolchain](https://rustup.rs/):
    ```sh
    curl -LsSf https://astral.sh/uv/install.sh | sh
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y -c clippy,rustfmt
    ```

## Setup

```sh
uv sync --all-extras
uv run pre-commit install
```

`uv sync` builds the native extension (`iqz._native`), and rebuilds it whenever `Cargo.toml` or `rust/*.rs` changes.

## Tests

```sh
uv run pytest -ra --cov --cov-report=html --cov-report=term -- tests
```

The tests include reference test vectors (`tests/data/`): blobs encoded by the original prototype, with the SHA-256 of the original data in `manifest.json`. Both backends must decode them exactly, and reproduce them byte for byte when built against the same libzstd version.

Benchmark single-threaded encode and decode throughput on the same vectors:
```sh
uv run python scripts/bench.py
```

!!! info

    The pre-commit hooks (`ruff + pyright + pytest + cargo fmt + cargo clippy`) can be run manually with `uv run pre-commit run --all-files`.

## Analysis

The measurements on the [Analysis](analysis/index.md) page are produced by a separate project in `analysis/`, with its own dependencies; see `analysis/README.md`. Its outputs (`analysis/results/`, and the figures and tables in `docs/analysis/`) are committed, so building the docs does not require the datasets.

## Docs

```sh
uv run --extra docs mkdocs serve
```
