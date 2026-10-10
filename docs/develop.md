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

## Layout

| Path | Contents |
| --- | --- |
| `src/iqz/_api.py` | Functional API, input coercion, and backend dispatch. |
| `src/iqz/_stream.py` | `Encoder` and `Decoder`. |
| `src/iqz/_format.py` | Blob parsing (frame split, shape recovery). |
| `src/iqz/_numpy.py` | Pure numpy backend. |
| `rust/lib.rs` | Native backend. |

Both backends implement the same two primitives on a C-contiguous `(n, M)` int16 chunk: `encode_chunk(x, level) -> bytes` and `decode_into(blob, out)`. Both produce byte-identical blobs when linked against the same libzstd version.

## Tests

```sh
uv run pytest -ra --cov --cov-report=html --cov-report=term -- tests
```

Check both backends against the reference test vectors (not included in the repository):
```sh
uv run python scripts/check_vectors.py path/to/testvectors
```

Benchmark single-threaded encode and decode throughput on the same vectors:
```sh
uv run python scripts/bench.py path/to/testvectors
```

!!! info

    The pre-commit hooks (`ruff + pyright + pytest + cargo fmt + cargo clippy`) can be run manually with `uv run pre-commit run --all-files`.

## Docs

```sh
uv run --extra docs mkdocs serve
```
