"""Check both backends against the reference test vectors.

Usage: `uv run python scripts/check_vectors.py [testvectors-dir]`

The directory (default `iqz-handoff/testvectors`) must contain
`manifest.json`, and a `<name>.npy` / `<name>.iqz` pair per vector. For each
vector and backend, checks that the stored blob decodes exactly, that the
round trip is exact, and compares the encoded size (and bytes) with the
reference encoding.
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np

import iqz

BACKENDS: list[Literal["native", "numpy"]] = ["numpy", "native"]


def main(root: Path) -> bool:
    """Check all vectors in `root`; returns whether all checks passed."""
    manifest = json.loads((root / "manifest.json").read_text())
    ok = True
    for v in manifest["vectors"]:
        x = np.load(root / f"{v['name']}.npy")
        blob = (root / f"{v['name']}.iqz").read_bytes()
        if hashlib.sha256(x.tobytes()).hexdigest() != v["raw_sha256"]:
            print(f"{v['name']}: raw data does not match manifest hash")
            ok = False
            continue

        sample = tuple(v["shape"][1:])
        for backend in BACKENDS:
            dec_ok = np.array_equal(
                iqz.decode(blob, sample, backend=backend), x)
            enc = iqz.encode(x, backend=backend)
            rt_ok = np.array_equal(
                iqz.decode(enc, sample, backend=backend), x)
            ok &= dec_ok and rt_ok
            print(
                f"{v['name']:28s} {backend:6s} "
                f"decode-stored {'OK' if dec_ok else 'FAIL':4s} "
                f"round-trip {'OK' if rt_ok else 'FAIL':4s} "
                f"size {len(enc):>9,d} B (reference "
                f"{v['encoded_bytes']:,d} B"
                f"{', identical' if enc == blob else ''})")
    return ok


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "iqz-handoff/testvectors")
    sys.exit(0 if main(root) else 1)
