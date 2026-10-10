# IQZ: Lossless I/Q Radar Compression

[![pypi version](https://img.shields.io/pypi/v/iqz.svg)](https://pypi.org/project/iqz/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/iqz)
![License - MIT](https://img.shields.io/badge/license-MIT-green)
![PyPI - Types](https://img.shields.io/pypi/types/iqz)
[![CI](https://github.com/RadarML/iqz/actions/workflows/ci.yml/badge.svg)](https://github.com/RadarML/iqz/actions/workflows/ci.yml)
![GitHub issues](https://img.shields.io/github/issues/RadarML/iqz)

A simple lossless compression codec for I/Q radar data: subtract each chunk's mean chirp, zigzag and byte-shuffle the residuals, and compress with zstd. It gets 2.31x on Tartan\* and 1.84x on I/Q-1M (vs. 1.62x and 1.43x for zstd alone), and decodes at ~3 GB/s per core with the native (Rust) backend.

```sh
pip install iqz
```

```python
import iqz

blob = iqz.encode(x)                        # x: (64, 3, 4, 1024) int16
y = iqz.decode(blob, shape=(3, 4, 1024))    # shape excludes the chirps axis
```

See the [documentation](https://radarml.github.io/iqz) for details.
