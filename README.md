# IQZ: Lossless I/Q Radar Compression

A simple lossless compression codec for I/Q radar data: subtract each chunk's mean chirp, zigzag and byte-shuffle the residuals, and compress with zstd. It gets 2.31x on Tartan\* and 1.87x on I/Q-1M (vs. 1.62x and 1.42x for zstd alone), and decodes at ~3.5 GB/s per core with the native (Rust) backend.

```sh
pip install git+https://github.com/RadarML/iqz.git
```

```python
import iqz

blob = iqz.encode(x)                        # x: (64, 3, 4, 1024) int16
y = iqz.decode(blob, shape=(3, 4, 1024))    # shape excludes the chirps axis
```

See the [documentation](https://radarml.github.io/iqz) for details.
