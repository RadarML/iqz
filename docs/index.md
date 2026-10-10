# IQZ: Lossless I/Q Radar Compression

`iqz` is a simple, fast, lossless compression codec for raw (I/Q) mmWave radar data.

Each chunk of chirps (typically 64) is encoded by subtracting the chunk's rounded mean chirp, which removes static clutter and TX-RX leakage; zigzag-encoding and byte-shuffling the residuals; and compressing with zstd. See the [format specification](format.md) for details.

| | Tartan* | I/Q-1M |
| --- | --- | --- |
| zstd-3 on raw bytes | 1.62x | 1.42x |
| **iqz** | **2.31x** | **1.87x** |
| Decode, 1 core (native) | ~3.7 GB/s | ~3.2 GB/s |

## Install

```sh
pip install git+https://github.com/RadarML/iqz.git
```

!!! info

    `iqz` includes a native (Rust) backend, so installing from source requires a [Rust toolchain](https://rustup.rs/). A pure numpy backend (using `zstandard`) is also included as a reference implementation.

## Usage

Encode and decode single chunks:
```python
import iqz

x = ...  # (64, 3, 4, 1024) int16
blob = iqz.encode(x)
y = iqz.decode(blob, shape=(3, 4, 1024))  # shape excludes the chirps axis
```

Encode a stream of chirps, pushed in batches of any size:
```python
encoder = iqz.Encoder(window=64)
blobs = []
for frame in frames:
    blobs.extend(encoder.push(frame))
blobs.extend(encoder.flush())
```

Decode a stream of blobs, and read chirps in batches of any size:
```python
decoder = iqz.Decoder(shape=(3, 4, 1024))
for blob in blobs:
    decoder.push(blob)
    while decoder.available >= 16:
        x = decoder.read(16)
```
