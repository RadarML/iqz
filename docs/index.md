# `iqz`: Lossless I/Q Radar Compression

`iqz` is a simple, fast, lossless compression codec for raw (I/Q) mmWave radar data which achieves modest compression ratios at fast speeds suitable for on-device encoding and in-dataloader decoding.

| | Tartan* | I/Q-1M |
| --- | --- | --- |
| zstd-3 on raw bytes | 1.62x | 1.43x |
| **iqz** | **2.31x** | **1.84x** |
| Decode, 1 core (native) | ~3.4 GB/s | ~2.9 GB/s |

Each chunk of chirps (typically 64) is encoded by subtracting the chunk's rounded mean chirp, which removes static clutter and TX-RX leakage; zigzag-encoding and byte-shuffling the residuals; and compressing with zstd.

![Encode and decode pipeline](analysis/figures/pipeline-light.svg#only-light)
![Encode and decode pipeline](analysis/figures/pipeline-dark.svg#only-dark)

## Usage

### Install

```sh
pip install git+https://github.com/RadarML/iqz.git
```

!!! info

    `iqz` includes a native (Rust) backend, so installing from source requires a [Rust toolchain](https://rustup.rs/). A pure numpy backend (using `zstandard`) is also included as a reference implementation.

### Basic Usage

=== "Encode and decode single chunks"

    ```python
    import iqz

    x = ...  # (64, 3, 4, 1024) int16
    blob = iqz.encode(x)
    y = iqz.decode(blob, shape=(3, 4, 1024))  # shape excludes the chirps axis
    ```

=== "Encode a stream of chirps"

    ```python
    encoder = iqz.Encoder(window=64)
    blobs = []
    for frame in frames:
        blobs.extend(encoder.push(frame))
    blobs.extend(encoder.flush())
    ```

=== "Decode a stream of blobs"

    ```python
    decoder = iqz.Decoder(shape=(3, 4, 1024))
    for blob in blobs:
        decoder.push(blob)
        while decoder.available >= 16:
            x = decoder.read(16)
    ```

### Performance Tuning

`decode` and `Decoder.read` return a new array on every call. Freshly allocated memory may have to be page-faulted in on first write, which can cost up to 40% of the decode throughput on chunks of a few MB, depending on the state of the memory allocator. For the full speed, decode into buffers which you allocate once and reuse:

- [`iqz.decode_into`][iqz.decode_into] decodes one chunk into an existing array, e.g. a slice of a preallocated batch:
    ```python
    batch = np.empty((8, 64, 3, 4, 1024), dtype=np.int16)
    for i, blob in enumerate(blobs[:8]):
        iqz.decode_into(blob, batch[i])
    ```
- [`Decoder.read_into`][iqz.Decoder.read_into] reads the next chirps of a stream into an existing array; chunks which fit entirely in the read are decoded directly into it.
    ```python
    out = np.empty((16, 3, 4, 1024), dtype=np.int16)
    while decoder.available >= 16:
        decoder.read_into(out)
    ```

The native backend releases the GIL while encoding and decoding, so multiple threads (e.g. dataloader workers) can decode in parallel.

## Format

An `iqz` blob encodes one **chunk** of `n` consecutive chirps, where each chirp is a vector of `M` int16 samples, flattened in their on-disk order. No reordering is applied; in particular, the TI `IIQQ` interleave is left as is. All arithmetic is wrapping 16-bit integer arithmetic.

### Encoding

Let `x[s, k]` be sample `k` of chirp `s`, for `0 <= s < n` and `0 <= k < M`.

1. **Mean chirp**: `m[k] = round(mean_s x[s, k])`, stored as int16. The reference implementations round half to even (`np.round`).
2. **Residual**: `r[s, k] = int16(x[s, k] - m[k])`, wrapping on overflow.
3. **Zigzag**: `u = uint16((r << 1) ^ (r >> 15))`, with an arithmetic right shift. This maps `0, -1, 1, -2, 2, ...` to `0, 1, 2, 3, 4, ...`.
4. **Byte shuffle**: `residuals = [hi(u[0]), ..., hi(u[N-1]), lo(u[0]), ..., lo(u[N-1])]`, where `N = n * M` and `u` is flattened in C order (chirp-major).
5. **Blob**: `u32le len(A) | A | B`, where `A = zstd(m as int16 LE bytes)` and `B = zstd(residuals)`. Both are standard zstd frames (level 3 by default), which must record their content size in the frame header.

!!! info

    The decoder does not depend on how `m` is chosen: any int16 vector gives an exact round trip, and the rounding mode only affects the compressed size (negligibly).

### Decoding

Decompress `A` and `B`; then, for each element,
```
x = int16(m[k] + unzigzag((hi << 8) | lo)),    unzigzag(u) = (u >> 1) ^ -(u & 1).
```

Each element is independent, so this loop vectorizes.

### Shape

`M` and `n` are recovered from the content sizes of the two frames, which are `2M` and `2nM` bytes; see [`iqz.info`][iqz.info]. Only the per-chirp sample shape (e.g. `(3, 4, 1024)` for 3 TX, 4 RX, and 1024 fast-time samples) needs to be stored elsewhere.

The blob has no magic number, version, or checksum; integrity checks are left to the container (e.g. per-file checksums).

## See Also

<div class="grid cards" markdown>

- :material-antenna: [`xwr`](https://radarml.github.io/xwr/)

    ---

    python interface for collecting raw time signal data from TI mmWave radars

- :material-golf-cart: [`red-rover`](https://radarml.github.io/red-rover/)

    ---

    multimodal mmWave radar data collection platform and dataset toolkit

- :material-database: [`i/q-1m`](https://radarml.github.io/red-rover/iq1m/)

    ---

    one million i/q frames across indoor, outdoor, and bike-mounted settings

</div>
