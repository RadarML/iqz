# Format Specification

An `iqz` blob encodes one **chunk** of `n` consecutive chirps, where each chirp is a vector of `M` int16 samples, flattened in their on-disk order. No reordering is applied; in particular, the TI `IIQQ` interleave is left as is. All arithmetic is wrapping 16-bit integer arithmetic.

## Encoding

Let `x[s, k]` be sample `k` of chirp `s`, for `0 <= s < n` and `0 <= k < M`.

1. **Mean chirp**: `m[k] = round(mean_s x[s, k])`, stored as int16. The reference implementations round half to even (`np.round`).
2. **Residual**: `r[s, k] = int16(x[s, k] - m[k])`, wrapping on overflow.
3. **Zigzag**: `u = uint16((r << 1) ^ (r >> 15))`, with an arithmetic right shift. This maps `0, -1, 1, -2, 2, ...` to `0, 1, 2, 3, 4, ...`.
4. **Byte shuffle**: `planes = [hi(u[0]), ..., hi(u[N-1]), lo(u[0]), ..., lo(u[N-1])]`, where `N = n * M` and `u` is flattened in C order (chirp-major).
5. **Blob**: `u32le len(A) | A | B`, where `A = zstd(m as int16 LE bytes)` and `B = zstd(planes)`. Both are standard zstd frames (level 3 by default), which must record their content size in the frame header.

!!! info

    The decoder does not depend on how `m` is chosen: any int16 vector gives an exact round trip, and the rounding mode only affects the compressed size (negligibly).

## Decoding

Decompress `A` and `B`; then, for each element,
```
x = int16(m[k] + unzigzag((hi << 8) | lo)),    unzigzag(u) = (u >> 1) ^ -(u & 1).
```

Each element is independent, so this loop vectorizes.

## Shape

`M` and `n` are recovered from the content sizes of the two frames, which are `2M` and `2nM` bytes; see [`iqz.info`][iqz.info]. Only the per-chirp sample shape (e.g. `(3, 4, 1024)` for 3 TX, 4 RX, and 1024 fast-time samples) needs to be stored elsewhere.

The blob has no magic number, version, or checksum; integrity checks are left to the container (e.g. per-file checksums).
