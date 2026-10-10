| Coder | Tartan\* ratio | vs. iqz | entropy decode MB/s | I/Q-1M ratio | vs. iqz | entropy decode MB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| zstd level 3, on the raw bytes (no transform) | 1.617× | −30.0% | 1,355 | 1.429× | −22.3% | 1,817 |
| **zstd level 3 (as used by iqz)** | 2.312× | — | 3,892 | 1.839× | — | 3,255 |
| zstd level -1 | 1.914× | −17.2% | 10,228 | 1.670× | −9.2% | 5,211 |
| lz4 | 1.871× | −19.1% | 12,596 | 1.582× | −14.0% | 7,669 |
| lzma | 2.337× | +1.1% | 125 | 1.881× | +2.3% | 154 |
| pcodec | 2.366× | +2.4% | 2,073 | 1.899× | +3.3% | 2,128 |
| FLAC, level 8 | 2.498× | +8.1% | 309 | 2.062× | +12.1% | 302 |
| JPEG-LS | 2.190× | −5.3% | 132 | 1.767× | −3.9% | 118 |
| JPEG 2000 | 2.204× | −4.6% | 23 | 1.782× | −3.1% | 20 |
| JPEG-XL | 2.287× | −1.1% | 40 | 1.810× | −1.6% | 35 |
| HTJ2K | 2.094× | −9.4% | 798 | 1.703× | −7.4% | 727 |
| Rice coding, best parameter per 512 values (estimate) | 2.331× | +0.8% | — | 1.867× | +1.5% | — |
| Bit-packing, width per 128 values (estimate) | 2.097× | −9.3% | — | 1.727× | −6.1% | — |
| Order-0 entropy: best symbol-by-symbol coder (bound) | 2.358× | +2.0% | — | 1.894× | +3.0% | — |
