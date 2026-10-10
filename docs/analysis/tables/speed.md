| Implementation | Tartan\* encode | decode | I/Q-1M encode | decode |
| --- | ---: | ---: | ---: | ---: |
| iqz, native (Rust) backend | 1,286 | 3,416 | 864 | 2,866 |
| iqz, numpy backend | 1,042 | 2,374 | 679 | 2,115 |
| zstd level 3 on the raw bytes, for reference | 250 | 1,355 | 393 | 1,817 |

MB/s of raw data, single thread, data in memory, decoding into a reused output buffer. CPU: AMD EPYC 9575F 64-Core Processor; libzstd 1.5.7.
