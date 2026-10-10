# iqz analysis

Reproducible measurements behind the design of `iqz`: compression ratios with bootstrap confidence intervals on Tartan\* and I/Q-1M, speed, and the alternatives considered. The rendered results are on the [Analysis](https://radarml.github.io/iqz/analysis/) page of the docs.

This is a standalone project with its own dependencies (including the official dataset loaders, [`tartanstar`](https://github.com/RadarML/tartanstar) and [`roverd`](https://github.com/RadarML/red-rover), and the external codecs that `iqz` is compared against); it depends on `iqz` from the parent directory.

## Running

```sh
cd analysis
uv sync
uv run iqz-analysis all --tartanstar /data1/tartanstar/extracted --iq1m /data0/iq1m --workers 64
```

Datasets are read only through their loaders, so local copies in any format the loaders support work. Either dataset can be omitted to skip it.

| Step | Output (`results/`) | Needs datasets |
| --- | --- | --- |
| `sample` | `traces_<ds>.csv` (survey), `sample_<ds>.csv` (10k chunks) | yes |
| `measure` | `measure_<group>_<ds>.csv`: per-chunk sizes for each variant group | yes |
| `sweep` | `sweep_<ds>.csv`: chunk sizes 16-256 on paired 256-chirp windows | yes |
| `speed` | `speed_<ds>.csv`, `speed_env.json`: single-thread throughput | yes |
| `noise` | `noise_<ds>.csv`: per-chunk thermal noise floor | yes |
| `stats` | `summary.json`: ratios and bootstrap intervals | no |
| `render` | `../docs/analysis/{figures,tables}/` | no |

Run `uv run iqz-analysis <step> --help` for each step's options. On a 256-core machine, `measure` and `sweep` take about 10 minutes with `--workers 128`; `speed` takes about 10 minutes, and should be run on an otherwise idle machine.

The sample uses the same seed and algorithm as the original design study, so it selects exactly the same chunks.

## Layout

| Path | Contents |
| --- | --- |
| `src/iqz_analysis/datasets.py` | Dataset adapters over the official loaders. |
| `src/iqz_analysis/sampling.py` | Survey sampling and the parallel map over sampled chunks. |
| `src/iqz_analysis/variants.py` | Measured variants (predictors, layouts, coders, levels), grouped. |
| `src/iqz_analysis/coders.py` | Entropy coders and size estimates. |
| `src/iqz_analysis/noise.py` | Thermal noise floor estimate and entropy bound. |
| `src/iqz_analysis/stats.py` | Ratios and trace-level bootstrap intervals. |
| `src/iqz_analysis/render.py` | Figures and markdown tables. |
| `src/iqz_analysis/_cli/` | `iqz-analysis` subcommands. |
