"""Render figures (SVG, light and dark) and markdown tables from results.

Only reads `results/`; does not need access to the datasets.
"""

import json
import os

import font_roboto
import matplotlib

matplotlib.use("svg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from .coders import CODERS, ESTIMATES  # noqa: E402
from .variants import LEVELS, ORDERS  # noqa: E402

DATASETS = ["tartanstar", "iq1m"]

THEMES = {
    "light": dict(
        fg="#182026", muted="#56636e", line="#d6dde2", accent="#0a716c",
        accent_soft="#dcefed", base="#7b8792", warm="#a9501c",
        warm_soft="#f6e6dc", surface="#ffffff"),
    "dark": dict(
        fg="#e2e8ec", muted="#98a6b1", line="#2b353d", accent="#52c4b9",
        accent_soft="#173533", base="#8794a0", warm="#e5925f",
        warm_soft="#3a261b", surface="#182026"),
}
"""Palette, matching the original design report."""

PREDICTORS = {
    "none": "None (zigzag + shuffle only)",
    "delta": "Previous chirp (delta)",
    "iqz": "**Chunk mean chirp (iqz)**",
    "best_of": "Better of delta and mean, per chunk",
    "mean_fast_delta": "Mean, then delta along fast time",
    "mean_fast_fixed2": "Mean, then 2nd-order fixed predictor (fast time)",
    "mean_fast_fixed3": "Mean, then 3rd-order fixed predictor (fast time)",
    "mean_fast_lpc4": "Mean, then 4th-order LPC per series (fast time)",
    "mean_rx": "Mean, then neighbouring RX antenna",
    "mean_tx": "Mean, then neighbouring TX antenna",
}

LAYOUTS = {
    "iqz": "**Stored order, IIQQ interleaved (iqz)**",
    **{f"order_{o}": f"Axis order `{o}`" for o in ORDERS},
    "frame_per_antenna": "Separate zstd frame per antenna",
    "frame_per_iq": "Separate zstd frame per I/Q component",
}

ENTROPY = {
    "zstd3": "**zstd level 3 (as used by iqz)**",
    **{k: c.label for k, c in CODERS.items() if k != "zstd3"},
}


def _load(results: str) -> tuple[dict, dict, dict, dict]:
    with open(os.path.join(results, "summary.json")) as f:
        summary = json.load(f)
    speed = {}
    for ds in DATASETS:
        path = os.path.join(results, f"speed_{ds}.csv")
        if os.path.exists(path):
            speed[ds] = pd.read_csv(path).set_index(["kind", "name"])
    env: dict = {}
    for name, target in (("speed_env.json", env),):
        path = os.path.join(results, name)
        if os.path.exists(path):
            with open(path) as f:
                target.update(json.load(f))
    return summary, speed, env


# -- formatting ---------------------------------------------------------------

def _x(r: float) -> str:
    return f"{r:.3f}×"


def _ci(ci: list[float], digits: int = 3) -> str:
    return f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def _pct(v: float) -> str:
    return f"{v * 100:+.1f}%".replace("-", "−")


def _pct_ci(ci: list[float]) -> str:
    return f"[{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}]".replace("-", "−")


def _mbps(v: float) -> str:
    return f"{v:,.0f}"


def _lab(summary: dict, ds: str) -> str:
    """Dataset label, escaped for markdown (`Tartan*`)."""
    return summary[ds]["label"].replace("*", "\\*")


def _table(header: list[str], rows: list[list[str]], align: str) -> str:
    sep = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _variant_table(summary: dict, labels: dict[str, str]) -> str:
    """Ratio, and change vs iqz, per dataset."""
    header = ["Variant"]
    for ds in DATASETS:
        lab = _lab(summary, ds)
        header += [f"{lab} ratio", "vs. iqz"]
    rows = []
    for key, label in labels.items():
        row = [label]
        for ds in DATASETS:
            v = summary[ds]["variants"].get(key)
            if v is None:
                row += ["—"] * 2
            elif key in ("iqz", "zstd3"):
                row += [_x(v["ratio"]), "—"]
            else:
                row += [_x(v["ratio"]),
                        f"{_pct(v['rel'])} {_pct_ci(v['rel_ci'])}"]
        rows.append(row)
    return _table(header, rows, "l" + "rr" * len(DATASETS))


# -- tables -------------------------------------------------------------------

def _headline(summary: dict) -> str:
    s = [summary[ds] for ds in DATASETS]
    all_rows = [d["groups"][0] for d in s]
    rows = [
        ["Traces"] + [f"{d['traces']}" for d in s],
        ["Chunks of 64 chirps"] + [f"{d['chunks_total'] / 1e6:.2f} M"
                                   for d in s],
        ["Radar data"] + [f"{d['raw_tb']:.2f} TB" for d in s],
        ["zstd-3 on raw bytes"] + [
            f"{_x(r['zstd']['ratio'])} {_ci(r['zstd']['ci'])}"
            for r in all_rows],
        ["**iqz**"] + [
            f"**{_x(r['iqz']['ratio'])}** {_ci(r['iqz']['ci'])}"
            for r in all_rows],
        ["Storage with iqz"] + [f"{d['tb']['iqz']:.2f} TB" for d in s],
    ]
    return _table([""] + [_lab(summary, ds) for ds in DATASETS], rows, "lrr")


def _ci_table(summary: dict) -> str:
    header = ["Group", "Traces", "Chunks", "zstd-3", "iqz",
              "95% CI (traces)", "95% CI (chunks)"]
    rows = []
    for ds in DATASETS:
        d = summary[ds]
        rows.append([f"**{_lab(summary, ds)}**"] + [""] * 6)
        last = None
        for r in d["groups"]:
            if r["group"] not in ("all", last):
                rows.append([f"*{r['group']}*"] + [""] * 6)
                last = r["group"]
            label = f"**{r['label']}**" if r["group"] == "all" else r["label"]
            rows.append([
                label, str(r["traces"]), f"{r['chunks']:,}",
                _x(r["zstd"]["ratio"]), _x(r["iqz"]["ratio"]),
                _ci(r["iqz"]["ci"]), _ci(r["iqz"]["chunk_ci"])])
    return _table(header, rows, "lrrrrrr")


def _chunking(summary: dict) -> str:
    header = ["Chirps per chunk"]
    for ds in DATASETS:
        lab = _lab(summary, ds)
        header += [f"{lab}, mean (iqz)", f"{lab}, delta"]
    sizes = summary[DATASETS[0]]["sweep"]["sizes"]
    rows = []
    for n in sizes:
        row = [f"**{n}**" if n == 64 else str(n)]
        for ds in DATASETS:
            sw = summary[ds]["sweep"]
            for kind in ("mean", "delta"):
                v = sw[kind][str(n)]
                row.append(_x(v["ratio"]) if n == 64 else
                           f"{_pct(v['rel'])} {_pct_ci(v['rel_ci'])}")
        rows.append(row)
    return _table(header, rows, "l" + "rr" * len(DATASETS))


def _entropy(summary: dict, speed: dict) -> str:
    header = ["Coder"]
    for ds in DATASETS:
        lab = _lab(summary, ds)
        header += [f"{lab} ratio", "vs. iqz", "entropy decode MB/s"]
    rows = []

    def speed_of(ds: str, kind: str, name: str) -> str:
        try:
            return _mbps(speed[ds].loc[(kind, name), "decode"])
        except KeyError:
            return "—"

    row = ["zstd level 3, on the raw bytes (no transform)"]
    for ds in DATASETS:
        v = summary[ds]["variants"]["zstd"]
        row += [_x(v["ratio"]), _pct(v["rel"]), speed_of(ds, "raw", "zstd")]
    rows.append(row)
    for key, label in ENTROPY.items():
        row = [label]
        for ds in DATASETS:
            v = summary[ds]["variants"][key]
            rel = "—" if key == "zstd3" else _pct(v["rel"])
            row += [_x(v["ratio"]), rel, speed_of(ds, "coder", key)]
        rows.append(row)
    for key, (label, _) in ESTIMATES.items():
        row = [label]
        for ds in DATASETS:
            v = summary[ds]["variants"][key]
            row += [_x(v["ratio"]), _pct(v["rel"]), "—"]
        rows.append(row)
    return _table(header, rows, "l" + "rrr" * len(DATASETS))


def _levels(summary: dict, speed: dict) -> str:
    header = ["zstd level"]
    for ds in DATASETS:
        lab = _lab(summary, ds)
        header += [f"{lab} ratio", "encode MB/s", "decode MB/s"]
    rows = []
    for lv in LEVELS:
        row = [f"**{lv}**" if lv == 3 else str(lv)]
        for ds in DATASETS:
            v = summary[ds]["variants"][f"level{lv}"]
            sp = speed[ds].loc[("level", f"level{lv}")]
            row += [_x(v["ratio"]), _mbps(sp["encode"]), _mbps(sp["decode"])]
        rows.append(row)
    return _table(header, rows, "l" + "rrr" * len(DATASETS))


def _speed(summary: dict, speed: dict, env: dict) -> str:
    header = ["Implementation"]
    for ds in DATASETS:
        lab = _lab(summary, ds)
        header += [f"{lab} encode", "decode"]
    rows = []
    for name, label in (("native", "iqz, native (Rust) backend"),
                        ("numpy", "iqz, numpy backend")):
        row = [label]
        for ds in DATASETS:
            sp = speed[ds].loc[("iqz", name)]
            row += [_mbps(sp["encode"]), _mbps(sp["decode"])]
        rows.append(row)
    row = ["zstd level 3 on the raw bytes, for reference"]
    for ds in DATASETS:
        sp = speed[ds].loc[("raw", "zstd")]
        row += [_mbps(sp["encode"]), _mbps(sp["decode"])]
    rows.append(row)
    note = (f"\nMB/s of raw data, single thread, data in memory, decoding "
            f"into a reused output buffer. CPU: {env.get('cpu', '?')}; "
            f"libzstd {env.get('libzstd', '?')}.\n")
    return _table(header, rows, "l" + "rr" * len(DATASETS)) + note


NOISE_BREAKDOWNS = ("Scene type", "Category")
"""Group breakdowns shown in the noise table (others are omitted)."""


def _noise(summary: dict) -> str:
    header = ["Group", "Noise σ, LSB", "Ratio limit", "iqz",
              "Gap, bits/value"]
    rows = []
    for ds in DATASETS:
        rows.append([f"**{_lab(summary, ds)}**"] + [""] * 4)
        for r in summary[ds]["noise"]["groups"]:
            if r["group"] not in ("all", *NOISE_BREAKDOWNS):
                continue
            label = f"**{r['label']}**" if r["group"] == "all" else r["label"]
            rows.append([
                label, f"{r['sigma']:.1f}",
                f"{_x(r['cap']['ratio'])} {_ci(r['cap']['ci'], 2)}",
                _x(r["iqz"]), f"{r['iqz_bits'] - r['bound_bits']:.1f}"])
    return _table(header, rows, "lrrrr")


# -- figures ------------------------------------------------------------------

def _register_fonts() -> None:
    """Register Roboto, the default text font of mkdocs-material.

    SVGs embedded as images cannot use the page's web fonts, so text is
    rendered as paths in the same font instead.
    """
    root = os.path.dirname(font_roboto.__file__)
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name.endswith(".ttf"):
                font_manager.fontManager.addfont(os.path.join(dirpath, name))


def _style(theme: dict) -> dict:
    return {
        "svg.fonttype": "path", "svg.hashsalt": "iqz",
        "font.family": "Roboto",
        "font.size": 10,
        "text.color": theme["fg"], "axes.labelcolor": theme["fg"],
        "axes.edgecolor": theme["muted"], "xtick.color": theme["muted"],
        "ytick.color": theme["muted"], "axes.facecolor": "none",
        "figure.facecolor": "none", "savefig.facecolor": "none",
        "savefig.transparent": True, "grid.color": theme["line"],
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    }


def _save(fig, docs: str, name: str, theme: str) -> None:
    fig.savefig(
        os.path.join(docs, "figures", f"{name}-{theme}.svg"),
        bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def _fig_ci(summary: dict, t: dict) -> plt.Figure:
    labels, zs, qs, caps, ys, heads = [], [], [], [], [], []
    y = 0
    for ds in DATASETS:
        d = summary[ds]
        limits = {(r["group"], r["label"]): r["cap"]
                  for r in d.get("noise", {}).get("groups", [])}
        heads.append((y, d["label"], True))
        y += 1
        last = None
        for r in d["groups"]:
            if r["group"] not in ("all", last):
                heads.append((y, r["group"].upper(), False))
                y += 0.8
                last = r["group"]
            labels.append((y, r["label"], r["traces"], r["group"] == "all"))
            zs.append(r["zstd"])
            qs.append(r["iqz"])
            caps.append(limits.get((r["group"], r["label"])))
            ys.append(y)
            y += 1
        y += 0.4

    fig, ax = plt.subplots(figsize=(8, 0.26 * y + 0.6))
    ys_ = np.array(ys)
    for series, color in ((zs, t["base"]), (qs, t["accent"])):
        lo = [s["ci"][0] for s in series]
        hi = [s["ci"][1] for s in series]
        ax.hlines(ys_, lo, hi, color=color, lw=2.5, capstyle="round")
        ax.scatter([s["ratio"] for s in series], ys_, s=36, color=color,
                   edgecolor=t["surface"], linewidth=1.5, zorder=3)
    has_cap = [c is not None for c in caps]
    if any(has_cap):
        cy = ys_[has_cap]
        cs = [c for c in caps if c is not None]
        ax.hlines(cy, [c["ci"][0] for c in cs], [c["ci"][1] for c in cs],
                  color=t["warm"], lw=2.5, capstyle="round")
        ax.scatter([c["ratio"] for c in cs], cy, s=40, marker="D",
                   color=t["warm"], edgecolor=t["surface"], linewidth=1.5,
                   zorder=3)
    xmin = min(s["ci"][0] for s in zs) - 0.05
    xmax = max([s["ci"][1] for s in qs]
               + [c["ci"][1] for c in caps if c is not None]) + 0.05
    ax.set_xlim(xmin, xmax)
    for yy, text, strong in heads:
        ax.text(xmin - 0.02 * (xmax - xmin), yy, text, ha="right",
                va="center", fontsize=11 if strong else 8,
                fontweight="bold" if strong else "normal",
                color=t["fg"] if strong else t["muted"],
                transform=ax.transData)
    ax.set_yticks([v[0] for v in labels])
    ax.set_yticklabels([f"{v[1]}  ({v[2]} traces)" for v in labels])
    for tick, v in zip(ax.get_yticklabels(), labels):
        tick.set_color(t["fg"])
        if v[3]:
            tick.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.invert_yaxis()
    ax.grid(axis="x", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Compression ratio (dot), with 95% trace-bootstrap CI")
    ax.scatter([], [], color=t["base"], label="zstd-3 on raw bytes")
    ax.scatter([], [], color=t["accent"], label="iqz")
    if any(has_cap):
        ax.scatter([], [], color=t["warm"], marker="D",
                   label="limit set by thermal noise")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=3)
    return fig


def _fig_hist(summary: dict, t: dict) -> plt.Figure:
    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(8, 4), sharex=True)
    for ax, ds in zip(axes, DATASETS):
        h = summary[ds]["hist"]
        bins = np.array(h["bins"])
        ax.bar(bins[:-1], h["counts"], width=np.diff(bins), align="edge",
               color=t["accent"], edgecolor=t["surface"], linewidth=0.5)
        lo, med, hi = h["pct"]
        ax.axvline(med, color=t["fg"], ls="--", lw=1.2)
        ax.axvspan(lo, hi, color=t["accent_soft"], zorder=0, lw=0)
        ax.set_title(
            f"{summary[ds]['label']}: median {med:.2f}×, "
            f"95% of chunks in {lo:.2f}–{hi:.2f}×",
            loc="left", fontsize=10, color=t["fg"])
        ax.set_ylabel("Chunks")
        ax.grid(axis="y", lw=0.8)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("Per-chunk compression ratio (iqz)")
    fig.tight_layout()
    return fig


def _fig_sweep(summary: dict, t: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(8, 3))
    for ax, ds in zip(axes, DATASETS):
        sw = summary[ds]["sweep"]
        sizes = sw["sizes"]
        for kind, color, label in (("mean", t["accent"], "mean (iqz)"),
                                   ("delta", t["base"], "delta")):
            ratio = [sw[kind][str(n)]["ratio"] for n in sizes]
            ax.plot(sizes, ratio, "o-", color=color, label=label, lw=2,
                    markeredgecolor=t["surface"])
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes, [str(n) for n in sizes])
        ax.minorticks_off()
        ax.set_title(summary[ds]["label"], loc="left", fontsize=10,
                     color=t["fg"])
        ax.set_xlabel("Chirps per chunk")
        ax.grid(lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Compression ratio")
    axes[0].legend()
    fig.tight_layout()
    return fig


def _fig_levels(summary: dict, speed: dict, t: dict) -> plt.Figure:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3))
    for ds, color in zip(DATASETS, (t["accent"], t["warm"])):
        lab = summary[ds]["label"]
        a1.plot(LEVELS, [summary[ds]["variants"][f"level{lv}"]["ratio"]
                         for lv in LEVELS], "o-", color=color, label=lab,
                markeredgecolor=t["surface"])
        sp = speed[ds].loc["level"]
        a2.plot(LEVELS, [sp.loc[f"level{lv}", "encode"] for lv in LEVELS],
                "o-", color=color, label=f"{lab} encode",
                markeredgecolor=t["surface"])
        a2.plot(LEVELS, [sp.loc[f"level{lv}", "decode"] for lv in LEVELS],
                "o--", color=color, label=f"{lab} decode",
                markeredgecolor=t["surface"], alpha=0.7)
    a1.set_ylabel("Compression ratio")
    a2.set_ylabel("MB/s (raw), single thread")
    a2.set_ylim(bottom=0)
    for ax in (a1, a2):
        ax.set_xlabel("zstd level")
        ax.set_xticks(LEVELS)
        ax.grid(lw=0.8)
        ax.set_axisbelow(True)
    a1.legend()
    a2.legend(fontsize=8)
    fig.tight_layout()
    return fig


_BOXES = [
    # x, y, w, h, kind, title, subtitles (in a 940 x 300 canvas, y down)
    (10, 38, 150, 74, "", "64-chirp chunk", ["(64, 3, 4, F) i16"]),
    (200, 38, 160, 74, "w", "subtract mean",
     ["r[s] = x[s] - m", "m = round(mean over s)"]),
    (400, 38, 140, 74, "w", "zigzag", ["(r<<1) ^ (r>>15)", "i16 to u16"]),
    (580, 38, 150, 74, "w", "byte shuffle",
     ["all high bytes,", "then all low bytes"]),
    (770, 38, 160, 74, "k", "zstd level 3", ["residuals"]),
    (770, 128, 160, 44, "k", "zstd level 3", ["mean chirp m"]),
    (10, 236, 150, 48, "", "original chunk", []),
    (200, 236, 530, 48, "w",
     "one fused loop: unshuffle + unzigzag + add m",
     ["every element independent; vectorizes"]),
    (770, 236, 160, 48, "k", "zstd decompress ×2", []),
]

_ARROWS = [
    [(160, 75), (198, 75)], [(360, 75), (398, 75)], [(540, 75), (578, 75)],
    [(730, 75), (768, 75)], [(280, 112), (280, 150), (768, 150)],
    [(768, 260), (732, 260)], [(200, 260), (162, 260)],
]


def _fig_pipeline(t: dict) -> plt.Figure:
    """Encode / decode block diagram."""
    from matplotlib.patches import FancyBboxPatch

    fig = plt.figure(figsize=(9.4, 3.0))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 940)
    ax.set_ylim(300, 0)
    ax.axis("off")
    fill = {"": (t["surface"], t["line"]), "w": (t["warm_soft"], t["warm"]),
            "k": (t["accent_soft"], t["accent"])}
    for x, y, w, h, kind, title, subs in _BOXES:
        face, edge = fill[kind]
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=6",
            facecolor=face, edgecolor=edge, linewidth=1.5))
        ty = y + (26 if h > 50 else 20)
        ax.text(x + w / 2, ty, title, ha="center", va="baseline",
                fontsize=10, fontweight="medium", color=t["fg"])
        for i, sub in enumerate(subs):
            ax.text(x + w / 2, ty + 18 + 16 * i, sub, ha="center",
                    va="baseline", fontsize=8.5, color=t["muted"])
    for path in _ARROWS:
        xs, ys = zip(*path)
        ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=t["muted"],
                lw=1.5, solid_joinstyle="miter")
        ax.annotate("", xy=path[-1], xytext=path[-2], arrowprops=dict(
            arrowstyle="-|>", color=t["muted"], lw=1.5,
            shrinkA=0, shrinkB=0))
    ax.text(10, 22, "ENCODE", color=t["accent"], fontsize=8.5)
    ax.text(10, 222, "DECODE", color=t["warm"], fontsize=8.5)
    ax.text(520, 143, "m stored alongside", ha="center", fontsize=8.5,
            color=t["muted"])
    return fig


# -- entry point --------------------------------------------------------------

def render(results: str, docs: str) -> None:
    """Render all figures and tables.

    Args:
        results: directory with `summary.json` and speed results.
        docs: output directory; figures go to `figures/`, tables to
            `tables/`.
    """
    summary, speed, env = _load(results)
    _register_fonts()
    os.makedirs(os.path.join(docs, "figures"), exist_ok=True)
    os.makedirs(os.path.join(docs, "tables"), exist_ok=True)

    tables = {
        "headline": _headline(summary),
        "ci": _ci_table(summary),
        "predictors": _variant_table(summary, PREDICTORS),
        "layouts": _variant_table(summary, LAYOUTS),
        "chunking": _chunking(summary),
        "entropy": _entropy(summary, speed),
        "levels": _levels(summary, speed),
        "speed": _speed(summary, speed, env),
        "noise": _noise(summary),
    }
    for name, text in tables.items():
        with open(os.path.join(docs, "tables", f"{name}.md"), "w") as f:
            f.write(text)

    for theme_name, theme in THEMES.items():
        with plt.rc_context(_style(theme)):
            _save(_fig_ci(summary, theme), docs, "ci", theme_name)
            _save(_fig_hist(summary, theme), docs, "hist", theme_name)
            _save(_fig_sweep(summary, theme), docs, "sweep", theme_name)
            _save(_fig_levels(summary, speed, theme), docs, "levels",
                  theme_name)
            _save(_fig_pipeline(theme), docs, "pipeline", theme_name)
