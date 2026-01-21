#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reads "summary_results.txt" at runtime. The file contains multiple plot blocks separated by blank lines.

Presentation/Beamer optimizations:
- 16:9 figure format
- Larger fonts (title, labels, ticks) for slide readability
- Legend overlay (inside plot), 2 columns, significantly larger font
- Discrete markers only (no lines)
- Distinct color + marker per series (bounded, safe)
- DIM (X axis) displayed as integer (no decimals)
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from itertools import cycle, islice
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator


# ============================
# Global presentation styling
# ============================

def set_presentation_style() -> None:
    """
    Matplotlib rcParams tuned for slide / beamer readability.
    Adjust the numbers here if you want even larger/smaller.
    """
    plt.rcParams.update({
        # general
        "figure.dpi": 120,
        "savefig.dpi": 150,

        # font sizes (slide friendly)
        "font.size": 16,              # base
        "axes.titlesize": 22,         # big title
        "axes.labelsize": 18,         # axis labels
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,

        # legend (roughly "double" compared to x-small)
        "legend.fontsize": 14,

        # line/marker defaults (we still override in plot)
        "lines.markersize": 7,

        # layout
        "figure.autolayout": False,
    })


# ============================
# Data model
# ============================

@dataclass
class PlotBlock:
    title: str
    subtitle: str
    y_name: str
    y_unit: str
    y_scale: str          # "linear" or "log"
    x_label: str
    series_labels: List[str]
    x: List[int]          # DIM parsed as integer
    ys: Dict[str, List[Optional[float]]]


# ============================
# Parsing helpers
# ============================

_SPLIT_RE = re.compile(r"\t+| {2,}")


def _split_fields(line: str) -> List[str]:
    line = line.strip("\n\r")
    if not line.strip():
        return []
    return [p.strip() for p in _SPLIT_RE.split(line.strip()) if p.strip()]


def _parse_float(cell: str) -> Optional[float]:
    if cell is None:
        return None
    s = cell.strip()
    if not s:
        return None
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def _normalize_scale(scale_raw: str) -> str:
    s = (scale_raw or "").strip().lower()
    if s in ("log", "log10", "logarithmic", "logarithmisch"):
        return "log"
    return "linear"


def _extract_title(line: str) -> str:
    fields = _split_fields(line)
    if len(fields) >= 2 and fields[0].lower() == "title":
        return " - ".join(fields[1:]).strip()
    return line.strip()


def _extract_subtitle(line: str) -> str:
    line = line.strip()
    if ":" in line:
        head, tail = line.split(":", 1)
        if head.strip().lower() in ("method", "subtitle"):
            return tail.strip()
    fields = _split_fields(line)
    if fields and fields[0].lower().startswith(("method", "subtitle")) and len(fields) > 1:
        return " ".join(fields[1:]).strip()
    return line.strip()


def _extract_y_meta(line: str) -> Tuple[str, str, str]:
    fields = _split_fields(line)
    if not fields:
        return ("", "", "linear")

    if fields[0].lower() == "y":
        y_name = fields[1] if len(fields) >= 2 else ""
        y_unit = fields[2] if len(fields) >= 3 else ""
        y_scale = _normalize_scale(fields[3] if len(fields) >= 4 else "")
        return (y_name, y_unit, y_scale)

    # fallback
    y_name = fields[0]
    y_unit = fields[1] if len(fields) >= 2 else ""
    y_scale = _normalize_scale(fields[2] if len(fields) >= 3 else "")
    return (y_name, y_unit, y_scale)


def _parse_block(block_text: str) -> Optional[PlotBlock]:
    lines = [ln for ln in (l.rstrip() for l in block_text.splitlines()) if ln.strip() != ""]
    if len(lines) < 4:
        return None

    title = _extract_title(lines[0])
    subtitle = _extract_subtitle(lines[1]) if len(lines) >= 2 else ""

    y_name, y_unit, y_scale = _extract_y_meta(lines[2])

    header = _split_fields(lines[3])
    if len(header) < 2:
        return None

    x_label = header[0]
    series_labels = header[1:]

    x_vals: List[int] = []
    ys: Dict[str, List[Optional[float]]] = {lab: [] for lab in series_labels}

    for ln in lines[4:]:
        raw = ln.strip("\n\r")
        if not raw.strip():
            continue

        if "\t" in raw:
            parts = [p.strip() for p in raw.split("\t")]
        else:
            parts = [p.strip() for p in re.split(r" {2,}", raw.strip())]

        while parts and parts[-1] == "":
            parts.pop()

        if not parts or not parts[0]:
            continue

        x0 = _parse_float(parts[0])
        if x0 is None:
            continue
        x_int = int(round(x0))
        x_vals.append(x_int)

        for idx, lab in enumerate(series_labels, start=1):
            val = _parse_float(parts[idx]) if idx < len(parts) else None
            ys[lab].append(val)

        for lab in series_labels:
            if len(ys[lab]) < len(x_vals):
                ys[lab].append(None)

    if not x_vals:
        return None

    return PlotBlock(
        title=title,
        subtitle=subtitle,
        y_name=y_name,
        y_unit=y_unit,
        y_scale=y_scale,
        x_label=x_label,
        series_labels=series_labels,
        x=x_vals,
        ys=ys,
    )


def parse_file(path: Path) -> List[PlotBlock]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n+", text.strip(), flags=re.MULTILINE)
    out: List[PlotBlock] = []
    for blk in blocks:
        pb = _parse_block(blk)
        if pb is not None:
            out.append(pb)
    return out


# ============================
# Styling (bounded, safe)
# ============================

def make_style_pairs(n: int) -> List[Tuple[object, str]]:
    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*", "h", "H", "8", "p"]

    if n <= 10:
        colors = plt.get_cmap("tab10").colors
    elif n <= 20:
        colors = plt.get_cmap("tab20").colors
    else:
        colors = cm.hsv(np.linspace(0, 1, n))

    return list(islice(zip(cycle(colors), cycle(markers)), n))


# ============================
# Plotting
# ============================

def plot_blocks(blocks: List[PlotBlock], save_dir: Optional[Path], show: bool) -> None:
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    for idx, b in enumerate(blocks, start=1):
        # 16:9 format: good default for slides
        fig, ax = plt.subplots(figsize=(12.8, 7.2))

        styles = make_style_pairs(len(b.series_labels))

        for label, (color, marker) in zip(b.series_labels, styles):
            xs, ys_vals = [], []
            for x, y in zip(b.x, b.ys[label]):
                if y is not None:
                    xs.append(x)
                    ys_vals.append(y)

            if xs:
                ax.plot(
                    xs,
                    ys_vals,
                    linestyle="dotted",
                    marker=marker,
                    color=color,
                    markersize=8,   # slightly larger for beamer
                    label=label,
                )

        # Labels and scales
        ax.set_xlabel(b.x_label)

        ylab = (b.y_name or "").strip()
        if (b.y_unit or "").strip():
            ylab = f"{ylab} ({b.y_unit.strip()})" if ylab else f"({b.y_unit.strip()})"
        ax.set_ylabel(ylab if ylab else "Y")

        ax.set_yscale(b.y_scale if b.y_scale in ("linear", "log") else "linear")

        # DIM ticks: integer only
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))

        # Title (big & readable)
        full_title = (b.title or f"Plot {idx}").strip()
        if (b.subtitle or "").strip():
            full_title = f"{full_title}\n{b.subtitle.strip()}"
        ax.set_title(full_title, pad=12)

        ax.grid(True, which="both")

        # Legend overlay: bigger and readable on beamer
        ax.legend(
            loc="lower right",
            ncol=2,
            fontsize=14,        # explicit: beamer readable
            frameon=True,
            framealpha=0.90,
            borderpad=0.6,
            labelspacing=0.4,
            columnspacing=1.0,
            handletextpad=0.6,
            handlelength=1.3,
            markerscale=1.2,    # legend markers a bit larger than plot markers
        )

        fig.tight_layout()

        if save_dir is not None:
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", (b.title or f"plot_{idx}")).strip("_")
            fig.savefig(save_dir / f"{idx:02d}_{safe}.png", dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)


# ============================
# CLI
# ============================

def main() -> None:
    ap = argparse.ArgumentParser(description="Plot multiple charts from summary_results.txt blocks.")
    ap.add_argument("file", nargs="?", default="summary_results.txt")
    ap.add_argument("--save-dir", default=None, help="Save PNGs into this directory.")
    ap.add_argument("--no-show", action="store_true", help="Do not open interactive windows.")
    args = ap.parse_args()

    set_presentation_style()

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")

    blocks = parse_file(path)
    if not blocks:
        raise SystemExit("No valid plot blocks found in file.")

    save_dir = Path(args.save_dir) if args.save_dir else None
    plot_blocks(blocks, save_dir=save_dir, show=(not args.no_show))


if __name__ == "__main__":
    main()

