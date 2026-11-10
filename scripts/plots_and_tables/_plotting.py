"""
Shared helper. Publication-style for scatter plots and + label styling.
"""
from __future__ import annotations

# Standard Import
import re
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

GENUS_ABBREV_RE = re.compile(r"\b([A-Z][a-z]+)(?=\s+[a-z])")

def abbreviate_species(name: str) -> str:
    """"Genus species" -> "G. species" (first letter of genus + period)."""
    return GENUS_ABBREV_RE.sub(lambda m: f"{m.group(1)[0]}.", name)

def scatter_with_labels(ax: plt.Axes, x: np.ndarray, y: np.ndarray, labels: list[str], colors, size: float = 20, fontsize: float = 11, style: str = "italic") -> list:
    """Plot points, then place text label per point (color match its own)"""
    if len(x):
        span = max(float(np.ptp(x)), float(np.ptp(y)), 1e-6)
        offset = 0.03 * span
    else:
        offset = 0.0
    ax.scatter(x, y, c=colors, s=size, zorder=3)
    text_colors = [colors] * len(x) if isinstance(colors, str) else list(colors)
    return [ax.text(xi + offset, yi + offset, abbreviate_species(label), fontsize=fontsize, color=ci, style=style, zorder=4) for xi, yi, ci, label in zip(x, y, text_colors, labels)]

def declutter_labels(texts: list, x: np.ndarray, y: np.ndarray, ax: plt.Axes) -> None:
    """Repel overlapping labels apart."""
    if texts:
        adjust_text(texts, x=x, y=y, target_x=x, target_y=y, ax=ax, min_arrow_len=0,
                    expand=(1.6, 1.9), force_text=(0.6, 0.9), force_static=(0.3, 0.3), max_move=(120, 120),
                    time_lim=20, iter_lim=2000,
                    arrowprops=dict(arrowstyle="->", color="0.2", lw=0.7, shrinkA=0, shrinkB=3))

def style_axes(ax: plt.Axes, xlabel: str, ylabel: str, margin: float = 0.15) -> None:
    """Axes styling"""
    ax.set_xlabel(xlabel, fontsize=16, color="black")
    ax.set_ylabel(ylabel, fontsize=16, color="black")
    ax.tick_params(labelsize=12, colors="black")
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_color("black")
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.grid(True, linewidth=0.6, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    ax.margins(margin)
