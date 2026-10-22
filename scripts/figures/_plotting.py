"""
Shared helper -- not a standalone CLI. Publication-style scatter + label styling used by
scatterChimera.py and scatterGene.py: text color matches its point's color, decluttered
(non-overlapping, arrow-leader-lined) labels via adjustText, genus names abbreviated (e.g.
"Methanosarcina acetivorans" -> "M. acetivorans") to save space, a background grid, no title, larger
axis labels, and generous margins so labels stay inside the figure instead of running off the edge.
"""
from __future__ import annotations

# Standard Import
import re
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

GENUS_ABBREV_RE = re.compile(r"\b([A-Z][a-z]+)(?=\s+[a-z])")

def abbreviate_species(name: str) -> str:
    """"Genus species" -> "G. species" (first letter of genus + period). Matches any capitalized word
    immediately followed by a lowercase word, so it also abbreviates each species mentioned inside a
    compound chimera label (e.g. "25% Mycobacterium tuberculosis + 75% Bacillus subtilis" ->
    "25% M. tuberculosis + 75% B. subtilis")."""
    return GENUS_ABBREV_RE.sub(lambda m: f"{m.group(1)[0]}.", name)

def scatter_with_labels(ax: plt.Axes, x: np.ndarray, y: np.ndarray, labels: list[str], colors, size: float = 20, fontsize: float = 11, style: str = "italic") -> list:
    """
    Plot points, then place one (not-yet-decluttered) text label per point -- colored to match its own
    point, genus-abbreviated, nudged a small fixed distance from its point so it never starts exactly
    on top of it. Call declutter_labels() afterward.

    AI written (double check) -- adjustText's own "pull text back toward its original position" force
    uses wherever the text started as that anchor; starting it exactly AT the point (dx=dy=0) meant an
    isolated label with nothing to repel from just stayed at distance 0 -- an invisible, zero-length
    arrow even with min_arrow_len=0 in declutter_labels(). Starting from a small diagonal offset instead
    gives every label a real non-zero rest distance, so its leader line/arrow is always visible.
    """
    if len(x):
        span = max(float(np.ptp(x)), float(np.ptp(y)), 1e-6)
        offset = 0.03 * span
    else:
        offset = 0.0
    ax.scatter(x, y, c=colors, s=size, zorder=3)
    text_colors = [colors] * len(x) if isinstance(colors, str) else list(colors)
    return [ax.text(xi + offset, yi + offset, abbreviate_species(label), fontsize=fontsize, color=ci, style=style, zorder=4) for xi, yi, ci, label in zip(x, y, text_colors, labels)]

def declutter_labels(texts: list, x: np.ndarray, y: np.ndarray, ax: plt.Axes) -> None:
    """Repel overlapping labels apart (arrow leader lines back to their real point) -- this is what fixes
    severe crowding in tight clusters (e.g. same-genus points that sit almost on top of each other).

    target_x/target_y pin the arrow's endpoint to the TRUE point coordinates (not wherever the label's
    own starting offset put it -- see scatter_with_labels), and min_arrow_len=0 (adjustText default is 5
    display units) means every label gets that connecting line, even ones adjustText barely had to move
    -- otherwise a label left sitting right next to an unrelated point (common in a dense cluster) reads
    as ambiguous about which point is really its. arrowstyle="->" (was a plain "-" line) and a darker
    "0.2" gray make the leader unambiguously an arrow pointing at its point, not just a stray line.

    expand/force_text/max_move are all pushed well past adjustText's defaults (1.05-1.2 / 0.1-0.2 / 10px)
    -- those are tuned for a handful of scattered labels, not the tight micro-clusters this project
    actually has (e.g. 5-7 near-identical rRNA gene copies landing almost on top of each other). Default
    settings leave those visibly overlapping; this trades a bit more label travel for (close to)
    guaranteed separation."""
    if texts:
        adjust_text(texts, x=x, y=y, target_x=x, target_y=y, ax=ax, min_arrow_len=0,
                    expand=(1.6, 1.9), force_text=(0.6, 0.9), force_static=(0.3, 0.3), max_move=(120, 120),
                    time_lim=20, iter_lim=2000,
                    arrowprops=dict(arrowstyle="->", color="0.2", lw=0.7, shrinkA=0, shrinkB=3))

def style_axes(ax: plt.Axes, xlabel: str, ylabel: str, margin: float = 0.15) -> None:
    """Paper-ready axis styling: large pure-black labels/ticks, a background grid, generous margins
    (so labels have room to stay inside the figure instead of being clipped at the edge), no title."""
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
