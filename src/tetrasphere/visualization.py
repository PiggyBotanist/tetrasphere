from __future__ import annotations

# Standard Import
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

PALETTE = px.colors.qualitative.Alphabet
OTHER_COLOR = "#999999" 

# =================================================================
# Dimensionality reduction
# =================================================================
def reduce_to_3d(embedding: np.ndarray, method: str, n_components: int = 3, seed: int = 42) -> np.ndarray:
    """Reduce (n_samples, dim_output) embeddings to (n_samples, n_components) via pca/umap/tsne. n_components=2 for a 2D scatter instead of 3D."""
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=n_components, random_state=seed).fit_transform(embedding)
    if method == "umap":
        import umap
        return umap.UMAP(n_components=n_components, random_state=seed).fit_transform(embedding)
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=n_components, random_state=seed, init="pca").fit_transform(embedding)
    raise ValueError(f"unknown reduction method {method!r}")

def reduce_to_2d_mds(embedding: np.ndarray, seed: int = 42, metric: str = "cosine") -> np.ndarray:
    """Reduce (n_samples, dim_output) embeddings to (n_samples, 2) via classical MDS on a precomputed dissimilarity matrix. Cosine by default, matching how the model itself measures distance (see PhylogenyEmbedder.forward)."""
    from sklearn.manifold import MDS
    from sklearn.metrics import pairwise_distances
    dissimilarity = pairwise_distances(embedding, metric=metric)
    return MDS(n_components=2, dissimilarity="precomputed", random_state=seed, normalized_stress=False).fit_transform(dissimilarity)

# =================================================================
# Tabbed 3D scatter (dropdown switches which column colors the points)
# =================================================================
def make_tabbed_figure(coords: np.ndarray, df: pd.DataFrame, color_modes: list[str], method: str, id_column: str = "id", color_map: dict[str, str] | None = None) -> go.Figure:
    """
    One 2D or 3D scatter figure (based on coords.shape[1]) with a dropdown ("tabs") to switch which
    column of df colors the points. Every mode's traces are pre-added (one trace per category, so each
    stays individually legend-clickable) and toggled via visibility rather than rebuilt on each dropdown
    selection.

    color_map optionally pins specific category values to specific colors (e.g. semantic
    correct/misclassified/unclassified colors), overriding the palette for just those values.
    """
    is_3d = coords.shape[1] == 3
    fig = go.Figure()
    trace_counts = []
    for mode in color_modes:
        counts_per_cat = df[mode].astype(str).value_counts()
        # Draw order: largest category first (bottom layer) down to smallest (top layer), so a handful of
        # rare-but-important points (e.g. misclassified) aren't hidden under a wall of majority-class points.
        # "other" always drawn dead last regardless of its size, so it reads as background.
        categories = sorted(counts_per_cat.index, key=lambda c: (c == "other", -counts_per_cat[c]))
        palette_i = 0
        for cat in categories:
            mask = (df[mode] == cat).to_numpy()
            if cat == "other":
                color = OTHER_COLOR
            elif color_map and cat in color_map:
                color = color_map[cat]
            else:
                color = PALETTE[palette_i % len(PALETTE)]
                palette_i += 1
            scatter_cls, coord_kwargs = (go.Scatter3d, dict(z=coords[mask, 2])) if is_3d else (go.Scatter, {})
            fig.add_trace(scatter_cls(
                x=coords[mask, 0], y=coords[mask, 1], **coord_kwargs,
                mode="markers", name=str(cat), legendgroup=mode,
                marker=dict(size=3 if is_3d else 6, color=color), text=df.loc[mask, id_column],
                visible=(mode == color_modes[0]),
            ))
        trace_counts.append(len(categories))

    buttons, start = [], 0
    for mode, count in zip(color_modes, trace_counts):
        visible = [False] * sum(trace_counts)
        visible[start:start + count] = [True] * count
        label = mode.removesuffix("_display")
        buttons.append(dict(label=label, method="update", args=[{"visible": visible}, {"title.text": f"{method}: colored by {label}"}]))
        start += count

    fig.update_layout(
        title=f"{method}: colored by {color_modes[0].removesuffix('_display')}",
        updatemenus=[dict(active=0, buttons=buttons, x=0.02, y=1.08, xanchor="left", yanchor="top")],
        height=700,
    )
    return fig
