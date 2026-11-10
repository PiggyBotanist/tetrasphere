# Figure & table scripts -- what's what

One script per paper figure/table, named by its number, plus a handful of support scripts that
don't have a number of their own. See `scripts/reproduce_figures.sh` to run everything in one
pass with the right flags already wired up.

## Main-text figures

| Script | Produces |
|---|---|
| `fig2a.py` | Figure 2A -- k-mer size sweep |
| `fig2b.py` | Figure 2B -- Tetrasphere vs. kf2vec, whole-genome |
| `fig2c.py` | Figure 2C -- fragment-length sweep, with/without denoiser vs. kf2vec |
| `fig2d.py` | Figure 2D -- clade-holdout generalization |
| `fig3a.py` | Figure 3A -- whole-genome 3D PCA by phylum |
| `fig3b.py` | Figure 3B -- chimera MDS |
| `fig4a.py` | Figure 4A -- CAMI embedding comparison (t-SNE) |
| `fig4b.py` | Figure 4B -- CAMI binning benchmark, 6-7 methods |

## Appendix figures & tables

| Script | Produces |
|---|---|
| `figS1.py` | Figure S1 -- train/val loss curves |
| `figS2.py` | Figure S2 -- class-imbalance (rarity-bucket) ablation plot |
| `figS2_evaluate.py` | Computes the data `figS2.py` plots (run once per domain after training; see `scripts/reproduce_training.sh`) |
| `figS3_S4.py` | Figures S3 (archaea) & S4 (bacteria) -- whole-genome PCA by taxonomy/split (one script, `--domain` picks which) |
| `tableS1.py` ... `tableS6.py` | Tables S1-S6 -- training-data/curriculum/k-mer-sweep/clade-holdout stats |
| `tableS7.py` | Table S7 -- CAMI sample filtering stats |
| `tableS8.py` | Table S8 -- ANI overlap vs. training set |
| `tableS8_build_queries.py`, `tableS8_skani_search.sh` | Upstream prep for `tableS8.py` (build query FASTAs, run external `skani` ANI search) -- see `cami_README.md`; not needed if using the shipped `ani_cache/` |
| `tableS9_S10.py` | Tables S9 & S10 -- HQ/LQ genome recovery (one script, both tables -- they're the same pivot over a different column) |
| `tableS11.py` | Table S11 -- per-genome bin quality |
| `cami_README.md` | Details the CAMI-benchmark table pipeline (S7, S8, S9/S10, S11) specifically |
| `ani_cache/` | Cached ANI search results `tableS8_build_queries.py`/`tableS8_skani_search.sh` would otherwise need to regenerate |

## Not a numbered paper figure

| Script | What it is |
|---|---|
| `marker_gene_mds.py` | Gene-level (16S/23S rRNA, gyrA, ...) embedding MDS -- supplementary diagnostic, not in the paper |
| `denoiser_reconstruction.py` | Denoiser reconstruction fidelity by fragment length -- supplementary diagnostic, not in the paper |

## Core pipeline support (not figure/table scripts)

Relocated here alongside the figure/table scripts because they're evaluate-and-write-a-table
steps, like those -- but they're genuinely part of the core precompute/split/train/evaluate
pipeline in `scripts/tetrasphere/` (same role as that folder's `05_evaluate_genome.py`), not tied
to any one figure. `scripts/reproduce_training.sh` calls them directly as part of training.

| Script | What it is |
|---|---|
| `evaluate_fragment.py` | Fragment-stage per-fold validation metrics (with denoiser) |
| `evaluate_fragment_noDenoiser.py` | Fragment-stage per-fold validation metrics, zero-shot genome-stage encoder (no denoiser) |

## Shared helpers (not standalone scripts)

| File | Used by |
|---|---|
| `_embedding.py` | Checkpoint loading + sequence embedding. Imported by `fig3b.py`, `marker_gene_mds.py`. |
| `_plotting.py` | Publication-style scatter/label styling. Imported by `fig3b.py`, `marker_gene_mds.py`. |
