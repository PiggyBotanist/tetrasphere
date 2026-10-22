# Tetrasphere

Tetrasphere maps $k$-mer composition into an embedding space where distance approximates
phylogenetic distance, and remains stable as sequences fragment. It has two components:

- **Embedding network** $M_\theta$: maps a composition vector to an embedding $z$, such that
  $\hat d(a,b) = 1 - \cos(z_a, z_b)$ approximates normalized patristic distance and
  $\hat\rho(a) = \lVert z_a \rVert_2$ approximates normalized root-to-tip depth.
- **Denoiser** $D_\psi$: maps a fragment's composition (plus its log length) toward an estimate of
  its parent genome's composition, before it is embedded.

Trained on the GTDB reference phylogeny (archaeal and bacterial domains), evaluated against
kf2vec (phylogenetic distance estimation) and COMEBin (metagenomic binning).

## Repository layout

```
config.py                 central path configuration (override via env vars, see below)
src/tetrasphere/          core model, loss, metrics, k-mer encoding, curriculum, data loading
src/metagenome/           CAMI contig embedding + DBSCAN clustering + bin scoring
src/io/, src/model/       shared I/O (FASTA/HDF5/Newick) and training/checkpoint utilities
scripts/download/         GTDB + CAMI data download
scripts/tetrasphere/      00-07: precompute features -> split -> train -> evaluate
scripts/kf2vec/           kf2vec benchmark wrapper (external tool, not vendored)
scripts/comebin/          COMEBin benchmark wrapper (external tool, not vendored)
scripts/figures/          one script per paper figure/panel
checkpoints/              one representative (fold 1) checkpoint per domain/stage
figures/, results/        script outputs land here by default
```

## Setup

```
pip install -r requirements.txt
```

By default, paths point at `/bls2/data` (this project's own data location) and write results
under this repo. Override with environment variables to run elsewhere:

```
export TETRASPHERE_DATA_ROOT=/path/to/data
export TETRASPHERE_RESULT_ROOT=/path/to/results
```

## Quickstart with the shipped checkpoints

The four checkpoints under `checkpoints/` (ar53/bac120, genome/fragment stage, fold 1) are enough
to run every figure script and `demo.ipynb` without training anything. See `UNIT_TEST.md` for the
full, step-by-step reproduction guide -- from downloading raw data through training every fold to
regenerating every figure, with the numbers to expect at each step.

## External tools

kf2vec and COMEBin are not vendored here (separate projects, separate licenses). `scripts/kf2vec/`
and `scripts/comebin/` wrap them; install instructions are in `UNIT_TEST.md`.
