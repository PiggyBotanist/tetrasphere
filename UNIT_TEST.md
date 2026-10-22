# Reproduction Guide

Step-by-step instructions to go from nothing to every figure in the paper. Each step says what to
run and what number/output to expect, so you can tell at a glance if something broke.

All commands assume you're at the repo root (`tetrasphere/`) with the environment set up:

```
pip install -r requirements.txt
```

Paths default to `/bls2/data` and this repo's own `results/`/`figures/`/`checkpoints/` folders.
Override with `TETRASPHERE_DATA_ROOT` / `TETRASPHERE_RESULT_ROOT` env vars if your data lives
elsewhere (see `config.py`).

---

## 0. Quick path: figures from the shipped checkpoints

`checkpoints/` already has one fold (fold 1) per domain/stage. If you just want to regenerate
figures and sanity-check the pipeline, skip to **Section 6** -- steps 1-5 (download/precompute/
split/train) are only needed to reproduce training itself or to get the other 4 folds.

---

## 1. Download data

```
python scripts/download/download_gtdb.py --release 232 --type ar53 bac120
python scripts/download/download_cami.py --dataset all
```

Expect: `{data_root}/gtdb/download/` with `{domain}_r232.tree.gz`, `{domain}_metadata_r232.tsv.gz`,
`{domain}_taxonomy_r232.tsv.gz`, and `representative_genomes/*.fna.gz` per domain. GTDB release 232
has 878,998 bacterial genomes (189,801 species representatives) and 22,343 archaeal genomes
(10,122 species representatives); after the CheckM2 completeness >=95% / contamination <=5% filter
applied in later steps, 88,346 bacterial and 2,613 archaeal genomes remain eligible.

CAMI download populates `{data_root}/cami/download/{sequence,metadata,bam}/`. This step aligns raw
reads into BAM files (needed later for COMEBin's coverage), so it is slow; `--workers`/`--threads`
control parallelism.

## 2. Precompute pairwise distance + k-mer features

```
python scripts/tetrasphere/00_precompute_distance.py --domain ar53 --output-file {data_root}/gtdb/feature/ar53_distance.hdf5
python scripts/tetrasphere/00_precompute_distance.py --domain bac120 --output-file {data_root}/gtdb/feature/bac120_distance.hdf5

python scripts/tetrasphere/00_precompute_kmerGenome.py --label ar53_4mer_clr --domain ar53 --k 4 --transform clr --output-file {data_root}/gtdb/feature/ar53_4mer_clr.hdf5
python scripts/tetrasphere/00_precompute_kmerGenome.py --label bac120_4mer_clr --domain bac120 --k 4 --transform clr --output-file {data_root}/gtdb/feature/bac120_4mer_clr.hdf5

python scripts/tetrasphere/00_precompute_kmerFragment.py --label ar53_frag --domain ar53 --k 4 --transform clr \
    --lengths 50000 25000 10000 5000 2500 1000 --output-prefix {data_root}/gtdb/feature/fragment_raw/ar53_4mer_clr
python scripts/tetrasphere/00_precompute_kmerFragment.py --label bac120_frag --domain bac120 --k 4 --transform clr \
    --lengths 50000 25000 10000 5000 2500 1000 --output-prefix {data_root}/gtdb/feature/fragment_raw/bac120_4mer_clr
```

Expect: `ar53_distance.hdf5` with `normalization_constant` attribute ~2.33265; `bac120_distance.hdf5`
~1.77469. These normalize both pairwise distance and depth into `[0, 2]`/`[0, 1]` respectively.

## 3. Build splits (5-fold CV + held-out eval, genome-level and fragment-level)

```
python scripts/tetrasphere/01_split_genome.py --label ar53_4mer_clr --domain ar53 \
    --feature-file {data_root}/gtdb/feature/ar53_4mer_clr.hdf5 --distance-file {data_root}/gtdb/feature/ar53_distance.hdf5 \
    --taxonomy-file {data_root}/gtdb/download/ar53_taxonomy_r232.tsv.gz --output-file {data_root}/gtdb/split/ar53_4mer_clr_genome.hdf5

python scripts/tetrasphere/02_split_fragment.py --domain ar53 --split-file {data_root}/gtdb/split/ar53_4mer_clr_genome.hdf5 \
    --distance-file {data_root}/gtdb/feature/ar53_distance.hdf5 --feature-file-prefix {data_root}/gtdb/feature/fragment_raw/ar53_4mer_clr \
    --lengths 50000 25000 10000 5000 2500 1000 --output-prefix {data_root}/gtdb/split/fragment/ar53_4mer_clr
```

Repeat both for `bac120`. For a clade-holdout split instead of the default genome-level split, add
`--holdout-level {genus,family,order,class}` to `01_split_genome.py`.

Expect: one `eval` group (~20% of genomes) plus `fold_1`..`fold_5` groups (each `train`+`val`) in
the output `.hdf5`, with every group's genomes disjoint from `eval`.

## 4. Train

Genome stage (per domain, repeat `--label`/`--experiment` per fold inside the script -- it trains
all 5 folds in one run):

```
python scripts/tetrasphere/03_train_genome.py --domain ar53 --label ar53_4mer_clr_genome --experiment ar53 \
    --split-file {data_root}/gtdb/split/ar53_4mer_clr_genome.hdf5
```

Fragment stage, after genome stage finishes (loads that fold's own frozen genome checkpoint):

```
python scripts/tetrasphere/04_train_fragment.py --label ar53_4mer_clr_fragment --experiment ar53 \
    --split-file-prefix {data_root}/gtdb/split/fragment/ar53_4mer_clr --lengths 50000 25000 10000 5000 2500 1000 \
    --genome-split-file {data_root}/gtdb/split/ar53_4mer_clr_genome.hdf5 --genome-experiment ar53 --genome-label ar53_4mer_clr_genome --genome-epoch 50
```

Defaults: AdamW, lr=1e-3, batch size 4096, dropout 0.1, no weight decay, 50 epochs, both stages.
Repeat both commands for `bac120`. Checkpoints land at
`checkpoints/{experiment}/{label}/fold_{n}/epoch_{k}.pt` (every 5 epochs).

**This step is the expensive one** -- full 5-fold x 2-stage x 2-domain training is what the shipped
`checkpoints/` (fold 1 only, both stages, both domains) stands in for. Everything in Section 6 only
needs fold 1.

## 5. Evaluate trained checkpoints

```
python scripts/tetrasphere/05_evaluate_genome.py --label ar53_4mer_clr_genome --experiment ar53 \
    --split-file {data_root}/gtdb/split/ar53_4mer_clr_genome.hdf5 --epoch 50
python scripts/tetrasphere/06_evaluate_fragment.py --label ar53_4mer_clr_fragment --experiment ar53 \
    --split-file-prefix {data_root}/gtdb/split/fragment/ar53_4mer_clr --lengths 50000 25000 10000 5000 2500 1000 \
    --genome-split-file {data_root}/gtdb/split/ar53_4mer_clr_genome.hdf5 --epoch 50
python scripts/tetrasphere/07_evaluate_fragment_noDenoiser.py --label ar53_4mer_clr_genome --experiment ar53 \
    --split-file-prefix {data_root}/gtdb/split/fragment/ar53_4mer_clr --lengths 50000 25000 10000 5000 2500 1000
```

These require all 5 folds' checkpoints (they average across folds). With only fold 1 shipped, use
the figure scripts in Section 6 instead, which work from a single fold.

Expect (ar53, fold 1, eval split, from `05_evaluate_genome.py`'s per-fold row):
`pairwise_mse ~= 0.0068`, `pairwise_r_squared ~= 0.899`, `pairwise_pearson_r ~= 0.951`.

## 6. Figures (work from the shipped fold-1 checkpoints alone)

```
# Fig 2A: k-mer size sweep (reads results/table/kmer_experiment/*.tsv, already included)
python scripts/figures/fig2a_kmer_sweep.py --domain ar53 --metric pairwise_mean_distortion
# Expect: k=4 mean distortion 0.0738 +/- 0.0022, minimum at k=4/k=5.

# Fig 2B: Tetrasphere vs. kf2vec, whole-genome, on kf2vec's own scored pairs
python scripts/figures/fig2b_kf2vec_comparison.py --domain ar53 --fold 1
# Expect: n=342232 pairs; Tetrasphere mse=0.0046 r_squared=0.948 pearson_r=0.974;
# kf2vec mse=0.0665 r_squared=0.239 pearson_r=0.775 spearman_r=0.883.
# Requires kf2vec's own completed classify+query output under config.kf2vec_root/ar53/fold_1/
# (see Section 7) -- this script reruns Tetrasphere live but reuses kf2vec's existing output
# rather than rerunning kf2vec itself.

# Fig 2C: fragment-length sweep, with/without denoiser vs. kf2vec zero-shot
python scripts/figures/fig2c_fragment_sweep.py --domain ar53 --fold 1
# Expect mean distortion (no denoiser) by length: 50kb=0.132, 10kb=0.371, 5kb=0.584, 2.5kb=0.735,
# 1kb=0.778. kf2vec's own zero-shot series (results/table/kf2vec/ar53_kf2vec_fragment_zeroshot.tsv,
# included) is evaluated at 2000bp, not 2500bp, at that one grid point -- not an identical length,
# only approximately comparable there.

# Fig 2D: denoiser reconstruction fidelity
python scripts/figures/fig2d_denoiser_reconstruction.py --domain ar53 --fold 1
# Expect Pearson r 0.981 (50kb) -> 0.944 (1kb), reconstruction MSE 0.0226 -> 0.0672,
# downstream-distortion/reconstruction-MSE ratio in [3.1, 3.6] across all lengths.

# Fig 2E: clade-holdout generalization (reads results/table/ar53_heldout_experiment/*.tsv, included)
python scripts/figures/fig2e_clade_holdout.py --domain ar53 --x heldout --y pairwise_mean_distortion --series dataset --drop fold mean std

# Fig 3A: whole-genome t-SNE colored by taxonomy
python scripts/figures/fig3a_taxonomy_tsne.py --domain bac120 --color-level class --inner-level family
# Expect top 8 classes: Gammaproteobacteria, Alphaproteobacteria, Bacteroidia, Actinomycetes,
# Clostridia, Bacilli, Minisyncoccia, Terriglobia.

# Fig 3B: chimera MDS (splices real M. tuberculosis / B. subtilis sequence)
python scripts/figures/fig3b_chimera_mds.py --domain bac120

# Fig 3C: marker-gene MDS (needs RefSeq gene annotations, config.refseq_root)
python scripts/figures/fig3c_marker_gene_mds.py --domain bac120 \
    --species "Escherichia coli" "Salmonella enterica" "Bacillus subtilis" "Listeria monocytogenes" \
    --gene "16S ribosomal RNA" "23S ribosomal RNA" "DNA gyrase, subunit A"
# Note: gene-product naming varies across RefSeq assemblies; a "no match" for one gene/species is a
# real annotation-naming gap, not a script bug -- the per-genome log line shows what matched.

# Fig 4: CAMI embedding comparison + binning recovery
python scripts/figures/fig4_cami_binning.py --samples cami_toy_low cami_toy_medium_m1_s001 ...
```

Fig 4 needs a COMEBin embedding already computed for each sample (Section 8) and uses DBSCAN
hyperparameters selected by silhouette score (unsupervised -- never against ground truth; see
`src/metagenome/cluster.py`). This is a real methodological change from an earlier, label-leaked
selection criterion. In practice, silhouette-selected DBSCAN collapses recovery toward 0 for every
representation on the toy samples tested here -- silhouette rewards few, well-separated clusters,
which is a poor proxy for recovering many fine-grained genome bins. Treat Fig 4B's numbers as
provisional until this is resolved; this is a known open issue, not a script bug.

## 7. kf2vec benchmark

kf2vec is a separate tool, not vendored here: install from its own repository and its own conda
environment (set `TETRASPHERE_KF2VEC_ENV` if the name differs from `kf2vec`).

```
python scripts/kf2vec/01_train.py --domain ar53 --split-file {data_root}/gtdb/split/ar53_4mer_clr_genome.hdf5 \
    --output-folder {data_root}/gtdb/kf2vec/ar53 --input-fasta {data_root}/gtdb/download/representative_genomes
python scripts/kf2vec/02_evaluate.py --domain ar53 --distance-file {data_root}/gtdb/feature/ar53_distance.hdf5 \
    --split-file {data_root}/gtdb/split/ar53_4mer_clr_genome.hdf5 --benchmark-folder {data_root}/gtdb/kf2vec/ar53
```

Expect: ar53 partitions into exactly 3 subtrees (logged by `01_train.py`'s tree-clustering step),
consistent across folds; bac120 partitions into ~80-82 subtrees, varying slightly per fold.
`02_evaluate.py` writes `results/table/kf2vec/{domain}_evaluation.tsv`; fold 1 eval row should
match Fig 2B's kf2vec numbers above.

## 8. COMEBin benchmark

COMEBin is a separate tool, not vendored here: install from https://github.com/ziyewang/COMEBin.

```
bash scripts/comebin/run_comebin_cami.sh <contig_fasta> <bam_dir> <output_dir>
```

Writes `<output_dir>/comebin_res/embeddings.tsv.gz`, which `scripts/figures/fig4_cami_binning.py`
reads directly (COMEBin's own clustering/bin-refinement step is not used -- only its learned
embedding is compared). Point `--comebin-archive` (or the `cami_archive` config path) at the
directory containing one `<sample>/comebin_res/` per CAMI sample.
