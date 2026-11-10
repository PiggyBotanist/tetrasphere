# CAMI appendix tables (Table S7, S8, S9, S10, S11)

These scripts build the CAMI-benchmark appendix tables. `skani` (ANI search) is an **external
tool** (not vendored), same treatment as kf2vec/COMEBin in the top-level README's "External
tools" section.

## Run order

1. `tableS8_build_queries.py` -- for every CAMI genome, picks its best-assembled sample and writes
   a query FASTA (one genome = one evaluation unit, same unit the binning benchmark scores).
   Needs an NCBI taxdump (`nodes.dmp`/`merged.dmp`) to resolve each genome's domain, and the
   CAMI sample-filter-stats table the Figure 4 binning pipeline produces (see note below).
2. `tableS8_skani_search.sh` -- sketches each domain's fold-1 training genomes (`skani sketch`) and
   searches every query against its domain's sketch (`skani search -n 1`), i.e. nearest-neighbor
   ANI only. Bacterial CAMI genomes search against bac120 fold-1 training genomes, archaeal
   against ar53.
3. `tableS8.py` -- buckets each genome's nearest-training-genome ANI into the appendix's bands
   (>=99%, 95-99%, 80-95%, <80%/no-hit), restricted to the bacterial domain (the appendix text
   excludes the 44 archaeal and 36 plasmid/virus genomes, since `skani`'s genome-sketch indexing
   is effectively bacteria-only here), and writes `results/table/tableS8_ani_overlap.tsv`.

## Already-computed cache

Steps 1-2 are expensive (an all-genomes-vs-all-training-genomes ANI search) and were already run
once; their output is cached at `tableS8_ani_cache/cami_ani_nearest_training_per_genome.tsv` (1,415
genomes, both domains, with each genome's nearest-training-genome ANI already resolved).
`tableS8.py` reads this cache by default via `--per-genome-table`, so a normal run does **not**
need `skani` installed or steps 1-2 re-run. Pass a different `--per-genome-table` (produced by
re-running 1-2 against this repo's own training genomes/CAMI download) to regenerate the cache
from scratch.

**Note**: `tableS8_build_queries.py` still has two paths with no default, needed only to re-run step 1
from scratch (irrelevant for the cached fast-path above): the NCBI taxdump location (`--taxdump`)
and the CAMI sample-filter-stats table (`--sample-filter-stats`) that the Figure 4 binning
pipeline produces.

## Tables S7, S9, S10, S11 (CAMI binning benchmark)

These three read the same fixed, precomputed per-sample tables as `fig4b.py` (which produces
`figures/figure4b.svg`), under `results/table/cami_binning/`:
- `table_embedding_euclidean_dbscan_scg_per_sample.tsv` -- the DBSCAN-on-embedding methods
  (4-mer baseline, COMEBin embedding, Tetrasphere x2), hyperparameters selected by single-copy-gene
  (SCG) count, euclidean metric. This is checkpoint-dependent for the two Tetrasphere rows, but
  since the bac120 main-domain checkpoint it was embedded with is unchanged by the current
  retrain (already complete before this pass started), these numbers are reproduced exactly by
  rerunning the same embedding+DBSCAN pipeline -- copied rather than rerun here only because
  rerunning the full grid search was out of scope for this porting pass.
- `table_cami_comparison_external_per_sample.tsv` (MetaBAT2, binny) and
  `table_comebin_full_pipeline_per_sample.tsv` (COMEBin's own pipeline) -- both entirely
  checkpoint-independent, never need rerunning regardless of Tetrasphere retraining.
- `table_sample_filter_stats.tsv` -- per-sample CAMI contig/genome counts pre/post the >5kb
  filter; feeds `tableS7.py` directly, and is also the file `tableS8_build_queries.py`
  above wants for `--sample-filter-stats`.
- `all_categories_quality_stats_matching_figure.tsv` -- per-(category, method) average
  completeness/contamination and tier-A fraction, confirmed numerically consistent with
  `figure4b.svg`'s own data, not an earlier/stale DBSCAN run.

Scripts:
- `tableS7.py` -- aggregates `table_sample_filter_stats.tsv` by sample type (the same 8 categories
  used everywhere else) + a Total row. Verified against the appendix text's own cited totals (35
  samples, 10,379 pre-filter genomes, 3,120 post-filter genomes, 291,512 post-filter contigs) --
  exact match.
- `tableS9_S10.py` -- pivots `fig4b.py`'s own `load_rows()`/`summarize()` output (imported
  directly, not reimplemented) into sample-type x method tables, one run writes both: HQ = tier_A
  + tier_B fraction (>=90%/<=10%, S9), LQ = sum of all tiers (>=70%/<=30%, S10), both as
  percentages with an Overall-mean row.
- `tableS11.py` -- reshapes `all_categories_quality_stats_matching_figure.tsv` into a tidy
  (sample_type, method, avg_completeness_pct, avg_contamination_pct, tier_A_fraction) table.

Run `python scripts/plots_and_tables/fig4b.py --output figures/figure4b.svg` first (or any time
after), then the table scripts in any order -- none of them re-run DBSCAN or re-embed anything;
they all just reshape the tables above.
