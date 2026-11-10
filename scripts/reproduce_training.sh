#!/usr/bin/env bash
# Finishes the Tetrasphere model training needed for every main-text and appendix figure/table.
#
# Most training is already complete on disk (main ar53/bac120 models, both heldout sweeps) -- this
# script only runs the two genuinely missing pieces:
#   1. kmer_bac120_finish: bac120 k-mer sweep (k=2,3,5,6,7) is only 3/5 folds; retrains all 5 cleanly.
#   2. reweighting_ablation: class-imbalance (Appendix Figure S2) ablation, both domains -- trains
#      it here from scratch.
#
# Every step is idempotent: if a fold's final checkpoint already exists, that step is skipped, so
# re-running this script after a crash only redoes unfinished work. External baselines (kf2vec,
# COMEBin, MetaBAT2, Binny) are NOT retrained here -- they're used as-is from wherever they stand.
#
# Usage: bash scripts/reproduce_training.sh   (run from the repo root; logs to log/, status to log/status.log)
set -uo pipefail
cd "$(dirname "$0")/.."
R=log
C=checkpoints
S="${TETRASPHERE_DATA_ROOT:-results}/gtdb/split"
TAX="${TETRASPHERE_DATA_ROOT:-results}/gtdb/download"
PY="${TETRASPHERE_PYTHON:-python}"
mkdir -p "$R"

step() {  # name, then the command
  local name=$1; shift
  echo "$(date '+%F %T') START $name" | tee -a "$R/status.log"
  if "$@" > "$R/$name.log" 2>&1; then
    echo "$(date '+%F %T') DONE $name" | tee -a "$R/status.log"
  else
    echo "$(date '+%F %T') FAILED $name (see $R/$name.log)" | tee -a "$R/status.log"
    return 1
  fi
}

done_fold5() { [ -f "$C/$1/$2/fold_5/epoch_50.pt" ]; }  # $1 experiment, $2 label

# ---------------------------------------------------------------------------
# Already complete on disk (main domains, both heldout sweeps) -- defined for
# documentation/full-reproducibility completeness, but a real run skips them.
# ---------------------------------------------------------------------------
main_domain() {  # $1 domain, $2 device
  local d=$1 dev=$2
  if done_fold5 "$d" "${d}_4mer_clr_genome"; then echo "main_domain $d: genome already complete, skipping"; else
    step ${d}_genome_train $PY scripts/tetrasphere/03_train_genome.py --label ${d}_4mer_clr_genome --experiment $d --split-file $S/${d}_4mer_clr_genome.hdf5 --device $dev || return 1
    step ${d}_genome_eval  $PY scripts/tetrasphere/05_evaluate_genome.py --label ${d}_4mer_clr_genome --experiment $d --split-file $S/${d}_4mer_clr_genome.hdf5 --epoch 50 --device $dev
  fi
  if done_fold5 "$d" "${d}_4mer_clr_fragment"; then echo "main_domain $d: fragment already complete, skipping"; else
    step ${d}_fragment_train $PY scripts/tetrasphere/04_train_fragment.py --label ${d}_4mer_clr_fragment --experiment $d --split-file-prefix $S/fragment/${d}_4mer_clr --lengths 50000 25000 10000 5000 2500 1000 --genome-split-file $S/${d}_4mer_clr_genome.hdf5 --genome-experiment $d --genome-label ${d}_4mer_clr_genome --genome-epoch 50 --device $dev || return 1
    step ${d}_fragment_eval $PY scripts/plots_and_tables/table_fragment_eval.py --label ${d}_4mer_clr_fragment --experiment $d --split-file-prefix $S/fragment/${d}_4mer_clr --lengths 50000 25000 10000 5000 2500 1000 --genome-split-file $S/${d}_4mer_clr_genome.hdf5 --epoch 50 --device $dev
    step ${d}_noDenoiser_eval $PY scripts/plots_and_tables/table_fragment_noDenoiser_eval.py --label ${d}_4mer_clr_genome --experiment $d --split-file-prefix $S/fragment/${d}_4mer_clr --lengths 50000 25000 10000 5000 2500 1000 --epoch 50 --device $dev
  fi
}

heldout() {  # $1 domain, $2 device
  local d=$1 dev=$2
  for level in genome species genus family order class; do
    if done_fold5 "${d}_heldout_experiment" "${d}_4mer_clr_${level}"; then echo "heldout $d/$level: already complete, skipping"; continue; fi
    step heldout_${d}_${level}_train $PY scripts/tetrasphere/03_train_genome.py --label ${d}_4mer_clr_${level} --experiment ${d}_heldout_experiment --split-file $S/${d}_4mer_clr_${level}.hdf5 --device $dev && \
    step heldout_${d}_${level}_eval  $PY scripts/tetrasphere/05_evaluate_genome.py --label ${d}_4mer_clr_${level} --experiment ${d}_heldout_experiment --split-file $S/${d}_4mer_clr_${level}.hdf5 --epoch 50 --device $dev
  done
}

# ---------------------------------------------------------------------------
# The real remaining work.
# ---------------------------------------------------------------------------
kmer_bac120_finish() {  # $1 device
  local dev=$1
  for k in 2 3 5 6 7; do
    if done_fold5 kmer_experiment "bac120_${k}mer_clr"; then echo "kmer bac120 k=$k: already complete, skipping"; continue; fi
    step kmer_bac120_${k}_train $PY scripts/tetrasphere/03_train_genome.py --label bac120_${k}mer_clr --experiment kmer_experiment --split-file $S/bac120_${k}mer_clr.hdf5 --device $dev && \
    step kmer_bac120_${k}_eval  $PY scripts/tetrasphere/05_evaluate_genome.py --label bac120_${k}mer_clr --experiment kmer_experiment --split-file $S/bac120_${k}mer_clr.hdf5 --epoch 50 --device $dev
  done
}

reweighting_ablation() {  # $1 device
  local dev=$1
  for d in ar53 bac120; do
    for w in none inverse_phylum; do
      local label=${d}_genome_${w}
      if done_fold5 "${d}_class_weighting" "$label"; then echo "reweighting $d/$w: already complete, skipping"; continue; fi
      if [ "$w" = inverse_phylum ]; then
        step reweight_${d}_${w}_train $PY scripts/tetrasphere/03_train_genome.py --label $label --experiment ${d}_class_weighting --split-file $S/${d}_4mer_clr_genome.hdf5 --weighting inverse_rank --weighting-rank phylum --taxonomy-file $TAX/${d}_taxonomy_r232.tsv.gz --device $dev
      else
        step reweight_${d}_${w}_train $PY scripts/tetrasphere/03_train_genome.py --label $label --experiment ${d}_class_weighting --split-file $S/${d}_4mer_clr_genome.hdf5 --weighting none --device $dev
      fi
    done
    step reweight_${d}_evaluate $PY scripts/plots_and_tables/figS2_precomputeTable.py --domain $d
  done
}

main_domain ar53 cuda:0   > $R/stream_ar53.log 2>&1 &
main_domain bac120 cuda:1 > $R/stream_bac120.log 2>&1 &
heldout ar53 cuda:0       > $R/stream_heldout_ar53.log 2>&1 &
heldout bac120 cuda:1     > $R/stream_heldout_bac120.log 2>&1 &
kmer_bac120_finish cuda:0  > $R/stream_kmer_bac120.log 2>&1 &
reweighting_ablation cuda:1 > $R/stream_reweighting.log 2>&1 &
wait
echo "$(date '+%F %T') ALL_STREAMS_FINISHED" >> $R/status.log
