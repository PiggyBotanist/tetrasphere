#!/usr/bin/env bash
# One-off fix: bac120's k-mer sweep (k=2,3,5,6,7) split files were built with only 3 CV folds,
# inconsistent with ar53's 5-fold sweep and the appendix's stated 5-fold methodology. Rebuilds
# those 5 split files with --n-folds 5 (the default, matching ar53), then retrains all 5 k-values
# from scratch on the new splits (the old fold boundaries don't carry over -- a different n-folds
# changes the whole partition, not just adds folds 4-5).
set -euo pipefail
cd "$(dirname "$0")/.."
R=log
DATA_ROOT="${TETRASPHERE_DATA_ROOT:-results}"
S=$DATA_ROOT/gtdb/split
FEAT=$DATA_ROOT/gtdb/feature
DL=$DATA_ROOT/gtdb/download
PY="${TETRASPHERE_PYTHON:-python}"

step() { local name=$1; shift; echo "$(date '+%F %T') START $name" | tee -a $R/status_kmerfix.log
  if "$@" > $R/$name.log 2>&1; then echo "$(date '+%F %T') DONE $name" | tee -a $R/status_kmerfix.log
  else echo "$(date '+%F %T') FAILED $name" | tee -a $R/status_kmerfix.log; return 1; fi; }

for k in 2 3 5 6 7; do
  step resplit_bac120_${k}mer $PY scripts/tetrasphere/01_split_genome.py --label bac120_${k}mer_clr --domain bac120 \
    --feature-file $FEAT/bac120_${k}mer_clr.hdf5 --distance-file $FEAT/bac120_distance.hdf5 \
    --taxonomy-file $DL/bac120_taxonomy_r232.tsv.gz --output-file $S/bac120_${k}mer_clr.hdf5 --n-folds 5
done

train_k() {  # $1 k, $2 device
  local k=$1 dev=$2
  step kmer5fold_bac120_${k}_train $PY scripts/tetrasphere/03_train_genome.py --label bac120_${k}mer_clr --experiment kmer_experiment --split-file $S/bac120_${k}mer_clr.hdf5 --device $dev && \
  step kmer5fold_bac120_${k}_eval  $PY scripts/tetrasphere/05_evaluate_genome.py --label bac120_${k}mer_clr --experiment kmer_experiment --split-file $S/bac120_${k}mer_clr.hdf5 --epoch 50 --device $dev
}

(train_k 2 cuda:0 && train_k 3 cuda:0 && train_k 5 cuda:0) &
(train_k 6 cuda:1 && train_k 7 cuda:1) &
wait
echo "$(date '+%F %T') KMERFIX_FINISHED" >> $R/status_kmerfix.log
