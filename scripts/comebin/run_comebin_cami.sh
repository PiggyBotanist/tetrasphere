#!/usr/bin/env bash
# Runs COMEBin's own, unmodified pipeline on one CAMI sample. COMEBin is a separate tool
# -- install it first: https://github.com/ziyewang/COMEBin
#
# Usage
# -----
#   bash scripts/comebin/run_comebin_cami.sh <contig_fasta> <bam_dir> <output_dir>
#
# <contig_fasta>  gold-standard assembly FASTA for the sample (see scripts/download/download_cami.py).
# <bam_dir>       directory of per-sample-read BAM files COMEBin uses to estimate coverage.
# <output_dir>    where COMEBin writes comebin_res/ (embeddings.tsv.gz is what later scripts read)
#                 and data_augmentation/ (per-view coverage, read by src/metagenome/coverage.py).
set -euo pipefail

contig_fasta=$1
bam_dir=$2
output_dir=$3

run_comebin.sh -a "$contig_fasta" -o "$output_dir" -p "$bam_dir"
