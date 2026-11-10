#!/usr/bin/env bash
# Nearest fold-1 training genome (skani ANI) for every prokaryotic CAMI genome from tableS8_build_queries.py.
# Bacterial CAMI genomes are searched against the bac120 fold-1 training genomes, archaeal against ar53.
# skani only reports hits with ANI >~ 80% and aligned fraction >= 15%; anything else = "no hit".
#
# Not needed for the cached fast-path (see cami_README.md) -- only run this if regenerating the ANI
# cache from scratch. Requires `skani` on PATH (external tool, not vendored).
#
# Usage
# -----
#   bash scripts/plots_and_tables/tableS8_skani_search.sh <query_dir> <ref_genome_dir> [threads]
#     query_dir:      the --output-dir passed to tableS8_build_queries.py (has queries.tsv + query/)
#     ref_genome_dir:  directory of {ar53,bac120}_fold1_train.txt genome-path lists
set -euo pipefail
OUT=${1:?query_dir required}
REF=${2:?ref_genome_dir required}
THREADS=${3:-48}

for pair in bacteria:bac120 archaea:ar53; do
    domain=${pair%%:*}; ref=${pair##*:}
    if [[ ! -d $OUT/db_$ref ]]; then
        skani sketch -t "$THREADS" -l "$REF/${ref}_fold1_train.txt" -o "$OUT/db_$ref"
    fi
    awk -F'\t' -v d="$domain" 'NR==1{for(i=1;i<=NF;i++) c[$i]=i; next} $c["domain"]==d {print $c["query"]}' "$OUT/queries.tsv" > "$OUT/query_$domain.txt"
    skani search -t "$THREADS" -n 1 -d "$OUT/db_$ref" --ql "$OUT/query_$domain.txt" -o "$OUT/search_$domain.tsv"
    echo "[$domain] $(wc -l < "$OUT/query_$domain.txt") queries, $(($(wc -l < "$OUT/search_$domain.tsv") - 1)) with a hit"
done
