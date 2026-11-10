#!/usr/bin/env bash
# Regenerates every committed figure/table from whatever checkpoints/results currently exist,
# using the paper's own figure numbering as the output filename. Each figure script also has its
# own standalone default output name (for ad-hoc exploration) -- this orchestrator always passes an
# explicit --output to produce the final, paper-numbered file instead.
#
# Domain note: main-text Figure 2 (all panels) and Figure 3 are computed on the BACTERIAL (bac120)
# GTDB tree (the manuscript states this explicitly); archaeal (ar53) results feed Appendix E's
# joint-domain tables (S4/S5/S6) instead of a separate main-text figure.
#
# kf2vec-bac120 gate: Figure 2B (hard requirement) and Figure 2C/Table S5's bac120 rows (soft --
# degrade gracefully without it) need kf2vec's own completed bac120 benchmark output, which may
# still be mid-training in a long-running background job unrelated to this script. Pass
# --include-kf2vec-bac120 once that's done; by default those two steps are skipped so everything
# else can still be regenerated now.
#
# Usage: bash scripts/reproduce_figures.sh [--include-kf2vec-bac120]
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${TETRASPHERE_PYTHON:-python}"
PT=scripts/plots_and_tables
F=figures
INCLUDE_KF2VEC_BAC120=0
[ "${1:-}" = "--include-kf2vec-bac120" ] && INCLUDE_KF2VEC_BAC120=1

run() {  # name, then the command -- logs but does not abort the whole script on one figure's failure
  local name=$1; shift
  echo "=== $name ==="
  if "$@"; then echo "OK: $name"; else echo "FAILED: $name (continuing)"; fi
}

mkdir -p "$F"

# ---------------------------------------------------------------------------
# Figure 2 (bac120)
# ---------------------------------------------------------------------------
run figure2a $PY $PT/fig2a.py --domain bac120 --metric pairwise_mean_distortion --output $F/figure2a.svg

if [ "$INCLUDE_KF2VEC_BAC120" = 1 ]; then
  run figure2b $PY $PT/fig2b.py --domain bac120 --fold 1 --output $F/figure2b.svg
else
  echo "=== figure2b === SKIPPED (needs --include-kf2vec-bac120; kf2vec's bac120 benchmark not yet complete)"
fi

# Precompute step (writes results/table/fig2c/{domain}_fold{N}.tsv for all 5 folds, one run per
# domain) -- always before plotting; figure2c, Table S5, and fig_denoiser_reconstruction.py (if
# you run it) all just read this, never compute it themselves.
run fig2c_precompute_ar53   $PY $PT/fig2c_precomputeTable.py --domain ar53
run fig2c_precompute_bac120 $PY $PT/fig2c_precomputeTable.py --domain bac120
run figure2c $PY $PT/fig2c.py --domain bac120 --fold 1 --output $F/figure2c.svg
run figure2d $PY $PT/fig2d.py --domain bac120 --x heldout --y pairwise_mean_distortion --series dataset --drop fold mean std --output $F/figure2d.svg

# ---------------------------------------------------------------------------
# Figure 3 (bac120)
# ---------------------------------------------------------------------------
run figure3a $PY $PT/fig3a.py --domain bac120 --color-level phylum --dim 3 --output $F/figure3a.svg
run figure3b $PY $PT/fig3b.py --domain bac120 --output $F/figure3b.svg

# ---------------------------------------------------------------------------
# Figure 4 (bac120)
# ---------------------------------------------------------------------------
# Panel A renders exactly one sample (the manuscript's own "CAMI medium (S001-M001)"); panel B
# (figure4b.svg) comes from fig4b.py instead, so there's no need to score this script's panel-B
# DBSCAN pass across other sample types here.
run figure4a $PY $PT/fig4a.py --domain bac120 --samples cami_toy_medium_m1_s001 --output-a $F/figure4a.svg --output-b $F/.figure4a_panelB_unused.png
run figure4b $PY $PT/fig4b.py --output $F/figure4b.svg

# ---------------------------------------------------------------------------
# Appendix figures
# ---------------------------------------------------------------------------
run appendix_figureS1 $PY $PT/figS1.py --output $F/appendix_figureS1.svg
run appendix_figureS2 $PY $PT/figS2.py --output $F/appendix_figureS2.svg
run appendix_figureS3 $PY $PT/figS3_S4.py --domain ar53   --output $F/appendix_figureS3.svg
run appendix_figureS4 $PY $PT/figS3_S4.py --domain bac120 --output $F/appendix_figureS4.svg

# ---------------------------------------------------------------------------
# Appendix tables (results/table/*.tsv)
# ---------------------------------------------------------------------------
run tableS1 $PY $PT/tableS1.py
run tableS2 $PY $PT/tableS2.py
run tableS3 $PY $PT/tableS3.py
run tableS4 $PY $PT/tableS4.py
# Table S5 aggregates fig2c_precomputeTable.py's per-fold tables (already generated above, for
# both domains, before any plotting happened).
run tableS5 $PY $PT/tableS5.py
run tableS6 $PY $PT/tableS6.py
run tableS7 $PY $PT/tableS7.py
run tableS8 $PY $PT/tableS8.py
run tableS9_S10 $PY $PT/tableS9_S10.py
run tableS11 $PY $PT/tableS11.py

# Clean up the throwaway panel-B PNG fig4a writes as a side effect (the orchestrator only wants
# figure4b.svg from fig4b.py instead) -- keep nothing under a leading "." in figures/.
rm -f $F/.figure4a_panelB_unused.png

echo
echo "Done. See output above for any FAILED steps (each is independent; fix and re-run just that script)."
