#!/usr/bin/env bash
# Run the whole data pipeline on the real files in data/raw/. It stops at the first error.
# Use from the project folder, in the (tadtalk) environment:
#   bash scripts/run_real.sh prep     2>&1 | tee run_prep.log     # tables and gene sets (about 15 minutes)
#   bash scripts/run_real.sh score    2>&1 | tee run_score.log    # GSVA TAD activity (the long step: hours)
#   bash scripts/run_real.sh finish   2>&1 | tee run_finish.log   # activity tables, null Cox, checks, tests
set -euo pipefail

step() { echo; echo "===== $(date +%H:%M:%S)  $* ====="; }

# R lives in the tadtalk-r conda environment; without conda (e.g. on a server) plain Rscript is used.
ENVS="$(command -v conda >/dev/null 2>&1 && conda env list || true)"
if grep -q "^tadtalk-r " <<< "$ENVS"; then
  RSCRIPT="conda run -n tadtalk-r --no-capture-output Rscript"
else
  RSCRIPT="Rscript"
fi

case "${1:-}" in
  prep)
    step "1/6 TADs and genes";           python -m scripts.prep_tads
    step "2/6 DepMap tables";            python -m scripts.prep_depmap
    step "3/6 cut cancers out of TCGA";  python -m scripts.extract_tcga
    step "4/6 TCGA matrices, survival";  python -m scripts.prep_tcga
    step "5/6 drug response";            python -m scripts.prep_drugs
    step "6/6 null gene sets";           python -m scripts.make_null_sets
    step "prep finished"
    ;;
  score)
    free -h
    step "1/4 GSVA: cell lines";  $RSCRIPT scripts/run_gsva.R data/raw/ccle_matrix_drug.csv data/raw/ccle_gsva.csv
    n=1
    for c in HNSC LUAD BRCA; do   # BRCA last: it is the biggest
      n=$((n + 1))
      step "$n/4 GSVA: TCGA $c";  $RSCRIPT scripts/run_gsva.R data/raw/tcga_${c}_matrix.csv data/raw/tcga_${c}_gsva.csv
    done
    step "score finished"
    ;;
  finish)
    step "1/5 activity tables";   python -m scripts.build_activity
    step "2/5 null Cox models";   python -m scripts.null_cox
    step "3/5 row counts"
    python -c "from tools.common import q; [print(t, q(f'SELECT COUNT(*) AS n FROM {t}').n[0]) for t in ['tad_genes','gene_tad','models','crispr','ccle_expr','drug_aac','survival','sets','ccle_activity','tcga_activity','null_cox']]"
    step "4/5 positive control: breast lines with the highest ERBB2-TAD activity (expect HER2+ lines)"
    python -c "
from tools.common import q, tad_of_gene
tad = tad_of_gene('ERBB2'); print('ERBB2 TAD:', tad)
print(q('''SELECT m.name, round(a.score, 2) AS activity FROM ccle_activity a JOIN models m USING (model_id)
           WHERE a.set_id = ? AND m.lineage = 'Breast' ORDER BY a.score DESC LIMIT 8''', [tad]).to_string(index=False))"
    step "5/5 tests";             python -m pytest -q
    step "finish finished"
    ;;
  *)
    echo "Usage: bash scripts/run_real.sh prep | score | finish"; exit 1
    ;;
esac
