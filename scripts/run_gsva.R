# TAD activity = GSVA enrichment score of each gene set in each sample (the TADac method).
# Usage (from the project folder, in the tadtalk-r environment):
#   Rscript scripts/run_gsva.R <input matrix csv> <output csv> [cores, default 1]
#   Rscript scripts/run_gsva.R data/raw/ccle_matrix_drug.csv data/raw/ccle_gsva.csv
# Each extra core adds about one more copy of the data in memory; with less than 8 GB of RAM, keep 1.
suppressPackageStartupMessages({
  library(GSVA)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
cores <- if (length(args) >= 3) as.integer(args[3]) else 1
expr <- fread(args[1])                                  # column 1 = gene, then one column per sample
mat <- as.matrix(expr[, -1])
rownames(mat) <- expr[[1]]
rm(expr)
invisible(gc())                                         # free the table copy before GSVA starts

sets <- fread("data/raw/sets_long.csv")
gene_sets <- split(sets$gene, sets$set_id)              # list: set_id -> genes

bp <- if (cores > 1) BiocParallel::MulticoreParam(cores) else BiocParallel::SerialParam()
param <- gsvaParam(mat, gene_sets, kcdf = "Gaussian", minSize = 3)
scores <- gsva(param, BPPARAM = bp)

fwrite(data.table(set_id = rownames(scores), scores), args[2])
cat("Scored", nrow(scores), "gene sets in", ncol(scores), "samples ->", args[2], "\n")
