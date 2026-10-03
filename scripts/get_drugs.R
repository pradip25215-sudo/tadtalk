# Download CTRPv2 drug response with PharmacoGx and save AAC as a table (rows = drugs, columns = cell lines).
# Run from the project folder, inside the tadtalk-r environment:  Rscript scripts/get_drugs.R
suppressPackageStartupMessages(library(PharmacoGx))

print(availablePSets())                      # every dataset PharmacoGx can download, with exact names
dir.create("data/raw/psets", recursive = TRUE, showWarnings = FALSE)
ctrp <- downloadPSet("CTRPv2_2015", saveDir = "data/raw/psets", timeout = 3600)

print(colnames(sensitivityProfiles(ctrp)))   # available measures; we use aac_recomputed
aac <- summarizeSensitivityProfiles(ctrp, sensitivity.measure = "aac_recomputed", summary.stat = "median")
write.csv(aac, "data/raw/ctrpv2_aac_wide.csv")
cat("Saved", nrow(aac), "drugs x", ncol(aac), "cell lines to data/raw/ctrpv2_aac_wide.csv\n")
