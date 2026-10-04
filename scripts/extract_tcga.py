"""Cut the chosen cancers out of the big TCGA expression table, and save the survival table.

Inputs in data/raw/:
  tcga_all_tpm.txt.gz          GEO GSE62944 tumour TPM: one row per gene, one column per TCGA sample (24 cancers)
  TCGA_CancerType_Samples.txt  sample barcode <TAB> cancer code, no header
  TCGA-CDR.xlsx                TCGA Clinical Data Resource (Liu et al. 2018): curated overall survival
Outputs in data/raw/:
  tcga_<CANCER>_expr.csv       gene, then one column per sample of that cancer (prep_tcga.py reads these)
  tcga_survival.csv            bcr_patient_barcode, type, OS, OS.time
Run:  python -m scripts.extract_tcga      (a few minutes: it reads a 3 GB table once)
"""
import gzip

import pandas as pd

from config import RAW

CANCERS = ["HNSC", "BRCA", "LUAD"]  # keep the same list in scripts/prep_tcga.py
BIG = RAW / "tcga_all_tpm.txt.gz"

# 1. Which sample belongs to which cancer.
samples = pd.read_csv(RAW / "TCGA_CancerType_Samples.txt", sep="\t", header=None, names=["sample", "cancer"])
cancer_of = dict(zip(samples["sample"], samples["cancer"]))

# 2. Read only the header and the first data row, to find the position of every sample column.
with gzip.open(BIG, "rt") as fh:
    header = fh.readline().rstrip("\n").split("\t")
    n_fields = len(fh.readline().rstrip("\n").split("\t"))
if len(header) == n_fields - 1:  # R style: the header has no label for the gene column
    header = ["gene"] + header
position = {name: i for i, name in enumerate(header) if i > 0}
wanted = {c: [s for s in position if cancer_of.get(s) == c] for c in CANCERS}
for c in CANCERS:
    print(f"{c}: {len(wanted[c])} samples found in the big table")

# 3. One pass over the big table, keeping only the wanted columns (float32 halves the memory).
use = sorted(position[s] for c in CANCERS for s in wanted[c])
parts = []
reader = pd.read_csv(BIG, sep="\t", header=None, skiprows=1, usecols=[0] + use, index_col=0,
                     dtype={i: "float32" for i in use}, chunksize=2000)
for n, chunk in enumerate(reader, 1):
    parts.append(chunk)
    print(f"  read {n * 2000} genes...", end="\r")
table = pd.concat(parts)
table.columns = [header[i] for i in table.columns]
table.index = table.index.astype(str)
table = table[~table.index.duplicated()]  # a few gene names appear twice; keep the first
table.index.name = "gene"
print(f"\nGenes: {len(table)}")

for c in CANCERS:
    table[wanted[c]].to_csv(RAW / f"tcga_{c}_expr.csv")
    print(f"Saved data/raw/tcga_{c}_expr.csv: {len(table)} genes x {len(wanted[c])} samples")

# 4. Survival: the TCGA-CDR sheet has one row per patient.
sheets = pd.read_excel(RAW / "TCGA-CDR.xlsx", sheet_name=None)  # every sheet, as a dict
need = ["bcr_patient_barcode", "type", "OS", "OS.time"]
cdr = next(df for df in sheets.values() if all(col in df.columns for col in need))
cdr[need].to_csv(RAW / "tcga_survival.csv", index=False)
print(f"Saved data/raw/tcga_survival.csv: {len(cdr)} patients")
