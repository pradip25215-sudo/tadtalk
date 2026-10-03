"""Match CTRPv2 cell-line names to DepMap models.

Outputs:  data/drug_aac.parquet           model_id, drug, aac
          data/raw/ccle_matrix_drug.csv   expression of the drug-screened cell lines only (GSVA input, Day 3)
Run:  python -m scripts.prep_drugs
"""
import re

import pandas as pd

from config import DATA, RAW


def clean(name):
    """Upper-case letters and digits only: 'NCI-H1975' -> 'NCIH1975' (DepMap's stripped-name style)."""
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


aac = pd.read_csv(RAW / "ctrpv2_aac_wide.csv", index_col=0)  # rows = drugs, columns = cell lines
long = aac.reset_index(names="drug").melt(id_vars="drug", var_name="cell", value_name="aac").dropna()

models = pd.read_parquet(DATA / "models.parquet")
name_to_model = {clean(n): m for n, m in zip(models["name"], models["model_id"])}
long["model_id"] = long["cell"].map(lambda c: name_to_model.get(clean(c)))

cells = long.drop_duplicates("cell")
print(f"Matched {cells['model_id'].notna().sum()} of {len(cells)} CTRPv2 cell lines to DepMap")
print("Unmatched examples:", cells.loc[cells["model_id"].isna(), "cell"].head(10).tolist())

long = long.dropna(subset=["model_id"])
long = long.groupby(["model_id", "drug"], as_index=False)["aac"].median()  # duplicates -> one value
long.to_parquet(DATA / "drug_aac.parquet", index=False)
print(f"Saved {len(long)} rows: {long['drug'].nunique()} drugs x {long['model_id'].nunique()} cell lines")

# GSVA input for Day 3: keep only the cell lines that have drug data. TAD activity of cell lines is used
# only together with drug response, and GSVA's memory and run time grow with the number of samples.
columns = pd.read_csv(RAW / "ccle_matrix.csv", nrows=0).columns
screened = set(long["model_id"])
keep = [c for c in columns[1:] if c in screened]
matrix = pd.read_csv(RAW / "ccle_matrix.csv", usecols=["gene"] + keep, dtype={c: "float32" for c in keep})
matrix = matrix[matrix[keep].std(axis=1) > 0]  # drop genes with the same value in every kept cell line
matrix.to_csv(RAW / "ccle_matrix_drug.csv", index=False)
print(f"GSVA input: {len(matrix)} genes x {len(keep)} drug-screened cell lines -> data/raw/ccle_matrix_drug.csv")
