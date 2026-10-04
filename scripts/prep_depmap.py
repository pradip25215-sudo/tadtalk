"""Turn DepMap downloads into clean tables.

Inputs in data/raw/:  Model.csv, CRISPRGeneEffect.csv and the protein-coding expression file
                      (OmicsExpressionProteinCodingGenesTPMLogp1.csv or OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv)
Outputs:  data/models.parquet     model_id, name, lineage
          data/crispr.parquet     model_id, gene, effect   (long format)
          data/ccle_expr.parquet  model_id, gene, expr     (long format)
          data/raw/ccle_matrix.csv  genes x all cell lines (prep_drugs.py cuts the GSVA input from it)
Run:  python -m scripts.prep_depmap
"""
import numpy as np
import pandas as pd

from config import DATA, RAW

# The expression file is named differently in different DepMap releases; take the one that is there.
EXPR_NAMES = ["OmicsExpressionProteinCodingGenesTPMLogp1.csv", "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"]
EXPR_FILE = next((RAW / n for n in EXPR_NAMES if (RAW / n).exists()), RAW / EXPR_NAMES[0])


def read_depmap_matrix(path):
    """Read a DepMap cell-line x gene CSV. Returns float32 values, rows = model ids, columns = gene symbols."""
    header = pd.read_csv(path, nrows=0).columns
    df = pd.read_csv(path, dtype={c: "float32" for c in header if " (" in c})  # float32 halves the memory
    if "IsDefaultEntryForModel" in df.columns:  # newer releases keep several profiles per cell line
        df = df[df["IsDefaultEntryForModel"].astype(str).str.lower().isin(["yes", "true"])]
    id_col = "ModelID" if "ModelID" in df.columns else df.columns[0]
    df = df.set_index(id_col)
    gene_cols = [c for c in df.columns if " (" in c]  # gene columns look like "EGFR (1956)"
    df = df[gene_cols]
    df.columns = [c.split(" (")[0] for c in gene_cols]
    df = df.loc[~df.index.duplicated(), ~df.columns.duplicated()]
    df.index.name = "model_id"
    return df


def to_long(wide, value_name, keep_genes):
    """Wide (models x genes) -> long (model_id, gene, value), only genes that sit in a TAD.

    'category' columns store each name once plus a small number per row, so 20+ million rows
    fit in a few hundred MB instead of several GB.
    """
    wide = wide.loc[:, wide.columns.isin(keep_genes)]
    n_models, n_genes = wide.shape
    long = pd.DataFrame({
        "model_id": pd.Categorical.from_codes(np.repeat(np.arange(n_models, dtype="int32"), n_genes), wide.index),
        "gene": pd.Categorical.from_codes(np.tile(np.arange(n_genes, dtype="int32"), n_models), wide.columns),
        value_name: wide.to_numpy().ravel(),
    })
    return long[long[value_name].notna()]


tad_genes = set(pd.read_parquet(DATA / "tad_genes.parquet")["gene"])

m = pd.read_csv(RAW / "Model.csv")
models = pd.DataFrame({"model_id": m["ModelID"], "name": m["StrippedCellLineName"],
                       "lineage": m["OncotreeLineage"]})
models.to_parquet(DATA / "models.parquet", index=False)
print(f"Cell lines: {len(models)}")
print(models["lineage"].value_counts().head(10).to_string())

crispr = read_depmap_matrix(RAW / "CRISPRGeneEffect.csv")
to_long(crispr, "effect", tad_genes).to_parquet(DATA / "crispr.parquet", index=False)
print(f"CRISPR: {crispr.shape[0]} cell lines x {crispr.shape[1]} genes")

expr = read_depmap_matrix(EXPR_FILE)
to_long(expr, "expr", tad_genes).to_parquet(DATA / "ccle_expr.parquet", index=False)
matrix = expr.T  # GSVA wants genes as rows, samples as columns
matrix = matrix[matrix.std(axis=1) > 0]  # drop genes with the same value everywhere
matrix.index.name = "gene"
matrix.to_csv(RAW / "ccle_matrix.csv")
print(f"Expression: {expr.shape[0]} cell lines x {expr.shape[1]} genes -> data/raw/ccle_matrix.csv")
