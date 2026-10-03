"""Turn GSVA outputs (sets x samples) into the long tables the tools read.

Outputs:  data/ccle_activity.parquet  set_id, model_id, score
          data/tcga_activity.parquet  cancer, sample, set_id, score
Run:  python -m scripts.build_activity
"""
import pandas as pd

from config import DATA, RAW


def long_scores(path, sample_col):
    """Wide GSVA table (one row per set, one column per sample) -> long table."""
    wide = pd.read_csv(path)
    return wide.melt(id_vars="set_id", var_name=sample_col, value_name="score")


ccle = long_scores(RAW / "ccle_gsva.csv", "model_id")
ccle.to_parquet(DATA / "ccle_activity.parquet", index=False)
print(f"CCLE: {ccle['set_id'].nunique()} sets x {ccle['model_id'].nunique()} cell lines")

parts = []
for f in sorted(RAW.glob("tcga_*_gsva.csv")):
    cancer = f.stem.split("_")[1]  # tcga_HNSC_gsva -> HNSC
    d = long_scores(f, "sample")
    d.insert(0, "cancer", cancer)
    parts.append(d)
    print(f"{cancer}: {d['set_id'].nunique()} sets x {d['sample'].nunique()} patients")
pd.concat(parts).to_parquet(DATA / "tcga_activity.parquet", index=False)
