"""Turn GSVA outputs (sets x samples) into the long tables the tools read.

Outputs:  data/ccle_activity.parquet  set_id, model_id, score
          data/tcga_activity.parquet  cancer, sample, set_id, score
Run:  python -m scripts.build_activity
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import DATA, RAW


def long_scores(path, sample_col):
    """Wide GSVA table (one row per set, one column per sample) -> long table.

    'category' columns store each name once plus a small number per row, so tens of millions of rows
    fit in a few hundred MB. Rows stay grouped by set_id, which makes "WHERE set_id = ..." fast.
    """
    header = pd.read_csv(path, nrows=0).columns
    wide = pd.read_csv(path, index_col="set_id", dtype={c: "float32" for c in header if c != "set_id"})
    n_sets, n_samples = wide.shape
    return pd.DataFrame({
        "set_id": pd.Categorical.from_codes(np.repeat(np.arange(n_sets, dtype="int32"), n_samples), wide.index),
        sample_col: pd.Categorical.from_codes(np.tile(np.arange(n_samples, dtype="int32"), n_sets), wide.columns),
        "score": wide.to_numpy().ravel(),
    })


def as_table(df):
    """pandas -> Arrow, with names stored as plain text so every part of the file has the same layout."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    for i, field in enumerate(table.schema):
        if pa.types.is_dictionary(field.type):
            table = table.set_column(i, field.name, table.column(i).cast(pa.string()))
    return table


ccle = long_scores(RAW / "ccle_gsva.csv", "model_id")
pq.write_table(as_table(ccle), DATA / "ccle_activity.parquet")
print(f"CCLE: {ccle['set_id'].nunique()} sets x {ccle['model_id'].nunique()} cell lines")
del ccle

writer = None  # one cancer at a time, appended to the same file, so memory stays low
for f in sorted(RAW.glob("tcga_*_gsva.csv")):
    cancer = f.stem.split("_")[1]  # tcga_HNSC_gsva -> HNSC
    d = long_scores(f, "sample")
    print(f"{cancer}: {d['set_id'].nunique()} sets x {d['sample'].nunique()} patients")
    d.insert(0, "cancer", cancer)
    d = d[["cancer", "sample", "set_id", "score"]]
    table = as_table(d)
    del d
    if writer is None:
        writer = pq.ParquetWriter(DATA / "tcga_activity.parquet", table.schema)
    writer.write_table(table)
if writer is not None:
    writer.close()
