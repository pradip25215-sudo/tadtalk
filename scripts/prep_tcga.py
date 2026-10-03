"""Clean TCGA expression and survival for the chosen cancers.

Inputs in data/raw/:  tcga_<CANCER>_expr.csv  (first column = gene, then one column per sample barcode)
                      tcga_survival.csv       (TCGA-CDR columns: bcr_patient_barcode, type, OS, OS.time)
Outputs:  data/raw/tcga_<CANCER>_matrix.csv  primary tumours only, one column per patient (GSVA input)
          data/survival.parquet              sample, cancer, OS, OS_time
Run:  python -m scripts.prep_tcga
"""
import numpy as np
import pandas as pd

from config import DATA, RAW

CANCERS = ["HNSC", "BRCA", "LUAD"]

for cancer in CANCERS:
    m = pd.read_csv(RAW / f"tcga_{cancer}_expr.csv", index_col=0)
    m.columns = [c.replace(".", "-") for c in m.columns]  # R sometimes writes TCGA.CV.7090 instead of TCGA-CV-7090
    if all(len(c) >= 15 for c in m.columns):  # full barcodes: keep primary tumours (sample type 01)
        m = m[[c for c in m.columns if c[13:15] == "01"]]
    m.columns = [c[:12] for c in m.columns]  # TCGA-CV-7090-01A-... -> patient id TCGA-CV-7090
    m = m.loc[:, ~m.columns.duplicated()]
    if m.to_numpy().max() > 50:  # looks like raw TPM, not log scale
        m = np.log2(m + 1)
    m = m[m.std(axis=1) > 0]
    m.index.name = "gene"
    m.to_csv(RAW / f"tcga_{cancer}_matrix.csv")
    print(f"{cancer}: {m.shape[0]} genes x {m.shape[1]} patients")

s = pd.read_csv(RAW / "tcga_survival.csv")
s = s.rename(columns={"bcr_patient_barcode": "sample", "type": "cancer", "OS.time": "OS_time"})
s = s[["sample", "cancer", "OS", "OS_time"]].dropna()
s = s[s["cancer"].isin(CANCERS) & (s["OS_time"] > 0)]
s["OS"] = s["OS"].astype(int)  # 1 = died, 0 = alive at last follow-up (censored)
s.to_parquet(DATA / "survival.parquet", index=False)
print(s.groupby("cancer").agg(patients=("sample", "size"), deaths=("OS", "sum")).to_string())
