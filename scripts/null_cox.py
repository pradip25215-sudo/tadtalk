"""Precompute the Cox z-score of every null gene set in every cancer (StatGuard uses them for survival).

Output: data/null_cox.parquet  cancer, set_id, kind, size_bin, z
Run:  python -m scripts.null_cox      (a few minutes)
"""
import warnings

import pandas as pd
from lifelines import CoxPHFitter

from config import DATA
from tools.common import q

warnings.filterwarnings("ignore")  # lifelines warns on a few near-constant sets; those are skipped

rows = []
for cancer in q("SELECT DISTINCT cancer FROM tcga_activity ORDER BY 1")["cancer"]:
    d = q("""SELECT a.set_id, s.kind, s.size_bin, a.score, v.OS, v.OS_time
             FROM tcga_activity a
             JOIN (SELECT DISTINCT set_id, kind, size_bin FROM sets WHERE kind <> 'tad') s USING (set_id)
             JOIN survival v ON v.sample = a.sample AND v.cancer = a.cancer
             WHERE a.cancer = ?""", [cancer])
    for (set_id, kind, bin_name), g in d.groupby(["set_id", "kind", "size_bin"]):
        g = g.assign(x=(g["score"] - g["score"].mean()) / g["score"].std())  # 1 unit = 1 SD
        try:
            cph = CoxPHFitter().fit(g[["OS_time", "OS", "x"]], duration_col="OS_time", event_col="OS")
            rows.append((cancer, set_id, kind, bin_name, float(cph.summary.loc["x", "z"])))
        except Exception:
            pass
    print(f"{cancer}: done")

out = pd.DataFrame(rows, columns=["cancer", "set_id", "kind", "size_bin", "z"])
out.to_parquet(DATA / "null_cox.parquet", index=False)
print(out.groupby(["cancer", "kind"]).size().to_string())
