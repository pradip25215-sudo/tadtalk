"""Build all gene sets for GSVA: every TAD plus null sets (random genes and neighbour blocks).

Output: data/sets.parquet        set_id, kind (tad / random / block), size_bin, gene
        data/raw/sets_long.csv   set_id, gene (the same sets, for R)
Run:  python -m scripts.make_null_sets
"""
import numpy as np
import pandas as pd

from config import DATA, RAW
from tools.common import BINS, size_bin

N_NULL = 200                     # null sets per kind and per size bin
rng = np.random.default_rng(42)  # fixed seed: the same sets every time you run it

tg = pd.read_parquet(DATA / "tad_genes.parquet").drop_duplicates("gene")
tg = tg.sort_values(["chrom", "gene_start"])
tads = tg.groupby("tad_id")["gene"].apply(list)
tads = tads[tads.str.len() >= 3]  # a TAD needs at least 3 genes for a GSVA score

rows = []
for tad_id, genes in tads.items():
    rows += [(tad_id, "tad", size_bin(len(genes)), g) for g in genes]

all_genes = tg["gene"].to_numpy()
by_chrom = {c: d["gene"].to_numpy() for c, d in tg.groupby("chrom")}
sizes = tads.str.len()

for lo, hi, name in BINS:
    pool = sizes[(sizes >= lo) & (sizes <= hi)].to_numpy()  # real TAD sizes in this bin
    if len(pool) == 0:
        continue
    for i in range(N_NULL):
        k = int(rng.choice(pool))
        # Random set: k genes from anywhere in the genome.
        genes = rng.choice(all_genes, size=k, replace=False)
        rows += [(f"random_{name}_{i}", "random", name, g) for g in genes]
        # Neighbour block: k consecutive genes on one chromosome, ignoring TAD borders.
        chroms = [c for c, g in by_chrom.items() if len(g) >= k]
        weights = np.array([len(by_chrom[c]) for c in chroms], dtype=float)
        c = rng.choice(chroms, p=weights / weights.sum())  # bigger chromosomes are picked more often
        start = int(rng.integers(0, len(by_chrom[c]) - k + 1))
        genes = by_chrom[c][start:start + k]
        rows += [(f"block_{name}_{i}", "block", name, g) for g in genes]

sets = pd.DataFrame(rows, columns=["set_id", "kind", "size_bin", "gene"])
sets.to_parquet(DATA / "sets.parquet", index=False)
sets[["set_id", "gene"]].to_csv(RAW / "sets_long.csv", index=False)
print(sets.drop_duplicates("set_id").groupby(["kind", "size_bin"]).size().unstack(fill_value=0))
