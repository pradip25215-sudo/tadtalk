"""FAKE sample data for testing the code only. Never use its results.

Use it only if a real download is slow: copy the whole project to a new folder (e.g. ~/tadtalk_sample),
and in THAT copy run:  python extras/make_sample_data.py data/raw
Then run the scripts in the guide's order. It writes fake files with the real names, so never run it
inside your real project folder.

Synthetic raw inputs that mimic the real formats (TAD BED, GENCODE GTF, DepMap, CTRPv2, TCGA).

Planted signals:
  EGFR TAD activity   <-> Erlotinib AAC in Lung lines
  ERBB2 TAD genes     essential (CRISPR) in Breast lines; ERBB2 TAD <-> Lapatinib in Breast
  CCND1 TAD activity  -> worse survival in HNSC
"""
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(sys.argv[1])
RAW.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)

special = {
    "chr7": ["SEC61G", "EGFR", "LANCL2", "VOPP1"],
    "chr11": ["CCND1", "ORAOV1", "FGF19", "FGF4", "FGF3", "ANO1"],
    "chr17": ["PNMT", "PGAP3", "ERBB2", "MIEN1", "GRB7", "STARD3", "TCAP"],
}
genes_rows, tad_rows = [], []
gid = 0
for chrom, n_tads in [("chr7", 70), ("chr11", 70), ("chr17", 60)]:
    pos = 1_000_000
    special_at = rng.integers(10, n_tads - 10)
    for t in range(n_tads):
        if t == special_at:
            names = special[chrom]
        else:
            k = int(rng.choice([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 25, 30, 45]))
            names = []
            for _ in range(k):
                gid += 1
                names.append(f"G{gid:04d}")
        start = pos
        size = 50_000 * len(names) + int(rng.integers(100_000, 400_000))
        end = start + size
        tad_rows.append((chrom, start, end))
        step = size // (len(names) + 1)
        for i, g in enumerate(names):
            gs = start + step * (i + 1)
            strand = "+" if rng.random() < 0.5 else "-"
            genes_rows.append((chrom, gs, gs + 20_000, strand, g))
        pos = end
    # a few genes outside any TAD
    for _ in range(3):
        gid += 1
        genes_rows.append((chrom, pos + 500_000, pos + 520_000, "+", f"G{gid:04d}"))

pd.DataFrame(tad_rows).to_csv(RAW / "tads.bed", sep="\t", header=False, index=False)
genes = pd.DataFrame(genes_rows, columns=["chrom", "start", "end", "strand", "gene"])
with gzip.open(RAW / "gencode.basic.gtf.gz", "wt") as fh:
    fh.write('##description: fake\n##provider: GENCODE\n')
    for r in genes.itertuples():
        # TSS inside the TAD for both strands: '-' strand genes have TSS = end
        s, e = (r.start, r.end) if r.strand == "+" else (r.start - 20_000, r.start)
        attr = f'gene_id "ENSG{r.Index:011d}.1"; gene_type "protein_coding"; gene_name "{r.gene}"; level 2;'
        fh.write(f"{r.chrom}\tHAVANA\tgene\t{s}\t{e}\t.\t{r.strand}\t.\t{attr}\n")
        fh.write(f"{r.chrom}\tHAVANA\ttranscript\t{s}\t{e}\t.\t{r.strand}\t.\t{attr} transcript_type \"protein_coding\";\n")
    fh.write('chr7\tHAVANA\tgene\t5\t50\t.\t+\t.\tgene_id "X"; gene_type "lncRNA"; gene_name "LINC0001";\n')

# gene -> TAD index (for simulation only)
genes = genes.sort_values(["chrom", "start"]).reset_index(drop=True)
tads = pd.DataFrame(tad_rows, columns=["chrom", "s", "e"])
tad_of = []
for r in genes.itertuples():
    tss = r.start if r.strand == "+" else r.end
    m = tads[(tads.chrom == r.chrom) & (tads.s <= tss) & (tads.e > tss)]
    tad_of.append(m.index[0] if len(m) else -1)
genes["tad"] = tad_of
gene_names = genes["gene"].tolist()
egfr_t = genes.loc[genes.gene == "EGFR", "tad"].item()
erbb2_t = genes.loc[genes.gene == "ERBB2", "tad"].item()
ccnd1_t = genes.loc[genes.gene == "CCND1", "tad"].item()


def expression(n_samples, drivers):
    """genes x samples log2(TPM+1); TAD genes share a latent factor; neighbouring TADs correlate a bit."""
    n_t = len(tads)
    base_f = rng.normal(0, 1, (n_t, n_samples))
    f = base_f.copy()
    for i in range(1, n_t):  # weak correlation between adjacent TADs
        f[i] = 0.85 * base_f[i] + 0.3 * f[i - 1]
    for t, vec in drivers.items():
        f[t] = vec
    ge = np.empty((len(genes), n_samples))
    for i, r in enumerate(genes.itertuples()):
        base = rng.uniform(1, 8)
        load = 0.9 if r.tad >= 0 else 0
        fac = f[r.tad] if r.tad >= 0 else 0
        ge[i] = np.clip(base + load * fac + rng.normal(0, 0.7, n_samples), 0, None)
    return ge


# ---- DepMap-like models
lin = ["Lung"] * 90 + ["Breast"] * 70 + ["Head and Neck"] * 55 + ["Skin"] * 35
n_m = len(lin)
model_ids = [f"ACH-{i:06d}" for i in range(1, n_m + 1)]
cell_names = [f"NCI-H{1000 + i}" if l == "Lung" else f"CL-{l[:2].upper()}{i}" for i, l in enumerate(lin)]
stripped = [c.replace("-", "").upper() for c in cell_names]
pd.DataFrame({"ModelID": model_ids, "CellLineName": cell_names, "StrippedCellLineName": stripped,
              "OncotreeLineage": lin, "OncotreePrimaryDisease": "x"}).to_csv(RAW / "Model.csv", index=False)

f_egfr = rng.normal(0, 1, n_m)
f_erbb2 = rng.normal(0, 1, n_m)
ge = expression(n_m, {egfr_t: f_egfr, erbb2_t: f_erbb2})
cols = [f"{g} ({1000 + i})" for i, g in enumerate(gene_names)]
expr = pd.DataFrame(ge.T.round(4), columns=cols)
expr.insert(0, "IsDefaultEntryForModel", "Yes")
expr.insert(0, "ModelID", model_ids)
expr.insert(0, "SequencingID", [f"CDS-{i:06d}" for i in range(n_m)])
dup = expr.iloc[:5].copy()
dup["IsDefaultEntryForModel"] = "No"
dup["SequencingID"] = [f"CDS-9{i:05d}" for i in range(5)]
pd.concat([expr, dup]).to_csv(RAW / "OmicsExpressionProteinCodingGenesTPMLogp1.csv", index=False)

eff = rng.normal(-0.05, 0.15, (n_m, len(genes)))
lin_arr = np.array(lin)
for j, r in enumerate(genes.itertuples()):
    if r.tad == erbb2_t:
        eff[lin_arr == "Breast", j] = rng.normal(-1.0, 0.2, (lin_arr == "Breast").sum())
eff[:, :15] -= 1.2  # some common-essential genes
crispr = pd.DataFrame(eff.round(4), index=model_ids, columns=cols)
crispr.iloc[3, 7] = np.nan
crispr.to_csv(RAW / "CRISPRGeneEffect.csv")  # first column unnamed = ModelID

# ---- CTRPv2-like AAC (drugs x cell names)
drugs = ["Erlotinib", "Lapatinib", "Paclitaxel", "Gefitinib", "Navitoclax", "Vorinostat", "Bortezomib",
         "Dasatinib", "Afatinib", "Neratinib", "Trametinib", "Selumetinib"]
aac = pd.DataFrame(np.clip(rng.normal(0.2, 0.08, (len(drugs), n_m)), 0, 1), index=drugs, columns=cell_names)
lung = lin_arr == "Lung"
aac.loc["Erlotinib", lung] = np.clip(0.25 + 0.07 * f_egfr[lung] + rng.normal(0, 0.05, lung.sum()), 0, 1)
brca = lin_arr == "Breast"
aac.loc["Lapatinib", brca] = np.clip(0.25 + 0.07 * f_erbb2[brca] + rng.normal(0, 0.05, brca.sum()), 0, 1)
aac = aac.drop(columns=cell_names[-6:])  # a few lines not screened
aac["UNKNOWN-CELL1"] = 0.2  # unmatched name
aac.iloc[2, 4] = np.nan
aac.to_csv(RAW / "ctrpv2_aac_wide.csv")

# ---- TCGA-like
surv = []
for c, n in [("HNSC", 300), ("BRCA", 300), ("LUAD", 250)]:
    fcc = rng.normal(0, 1, n)
    ge = expression(n, {ccnd1_t: fcc})
    pids = [f"TCGA-{c[:2]}-{i:04d}" for i in range(n)]
    samples = [f"{p}-01A-11R-A000-07" for p in pids]
    m = pd.DataFrame(ge.round(4), index=gene_names, columns=samples)
    m["TCGA-" + c[:2] + "-0001-11A-11R-A000-07"] = m.iloc[:, 1]  # a normal sample
    m.index.name = "gene"
    m.to_csv(RAW / f"tcga_{c}_expr.csv")
    beta = 0.7 if c == "HNSC" else 0.0
    t = rng.exponential(1500 / np.exp(beta * fcc))
    cens = rng.uniform(300, 3500, n)
    for p, ti, ci in zip(pids, t, cens):
        surv.append((p, c, int(ti <= ci), round(min(ti, ci), 1)))
pd.DataFrame(surv + [("TCGA-XX-0001", "OV", 1, 100.0)],
             columns=["bcr_patient_barcode", "type", "OS", "OS.time"]).to_csv(RAW / "tcga_survival.csv", index=False)
print("fake raw data written to", RAW)
