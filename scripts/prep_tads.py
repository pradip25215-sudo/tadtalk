"""Build data/tad_genes.parquet: which gene sits in which TAD.

Inputs in data/raw/:
  tads.bed                     TAD calls: chrom, start, end (tab-separated, no header)
  gencode.basic.gtf.gz         GENCODE gene annotation, SAME genome build as the TADs
A gene belongs to the TAD that contains its transcription start site (TSS).
Run from the project folder:  python -m scripts.prep_tads
"""
import duckdb
import pandas as pd

from config import DATA, RAW

# 1. TADs: one row per domain. BED is 0-based: start is included, end is not.
tads = pd.read_csv(RAW / "tads.bed", sep="\t", header=None, usecols=[0, 1, 2],
                   names=["chrom", "tad_start", "tad_end"], comment="#")
tads["chrom"] = tads["chrom"].astype(str)
if not tads["chrom"].str.startswith("chr").all():  # "7" -> "chr7", to match GENCODE
    tads["chrom"] = "chr" + tads["chrom"].str.replace("chr", "", regex=False)
tads = tads.drop_duplicates().sort_values(["chrom", "tad_start"])
tads["tad_id"] = tads["chrom"] + ":" + tads["tad_start"].astype(str) + "-" + tads["tad_end"].astype(str)
print(f"TADs: {len(tads)}")

# 2. Genes: protein-coding genes and their TSS. Read the big GTF in chunks, keep "gene" rows.
cols = ["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attr"]
parts = []
for chunk in pd.read_csv(RAW / "gencode.basic.gtf.gz", sep="\t", comment="#", header=None,
                         names=cols, quoting=3, chunksize=500_000, low_memory=False):
    parts.append(chunk[chunk["feature"] == "gene"])
genes = pd.concat(parts)
genes["gene"] = genes["attr"].str.extract(r'gene_name "([^"]+)"', expand=False)
genes["type"] = genes["attr"].str.extract(r'gene_type "([^"]+)"', expand=False)
genes = genes[genes["type"] == "protein_coding"].drop_duplicates("gene")
# TSS = start on the + strand, end on the - strand. GTF is 1-based, so subtract 1 to compare with BED.
genes["tss"] = genes["start"].where(genes["strand"] == "+", genes["end"])
genes["tss0"] = genes["tss"] - 1
genes = genes[["gene", "chrom", "tss", "tss0"]]
print(f"Protein-coding genes: {len(genes)}")

# 3. Interval join: gene TSS inside TAD. If TADs overlap, keep the smallest TAD for each gene.
con = duckdb.connect()
con.register("tads", tads)
con.register("genes", genes)
tad_genes = con.execute("""
    SELECT t.tad_id, t.chrom, t.tad_start, t.tad_end, g.gene, g.tss AS gene_start
    FROM genes g
    JOIN tads t ON g.chrom = t.chrom AND g.tss0 >= t.tad_start AND g.tss0 < t.tad_end
    QUALIFY row_number() OVER (PARTITION BY g.gene ORDER BY t.tad_end - t.tad_start) = 1
    ORDER BY t.chrom, gene_start
""").df()

tad_genes.to_parquet(DATA / "tad_genes.parquet", index=False)
per_tad = tad_genes.groupby("tad_id").size()
print(f"Genes placed in a TAD: {len(tad_genes)} of {len(genes)}")
print(f"TADs with at least 1 gene: {len(per_tad)}; with at least 3 genes: {(per_tad >= 3).sum()}")
print(f"Genes per TAD: median {per_tad.median():.0f}, max {per_tad.max()}")
