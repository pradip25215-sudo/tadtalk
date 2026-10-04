"""Build the TAD tables: which genes sit in which TAD, and which TAD is used for each gene.

Inputs in data/raw/:
  gencode.basic.gtf.gz   GENCODE gene annotation, SAME genome build as the TADs (hg19 for the lab TAD set)
  and ONE of:
  union_tads.txt         the lab's union TAD file: one TAD per line, "chr10_100025000_102750000<TAB>gene<TAB>gene..."
  tads.bed               plain TAD calls: chrom, start, end (tab-separated, no header)

Outputs:
  data/tad_genes.parquet  tad_id, chrom, tad_start, tad_end, gene, gene_start   (all protein-coding genes of each TAD)
  data/gene_tad.parquet   gene, tad_id                                          (the ONE TAD used when you ask about a gene)

Union TADs are nested: one gene can sit in several TADs. For each gene we use the SMALLEST TAD that still has
at least 3 protein-coding genes ("the tightest domain around the gene"). Only TADs chosen this way are kept.
Run from the project folder:  python -m scripts.prep_tads
"""
import duckdb
import pandas as pd

from config import DATA, RAW

MIN_GENES = 3  # a TAD needs at least 3 genes for a GSVA score


def read_gencode_genes():
    """Protein-coding genes and their TSS. Reads the big GTF in chunks and keeps only the "gene" rows."""
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
    return genes[["gene", "chrom", "tss", "tss0"]]


def from_union_file(genes):
    """Lab format: the file already lists the genes of every TAD. GENCODE adds the position and the gene type."""
    rows = []
    n_tads = 0
    with open(RAW / "union_tads.txt") as fh:
        for line in fh:
            parts = line.split()  # the TAD id, then its genes (separated by tabs or spaces)
            if len(parts) < 2 or not parts[0].startswith("chr"):
                continue
            chrom, start, end = parts[0].rsplit("_", 2)  # "chr10_100025000_102750000"
            n_tads += 1
            for g in dict.fromkeys(parts[1:]):  # unique genes, file order
                rows.append((f"{chrom}:{start}-{end}", chrom, int(start), int(end), g))
    member = pd.DataFrame(rows, columns=["tad_id", "chrom", "tad_start", "tad_end", "gene"])
    print(f"TADs in the lab file: {n_tads}; gene entries: {len(member)}")
    # Keep protein-coding genes that GENCODE places on the same chromosome.
    member = member.merge(genes[["gene", "chrom", "tss"]], on=["gene", "chrom"])
    return member.rename(columns={"tss": "gene_start"})


def from_bed(genes):
    """Plain TAD calls: a gene belongs to every TAD that contains its TSS."""
    tads = pd.read_csv(RAW / "tads.bed", sep="\t", header=None, usecols=[0, 1, 2],
                       names=["chrom", "tad_start", "tad_end"], comment="#")
    tads["chrom"] = tads["chrom"].astype(str)
    if not tads["chrom"].str.startswith("chr").all():  # "7" -> "chr7", to match GENCODE
        tads["chrom"] = "chr" + tads["chrom"].str.replace("chr", "", regex=False)
    tads = tads.drop_duplicates().sort_values(["chrom", "tad_start"])
    tads["tad_id"] = tads["chrom"] + ":" + tads["tad_start"].astype(str) + "-" + tads["tad_end"].astype(str)
    print(f"TADs in the BED file: {len(tads)}")
    con = duckdb.connect()
    con.register("tads", tads)
    con.register("genes", genes)
    return con.execute("""
        SELECT t.tad_id, t.chrom, t.tad_start, t.tad_end, g.gene, g.tss AS gene_start
        FROM genes g
        JOIN tads t ON g.chrom = t.chrom AND g.tss0 >= t.tad_start AND g.tss0 < t.tad_end
    """).df()


genes = read_gencode_genes()
print(f"Protein-coding genes in GENCODE: {len(genes)}")

member = from_union_file(genes) if (RAW / "union_tads.txt").exists() else from_bed(genes)

# 1. Keep TADs with at least MIN_GENES protein-coding genes.
member["n_genes"] = member.groupby("tad_id")["gene"].transform("size")
n_with_gene = member["tad_id"].nunique()
member = member[member["n_genes"] >= MIN_GENES]
print(f"TADs with at least 1 protein-coding gene: {n_with_gene}; with at least {MIN_GENES}: {member['tad_id'].nunique()}")

# 2. For each gene: the smallest of those TADs (ties: fewer genes, then the id).
member["tad_bp"] = member["tad_end"] - member["tad_start"]
gene_tad = (member.sort_values(["gene", "tad_bp", "n_genes", "tad_id"])
            .drop_duplicates("gene")[["gene", "tad_id"]])

# 3. Keep the TADs that are used for at least one gene, plus any ids listed in data/raw/extra_tads.txt.
keep = set(gene_tad["tad_id"])
extra_file = RAW / "extra_tads.txt"  # optional: one TAD id per line, e.g. a TAD from your thesis
if extra_file.exists():
    for line in extra_file.read_text().split():
        chrom, start, end = line.replace(":", "_").replace("-", "_").rsplit("_", 2)
        tad_id = f"{chrom}:{start}-{end}"
        if tad_id in set(member["tad_id"]):
            keep.add(tad_id)
        else:
            print(f"extra_tads.txt: {line} is not a TAD with at least {MIN_GENES} protein-coding genes; skipped")
tad_genes = member[member["tad_id"].isin(keep)]
tad_genes = tad_genes[["tad_id", "chrom", "tad_start", "tad_end", "gene", "gene_start"]]
tad_genes = tad_genes.sort_values(["chrom", "tad_start", "tad_end", "gene_start"])

tad_genes.to_parquet(DATA / "tad_genes.parquet", index=False)
gene_tad.to_parquet(DATA / "gene_tad.parquet", index=False)

per_tad = tad_genes.groupby("tad_id").size()
kb = (tad_genes.drop_duplicates("tad_id")["tad_end"] - tad_genes.drop_duplicates("tad_id")["tad_start"]) / 1000
print(f"TADs kept (the tightest TAD of at least one gene): {len(per_tad)}")
print(f"Genes with a TAD: {len(gene_tad)} of {len(genes)}")
print(f"Genes per TAD: median {per_tad.median():.0f}, max {per_tad.max()}; TAD size: median {kb.median():.0f} kb")
bins = pd.cut(per_tad, [2, 5, 10, 20, 40, 10**6], labels=["3-5", "6-10", "11-20", "21-40", "41+"])
print("TADs per size bin:", bins.value_counts().sort_index().to_dict())
