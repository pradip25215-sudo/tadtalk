"""Helpers shared by all tools: SQL on the data folder, TAD lookup, cell-line groups, plots."""
import re

import duckdb
import matplotlib

matplotlib.use("Agg")  # draw plots into files, never open a window
import matplotlib.pyplot as plt  # noqa: E402

from config import DATA, PLOTS  # noqa: E402

MIN_N = 15  # fewer samples than this = too noisy to report a correlation

# Size bins: a TAD is only compared with null gene sets of similar size.
BINS = [(3, 5, "3-5"), (6, 10, "6-10"), (11, 20, "11-20"), (21, 40, "21-40"), (41, 10**6, "41+")]

# Every Parquet file in data/ becomes a SQL table named after the file:
# data/tad_genes.parquet -> table tad_genes, data/models.parquet -> table models, ...
_con = duckdb.connect()
_TABLES = set()
for _f in sorted(DATA.glob("*.parquet")):
    _con.execute(f"CREATE VIEW {_f.stem} AS SELECT * FROM read_parquet('{_f.as_posix()}')")
    _TABLES.add(_f.stem)


def q(sql, params=None):
    """Run SQL on the data tables and return a pandas DataFrame."""
    return _con.cursor().execute(sql, params or []).df()


def size_bin(n_genes):
    """Name of the size bin for a gene set with n_genes genes (None if fewer than 3)."""
    for lo, hi, name in BINS:
        if lo <= n_genes <= hi:
            return name
    return None


def tad_id_text(text):
    """Accept both 'chr7:55000000-55500000' and the lab style 'chr7_55000000_55500000'."""
    t = text.strip()
    if t.lower().startswith("chr") and t.count("_") >= 2 and ":" not in t:
        chrom, start, end = t.rsplit("_", 2)
        return f"{chrom}:{start}-{end}"
    return t


def tad_of_gene(gene):
    """The TAD id used for a gene: the smallest TAD around it with at least 3 genes. None if the gene has no TAD."""
    if "gene_tad" in _TABLES:
        rows = q("SELECT tad_id FROM gene_tad WHERE upper(gene) = upper(?)", [gene.strip()])
    else:  # older data folder without gene_tad: every gene sits in exactly one TAD
        rows = q("SELECT DISTINCT tad_id FROM tad_genes WHERE upper(gene) = upper(?)", [gene.strip()])
    return None if rows.empty else rows["tad_id"].iloc[0]


def resolve_target(target):
    """Turn a gene symbol or a TAD id into its TAD.

    Returns (info, None) when found, or (None, {"error": ...}) when not.
    info = tad_id, genes, gene (the gene asked about, or None), size_bin.
    """
    t = target.strip()
    tad_id = tad_of_gene(t) or tad_id_text(t)
    rows = q("SELECT tad_id, gene FROM tad_genes WHERE upper(tad_id) = upper(?) ORDER BY gene_start", [tad_id])
    if rows.empty:
        return None, {"error": f"'{target}' is not a gene or TAD id in this TAD set."}
    genes = rows["gene"].tolist()
    asked_gene = next((g for g in genes if g.upper() == t.upper()), None)
    return {
        "tad_id": rows["tad_id"].iloc[0],
        "genes": genes,
        "gene": asked_gene,
        "size_bin": size_bin(len(genes)),
    }, None


def models_in(lineage):
    """DepMap model ids of one cell-line lineage, e.g. 'Lung'. 'all' = every lineage."""
    if lineage.strip().lower() in {"all", "any", "pan-cancer", "pancancer"}:
        return q("SELECT model_id FROM models")["model_id"].tolist()
    sql = "SELECT model_id FROM models WHERE lower(lineage) = lower(?)"
    return q(sql, [lineage.strip()])["model_id"].tolist()


def lineages():
    """All cell-line lineages in the data."""
    return q("SELECT DISTINCT lineage FROM models WHERE lineage IS NOT NULL ORDER BY 1")["lineage"].tolist()


def cancers():
    """All TCGA cancer codes that have TAD activity scores."""
    return q("SELECT DISTINCT cancer FROM tcga_activity ORDER BY 1")["cancer"].tolist()


def null_ids(bin_name):
    """Ids and kinds (random / block) of the null gene sets in one size bin."""
    return q("SELECT DISTINCT set_id, kind FROM sets WHERE kind <> 'tad' AND size_bin = ?", [bin_name])


def r3(x):
    """Round to 3 decimals and return a plain Python float (JSON-safe)."""
    return round(float(x), 3)


def sig3(p):
    """Keep 3 significant digits of a p-value, e.g. 0.000123456 -> 0.000123."""
    return float(f"{float(p):.3g}")


def save_plot(fig, name):
    """Save a matplotlib figure into plots/ and return the file path as text."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    path = PLOTS / f"{safe}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)
