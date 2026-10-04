"""Tool 1: which TAD holds a gene, and which genes share it."""
from tools.common import q, tad_of_gene


def find_tad(gene: str) -> dict:
    """Find the TAD (topologically associating domain) that contains a gene.

    Use for questions like "Which TAD holds EGFR?" or "Which genes share a TAD with MYC?".
    Returns the TAD id (chrom:start-end), its size and all protein-coding genes in it, in genome order.
    TADs in this set are nested, so a gene can sit in several TADs: the one returned is the smallest TAD
    around the gene with at least 3 genes, and other_tads lists the larger ones (their ids work as a target
    in the other tools).
    """
    name = gene.strip()
    tad_id = tad_of_gene(name)
    if tad_id is None:
        return {"error": f"{gene} is not in any TAD of this TAD set (a TAD needs at least 3 protein-coding genes). "
                         "Check the gene symbol."}
    df = q("SELECT tad_id, tad_start, tad_end, gene FROM tad_genes WHERE tad_id = ? ORDER BY gene_start", [tad_id])
    others = q("""SELECT DISTINCT tad_id FROM tad_genes WHERE upper(gene) = upper(?) AND tad_id <> ?
                  ORDER BY tad_end - tad_start""", [name, tad_id])["tad_id"].tolist()
    first = df.iloc[0]
    result = {
        "gene": name.upper(),
        "tad_id": tad_id,
        "size_kb": round((int(first["tad_end"]) - int(first["tad_start"])) / 1000),
        "n_genes": len(df),
        "genes": df["gene"].tolist(),
    }
    if others:
        result["other_tads"] = others[:5]
    return result


if __name__ == "__main__":
    print(find_tad("EGFR"))
