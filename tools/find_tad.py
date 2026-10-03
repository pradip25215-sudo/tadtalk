"""Tool 1: which TAD holds a gene, and which genes share it."""
from tools.common import q


def find_tad(gene: str) -> dict:
    """Find the TAD (topologically associating domain) that contains a gene.

    Use for questions like "Which TAD holds EGFR?" or "Which genes share a TAD with MYC?".
    Returns the TAD id (chrom:start-end), its size and all genes in it, in genome order.
    """
    df = q(
        """SELECT tad_id, chrom, tad_start, tad_end, gene FROM tad_genes
           WHERE tad_id IN (SELECT tad_id FROM tad_genes WHERE upper(gene) = upper(?))
           ORDER BY gene_start""",
        [gene.strip()],
    )
    if df.empty:
        return {"error": f"{gene} is not in any TAD of this TAD set. Check the gene symbol."}
    first = df.iloc[0]
    return {
        "gene": gene.strip().upper(),
        "tad_id": first["tad_id"],
        "size_kb": round((int(first["tad_end"]) - int(first["tad_start"])) / 1000),
        "n_genes": len(df),
        "genes": df["gene"].tolist(),
    }


if __name__ == "__main__":
    print(find_tad("EGFR"))
