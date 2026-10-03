"""Drug tools: link TAD activity to drug response (CTRPv2 AAC) in cancer cell lines."""
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

from tools.common import MIN_N, lineages, models_in, null_ids, q, r3, resolve_target, save_plot, sig3
from tools.statguard import statguard


def _drug_response(drug, models):
    """AAC of one drug in the given cell lines. Returns (table model_id/aac, None) or (None, error)."""
    d = q("SELECT model_id, aac FROM drug_aac WHERE upper(drug) = upper(?)", [drug.strip()])
    if d.empty:
        like = q("SELECT DISTINCT drug FROM drug_aac WHERE drug ILIKE ? ORDER BY 1 LIMIT 8",
                 [f"%{drug.strip()[:4]}%"])["drug"].tolist()
        return None, {"error": f"Drug '{drug}' is not in CTRPv2.", "similar_names": like}
    return d[d["model_id"].isin(models)], None


def _spearman_per_set(activity, response):
    """Spearman rho between activity and AAC for every set_id. Returns a Series indexed by set_id."""
    d = activity.merge(response, on="model_id")
    return d.groupby("set_id")[["score", "aac"]].apply(lambda g: g["score"].corr(g["aac"], method="spearman"))


def _null_rhos(bin_name, response):
    """Rho of every null set in one size bin, split into random sets and neighbour blocks."""
    ids = null_ids(bin_name)
    act = q("""SELECT a.set_id, a.model_id, a.score FROM ccle_activity a
               JOIN (SELECT DISTINCT set_id FROM sets WHERE kind <> 'tad' AND size_bin = ?) s USING (set_id)
               WHERE list_contains(?, a.model_id)""",
            [bin_name, response["model_id"].tolist()])
    rho = _spearman_per_set(act, response)
    kind = ids.set_index("set_id")["kind"]
    return rho[kind.reindex(rho.index) == "random"], rho[kind.reindex(rho.index) == "block"]


def drug_link(target: str, drug: str, lineage: str = "all") -> dict:
    """Test if a TAD's activity is linked to response to one drug in cancer cell lines.

    target: a gene symbol (its TAD is used) or a TAD id.  drug: a CTRPv2 drug name, e.g. "erlotinib".
    lineage: a cell-line lineage such as "Lung" or "Breast", or "all".
    Returns Spearman rho for the TAD and for the gene alone, the StatGuard check and a scatter plot.
    Drug response is AAC: higher AAC = the drug kills the cells more. Positive rho = higher TAD
    activity goes with more sensitivity.
    """
    tad, err = resolve_target(target)
    if err:
        return err
    if tad["size_bin"] is None:
        return {"error": f"TAD {tad['tad_id']} has fewer than 3 genes, so it has no activity score."}
    models = models_in(lineage)
    if not models:
        return {"error": f"Unknown lineage '{lineage}'.", "valid_lineages": lineages() + ["all"]}
    resp, err = _drug_response(drug, models)
    if err:
        return err

    act = q("SELECT model_id, score FROM ccle_activity WHERE set_id = ?", [tad["tad_id"]])
    d = act.merge(resp, on="model_id")
    if len(d) < MIN_N:
        return {"error": f"Only {len(d)} {lineage} cell lines have both TAD activity and {drug} data; "
                         f"need at least {MIN_N}. Try lineage 'all'."}
    rho, p = spearmanr(d["score"], d["aac"])

    result = {"target": target, "tad_id": tad["tad_id"], "n_genes_in_tad": len(tad["genes"]),
              "drug": drug, "lineage": lineage, "n_cell_lines": len(d),
              "rho_tad": r3(rho), "p_tad": sig3(p)}

    if tad["gene"]:  # the single gene, for comparison with its TAD
        e = q("SELECT model_id, expr FROM ccle_expr WHERE gene = ?", [tad["gene"]]).merge(resp, on="model_id")
        if len(e) >= MIN_N:
            rho_g, p_g = spearmanr(e["expr"], e["aac"])
            result.update({"rho_gene": r3(rho_g), "p_gene": sig3(p_g)})

    null_random, null_block = _null_rhos(tad["size_bin"], resp)
    result.update(statguard(rho, null_random, null_block, p_obs=p))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(d["score"], d["aac"], s=18, alpha=0.7)
    ax.set_xlabel(f"TAD activity (GSVA)\n{tad['tad_id']}")
    ax.set_ylabel(f"{drug} response (AAC)")
    ax.set_title(f"{lineage} cell lines: rho = {rho:.2f}, n = {len(d)}")
    result["plot"] = save_plot(fig, f"drug_{tad['tad_id']}_{drug}_{lineage}")
    result["note"] = "Cell-line result: a hypothesis, not treatment advice."
    return result


def top_tads_for_drug(drug: str, lineage: str = "all", k: int = 10) -> dict:
    """Rank all TADs by how strongly their activity is linked to response to one drug.

    Use for "Which TADs are linked to lapatinib response in breast cell lines?".
    Returns the top k TADs with Spearman rho, p, FDR (Benjamini-Hochberg over all TADs) and the
    StatGuard check, plus a bar plot. Higher AAC = more sensitive, so positive rho = more sensitive.
    """
    models = models_in(lineage)
    if not models:
        return {"error": f"Unknown lineage '{lineage}'.", "valid_lineages": lineages() + ["all"]}
    resp, err = _drug_response(drug, models)
    if err:
        return err
    if len(resp) < MIN_N:
        return {"error": f"Only {len(resp)} {lineage} cell lines have {drug} data; need at least {MIN_N}."}

    act = q("""SELECT a.set_id, a.model_id, a.score FROM ccle_activity a
               JOIN (SELECT DISTINCT set_id FROM sets WHERE kind = 'tad') s USING (set_id)
               WHERE list_contains(?, a.model_id)""", [resp["model_id"].tolist()])
    d = act.merge(resp, on="model_id")

    def test(g):
        r = spearmanr(g["score"], g["aac"])
        return pd.Series({"rho": r.statistic, "p": r.pvalue, "n": len(g)})

    stats = d.groupby("set_id")[["score", "aac"]].apply(test).dropna()
    stats["fdr"] = multipletests(stats["p"], method="fdr_bh")[1]
    top = stats.sort_values("p").head(int(k))

    info = q("SELECT DISTINCT set_id, size_bin FROM sets WHERE kind = 'tad'").set_index("set_id")["size_bin"]
    genes = q("SELECT tad_id, list(gene ORDER BY gene_start) AS genes FROM tad_genes GROUP BY tad_id")
    genes = genes.set_index("tad_id")["genes"]
    nulls = {}
    rows = []
    for tad_id, s in top.iterrows():
        b = info[tad_id]
        if b not in nulls:
            nulls[b] = _null_rhos(b, resp)
        guard = statguard(s["rho"], *nulls[b], p_obs=s["fdr"])
        rows.append({"tad_id": tad_id, "genes": list(genes[tad_id])[:8], "rho": r3(s["rho"]),
                     "p": sig3(s["p"]), "fdr": sig3(s["fdr"]),
                     "p_random": guard["p_random"], "p_block": guard["p_block"], "verdict": guard["verdict"]})

    fig, ax = plt.subplots(figsize=(6, 0.35 * len(rows) + 1.2))
    labels = [f"{r['tad_id']} ({r['genes'][0]}...)" for r in rows][::-1]
    colors = ["tab:green" if r["verdict"] == "PASS" else "tab:gray" for r in rows][::-1]
    ax.barh(labels, [r["rho"] for r in rows][::-1], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(f"Spearman rho with {drug} AAC (green = StatGuard PASS)")
    ax.set_title(f"Top TADs for {drug}, {lineage} cell lines")
    return {"drug": drug, "lineage": lineage, "n_cell_lines": int(stats["n"].max()),
            "n_tads_tested": len(stats), "top_tads": rows,
            "plot": save_plot(fig, f"top_tads_{drug}_{lineage}"),
            "note": "FDR corrects for testing every TAD. Cell-line result: a hypothesis, not treatment advice."}


if __name__ == "__main__":
    print(drug_link("EGFR", "erlotinib", "Lung"))
