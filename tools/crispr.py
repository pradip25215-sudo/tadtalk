"""CRISPR tools: are the genes of a TAD essential for cancer cell growth (DepMap gene effect)?"""
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from tools.common import lineages, models_in, q, r3, resolve_target, save_plot, sig3
from tools.statguard import statguard

ESSENTIAL = -0.5  # gene effect below this = the gene is needed for growth


def _gene_means(models):
    """Mean gene effect of every gene across the screened cell lines. Returns (table gene/effect, n lines)."""
    gm = q("""SELECT gene, avg(effect) AS effect FROM crispr
              WHERE list_contains(?, model_id) GROUP BY gene""", [models])
    n = q("SELECT count(DISTINCT model_id) AS n FROM crispr WHERE list_contains(?, model_id)", [models])["n"][0]
    return gm, int(n)


def _null_means(bin_name, gm):
    """Mean effect of every null set in one size bin, split into random sets and neighbour blocks."""
    ns = q("SELECT set_id, kind, gene FROM sets WHERE kind <> 'tad' AND size_bin = ?", [bin_name])
    m = ns.merge(gm, on="gene").groupby(["set_id", "kind"], as_index=False)["effect"].mean()
    return m.loc[m["kind"] == "random", "effect"], m.loc[m["kind"] == "block", "effect"]


def crispr_link(target: str, lineage: str = "all") -> dict:
    """Test if the genes of one TAD are essential (needed for growth) in cancer cell lines, using DepMap CRISPR.

    target: a gene symbol (its TAD is used) or a TAD id.  lineage: e.g. "Breast", or "all".
    Gene effect (Chronos): 0 = no effect, -1 = a typical essential gene; more negative = more essential.
    Returns each gene's mean effect, the TAD mean, the share of genes below -0.5, and the StatGuard
    check (one-sided: is the TAD more essential than random gene sets and neighbour blocks?).
    """
    tad, err = resolve_target(target)
    if err:
        return err
    if tad["size_bin"] is None:
        return {"error": f"TAD {tad['tad_id']} has fewer than 3 genes."}
    models = models_in(lineage)
    if not models:
        return {"error": f"Unknown lineage '{lineage}'.", "valid_lineages": lineages() + ["all"]}

    gm, n_lines = _gene_means(models)
    if n_lines == 0:
        return {"error": f"No {lineage} cell lines have CRISPR data."}
    in_tad = gm[gm["gene"].isin(tad["genes"])].sort_values("effect")
    if len(in_tad) < 3:
        return {"error": f"Fewer than 3 genes of {tad['tad_id']} were screened by CRISPR."}
    others = gm.loc[~gm["gene"].isin(tad["genes"]), "effect"]
    obs = in_tad["effect"].mean()
    p = mannwhitneyu(in_tad["effect"], others, alternative="less").pvalue  # TAD genes lower than the rest?
    guard = statguard(obs, *_null_means(tad["size_bin"], gm), p_obs=p, two_sided=False)

    fig, ax = plt.subplots(figsize=(5, 0.3 * len(in_tad) + 1.2))
    ax.barh(in_tad["gene"][::-1], in_tad["effect"][::-1], color="tab:purple")
    ax.axvline(ESSENTIAL, color="red", ls="--", lw=1)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(f"Mean CRISPR gene effect, {lineage} cell lines\n(red line = -0.5, essential cut-off)")
    ax.set_title(f"{tad['tad_id']}")

    return {"target": target, "tad_id": tad["tad_id"], "lineage": lineage, "n_cell_lines": n_lines,
            "gene_effects": {g: r3(e) for g, e in zip(in_tad["gene"], in_tad["effect"])},
            "tad_mean_effect": r3(obs), "share_essential": r3((in_tad["effect"] < ESSENTIAL).mean()),
            "p_vs_other_genes": sig3(p), **guard,
            "plot": save_plot(fig, f"crispr_{tad['tad_id']}_{lineage}"),
            "note": "Cell-line result: shows dependency, not a drug target by itself."}


def top_essential_tads(lineage: str = "all", k: int = 10) -> dict:
    """Rank all TADs by how essential their genes are (lowest mean CRISPR gene effect) in one lineage.

    Use for "Which TADs are most essential in breast cancer cell lines?".
    Returns the top k TADs with mean effect, share of essential genes and the StatGuard check.
    """
    models = models_in(lineage)
    if not models:
        return {"error": f"Unknown lineage '{lineage}'.", "valid_lineages": lineages() + ["all"]}
    gm, n_lines = _gene_means(models)
    if n_lines == 0:
        return {"error": f"No {lineage} cell lines have CRISPR data."}
    tads = q("SELECT set_id, size_bin, gene FROM sets WHERE kind = 'tad'").merge(gm, on="gene")
    per_tad = tads.groupby(["set_id", "size_bin"], as_index=False).agg(
        mean_effect=("effect", "mean"), share_essential=("effect", lambda e: (e < ESSENTIAL).mean()),
        n_genes=("gene", "size"))
    top = per_tad[per_tad["n_genes"] >= 3].nsmallest(int(k), "mean_effect")

    nulls, rows = {}, []
    for t in top.itertuples():
        if t.size_bin not in nulls:
            nulls[t.size_bin] = _null_means(t.size_bin, gm)
        in_tad = tads["set_id"] == t.set_id
        others = gm.loc[~gm["gene"].isin(tads.loc[in_tad, "gene"]), "effect"]
        p = mannwhitneyu(tads.loc[in_tad, "effect"], others, alternative="less").pvalue
        guard = statguard(t.mean_effect, *nulls[t.size_bin], p_obs=p, two_sided=False)
        rows.append({"tad_id": t.set_id, "n_genes": int(t.n_genes), "mean_effect": r3(t.mean_effect),
                     "share_essential": r3(t.share_essential), "p": sig3(p), **guard})
    return {"lineage": lineage, "n_cell_lines": n_lines, "top_tads": rows,
            "note": "Ranked by mean gene effect; TADs full of known common-essential genes rank high in every lineage."}


if __name__ == "__main__":
    print(crispr_link("ERBB2", "Breast"))
