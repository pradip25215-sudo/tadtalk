"""Survival tool: does a TAD's activity predict overall survival of TCGA patients?"""
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

from tools.common import cancers, q, r3, resolve_target, save_plot, sig3
from tools.statguard import statguard


def survival(target: str, cancer: str) -> dict:
    """Test if a TAD's activity predicts overall survival of TCGA patients with one cancer type.

    target: a gene symbol (its TAD is used) or a TAD id.  cancer: a TCGA code such as "HNSC", "BRCA", "LUAD".
    Returns the Cox hazard ratio per 1 SD of TAD activity (HR > 1 = higher activity, worse survival)
    with 95% CI and p, a log-rank test for high vs low activity (median split), the StatGuard check
    and a Kaplan-Meier plot.
    """
    tad, err = resolve_target(target)
    if err:
        return err
    if tad["size_bin"] is None:
        return {"error": f"TAD {tad['tad_id']} has fewer than 3 genes, so it has no activity score."}
    c = cancer.strip().upper()
    act = q("SELECT sample, score FROM tcga_activity WHERE cancer = ? AND set_id = ?", [c, tad["tad_id"]])
    if act.empty:
        if c not in cancers():
            return {"error": f"No TCGA data for cancer '{cancer}'.", "valid_cancers": cancers()}
        return {"error": f"TAD {tad['tad_id']} has no activity score in TCGA-{c}: fewer than 3 of its genes are in the "
                         "TCGA expression table (it uses some older gene names). Try another gene or TAD."}
    d = act.merge(q("SELECT sample, OS, OS_time FROM survival WHERE cancer = ?", [c]), on="sample")
    deaths = int(d["OS"].sum())
    if len(d) < 30 or deaths < 10:
        return {"error": f"Too few patients ({len(d)}) or deaths ({deaths}) in {c} for a survival test."}

    d["x"] = (d["score"] - d["score"].mean()) / d["score"].std()  # 1 unit = 1 SD of TAD activity
    cph = CoxPHFitter().fit(d[["OS_time", "OS", "x"]], duration_col="OS_time", event_col="OS")
    s = cph.summary.loc["x"]

    high = d["score"] > d["score"].median()
    lr = logrank_test(d.loc[high, "OS_time"], d.loc[~high, "OS_time"], d.loc[high, "OS"], d.loc[~high, "OS"])

    nz = q("SELECT kind, z FROM null_cox WHERE cancer = ? AND size_bin = ?", [c, tad["size_bin"]])
    guard = statguard(s["z"], nz.loc[nz["kind"] == "random", "z"], nz.loc[nz["kind"] == "block", "z"],
                      p_obs=s["p"])

    fig, ax = plt.subplots(figsize=(5, 4))
    for mask, label in [(high, "High TAD activity"), (~high, "Low TAD activity")]:
        km = KaplanMeierFitter().fit(d.loc[mask, "OS_time"] / 365.25, d.loc[mask, "OS"],
                                     label=f"{label} (n={int(mask.sum())})")
        km.plot_survival_function(ax=ax, ci_show=False)
    ax.set_xlabel("Years")
    ax.set_ylabel("Overall survival")
    ax.set_title(f"TCGA-{c}: {tad['tad_id']}\nHR = {s['exp(coef)']:.2f} per SD, log-rank p = {lr.p_value:.2g}")

    return {"target": target, "tad_id": tad["tad_id"], "cancer": c, "n_patients": len(d), "deaths": deaths,
            "hr_per_sd": r3(s["exp(coef)"]), "hr_ci95": [r3(s["exp(coef) lower 95%"]), r3(s["exp(coef) upper 95%"])],
            "p_cox": sig3(s["p"]), "p_logrank": sig3(lr.p_value), **guard,
            "plot": save_plot(fig, f"survival_{tad['tad_id']}_{c}"),
            "note": "HR > 1: higher TAD activity goes with shorter survival. Association, not causation."}


if __name__ == "__main__":
    print(survival("CCND1", "HNSC"))
