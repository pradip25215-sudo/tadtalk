"""StatGuard: is a TAD's result stronger than random gene sets and random neighbour blocks?"""
import numpy as np


def statguard(obs, null_random, null_block, p_obs=0.0, alpha=0.05, two_sided=True):
    """Compare one statistic with the same statistic for null gene sets of the same size.

    obs          the TAD's statistic (Spearman rho, Cox z, or mean CRISPR effect)
    null_random  the statistic for random gene sets (genes from anywhere)
    null_block   the statistic for neighbour blocks (consecutive genes, TAD borders ignored)
    p_obs        the ordinary p-value (or FDR) of the TAD's own test
    two_sided    True: a stronger value in either direction counts
                 False: only LOWER values count (e.g. more negative CRISPR effect = more essential)
    PASS only if p_obs < alpha AND the TAD beats both nulls (empirical p < alpha).
    """
    nr = np.asarray(null_random, dtype=float)
    nb = np.asarray(null_block, dtype=float)
    nr, nb = nr[~np.isnan(nr)], nb[~np.isnan(nb)]

    def empirical_p(null):
        # (1 + number of null sets at least as extreme) / (1 + number of null sets)
        if two_sided:
            as_extreme = np.sum(np.abs(null) >= abs(obs))
        else:
            as_extreme = np.sum(null <= obs)
        return (1 + as_extreme) / (1 + len(null))

    p_random, p_block = empirical_p(nr), empirical_p(nb)
    passed = p_obs < alpha and p_random < alpha and p_block < alpha
    return {
        "p_random": round(float(p_random), 4),
        "p_block": round(float(p_block), 4),
        "n_null_sets": int(len(nr) + len(nb)),
        "verdict": "PASS" if passed else "FAIL",
    }
