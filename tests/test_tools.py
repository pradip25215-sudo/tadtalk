"""Automatic checks for the tools. Run:  python -m pytest -q"""
import json
from pathlib import Path

import numpy as np

from tools.find_tad import find_tad
from tools.statguard import statguard

# Change these to a gene, drug, lineage and cancer that exist in YOUR data.
GENE, DRUG, LINEAGE, CANCER = "EGFR", "erlotinib", "Lung", "HNSC"


def check_result(r):
    """Every tool result must be an error OR a JSON-safe dict (the agent sends it to the LLM as JSON)."""
    assert isinstance(r, dict)
    json.dumps(r)
    if "plot" in r:
        assert Path(r["plot"]).exists()


# ---- Day 1
def test_find_tad():
    r = find_tad(GENE)
    assert "error" not in r, r
    assert GENE in r["genes"]
    check_result(r)


def test_unknown_gene_gives_error():
    assert "error" in find_tad("NOTAGENE123")


# ---- Day 4
def test_statguard_strong_result_passes():
    null = np.random.default_rng(0).normal(0, 0.1, 200)
    assert statguard(0.6, null, null, p_obs=0.001)["verdict"] == "PASS"


def test_statguard_typical_result_fails():
    null = np.random.default_rng(0).normal(0, 0.1, 200)
    assert statguard(0.05, null, null, p_obs=0.3)["verdict"] == "FAIL"


def test_statguard_one_sided():
    null = np.random.default_rng(0).normal(0, 0.1, 200)
    assert statguard(-0.8, null, null, p_obs=0.001, two_sided=False)["verdict"] == "PASS"
    assert statguard(0.8, null, null, p_obs=0.001, two_sided=False)["verdict"] == "FAIL"


def test_drug_link():
    from tools.drug import drug_link
    r = drug_link(GENE, DRUG, LINEAGE)
    assert "error" not in r, r
    assert r["n_cell_lines"] >= 15 and 0 < r["p_tad"] <= 1 and -1 <= r["rho_tad"] <= 1
    check_result(r)


def test_top_tads_for_drug():
    from tools.drug import top_tads_for_drug
    r = top_tads_for_drug(DRUG, LINEAGE, k=5)
    assert len(r["top_tads"]) == 5
    check_result(r)


def test_bad_drug_name_gives_error():
    from tools.drug import drug_link
    assert "error" in drug_link(GENE, "notadrug", LINEAGE)


# ---- Day 5
def test_survival():
    from tools.survival import survival
    r = survival(GENE, CANCER)
    assert "error" not in r, r
    assert r["hr_per_sd"] > 0 and r["hr_ci95"][0] <= r["hr_per_sd"] <= r["hr_ci95"][1]
    check_result(r)


def test_crispr_link():
    from tools.crispr import crispr_link
    r = crispr_link(GENE, LINEAGE)
    assert "error" not in r, r
    assert 0 <= r["share_essential"] <= 1
    check_result(r)


# ---- literature (no internet needed: the NCBI answer is faked)
def test_pubmed_search(monkeypatch):
    from tools import literature
    fake = {"esearch": {"esearchresult": {"idlist": ["22495300"]}},
            "esummary": {"result": {"uids": ["22495300"], "22495300": {
                "title": "Topological domains in mammalian genomes.", "source": "Nature", "pubdate": "2012 Apr 11",
                "authors": [{"name": "Dixon JR"}, {"name": "Selvaraj S"}, {"name": "Yue F"}, {"name": "Kim A"}]}}}}
    monkeypatch.setattr(literature, "_get", lambda endpoint, **params: fake[endpoint])
    r = literature.pubmed_search("TAD")
    assert r["papers"][0]["pmid"] == "22495300" and r["papers"][0]["year"] == "2012"
    check_result(r)


def test_pubmed_search_offline_gives_error(monkeypatch):
    from tools import literature

    def no_internet(endpoint, **params):
        raise OSError("no internet")
    monkeypatch.setattr(literature, "_get", no_internet)
    assert "error" in literature.pubmed_search("TAD")


def test_reference_check():
    from langchain_core.messages import ToolMessage

    from agent.agent import reference_check
    found = ToolMessage(content=json.dumps({"papers": [{"pmid": "22495300"}, {"pmid": "26700815"}]}), tool_call_id="1")
    text = "TADs were described in 2012 (PMID 22495300, 26700815) and later (PMID: 11111111)."
    verified, unverified = reference_check(text, [found])
    assert verified == ["22495300", "26700815"] and unverified == ["11111111"]


def test_gene_alias():
    from tools.common import official_symbol, resolve_target
    assert official_symbol("her2") == "ERBB2" and official_symbol("EGFR") == "EGFR"
    by_alias, _ = resolve_target("HER2")
    by_symbol, _ = resolve_target("ERBB2")
    assert by_alias["tad_id"] == by_symbol["tad_id"] and by_alias["gene"] == "ERBB2"
    assert find_tad("HER2")["gene"] == "ERBB2"
