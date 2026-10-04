"""Literature tool: search PubMed, so general answers can point to real papers."""
import json
import urllib.parse
import urllib.request

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"  # NCBI's free search service for PubMed


def _get(endpoint, **params):
    """Call one NCBI E-utilities endpoint and return its JSON answer."""
    params.update({"retmode": "json", "tool": "tadtalk"})
    url = f"{EUTILS}/{endpoint}.fcgi?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.load(response)


def pubmed_search(query: str, max_results: int = 5) -> dict:
    """Search PubMed for published papers and return their titles, journals, years and PMIDs.

    Use for general questions about genomics or cancer biology when a claim should be backed by papers,
    when the user asks for evidence or references, or to check what is known about a gene, TAD or drug
    found by the other tools. query: a few plain keywords, e.g. "EGFR erlotinib resistance lung" or
    "topologically associating domain disruption oncogene". Cite papers only by the PMIDs returned here.
    """
    n = max(1, min(int(max_results), 10))
    try:
        ids = _get("esearch", db="pubmed", term=query, retmax=n, sort="relevance").get("esearchresult", {}).get("idlist", [])
        if not ids:
            return {"query": query, "n_papers": 0, "papers": [], "note": "No PubMed results. Try fewer or simpler keywords."}
        summary = _get("esummary", db="pubmed", id=",".join(ids)).get("result", {})
    except Exception as e:  # no internet, a timeout, or NCBI is busy
        return {"error": f"PubMed search failed ({type(e).__name__}: {e}). Answer without references and say so."}
    papers = []
    for pmid in ids:
        item = summary.get(pmid, {})
        if not item.get("title"):
            continue
        authors = [a.get("name", "") for a in item.get("authors", [])]
        papers.append({
            "pmid": pmid,
            "title": item["title"],
            "journal": item.get("source", ""),
            "year": str(item.get("pubdate", ""))[:4],
            "authors": ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return {"query": query, "n_papers": len(papers), "papers": papers,
            "note": "Titles only: do not claim details that a title does not state."}


if __name__ == "__main__":
    print(json.dumps(pubmed_search("topologically associating domain disruption oncogene activation", 3), indent=1))
