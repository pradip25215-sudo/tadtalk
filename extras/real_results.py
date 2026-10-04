"""Print real results straight from the tools (no LLM, no API cost), to choose demo and benchmark examples.

Run:  python -m extras.real_results 2>&1 | tee real_results.txt
"""
import json
import time

from tools.common import q
from tools.crispr import crispr_link, top_essential_tads
from tools.drug import drug_link, top_tads_for_drug
from tools.find_tad import find_tad
from tools.survival import survival


def show(title, r):
    r = dict(r)
    for k in ("genes", "gene_effects"):  # keep long lists short
        if k in r and len(r[k]) > 12:
            r[k] = list(r[k])[:12] + ["..."]
    print(f"\n### {title}\n{json.dumps(r, indent=1, default=str)}", flush=True)


print("lineages:", q("SELECT lineage, count(*) n FROM models GROUP BY 1 ORDER BY 2 DESC")["lineage"].tolist())
print("drug names:", q("""SELECT DISTINCT drug FROM drug_aac WHERE drug ILIKE '%erlo%' OR drug ILIKE '%lapa%'
      OR drug ILIKE '%gefi%' OR drug ILIKE '%afat%' OR drug ILIKE '%pacli%' OR drug ILIKE '%tram%'
      OR drug ILIKE '%nerat%' ORDER BY 1""")["drug"].tolist())
for g in ["EGFR", "ERBB2", "CCND1", "MYC", "KRAS", "TP53", "PIK3CA", "BRCA1"]:
    show(f"find_tad {g}", find_tad(g))
for t, d, l in [("EGFR", "erlotinib", "Lung"), ("EGFR", "erlotinib", "all"), ("ERBB2", "lapatinib", "Breast"),
                ("ERBB2", "lapatinib", "all"), ("EGFR", "gefitinib", "Lung"), ("EGFR", "afatinib", "Head and Neck")]:
    show(f"drug_link {t} {d} {l}", drug_link(t, d, l))
for t, c in [("CCND1", "HNSC"), ("EGFR", "HNSC"), ("EGFR", "LUAD"), ("ERBB2", "BRCA"), ("MYC", "BRCA")]:
    show(f"survival {t} {c}", survival(t, c))
for t, l in [("ERBB2", "Breast"), ("EGFR", "Lung"), ("KRAS", "Lung"), ("MYC", "all")]:
    show(f"crispr_link {t} {l}", crispr_link(t, l))
for d, l in [("lapatinib", "Breast"), ("erlotinib", "Lung"), ("erlotinib", "Head and Neck")]:
    t0 = time.time()
    r = top_tads_for_drug(d, l, 5)
    show(f"top_tads_for_drug {d} {l} ({time.time() - t0:.0f}s)", r)
show("top_essential_tads Head and Neck", top_essential_tads("Head and Neck", 5))
