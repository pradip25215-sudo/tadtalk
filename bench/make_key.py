"""Build the answer key by calling the tools directly, with no LLM.  Run:  python -m bench.make_key"""
import json

import pandas as pd

from config import ROOT
from tools.crispr import crispr_link
from tools.drug import drug_link
from tools.find_tad import find_tad
from tools.survival import survival

BENCH = ROOT / "bench"
qs = pd.read_csv(BENCH / "questions.csv", dtype=str).fillna("")

key = {}
for r in qs.itertuples():
    if r.type == "lookup":
        result = find_tad(r.target)
    elif r.type == "drug":
        result = drug_link(r.target, r.drug, r.lineage)
    elif r.type == "survival":
        result = survival(r.target, r.cancer)
    elif r.type == "crispr":
        result = crispr_link(r.target, r.lineage)
    else:
        raise ValueError(f"{r.id}: unknown type {r.type}")
    if "error" in result:
        print(f"{r.id}: {result['error']}  -> change or delete this question")
    key[r.id] = result

(BENCH / "key.json").write_text(json.dumps(key, indent=1))
print(f"Answer key saved for {len(key)} questions: bench/key.json")
