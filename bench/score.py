"""Score the saved answers against the answer key.  Run:  python -m bench.score

Scores (simple, transparent rules; read them before trusting the numbers):
  correct              lookup: names at least 80% of the TAD's genes
                       drug / survival / crispr: states the main number (rho, HR or mean effect)
                       within 0.01 (HR: 0.02) AND the right StatGuard verdict (PASS / FAIL)
  unsupported_numbers  decimal numbers in the answer that match nothing in the answer key
  overclaim            the key says FAIL but the answer says PASS or "significant"
"""
import json
import re

import pandas as pd

from config import ROOT

BENCH = ROOT / "bench"
MAIN = {"drug": ("rho_tad", 0.01), "survival": ("hr_per_sd", 0.02), "crispr": ("tad_mean_effect", 0.01)}
ALLOWED = [0.05, 0.5, -0.5]  # thresholds of the method itself
NUMBER = re.compile(r"-?\d+\.\d+(?:e-?\d+)?|-?\d+e-?\d+")


def numbers(text):
    """Decimal numbers in a text, e.g. 0.77, -1.2, 3.6e-12."""
    return [float(x) for x in NUMBER.findall(text.lower().replace("−", "-"))]


def flatten(x):
    """Every number inside a tool result (nested dicts and lists)."""
    if isinstance(x, bool):
        return []
    if isinstance(x, (int, float)):
        return [float(x)]
    if isinstance(x, dict):
        return [n for v in x.values() for n in flatten(v)]
    if isinstance(x, list):
        return [n for v in x for n in flatten(v)]
    return []


def score_one(q, answer, key):
    text = answer.lower()
    if q.type == "lookup":
        found = [g for g in key["genes"] if re.search(rf"\b{re.escape(g.lower())}\b", text)]
        correct, overclaim = len(found) / len(key["genes"]) >= 0.8, False
    else:
        field, tol = MAIN[q.type]
        correct = (any(abs(n - key[field]) <= tol for n in numbers(answer))
                   and re.search(rf"\b{key['verdict'].lower()}\b", text) is not None)
        claims = re.search(r"\bpass\b", text) or re.search(r"(?<!not )(?<!no )(?<!non-)\bsignificant", text)
        overclaim = key["verdict"] == "FAIL" and claims is not None
    reference = flatten(key) + numbers(q.question) + ALLOWED
    unsupported = sum(not any(abs(n - b) <= max(0.005, 0.05 * abs(b)) for b in reference) for n in numbers(answer))
    return correct, unsupported, overclaim


def markdown(df):
    lines = ["| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |",
             "|" + " --- |" * (len(df.columns) + 1)]
    lines += ["| " + " | ".join([str(i)] + [str(v) for v in row]) + " |" for i, row in df.iterrows()]
    return "\n".join(lines)


qs = pd.read_csv(BENCH / "questions.csv", dtype=str).fillna("")
key = json.loads((BENCH / "key.json").read_text())
rows = []
for system in ["tadtalk", "llm_only"]:
    path = BENCH / f"answers_{system}.json"
    if not path.exists():
        continue
    answers = json.loads(path.read_text())
    for q in qs.itertuples():
        if "error" in key[q.id] or q.id not in answers:
            continue
        correct, unsupported, overclaim = score_one(q, answers[q.id]["answer"], key[q.id])
        rows.append({"system": system, "id": q.id, "type": q.type, "correct": correct,
                     "unsupported_numbers": unsupported, "overclaim": overclaim,
                     "seconds": answers[q.id]["seconds"]})

df = pd.DataFrame(rows)
df.to_csv(BENCH / "scored.csv", index=False)  # one row per answer: open it to see what failed
summary = df.groupby("system").agg(questions=("id", "count"), accuracy=("correct", "mean"),
                                   unsupported_numbers_per_answer=("unsupported_numbers", "mean"),
                                   overclaims=("overclaim", "sum"), seconds_per_answer=("seconds", "mean"))
summary["accuracy"] = (100 * summary["accuracy"]).round().astype(int).astype(str) + "%"
by_type = df.pivot_table(index="type", columns="system", values="correct", aggfunc="mean").mul(100).round()
report = f"## Summary\n\n{markdown(summary.round(2))}\n\n## Accuracy by question type (%)\n\n{markdown(by_type)}\n"
(BENCH / "results.md").write_text(report)
print(report)
