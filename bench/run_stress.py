"""Stress test: does TADTalk still pick the right tool and inputs when questions are worded differently?

The questions use new wording, typos, drug brand names, TAD ids, bad inputs, two-step requests,
general questions and out-of-scope requests. Each one is scored on what the agent DID:
  expect_tool   the tool that must be called ("none" = no tool at all; empty = any)
  expect_args   inputs that must match (ignoring upper/lower case). "target~" means "contains".
                A gene target also matches when the agent passed that gene's TAD id.
                A list, e.g. lineage ["Breast", "Lung"], means the tool must be called once for each.
  answer_has    a pattern the answer must contain;  answer_lacks  a pattern it must not contain
Run:  python -m bench.run_stress             (asks the LLM)
      python -m bench.run_stress --rescore   (scores the saved answers again, no LLM call)
Output: bench/answers_stress.json, bench/scored_stress.csv, bench/results_stress.md
"""
import json
import re
import uuid

import pandas as pd
from langchain_core.messages import HumanMessage

from agent.agent import build_agent, this_turn, tool_calls
from config import ROOT
from tools.common import official_symbol, tad_id_text, tad_of_gene

BENCH = ROOT / "bench"
ANALYSIS_TOOLS = {"find_tad", "drug_link", "top_tads_for_drug", "survival", "crispr_link", "top_essential_tads"}


def value_matches(key, want, got):
    """Does one input of a tool call match what was expected?"""
    got = str(got).strip().lower()
    if key.endswith("~"):
        return str(want).lower() in got
    want = str(want).lower()
    if key in ("target", "gene"):
        got = official_symbol(got).lower()  # the tools accept common names: HER2 means ERBB2
    if key in ("target", "gene") and got != want:  # the agent may pass the gene's TAD id instead of the gene
        tad = tad_of_gene(want)
        return tad is not None and tad_id_text(got).lower() == tad.lower()
    return got == want


def call_matches(call, tool, args):
    name, given = call
    if name != tool:
        return False
    return all(value_matches(k, v, given.get(k.rstrip("~"), "")) for k, v in args.items())


def check(row, calls, answer):
    """Returns (passed, reason)."""
    expected = json.loads(row.expect_args or "{}")
    if row.expect_tool == "none":
        used = [c[0] for c in calls if c[0] in ANALYSIS_TOOLS]
        if used:
            return False, f"called {used} but no analysis tool was needed"
    elif row.expect_tool:
        lists = {k: v for k, v in expected.items() if isinstance(v, list)}
        variants = [expected]
        for k, values in lists.items():  # one required call per listed value
            variants = [{**e, k: v} for e in variants for v in values]
        for want in variants:
            if not any(call_matches(c, row.expect_tool, want) for c in calls):
                return False, f"no call {row.expect_tool}({want}); calls were {calls}"
    if row.kind != "bad input" and re.search(r"not in any TAD|not a gene or TAD id", answer, flags=re.I):
        return False, "the answer says the gene or TAD was not found"
    if row.answer_has and not re.search(row.answer_has, answer, flags=re.I):
        return False, f"answer does not contain /{row.answer_has}/"
    if row.answer_lacks and re.search(row.answer_lacks, answer, flags=re.I):
        return False, f"answer contains /{row.answer_lacks}/"
    return True, ""


def main(model=None, rescore=False):
    qs = pd.read_csv(BENCH / "stress_questions.csv", dtype=str).fillna("")
    saved = json.loads((BENCH / "answers_stress.json").read_text()) if rescore else {}
    agent = None if rescore else build_agent(model)
    threads, answers, rows = {}, {}, []
    for r in qs.itertuples():
        if rescore:  # score the saved answers again: no LLM call, no cost
            calls = [(name, args) for name, args in saved[r.id]["tools"]]
            answer = saved[r.id]["answer"]
        else:
            # questions with the same 'thread' value share one chat (to test follow-ups); the others get a fresh chat
            thread_id = threads.setdefault(r.thread, str(uuid.uuid4())) if r.thread else str(uuid.uuid4())
            out = agent.invoke({"messages": [HumanMessage(r.question)]}, config={"configurable": {"thread_id": thread_id}})
            calls = tool_calls(this_turn(out["messages"]))
            answer = out["messages"][-1].text
        passed, reason = check(r, calls, answer)
        answers[r.id] = {"question": r.question, "tools": calls, "answer": answer}
        rows.append({"id": r.id, "kind": r.kind, "correct": passed, "reason": reason})
        print(f"{r.id} [{r.kind}] {'right' if passed else 'WRONG: ' + reason}")

    (BENCH / "answers_stress.json").write_text(json.dumps(answers, indent=1))
    df = pd.DataFrame(rows)
    df.to_csv(BENCH / "scored_stress.csv", index=False)
    by_kind = df.groupby("kind", sort=False)["correct"].agg(["sum", "count"])
    lines = [f"## Stress test ({len(df)} hard questions)", "",
             f"Accuracy: {int(df['correct'].sum())} of {len(df)} = {100 * df['correct'].mean():.0f}%", "",
             "| kind | correct | questions |", "| --- | --- | --- |"]
    lines += [f"| {k} | {int(v['sum'])} | {int(v['count'])} |" for k, v in by_kind.iterrows()]
    wrong = df[~df["correct"]]
    lines += ["", "Wrong:" if len(wrong) else "Wrong: none"] + [f"- {w.id}: {w.reason}" for w in wrong.itertuples()] + [""]
    report = "\n".join(lines)
    (BENCH / "results_stress.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    import sys
    main(rescore="--rescore" in sys.argv)
