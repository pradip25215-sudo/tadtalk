"""General-knowledge check: multiple-choice questions on genomics and cancer biology, answered by TADTalk.

Each question gets a fresh chat. The agent must start its reply with "Answer: X".
Run:  python -m bench.run_general
Output: bench/answers_general.json, bench/scored_general.csv, bench/results_general.md
"""
import json
import re
import time
import uuid

import pandas as pd
from langchain_core.messages import HumanMessage

from agent.agent import build_agent, this_turn, tool_calls
from config import ROOT

BENCH = ROOT / "bench"
FORMAT = ("\n\nThis is a multiple-choice question. The first line of your reply must be exactly "
          "'Answer: X' where X is A, B, C or D. Then explain your choice in one or two sentences.")


def main(model=None):
    qs = pd.read_csv(BENCH / "general_questions.csv", dtype=str)
    agent = build_agent(model)
    answers, rows = {}, []
    for r in qs.itertuples():
        prompt = f"{r.question}\nA. {r.A}\nB. {r.B}\nC. {r.C}\nD. {r.D}{FORMAT}"
        start = time.time()
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}  # a fresh chat for every question
        out = agent.invoke({"messages": [HumanMessage(prompt)]}, config=config)
        text = out["messages"][-1].text
        found = re.search(r"Answer:\W*([A-D])\b", text)
        chosen = found.group(1) if found else ""
        answers[r.id] = {"answer": text, "tools": [c[0] for c in tool_calls(this_turn(out["messages"]))]}
        rows.append({"id": r.id, "topic": r.topic, "chosen": chosen, "key": r.answer,
                     "correct": chosen == r.answer, "seconds": round(time.time() - start, 1)})
        print(f"{r.id}: chose {chosen or '?'}, key {r.answer}, {'right' if chosen == r.answer else 'WRONG'}")

    (BENCH / "answers_general.json").write_text(json.dumps(answers, indent=1))
    df = pd.DataFrame(rows)
    df.to_csv(BENCH / "scored_general.csv", index=False)
    by_topic = df.groupby("topic")["correct"].agg(["sum", "count"])
    lines = [f"## General knowledge (multiple choice, {len(df)} questions)", "",
             f"Accuracy: {int(df['correct'].sum())} of {len(df)} = {100 * df['correct'].mean():.0f}%", "",
             "| topic | correct | questions |", "| --- | --- | --- |"]
    lines += [f"| {t} | {int(v['sum'])} | {int(v['count'])} |" for t, v in by_topic.iterrows()]
    wrong = df.loc[~df["correct"], "id"].tolist()
    lines += ["", f"Wrong: {', '.join(wrong) if wrong else 'none'}", ""]
    report = "\n".join(lines)
    (BENCH / "results_general.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
