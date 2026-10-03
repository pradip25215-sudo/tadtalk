"""Ask every benchmark question to TADTalk, or to the same LLM with no tools.

Run:  python -m bench.run --system tadtalk
      python -m bench.run --system llm_only
"""
import argparse
import json
import time
import uuid

import pandas as pd
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from agent.agent import build_agent, this_turn, tool_calls
from config import MODEL, ROOT

BENCH = ROOT / "bench"
BASELINE_PROMPT = ("You are an expert in cancer genomics. Answer the question from your own knowledge. "
                   "Give specific numbers (correlation, hazard ratio, p-value) when you can. Be short.")


def main(system, model=None):
    qs = pd.read_csv(BENCH / "questions.csv", dtype=str).fillna("")
    if system == "tadtalk":
        agent = build_agent(model)
    else:
        llm = model or init_chat_model(MODEL, temperature=0)

    answers = {}
    for r in qs.itertuples():
        start = time.time()
        if system == "tadtalk":
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}  # a fresh chat for every question
            out = agent.invoke({"messages": [HumanMessage(r.question)]}, config=config)
            text, calls = out["messages"][-1].text, tool_calls(this_turn(out["messages"]))
        else:
            text, calls = llm.invoke([SystemMessage(BASELINE_PROMPT), HumanMessage(r.question)]).text, []
        answers[r.id] = {"answer": text, "tools": calls, "seconds": round(time.time() - start, 1)}
        print(f"{r.id}: {answers[r.id]['seconds']} s, tools: {[c[0] for c in calls]}")

    (BENCH / f"answers_{system}.json").write_text(json.dumps(answers, indent=1))
    print(f"Saved bench/answers_{system}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["tadtalk", "llm_only"], required=True)
    main(parser.parse_args().system)
