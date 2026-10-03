"""TADTalk web app.  Run from the project folder:  streamlit run ui/app.py"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so "import tools/agent/config" works

import streamlit as st  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

from agent.agent import build_agent, this_turn, tool_calls, tool_results  # noqa: E402

st.set_page_config(page_title="TADTalk", layout="centered")

EXAMPLES = [
    "Which TAD contains EGFR, and which genes share it?",
    "Is the EGFR TAD a better erlotinib marker than EGFR alone in lung cell lines?",
    "Which TADs are most linked to lapatinib response in breast cell lines?",
    "Does the CCND1 TAD predict survival in HNSC?",
    "Are the genes in the ERBB2 TAD essential in breast cancer cell lines?",
]


@st.cache_resource  # build the agent once, not on every click (Streamlit reruns this file on each action)
def get_agent():
    return build_agent()


def show_turn(turn):
    """Draw one chat turn: text, plots, and the raw tool results."""
    with st.chat_message(turn["role"]):
        st.markdown(turn["text"])
        for plot in turn.get("plots", []):
            st.image(plot)
        if turn.get("calls"):
            with st.expander("How this was computed"):
                for (name, args), result in zip(turn["calls"], turn["results"]):
                    st.markdown(f"**{name}** `{args}`")
                    st.json(result, expanded=False)


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []

with st.sidebar:
    st.header("TADTalk")
    st.write("Cancer questions at the level of 3D chromatin domains (TADs).")
    st.markdown("**Data:** TCGA (survival), DepMap CRISPR, CTRPv2 drug screens, "
                "TAD calls from HNSCC Hi-C. **StatGuard** checks every result against "
                "random gene sets and neighbour blocks of the same size.")
    st.warning("Research use only. Cell-line results are hypotheses, not treatment advice.")
    if st.button("New chat"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
    st.subheader("Try asking")
    for q in EXAMPLES:
        st.caption(q)

st.title("TADTalk")
for turn in st.session_state.history:
    show_turn(turn)

if question := st.chat_input("Ask about a gene, TAD, drug or cancer type"):
    user_turn = {"role": "user", "text": question}
    st.session_state.history.append(user_turn)
    show_turn(user_turn)
    with st.spinner("Thinking and running analyses..."):
        try:
            result = get_agent().invoke({"messages": [HumanMessage(question)]},
                                        config={"configurable": {"thread_id": st.session_state.thread_id}})
        except Exception as e:  # e.g. no internet, or a wrong API key
            st.error(f"The model call failed: {e}")
            st.stop()
    turn = this_turn(result["messages"])
    results = tool_results(turn)
    answer = {"role": "assistant", "text": result["messages"][-1].text,
              "plots": [r["plot"] for r in results if isinstance(r, dict) and Path(str(r.get("plot", ""))).is_file()],
              "calls": tool_calls(turn), "results": results}
    st.session_state.history.append(answer)
    show_turn(answer)
