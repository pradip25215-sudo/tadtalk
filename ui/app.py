"""TADTalk web app.  Run from the project folder:  streamlit run ui/app.py"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so "import tools/agent/config" works

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

from agent.agent import build_agent, reference_check, this_turn, tool_calls, tool_results  # noqa: E402
from config import ROOT  # noqa: E402
from tools.common import q  # noqa: E402

st.set_page_config(page_title="TADTalk", page_icon="🧬", layout="centered")

# Questions shown on the empty screen. Left: computed from the data. Right: general knowledge with references.
DATA_EXAMPLES = [
    "Which TAD contains ERBB2, and which genes share it?",
    "Is the ERBB2 TAD a better lapatinib marker than ERBB2 alone in breast cell lines?",
    "Is the EGFR TAD a better erlotinib marker than EGFR alone in lung cell lines?",
    "Does the CCND1 TAD predict survival in HNSC?",
    "Which TADs are most linked to erlotinib response in lung cell lines?",
]
GENERAL_EXAMPLES = [
    "What is a TAD, and how can a broken TAD border activate an oncogene? Give references.",
    "Why do some lung cancers stop responding to erlotinib?",
    "How does a Cox hazard ratio differ from a Kaplan-Meier curve?",
]

# The look: colours come from a Hi-C contact map (white to crimson); PASS is green, FAIL is a calm grey,
# because FAIL is a finding ("no better than chance"), not an error.
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, p, li, h1, h2, h3, label, textarea, input, .stMarkdown, div[data-testid="stButton"] button p,
.tt-head, .tt-facts, .tt-strip { font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif; }
code { font-family: 'IBM Plex Mono', Consolas, monospace; }
.block-container { padding-top: 4.5rem; max-width: 860px; }
.tt-head { display: flex; gap: 20px; align-items: center; margin-bottom: 4px; }
.tt-head h1 { font-size: 2.5rem; font-weight: 600; letter-spacing: -0.02em; margin: 0; padding: 0; color: #0F1F3D; }
.tt-head p { margin: 4px 0 0 0; color: #46536B; font-size: 1.02rem; line-height: 1.45; max-width: 34rem; }
.tt-facts { display: grid; grid-template-columns: repeat(4, 1fr); border-top: 1px solid #D9DEE7;
            border-bottom: 1px solid #D9DEE7; margin: 18px 0 22px 0; }
.tt-facts div { padding: 10px 14px 10px 0; }
.tt-facts b { display: block; font-size: 1.25rem; font-weight: 600; color: #0F1F3D; }
.tt-facts span { color: #5B6475; font-size: 0.86rem; }
.tt-strip { display: flex; flex-wrap: wrap; gap: 6px 22px; align-items: center; background: #FFFFFF;
            border: 1px solid #D9DEE7; border-left: 5px solid #5B6475; border-radius: 6px;
            padding: 10px 14px; margin: 2px 0 10px 0; }
.tt-strip.pass { border-left-color: #0E8F6E; }
.tt-badge { font-weight: 600; font-size: 0.86rem; padding: 2px 10px; border-radius: 999px; color: #FFFFFF; background: #5B6475; }
.tt-strip.pass .tt-badge { background: #0E8F6E; }
.tt-strip .tt-what { color: #0F1F3D; font-weight: 500; }
.tt-strip .tt-fig { color: #46536B; font-size: 0.92rem; }
.tt-strip .tt-fig b { color: #0F1F3D; font-weight: 600; }
.tt-kind { color: #5B6475; font-size: 0.9rem; margin: 0 0 6px 0; }
div[data-testid="stButton"] > button { justify-content: flex-start; border-radius: 6px; height: auto;
            border: 1px solid #D9DEE7; background: #FFFFFF; color: #0F1F3D; padding: 10px 14px; }
div[data-testid="stButton"] > button > div { justify-content: flex-start; width: 100%; }
div[data-testid="stButton"] > button div, div[data-testid="stButton"] > button span, div[data-testid="stButton"] > button p {
            text-align: left; white-space: normal; overflow: visible; text-overflow: clip; line-height: 1.35; }
div[data-testid="stButton"] > button:hover { border-color: #C81D4E; color: #C81D4E; }
div[data-testid="stButton"] > button:focus-visible { outline: 2px solid #C81D4E; outline-offset: 2px; }
section[data-testid="stSidebar"] h3 { font-size: 1rem; margin-bottom: 2px; }
@media (max-width: 640px) { .tt-facts { grid-template-columns: repeat(2, 1fr); } .tt-head h1 { font-size: 2rem; } }
</style>
"""

# A small Hi-C style picture: nested triangles along the diagonal, the way TADs look in a contact map.
LOGO = """
<svg width="92" height="58" viewBox="0 0 92 58" role="img" aria-label="Nested TADs in a Hi-C contact map">
  <polygon points="2,56 46,4 90,56" fill="#FBE3EA"/>
  <polygon points="2,56 24,30 46,56" fill="#F2A7BC"/>
  <polygon points="46,56 68,30 90,56" fill="#E56C8E"/>
  <polygon points="52,56 62,44 72,56" fill="#C81D4E"/>
  <polygon points="10,56 17,48 24,56" fill="#C81D4E"/>
  <line x1="0" y1="56.5" x2="92" y2="56.5" stroke="#0F1F3D" stroke-width="1.5"/>
</svg>
"""


@st.cache_resource  # build the agent once, not on every click (Streamlit reruns this file on each action)
def get_agent():
    return build_agent()


@st.cache_data  # the data do not change while the app runs
def facts():
    """What is inside the data, counted from the tables."""
    one = lambda sql: int(q(sql).iloc[0, 0])  # noqa: E731
    return [
        (f"{one('SELECT count(DISTINCT tad_id) FROM tad_genes'):,}", "TADs"),
        (f"{one('SELECT count(DISTINCT model_id) FROM drug_aac'):,}", "cell lines with drug data"),
        (f"{one('SELECT count(DISTINCT drug) FROM drug_aac'):,}", "drugs (CTRPv2)"),
        (f"{one('SELECT count(*) FROM survival'):,}", "TCGA patients"),
    ]


@st.cache_data
def benchmark_lines():
    """One line per benchmark that has been run (read from the bench/ folder)."""
    lines = []
    data, general = ROOT / "bench" / "scored.csv", ROOT / "bench" / "scored_general.csv"
    if data.exists():
        d = pd.read_csv(data)
        for system, name in [("tadtalk", "TADTalk"), ("llm_only", "Same LLM, no tools")]:
            s = d[d["system"] == system]
            if len(s):
                lines.append(f"{name}: {int(s['correct'].sum())} of {len(s)} data questions")
    if general.exists():
        g = pd.read_csv(general)
        lines.append(f"TADTalk: {int(g['correct'].sum())} of {len(g)} general questions")
    stress = ROOT / "bench" / "scored_stress.csv"
    if stress.exists():
        t = pd.read_csv(stress)
        lines.append(f"TADTalk: {int(t['correct'].sum())} of {len(t)} hard questions (new wording, typos, bad inputs)")
    return lines


def fmt(x):
    """Short text for a number: 0.575, 4.7e-04, 1,080."""
    if isinstance(x, int):
        return f"{x:,}"
    return f"{x:.1e}" if 0 < abs(x) < 0.001 else f"{x:.3g}"


def verdict_strip(name, result):
    """The key figures of one test, straight from the tool result (not from the LLM's text)."""
    if not isinstance(result, dict) or "verdict" not in result:
        return ""
    kinds = {
        "drug_link": ("TAD activity and drug response", [("rho", "rho_tad"), ("p", "p_tad"), ("cell lines", "n_cell_lines"),
                                                         ("gene alone rho", "rho_gene")]),
        "survival": ("TAD activity and survival", [("HR per SD", "hr_per_sd"), ("p", "p_cox"), ("patients", "n_patients")]),
        "crispr_link": ("Gene essentiality in the TAD", [("mean effect", "tad_mean_effect"), ("p", "p_vs_other_genes"),
                                                         ("cell lines", "n_cell_lines")]),
    }
    if name not in kinds:
        return ""
    what, fields = kinds[name]
    figures = "".join(f'<span class="tt-fig">{label} <b>{fmt(result[key])}</b></span>'
                      for label, key in fields if key in result)
    passed = result["verdict"] == "PASS"
    return (f'<div class="tt-strip {"pass" if passed else ""}"><span class="tt-badge">StatGuard {result["verdict"]}</span>'
            f'<span class="tt-what">{what}</span>{figures}</div>')


def show_turn(turn):
    """Draw one chat turn: key figures, text, plots, reference check and the raw tool results."""
    is_user = turn["role"] == "user"
    with st.chat_message(turn["role"], avatar=":material/person:" if is_user else ":material/genetics:"):
        strips = "".join(verdict_strip(name, result) for (name, _), result in zip(turn.get("calls", []), turn.get("results", [])))
        if strips:
            st.markdown(strips, unsafe_allow_html=True)
        st.markdown(turn["text"])
        for plot in turn.get("plots", []):
            st.image(plot)
        if turn.get("unverified"):
            st.warning("Reference check: PMID " + ", ".join(turn["unverified"]) +
                       " was not returned by the PubMed search. Verify before using it.")
        elif turn.get("verified"):
            st.caption(f"Reference check: all {len(turn['verified'])} cited PMIDs were returned by the PubMed search.")
        if turn.get("calls"):
            with st.expander("How this was computed"):
                for (name, args), result in zip(turn["calls"], turn["results"]):
                    st.markdown(f"**{name}** `{args}`")
                    st.json(result, expanded=False)


def explain(error):
    """Turn a failed model call into a message that says what to do."""
    text = str(error)
    if "api key" in text.lower() or "authentication" in text.lower() or "401" in text:
        return ("No working API key. Put your key in the `.env` file in the project folder as "
                "`ANTHROPIC_API_KEY=...`, then stop the app with Ctrl+C and start it again.")
    if "credit" in text.lower():
        return "The API account has no credit left. Add credit in the Claude Console, then ask again."
    return f"The model call failed: {text}"


st.markdown(STYLE, unsafe_allow_html=True)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []

with st.sidebar:
    st.markdown("### What this is")
    st.write("A chat assistant for cancer genomics. Questions about TADs, drugs, survival and gene essentiality "
             "are computed from real data. General questions are answered from the model's knowledge, with "
             "references checked against PubMed.")
    st.markdown("### Where the numbers come from")
    st.write("TCGA (patient survival), DepMap (CRISPR gene effect), CTRPv2 (drug response) and the lab's "
             "non-redundant union TAD set (hg19).")
    st.markdown("### How StatGuard decides")
    st.write("A result passes only if the TAD's own test is significant and the TAD also beats 95% of "
             "200 random gene sets and 200 blocks of neighbouring genes of the same size.")
    lines = benchmark_lines()
    if lines:
        st.markdown("### Measured accuracy")
        for line in lines:
            st.write(line)
    if st.button("Start a new chat"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()
    st.caption("Research use only. Cell-line results are hypotheses, not treatment advice.")

st.markdown(f'<div class="tt-head">{LOGO}<div><h1>TADTalk</h1>'
            '<p>Ask how 3D chromatin domains relate to drug response, survival and gene essentiality in cancer. '
            'Every number is computed, and checked against chance.</p></div></div>', unsafe_allow_html=True)
st.markdown('<div class="tt-facts">' + "".join(f"<div><b>{n}</b><span>{label}</span></div>" for n, label in facts())
            + "</div>", unsafe_allow_html=True)

for turn in st.session_state.history:
    show_turn(turn)

typed = st.chat_input("Ask about a gene, TAD, drug, cancer type, or any genomics question")
question = typed or st.session_state.pop("pending", None)

if not st.session_state.history and not question:  # empty screen: offer questions to start with
    left, right = st.columns(2)
    with left:
        st.markdown('<p class="tt-kind">Computed from the data</p>', unsafe_allow_html=True)
        for i, example in enumerate(DATA_EXAMPLES):
            if st.button(example, key=f"data{i}", use_container_width=True):
                st.session_state.pending = example
                st.rerun()
    with right:
        st.markdown('<p class="tt-kind">General knowledge, with references</p>', unsafe_allow_html=True)
        for i, example in enumerate(GENERAL_EXAMPLES):
            if st.button(example, key=f"general{i}", use_container_width=True):
                st.session_state.pending = example
                st.rerun()

if question:
    user_turn = {"role": "user", "text": question}
    st.session_state.history.append(user_turn)
    show_turn(user_turn)
    with st.spinner("Choosing tools and running the analysis..."):
        try:
            result = get_agent().invoke({"messages": [HumanMessage(question)]},
                                        config={"configurable": {"thread_id": st.session_state.thread_id}})
        except Exception as e:  # e.g. no internet, a wrong API key, or no credit
            st.session_state.history.pop()  # so asking again does not show the question twice
            st.error(explain(e))
            st.stop()
    turn = this_turn(result["messages"])
    results = tool_results(turn)
    text = result["messages"][-1].text
    verified, unverified = reference_check(text, result["messages"])
    answer = {"role": "assistant", "text": text, "verified": verified, "unverified": unverified,
              "plots": list(dict.fromkeys(r["plot"] for r in results
                                          if isinstance(r, dict) and Path(str(r.get("plot", ""))).is_file())),
              "calls": tool_calls(turn), "results": results}
    st.session_state.history.append(answer)
    show_turn(answer)
