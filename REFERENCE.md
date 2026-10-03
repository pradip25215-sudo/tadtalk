# TADTalk reference code

This is the tested reference code for the 10-day guide.

- Learn by typing each file yourself, following the guide day by day.
- Use this folder only to compare when your version fails.
- `agent/agent.py` here is the full Day 6 version; Day 1 uses the short version shown in the guide.
- `tools/hic.py` is the optional Day 9 tool; add it to `TOOLS` in `agent/agent.py` if you use it.
- `extras/make_sample_data.py` makes FAKE data for testing only. Read its first lines before using it.

Tested with Python 3.11, LangChain 1.4, LangGraph 1.2, DuckDB 1.5, lifelines 0.30, Streamlit 1.64 and
GSVA 1.50 (R), on sample data with the same formats as DepMap, CTRPv2, TCGA and GENCODE.
