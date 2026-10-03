"""The TADTalk agent: an LLM that answers questions by calling the analysis tools."""
import functools
import json

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from config import MODEL
from tools.common import cancers, lineages
from tools.crispr import crispr_link, top_essential_tads
from tools.drug import drug_link, top_tads_for_drug
from tools.find_tad import find_tad
from tools.survival import survival


def safe(tool_fn):
    """Wrap a tool so an unexpected crash comes back to the LLM as an error message instead of stopping the app."""
    @functools.wraps(tool_fn)  # keeps the name, docstring and arguments that the LLM reads
    def wrapper(*args, **kwargs):
        try:
            return tool_fn(*args, **kwargs)
        except Exception as e:
            return {"error": f"{tool_fn.__name__} failed: {type(e).__name__}: {e}"}
    return wrapper


TOOLS = [safe(t) for t in [find_tad, drug_link, top_tads_for_drug, survival, crispr_link, top_essential_tads]]

RULES = """You are TADTalk, an assistant for cancer questions about 3D chromatin domains (TADs).
Rules:
1. Every number you give must come from a tool result in this conversation. Never estimate or recall numbers.
2. For every test, report n, the effect (rho or HR), the p-value (or FDR) and the StatGuard verdict,
   written exactly as "StatGuard: PASS" or "StatGuard: FAIL". If FAIL, say the result is not stronger
   than random genes or random neighbouring genes.
3. Name the TAD id and the data used (CTRPv2 drug screens, DepMap CRISPR, TCGA).
4. Drug and CRISPR results come from cell lines: call them hypotheses, never treatment advice.
5. If a tool returns an error, explain it and suggest a valid input. Never guess a result.
6. If a question is outside these tools, say so in one sentence and list what you can answer.
7. Be short: 2-4 sentences, then the key numbers."""


def system_prompt():
    """The rules plus the valid inputs, read from the data, so the LLM does not guess names."""
    return (f"{RULES}\n\nValid TCGA cancer codes: {', '.join(cancers())}."
            f"\nValid cell-line lineages: {', '.join(lineages())}, or 'all'.")


def build_agent(model=None):
    """Create the agent. Normally uses the model in .env; tests can pass a fake model."""
    llm = model or init_chat_model(MODEL, temperature=0)  # temperature 0 = the same answer every time
    return create_agent(model=llm, tools=TOOLS, system_prompt=system_prompt(), checkpointer=InMemorySaver())


def this_turn(messages):
    """Messages after the user's latest question: this turn's tool calls, tool results and answer."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i + 1:]
    return messages


def tool_calls(turn):
    """(tool name, arguments) for every tool the LLM called in this turn."""
    return [(c["name"], c["args"]) for m in turn if isinstance(m, AIMessage) for c in m.tool_calls]


def tool_results(turn):
    """Every tool result of this turn, as Python dicts."""
    out = []
    for m in turn:
        if isinstance(m, ToolMessage):
            try:
                out.append(json.loads(m.content))
            except (TypeError, ValueError):
                out.append({"raw": str(m.content)})
    return out
