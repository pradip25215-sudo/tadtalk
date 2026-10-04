"""Chat with TADTalk in the terminal.  Run:  python -m agent.cli"""
import uuid

from langchain_core.messages import HumanMessage

from agent.agent import build_agent, reference_check, this_turn, tool_calls


def main():
    agent = build_agent()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}  # one conversation = one thread id
    print("TADTalk is ready. Ask a question, or type 'quit'.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        result = agent.invoke({"messages": [HumanMessage(question)]}, config=config)
        for name, args in tool_calls(this_turn(result["messages"])):
            print(f"  [tool] {name}({args})")  # see which tool the LLM chose, with which inputs
        answer = result["messages"][-1].text
        print(f"\nTADTalk: {answer}")
        verified, unverified = reference_check(answer, result["messages"])
        if unverified:
            print(f"  [reference check] NOT returned by PubMed search, verify before use: {unverified}")
        elif verified:
            print(f"  [reference check] all {len(verified)} cited PMIDs were returned by PubMed search")


if __name__ == "__main__":
    main()
