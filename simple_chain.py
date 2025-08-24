# src/langchain_rpa_simkit/simple_chain.py
from __future__ import annotations
import os
from typing import Iterable

from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Optional: real LLM if installed
def _maybe_openai_llm() -> Runnable:
    """
    Returns a runnable that behaves like an LLM.
    - If langchain-openai is installed AND OPENAI_API_KEY is set, use ChatOpenAI.
    - Otherwise, return a deterministic fallback Runnable that pretends to be an LLM.
    """
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
        if os.getenv("OPENAI_API_KEY"):
            # 'gpt-4o-mini' is cheap/fast; change to your preference.
            return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    except Exception:
        pass

    # Fallback: turn the messages into a simple bullet plan (no external calls).
    def fake_llm(messages: Iterable[dict]) -> str:
        # Extract the latest user content from the chat messages
        last_user = ""
        for m in messages:
            if m.get("role") == "user":
                last_user = m.get("content", last_user)

        # A tiny deterministic planner for demonstration
        task = last_user.strip() or "the task"
        return (
            f"Plan for: {task}\n"
            f"1) Understand requirements\n"
            f"2) Identify tools/resources\n"
            f"3) Execute main steps\n"
            f"4) Verify results\n"
            f"5) Report outcome"
        )

    return RunnableLambda(fake_llm)


# Build the chain once; re-use it.
_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a concise planner. Produce clear, numbered steps."),
        ("human", "Create a short plan for: {task}"),
    ]
)

_llm = _maybe_openai_llm()
_chain: Runnable = _prompt | _llm | StrOutputParser()


def plan(task: str) -> str:
    """
    Synchronous single-call API.
    """
    return _chain.invoke({"task": task})


def plan_batch(tasks: list[str]) -> list[str]:
    """
    Batch execution to show LCEL batching.
    """
    return list(_chain.batch([{"task": t} for t in tasks]))


def plan_stream(task: str):
    """
    Streaming generator (works with real LLMs; with fallback, yields once).
    """
    for chunk in _chain.stream({"task": task}):
        yield chunk

