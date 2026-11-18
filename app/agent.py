"""LangGraph tool-calling agent with persistent, per-thread memory."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

SYSTEM_PROMPT = """You are an electrical engineering assistant. You may ONLY state facts that come from your tools.
Tools:
- search_knowledge_base: reference material (IP ratings, formulas, limits). Ground your answer ONLY in
  what it returns and cite sources inline exactly as given, e.g. (source: ip_ratings.md#1).
- voltage_drop: use for EVERY voltage-drop question. Report its numbers exactly. When comparing two
  results, compare the numbers: a higher voltage drop is worse. A smaller cable size (mm²) has higher
  resistance, so it INCREASES the voltage drop.
- calculator: any other arithmetic. Never do math in your head.
Rules:
- If search_knowledge_base returns NO_RESULTS, reply that the topic is not covered in your knowledge
  base. Never answer from general knowledge.
- Answer only what was asked. Do not add unrelated topics.
- Use details the user gave earlier in this conversation (e.g. current, length, voltage) for follow-ups.
- Be concise: 2-4 sentences."""

NOT_COVERED = "That isn't covered in my knowledge base, so I can't answer it reliably."

_SOURCE_RE = re.compile(r"\[source: ([^\]]+)\]")
_CITE_RE = re.compile(r"\(sources?: ([^)]+)\)")  # (source: a) or (source: a, source: b) / (sources: a; b)


def _cite_ids(group: str) -> list[str]:
    return [re.sub(r"^sources?:\s*", "", part.strip()) for part in re.split(r"[,;]", group) if part.strip()]
_SCORE_RE = re.compile(r"\[source: ([^\]]+)\] \(relevance ([0-9.\-]+)\)")


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    context: str  # knowledge-base passages retrieved for the latest question (overwritten each turn)


def build_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    checkpointer: BaseCheckpointSaver | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    retrieve: Callable[[str], str] | None = None,
):
    """Graph: START -> [retrieve] -> agent -(tool calls?)-> tools -> agent ... -> END.

    If `retrieve` is given, every question is grounded first: the knowledge base is searched with the
    user's message and the passages are injected into the system prompt, so the model can't skip retrieval.
    """
    bound = model.bind_tools(tools)

    def retrieve_context(state: AgentState) -> dict[str, Any]:
        question = next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
        return {"context": retrieve(str(question))}

    def call_model(state: AgentState) -> dict[str, Any]:
        prompt = system_prompt
        if state.get("context"):
            prompt += (
                "\n\nKNOWLEDGE BASE PASSAGES retrieved for the latest question. Use ONLY these for reference facts "
                "and cite them as (source: <id>) using the exact ids shown:\n" + state["context"]
            )
        response = bound.invoke([SystemMessage(content=prompt), *state["messages"]])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    if retrieve is not None:
        graph.add_node("retrieve", retrieve_context)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "agent")
    else:
        graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # -> "tools" or END
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)


def sqlite_checkpointer(path: str):
    """Durable memory: conversations survive restarts (one row-set per thread_id)."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


class Assistant:
    """Thin service wrapper the API and CLI talk to. Also enforces the grounding guardrail."""

    def __init__(self, graph):
        self.graph = graph

    def ask(self, message: str, thread_id: str = "default") -> dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        state = self.graph.invoke({"messages": [HumanMessage(content=message)], "context": ""}, config)
        messages = state["messages"]
        # only look at messages produced in this turn (after the latest human message)
        last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        turn = messages[last_human + 1 :]

        tools_used: list[str] = []
        evidence: list[str] = [state.get("context") or ""]  # everything the model was actually shown
        for m in turn:
            if isinstance(m, AIMessage):
                tools_used += [tc["name"] for tc in m.tool_calls]
            if isinstance(m, ToolMessage) and m.name == "search_knowledge_base":
                evidence.append(str(m.content))

        sources: list[str] = []
        relevance: dict[str, float] = {}
        for text in evidence:
            for src in _SOURCE_RE.findall(text):
                if src not in sources:
                    sources.append(src)
            for src, score in _SCORE_RE.findall(text):
                relevance[src] = max(relevance.get(src, -1.0), float(score))

        answer = next((m.content for m in reversed(turn) if isinstance(m, AIMessage) and m.content), "")

        # Guardrail 1: drop any citation that doesn't match a passage actually retrieved (or a tool used).
        allowed = set(sources) | set(tools_used)
        removed = [c for g in _CITE_RE.findall(answer) for c in _cite_ids(g) if c not in allowed]
        if removed:
            def _keep_valid(mt: re.Match) -> str:
                valid = [c for c in _cite_ids(mt.group(1)) if c in allowed]
                return f"(source: {', '.join(valid)})" if valid else ""

            answer = _CITE_RE.sub(_keep_valid, answer)
            answer = re.sub(r"[ \t]+([.,;])", r"\1", answer).strip()

        # Guardrail 2: a reference answer (knowledge base consulted, no computation tool used) must carry at
        # least one valid citation - otherwise it came from the model's memory, so refuse instead.
        used_kb = any(evidence) or "search_knowledge_base" in tools_used
        computed = bool(set(tools_used) - {"search_knowledge_base"})
        has_valid_cite = any(c in sources for g in _CITE_RE.findall(answer) for c in _cite_ids(g))
        if used_kb and not computed and not has_valid_cite:
            answer = NOT_COVERED

        return {
            "answer": answer, "sources": sources, "relevance": relevance, "tools_used": tools_used,
            "citations_removed": removed, "thread_id": thread_id,
        }
