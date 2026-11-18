from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent import Assistant, build_agent
from app.tools import calculator, make_kb_search_tool
from tests.conftest import ScriptedChatModel, tool_call


def make(responses, store, checkpointer):
    model = ScriptedChatModel(responses=responses, calls=[])
    graph = build_agent(model, [calculator, make_kb_search_tool(store)], checkpointer=checkpointer)
    return model, Assistant(graph)


def test_answers_directly_without_tools(store, checkpointer):
    _, bot = make([AIMessage(content="Hello! How can I help?")], store, checkpointer)
    r = bot.ask("hi", thread_id="t1")
    assert r["answer"] == "Hello! How can I help?"
    assert r["tools_used"] == [] and r["sources"] == []


def test_calls_calculator_and_uses_result(store, checkpointer):
    model, bot = make(
        [tool_call("calculator", {"expression": "2 * 18 * 25 * 7.41 / 1000"}), AIMessage(content="About 6.67 V.")],
        store, checkpointer,
    )
    r = bot.ask("voltage drop for 18 A over 25 m of 2.5mm2?", thread_id="t1")
    assert r["tools_used"] == ["calculator"]
    tool_msg = [m for m in model.calls[1] if isinstance(m, ToolMessage)][0]
    assert tool_msg.content == "6.669"  # real tool output fed back into the model
    assert r["answer"] == "About 6.67 V."


def test_rag_flow_returns_sources(store, checkpointer):
    _, bot = make(
        [tool_call("search_knowledge_base", {"query": "IP65 meaning dust water jets"}),
         AIMessage(content="IP65 = dust-tight + water jets (source: ip_ratings.md#1).")],
        store, checkpointer,
    )
    r = bot.ask("What does IP65 mean?", thread_id="t1")
    assert r["tools_used"] == ["search_knowledge_base"]
    assert any(s.startswith("ip_ratings.md#") for s in r["sources"])


def test_memory_persists_within_a_thread_and_is_isolated_across_threads(store, checkpointer):
    model, bot = make(
        [AIMessage(content="Noted, 400 V."), AIMessage(content="You said 400 V."), AIMessage(content="I don't know.")],
        store, checkpointer,
    )
    bot.ask("My supply is 400 V.", thread_id="alice")
    bot.ask("What supply voltage did I give you?", thread_id="alice")
    seen_by_turn2 = [m.content for m in model.calls[1] if isinstance(m, HumanMessage)]
    assert "My supply is 400 V." in seen_by_turn2  # earlier turn was remembered

    bot.ask("What supply voltage did I give you?", thread_id="bob")
    seen_by_bob = [m.content for m in model.calls[2] if isinstance(m, HumanMessage)]
    assert "My supply is 400 V." not in seen_by_bob  # other threads can't see it


def test_guardrail_refuses_when_knowledge_base_has_nothing(embeddings, checkpointer):
    import uuid

    from app.rag import build_vector_store

    empty = build_vector_store([], embeddings, collection_name=f"e-{uuid.uuid4().hex}")
    _, bot = make(
        [tool_call("search_knowledge_base", {"query": "boiling point of mercury"}), AIMessage(content="356.7 °C")],
        empty, checkpointer,
    )
    r = bot.ask("What is the boiling point of mercury?", thread_id="t1")
    assert r["answer"] != "356.7 °C" and "isn't covered" in r["answer"]  # model's unsupported answer is blocked


def test_forced_retrieval_strips_fake_citations_and_refuses_ungrounded_answers(store, checkpointer):
    from app.tools import make_kb_search_tool

    kb = make_kb_search_tool(store, k=2)
    model = ScriptedChatModel(
        responses=[AIMessage(content="IP65 is dust-tight with water-jet protection (source: ip_ratings.md#2, source: ip_ratings.md#65)."),
                   AIMessage(content="It is 356.7 °C (source: chemistry.md#1).")],
        calls=[],
    )
    bot = Assistant(build_agent(model, [kb], checkpointer=checkpointer, retrieve=lambda q: kb.invoke({"query": q})))

    r = bot.ask("What does IP65 mean for dust and water jets?", thread_id="t1")
    assert "[source: ip_ratings.md#2]" in model.calls[0][0].content  # passages injected before the model answered
    assert r["citations_removed"] == ["ip_ratings.md#65"] and "#65" not in r["answer"]
    assert "(source: ip_ratings.md#2)" in r["answer"]

    r2 = bot.ask("What is the boiling point of mercury?", thread_id="t1")
    assert "isn't covered" in r2["answer"]  # only citation was fake -> answer refused
