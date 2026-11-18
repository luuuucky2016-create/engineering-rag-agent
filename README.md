# Engineering RAG Agent

![CI](https://github.com/luuuucky2016-create/engineering-rag-agent/actions/workflows/ci.yml/badge.svg)

A generative AI assistant for electrical-engineering reference questions. It combines **Retrieval-Augmented Generation** with a **LangGraph tool-calling agent**, keeps **persistent multi-turn memory**, and ships as a **FastAPI** service with **Docker** and **GitHub Actions CI**. It runs fully local and free on **Ollama**, or on OpenAI by changing one env var.

```
            ┌──────────── LangGraph ────────────┐
 user ─► FastAPI /chat ─► agent (LLM) ──tool call?──► tools ─┐
            │               ▲                     │            │
            │               └──────── results ◄───┴────────────┘
            │        memory: SQLite checkpointer (per thread_id)
            │
 tools:  search_knowledge_base ─► Chroma (cosine, relevance cutoff) ◄─ chunk + embed ◄─ data/docs/*.md
         voltage_drop (deterministic IEC 60228 cable calc)   calculator (safe AST, no eval())
```

## What it does
- **RAG pipeline:** loads the docs, splits them into chunks with stable ids (`file.md#n`), embeds them into **Chroma**, retrieves the top-k chunks and returns answers **with source citations**. **Every question is grounded first:** a `retrieve` node searches the knowledge base before the model runs, so the model can't skip retrieval. Chunks below a cosine-relevance cutoff (`MIN_RELEVANCE`, default 0.6, calibrated on real runs: relevant passages scored 0.65–0.79, off-topic ones 0.49–0.55) are discarded.
- **Grounding guardrails (in code, not just the prompt):** any `(source: …)` citation that doesn't match a passage actually retrieved is stripped, and a reference answer left with no valid citation is replaced with "not covered". In testing, the 7B model invented a citation (`ip_ratings.md#65`); this guardrail is why that can't reach the user.
- **Tool-calling agent:** a hand-built LangGraph `StateGraph` (agent ⇄ ToolNode loop). It picks between `search_knowledge_base`, `voltage_drop` and `calculator`. The LLM decides what to do, and code does the physics: voltage drop comes from a deterministic tool, not the model's arithmetic.
- **Persistent memory:** a LangGraph SQLite checkpointer stores each conversation by `thread_id`. Memory survives restarts and is isolated between threads.
- **Service:** `POST /chat` returns `answer`, `sources`, `tools_used` and `thread_id`. `GET /health` reports status. Interactive docs are at `/docs`.
- **Quality:** 18 automated tests run **fully offline**. Fake chat models, fake embeddings and a fake assistant are injected through dependencies, so the tests need no API keys or model downloads. CI runs the tests and builds the Docker image on every push.

## Quick start (local, free)
1. Install [Ollama](https://ollama.com/download) and open it once.
2. Run:
```bash
./run_demo.sh            # pulls models, installs deps, runs the 18 tests, runs a live 4-turn demo
```
Default model is `qwen2.5:7b` (works on 8 GB Apple Silicon, ~10–30 s per answer). `qwen2.5:3b` is faster but got engineering answers wrong in testing.

### Run the API
```bash
source .venv/bin/activate
uvicorn app.api:app --reload
# open http://localhost:8000/docs, or:
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message":"What does IP66 mean?","thread_id":"me"}'
```

### Docker (API + Ollama + models)
```bash
docker compose up --build
```

### Interactive terminal chat
```bash
python -m app.cli
```

## Live run (Ollama, qwen2.5:7b, MacBook Pro, 8 GB RAM)
The real transcript of the final run is in [`demo_transcript.md`](demo_transcript.md): IP65 answered with real citations, 6.09 V / 1.52% and the 10.24 V / 2.56% follow-up from the `voltage_drop` tool, and the off-topic question refused.

On a Mac you can double-click `Run Demo.command` (or run `bash "Run Demo.command"`) to reproduce it.

## Demo checks (the right answers)
| Question | Expected |
|---|---|
| 3-phase, 32 A, 60 m, 10 mm² Cu (1.83 Ω/km), 400 V | Vd = √3·32·60·1.83/1000 ≈ **6.09 V ≈ 1.52 %**, within 3 % / 5 % |
| Same circuit with 6 mm² (3.08 Ω/km) | ≈ **10.24 V ≈ 2.56 %**, still within limits |
| Boiling point of mercury | Not in the knowledge base, so the agent should say it doesn't know |

## Project layout
```
app/
  rag.py       load -> chunk -> embed -> Chroma -> format with [source] tags
  tools.py     calculator (safe AST) + search_knowledge_base
  agent.py     LangGraph graph, SQLite memory, Assistant wrapper (answer/sources/tools)
  api.py       FastAPI app, dependency-injected assistant
  factory.py   wires real models/stores from env settings
  llm.py       Ollama / OpenAI model factories
data/docs/     knowledge base (add your own .md / .txt files)
tests/         18 offline tests with fakes
scripts/demo.py  live scripted demo -> demo_transcript.md
```

## Configuration
See `.env.example`. Use `LLM_PROVIDER=openai` with `OPENAI_API_KEY` to run on OpenAI instead of Ollama.

> The documents in `data/docs` are short sample reference notes for the demo. For design work, always check the governing code and the project specification.
