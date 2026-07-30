# Writer Agent

Writer Agent is a multi-agent research and writing workflow built with
LangGraph. It turns a user request into a reviewed final response by planning
the work, gathering information, analysing it, producing a draft, and checking
the result before returning it.

The project currently includes:

- supervisor-led task planning
- research, data-analysis, and writing agents
- specialist and final-output review
- bounded retries and replanning
- resumable workflows backed by PostgreSQL
- dark-only Streamlit product interface
- rerun-safe background task execution
- durable recent-task history and checkpoint resume
- multi-turn conversations with immutable answer versions
- supervisor-selected specialist reuse for efficient revisions

## Requirements

- Python 3.11 or newer
- Docker with Docker Compose
- a Groq API key
- a Tavily API key

## Setup

Create a virtual environment and install the project:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Create the environment file:

```bash
cp .env.example .env
```

Add your provider credentials to `.env`:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
TAVILY_API_KEY=your_tavily_api_key
DATABASE_URL=postgresql://writer_agent:writer_agent_dev@localhost:5432/writer_agent?sslmode=disable
LANGGRAPH_STRICT_MSGPACK=true
WRITER_AGENT_USER_ID=local-user
```

Start PostgreSQL:

```bash
docker compose up -d --wait checkpoint-db
```

## Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501`.

The app:

- accepts one substantial writing brief per task
- creates an idempotent, durable background job
- displays planning, research, analysis, writing, and review progress
- keeps running independently of Streamlit page reruns
- restores recent tasks from PostgreSQL
- returns final content only when the workflow completes
- exposes sources and a chronological workflow feed on demand
- records meaningful plans, searches, specialist outputs, reviews, retries,
  and replans without exposing hidden reasoning or operational internals
- supports Markdown download and clipboard copy
- accepts follow-up instructions on completed documents
- preserves every completed version within one visible conversation
- reuses passed research or analysis when the supervisor determines it remains
  valid, while rerunning affected specialists for changed or current information
- records the latest checkpointed state of each run as
  `tests/runs/<thread-id>.json`

The UI uses the local Compose database URL by default. Set `DATABASE_URL` to
override it. `WRITER_AGENT_USER_ID` is optional and defaults to `local-user`.
`GROQ_MODEL` is optional and defaults to `openai/gpt-oss-120b`.
Set `WRITER_AGENT_RUN_DIR` to store thread-named JSON run records somewhere
other than `tests/runs`.

## Python API

```python
from writer_agent import PersistentWriterAgent

with PersistentWriterAgent() as agent:
    result = agent.start(
        "example-thread-001",
        {
            "user_id": "user_001",
            "user_request": "Write an overview of retrieval-augmented generation.",
        },
    )

print(result["status"])
print(result["final_answer"])
```

Use a new thread ID for each new request.

## UI architecture

```text
Streamlit
→ WriterAgentService
→ Postgres task index + background runner
→ PersistentWriterAgent
→ LangGraph checkpoints
```

Streamlit stores only the selected task ID in session state. The task index is
the UI source of truth, so reruns render persisted state instead of restarting
the workflow. Each agent run owns its own Postgres checkpoint connection.

Each visible conversation contains one or more immutable runs. A follow-up
creates a new checkpoint thread linked to its parent run. The supervisor plans
the revision from the previous effective request, reviewed answer, and passed
specialist artifacts; only artifacts explicitly selected for reuse are copied
into the new run.

## Test

Run deterministic tests:

```bash
python -m unittest discover -s tests -v
```

Run Postgres integrations:

```bash
RUN_POSTGRES_TESTS=1 \
DATABASE_URL=postgresql://writer_agent:writer_agent_dev@localhost:5432/writer_agent?sslmode=disable \
python -m unittest discover -s tests -v
```

Live Groq/Tavily validation remains opt-in:

```bash
RUN_LIVE_TESTS=1 python -m unittest tests.test_full_graph_student_protest -v
```

## Stop PostgreSQL

```bash
docker compose stop checkpoint-db
```

The database volume is retained so workflows remain available after the
container starts again.
