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
TAVILY_API_KEY=your_tavily_api_key
DATABASE_URL=postgresql://writer_agent:writer_agent_dev@localhost:5432/writer_agent?sslmode=disable
LANGGRAPH_STRICT_MSGPACK=true
```

Start PostgreSQL:

```bash
docker compose up -d --wait checkpoint-db
```

## Run

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

## Stop PostgreSQL

```bash
docker compose stop checkpoint-db
```

The database volume is retained so workflows remain available after the
container starts again.
