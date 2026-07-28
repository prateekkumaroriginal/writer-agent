# Writer Agent

A LangGraph prototype that plans a request, executes sequential research, data,
and writing subtasks, reviews each result, and returns or escalates the final
response.

The reusable implementation lives in `src/writer_agent`. The original
`phase-1.ipynb` remains as an experiment and record of the first end-to-end run.

## Setup

Create and activate a virtual environment, then install the package:

```bash
python -m pip install -e .
```

Add the provider credentials to `.env`:

```dotenv
GROQ_API_KEY=...
TAVILY_API_KEY=...
```

## Usage

```python
from writer_agent import build_supervisor_graph

graph = build_supervisor_graph()
result = graph.invoke(
    {
        "user_id": "user_001",
        "user_request": "Write an overview of retrieval-augmented generation.",
    }
)

print(result["status"])
print(result["final_answer"])
```

## Tests

The retry and replanning control paths use deterministic tests and do not call
external providers:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The live full-graph test uses the exact request “Write a report on Student
Protest on Jantar Mantar” and requires Groq and Tavily credentials:

```bash
RUN_LIVE_TESTS=1 PYTHONPATH=src python -m unittest \
  tests.test_full_graph_student_protest -v
```

The complete returned graph state is written to
`tests/runs/student-protest-jantar-mantar.json`. The `tests/runs` directory is
ignored by Git.

## Package layout

- `state.py`: workflow state and runtime types
- `schemas.py`: structured LLM response models
- `prompts.py`: agent system prompts
- `search.py`: Tavily adapter
- `helpers.py`: state and context-formatting helpers
- `parent_nodes.py`: supervisor and final-review nodes
- `specialist_nodes.py`: specialist execution and review nodes
- `graph.py`: specialist and supervisor graph assembly

The notebook records the original prototype. The package is the canonical
implementation and now includes bounded final-writing retries and
supervisor-driven replanning.
