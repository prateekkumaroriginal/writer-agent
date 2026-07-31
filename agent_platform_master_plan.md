# Multi-Agent Platform Phase Plan

## Phase 1 — Core Agent Architecture

### Goal

Prove that the multi-agent orchestration loop works end to end without memory, human approval, or complex production infrastructure.

### Implemented

- Parent LangGraph `StateGraph`
- Specialist execution subgraph
- LLM supervisor planner
- LLM search query generator
- Real search adapter
- LLM research synthesiser
- LLM data agent
- LLM writing agent
- LLM subtask reviewer
- LLM final reviewer
- Sequential subtask execution
- Per-subtask retry loop
- Bounded final-writing revision loop
- Bounded supervisor-driven replanning
- Escalation on unrecoverable failure
- Successful end-user workflow test
- Ambiguous-request escalation test
- `final_answer` success-only invariant test

### Parent graph

```text
START
→ initialise_task
→ supervisor_plan
→ execute_specialists
→ final_review
→ return_response / prepare_final_retry / prepare_replan / escalate
→ END
```

### Specialist subgraph

```text
START
→ pick_next_subtask
→ specialist agent
→ review_agent
→ mark_subtask_complete / retry_subtask / request_replan / mark_subtask_failed / escalate_current_subtask
→ pick_next_subtask or END
```

### Included agents

#### Supervisor planner

Creates:

- `plan: str`
- `plan_confidence: float`
- ordered planned subtasks

The program converts planned subtasks into runtime subtasks by adding:

- `id`
- `tools_allowed`
- `status`
- `retry_count`

#### Search query generator

Creates a concise provider-safe query for the research tool.

The query generator is universal and should not hardcode a market such as project management.

The program enforces provider limits.

#### Research agent

Uses:

- user request
- current research subtask
- LLM-generated search query
- real search results

Returns a `SubtaskResult` containing:

- source-grounded summary
- findings
- uncertainties
- confidence
- sources from the search provider
- errors, if any

#### Data agent

Uses:

- user request
- current data subtask
- passed research context only

Returns structured analysis as `content`.

It should not invent facts beyond the research context.

#### Writing agent

Uses:

- user request
- current writing subtask
- passed upstream specialist context from non-writing agents
- final-review revision feedback, when retrying

It must not use previous writing outputs.

Returns user-facing writing output as:

```python
output = {
    "content": "..."
}
```

#### Review agent

Reviews one specialist output against:

- the current subtask objective
- expected output
- review criteria
- specialist result
- confidence
- errors

It returns:

```python
{
    "subtask_id": str | None,
    "passed": bool,
    "score": float,
    "issues": list[str],
    "action": "pass" | "retry" | "replan" | "escalate",
}
```

`replan` returns control to the supervisor with the reviewer issues. The
supervisor creates a corrected plan, clears stale plan results, and executes the
replacement plan. Replanning is bounded.

#### Final reviewer

Receives only the final writing output.

It returns:

```python
{
    "passed": bool,
    "score": float,
    "issues": list[str],
    "action": "return" | "retry" | "replan" | "escalate",
}
```

`retry` reopens only the final passed writing subtask and supplies the review
issues as revision feedback. `replan` returns the issues to the supervisor.
Both loops are bounded and escalate when their limits are exhausted.

## Phase 2 — Persistence and Checkpointing

### Goal

Make workflows resumable and inspectable.

This phase adds durable execution state, not long-term semantic memory.

### Implemented

- Checkpointer injection in the supervisor graph builder
- Postgres-backed production runtime with automatic schema setup
- Local Docker Postgres service with persistent storage and health checks
- Required thread/session IDs
- Resume from the latest checkpoint
- Latest-state and checkpoint-history inspection
- Per-run metadata in workflow state and checkpoint configuration
- Protection against accidental reuse of an existing thread for a new request
- Deterministic interrupt/resume and process-restart persistence tests
- Real Postgres cross-connection resume integration test

Postgres is the sole checkpoint backend for local and deployed workflows.

### Important distinction

Checkpointing is not memory.

Checkpointing stores workflow state so a graph run can be resumed or inspected.

Long-term memory stores reusable facts, preferences, or project knowledge across runs.

### Original scope

- LangGraph checkpointer
- thread/session IDs
- durable checkpoint storage
- ability to resume interrupted runs
- basic run metadata
- basic state inspection

### Likely storage

Use a durable checkpointer such as Postgres-based checkpointing when moving beyond local development.

For local experiments, a simpler checkpointer may be acceptable first.

### Phase 2 implementation outline

1. Add a `thread_id` or equivalent run/session identifier.
2. Compile the graph with a checkpointer.
3. Pass thread config when invoking the graph.
4. Verify the graph can resume from checkpoints.
5. Add a smoke test that invokes the same thread twice and confirms state continuity.
6. Keep long-term memory out of scope.

### Phase 2 should not add

- ChromaDB
- semantic memory
- user preference memory
- human approval UI
- parallel execution
- dependency graphs

## Phase 3A — Multi-Turn Iterations

### Status

Implemented.

### Goal

Allow users to revise or extend completed work inside one visible conversation
without overwriting checkpoint history or blindly rerunning every specialist.

### Implemented behavior

- one durable conversation containing multiple immutable task runs
- one new LangGraph checkpoint thread per follow-up
- parent/child run linkage and ordered version history
- supervisor initial, revision, and replan modes
- standalone effective requests for follow-up execution
- selective reuse of passed research and data artifacts
- deterministic freshness and calculation guards
- low-confidence revision fallback to fresh full planning
- writing and final review for every revised user-facing answer
- Streamlit follow-up input and version navigation
- migration of existing tasks into one-turn conversations

## Phase 3B — Long-Term Memory

### Status

Implemented with ChromaDB replacing the originally planned pgvector index.

### Goal

Add durable memory after checkpointing and multi-turn iteration are stable.

### Possible memory types

- bounded core preferences considered for every writing task
- contextual preferences and facts retrieved only when relevant

### Planned storage

- PostgreSQL tables for structured memory
- ChromaDB for contextual vector retrieval

### Rules

- Do not inject all memory into every prompt.
- Retrieve only relevant memory.
- Keep memory separate from checkpointing.
- Save only durable user-provided facts and preferences.
- Activate saved memories automatically for all of the user's writing.
- Keep one-off revision instructions in conversation context, not memory.
- Let the memory agent add, edit, or delete memory from explicit user messages.
- Show every successful memory mutation and its affected content in the
  chronological workflow UI.
- Provide a user-scoped, read-only view of currently stored memories.

## Phase 4 — Human-in-the-Loop

### Goal

Introduce human approval and correction points.

### Add approval for

- low-confidence plans
- high-impact actions
- failed reviews
- sensitive tool actions
- final answer before return, where required
- memory writes

### Behaviour

- escalation can pause instead of ending
- human feedback can modify plan, subtasks, or output
- graph resumes after human decision

This phase likely needs product and UI decisions, so it remains deferred.

## Phase 5 — Agent Specialisation and Subgraphs

### Goal

Extract more complex agent internals into their own subgraphs only when needed.

### Candidate subgraphs

- research subgraph
- data analysis subgraph
- writing/revision subgraph
- source verification subgraph

### Example research subgraph

```text
generate search queries
→ run multiple searches
→ filter sources
→ extract evidence
→ synthesise findings
→ self-check source coverage
```

### Rule

Do not create subgraphs just for neatness.

Create them only when a node becomes internally complex enough to deserve its own loop.

## Phase 6 — Advanced Planning and Execution

### Goal

Support more complex workflows.

### Add

- subtask dependencies
- parallel execution where safe
- dynamic replanning
- agent reassignment
- partial retries
- multi-step tool plans
- richer final synthesis

This is where a `dependencies` field may make sense.

It was intentionally excluded from Phase 1.
