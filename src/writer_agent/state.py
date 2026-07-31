"""Runtime state types for the workflow graphs."""

import operator
from typing import Annotated, Any, Literal, TypeAlias, TypedDict

AgentType: TypeAlias = Literal["research", "data", "writing"]
PlanningMode: TypeAlias = Literal["initial", "revision"]
RevisionIntent: TypeAlias = Literal[
    "edit",
    "extend",
    "verify",
    "reanalyze",
    "replace",
    "question",
]
SubtaskStatus: TypeAlias = Literal[
    "pending",
    "running",
    "passed",
    "failed",
    "escalated",
]
ReviewAction: TypeAlias = Literal["pass", "retry", "replan", "escalate"]
FinalReviewAction: TypeAlias = Literal[
    "return",
    "retry",
    "replan",
    "escalate",
]
WorkflowEventKind: TypeAlias = Literal[
    "plan",
    "search",
    "research",
    "analysis",
    "draft",
    "review",
    "replan",
    "memory",
]
WorkflowStatus: TypeAlias = Literal[
    "initialised",
    "planning",
    "executing",
    "reviewing",
    "completed",
    "failed",
    "escalated",
]


class Subtask(TypedDict):
    id: str
    agent_type: AgentType
    objective: str
    expected_output: str
    tools_allowed: list[str]
    review_criteria: list[str]
    status: SubtaskStatus
    retry_count: int


class SubtaskResult(TypedDict):
    subtask_id: str
    agent_type: AgentType
    output: Any
    confidence: float
    sources: list[dict]
    errors: list[str]


class ReviewReport(TypedDict):
    subtask_id: str | None
    passed: bool
    score: float
    issues: list[str]
    action: ReviewAction


class FinalReview(TypedDict):
    passed: bool
    score: float
    issues: list[str]
    action: FinalReviewAction


class WorkflowEvent(TypedDict, total=False):
    id: str
    kind: WorkflowEventKind
    title: str
    content: str
    details: list[str]
    decision: str


def merge_workflow_events(
    current: list[WorkflowEvent],
    incoming: list[WorkflowEvent],
) -> list[WorkflowEvent]:
    """Append workflow events while preserving first-seen order by ID."""
    merged: list[WorkflowEvent] = []
    seen_ids: set[str] = set()
    for event in [*current, *incoming]:
        event_id = event.get("id")
        if not event_id or event_id in seen_ids:
            continue
        seen_ids.add(event_id)
        merged.append(event)
    return merged


class SupervisorState(TypedDict, total=False):
    task_id: str
    thread_id: str
    conversation_id: str
    turn_number: int
    user_id: str
    user_request: str
    effective_request: str
    memory_context: str
    run_metadata: dict[str, Any]
    planning_mode: PlanningMode

    previous_effective_request: str
    previous_final_answer: str
    previous_subtasks: list[Subtask]
    previous_subtask_results: dict[str, SubtaskResult]
    revision_intent: RevisionIntent
    reuse_previous_answer: bool
    reused_agent_types: list[AgentType]

    status: WorkflowStatus
    error: str | None

    plan: str
    plan_confidence: float
    subtasks: list[Subtask]

    current_subtask_id: str | None
    subtask_results: dict[str, SubtaskResult]

    review_reports: Annotated[list[ReviewReport], operator.add]
    final_review: FinalReview | None
    workflow_events: Annotated[list[WorkflowEvent], merge_workflow_events]

    max_retries: int
    final_retry_count: int
    max_final_retries: int
    revision_feedback: list[str]

    replan_count: int
    max_replans: int
    replan_requested: bool
    replan_feedback: list[str]

    escalation_reason: str | None

    final_answer: str | None
