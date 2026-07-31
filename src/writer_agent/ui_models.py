"""Stable, user-facing models for the Streamlit application."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from writer_agent.state import SupervisorState

TaskStatus = Literal[
    "queued",
    "running",
    "completed",
    "escalated",
    "failed",
    "interrupted",
]
TaskStage = Literal[
    "queued",
    "planning",
    "researching",
    "analysing",
    "writing",
    "reviewing",
    "completed",
    "attention",
    "failed",
]
StepStatus = Literal["done", "current", "upcoming", "error"]
WorkflowEventKind = Literal[
    "plan",
    "search",
    "research",
    "analysis",
    "draft",
    "review",
    "replan",
    "memory",
]

STAGE_DEFINITIONS: tuple[tuple[str, str], ...] = (
    ("planning", "Plan"),
    ("researching", "Research"),
    ("analysing", "Analyse"),
    ("writing", "Write"),
    ("reviewing", "Review"),
)
TERMINAL_TASK_STATUSES = frozenset({"completed", "escalated", "failed"})
ACTIVE_TASK_STATUSES = frozenset({"queued", "running"})


class SourceView(BaseModel):
    """A safe source reference shown below the final document."""

    model_config = ConfigDict(frozen=True)

    title: str
    url: str
    snippet: str = ""


class StepView(BaseModel):
    """One stable, plain-language workflow step."""

    model_config = ConfigDict(frozen=True)

    key: str
    label: str
    status: StepStatus


class ReviewView(BaseModel):
    """Sanitized final-review information."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)


class WorkflowEventView(BaseModel):
    """One chronological artifact safe to expose in the Workflow tab."""

    model_config = ConfigDict(frozen=True)

    id: str
    kind: WorkflowEventKind
    title: str
    created_at: datetime
    content: str = ""
    details: list[str] = Field(default_factory=list)
    decision: str = ""
    subtask_name: str = ""
    objective: str = ""
    agent: str = ""
    review_criteria: list[str] = Field(default_factory=list)
    attempt: int | None = Field(default=None, ge=1)
    retry_count: int | None = Field(default=None, ge=0)
    sources: list[SourceView] = Field(default_factory=list)


class TaskView(BaseModel):
    """Complete frontend projection for one writing task."""

    model_config = ConfigDict(frozen=True)

    id: str
    thread_id: str
    user_id: str
    conversation_id: str = ""
    parent_task_id: str | None = None
    turn_number: int = Field(default=1, ge=1)
    title: str
    request: str
    status: TaskStatus
    stage: TaskStage
    status_message: str
    progress_current: int = Field(ge=0, le=5)
    progress_total: int = 5
    steps: list[StepView]
    final_answer: str | None = None
    sources: list[SourceView] = Field(default_factory=list)
    plan: str | None = None
    plan_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    review: ReviewView | None = None
    workflow_events: list[WorkflowEventView] = Field(default_factory=list)
    can_resume: bool = False
    created_at: datetime
    updated_at: datetime

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_TASK_STATUSES

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_TASK_STATUSES


class TaskSummary(BaseModel):
    """Compact task representation used in the sidebar."""

    model_config = ConfigDict(frozen=True)

    id: str
    conversation_id: str = ""
    turn_number: int = Field(default=1, ge=1)
    title: str
    status: TaskStatus
    stage: TaskStage
    updated_at: datetime


class TaskVersionSummary(BaseModel):
    """One selectable answer version inside a conversation."""

    model_config = ConfigDict(frozen=True)

    id: str
    turn_number: int = Field(ge=1)
    request: str
    status: TaskStatus
    updated_at: datetime


class TaskProjection(BaseModel):
    """Persistable subset derived from raw LangGraph state."""

    model_config = ConfigDict(frozen=True)

    status: TaskStatus
    stage: TaskStage
    status_message: str
    progress_current: int
    steps: list[StepView]
    final_answer: str | None
    sources: list[SourceView]
    plan: str | None
    plan_confidence: float | None
    review: ReviewView | None
    workflow_events: list[WorkflowEventView]
    can_resume: bool


def title_from_request(request: str, *, limit: int = 58) -> str:
    """Create a predictable task title without making another model call."""
    normalized = " ".join(request.split())
    if not normalized:
        return "Untitled writing task"
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip(" ,.;:-") + "…"


def stage_from_state(state: SupervisorState) -> TaskStage:
    """Map detailed workflow state to one of five user-facing stages."""
    workflow_status = state.get("status")
    if workflow_status == "completed":
        return "completed"
    if workflow_status == "escalated":
        return "attention"
    if workflow_status == "failed":
        return "failed"

    current_id = state.get("current_subtask_id")
    current_subtask = next(
        (
            item
            for item in state.get("subtasks", [])
            if item.get("id") == current_id
        ),
        None,
    )
    if current_subtask is not None:
        return {
            "research": "researching",
            "data": "analysing",
            "writing": "writing",
        }.get(current_subtask.get("agent_type"), "planning")

    if workflow_status == "reviewing":
        return "reviewing"
    if workflow_status in {"planning", "initialised"}:
        return "planning"
    if workflow_status == "executing":
        return _stage_from_completed_subtasks(state)
    return "queued"


def _stage_from_completed_subtasks(state: SupervisorState) -> TaskStage:
    """Infer the next visible stage between specialist nodes."""
    passed_types = {
        item.get("agent_type")
        for item in state.get("subtasks", [])
        if item.get("status") == "passed"
    }
    if "writing" in passed_types:
        return "reviewing"
    if "data" in passed_types:
        return "writing"
    if "research" in passed_types:
        return "analysing"
    return "researching"


def steps_for_stage(stage: TaskStage) -> list[StepView]:
    """Build the five-step progress display."""
    stage_keys = [key for key, _ in STAGE_DEFINITIONS]
    if stage == "completed":
        current_index = len(stage_keys)
    elif stage in {"attention", "failed"}:
        current_index = len(stage_keys) - 1
    elif stage == "queued":
        current_index = -1
    else:
        current_index = stage_keys.index(stage)

    steps: list[StepView] = []
    for index, (key, label) in enumerate(STAGE_DEFINITIONS):
        if stage == "completed" or index < current_index:
            status: StepStatus = "done"
        elif index == current_index:
            status = "error" if stage in {"attention", "failed"} else "current"
        else:
            status = "upcoming"
        steps.append(StepView(key=key, label=label, status=status))
    return steps


def sources_from_state(state: SupervisorState) -> list[SourceView]:
    """Flatten and de-duplicate safe HTTP research sources."""
    sources: list[SourceView] = []
    seen_urls: set[str] = set()
    for result in state.get("subtask_results", {}).values():
        for raw_source in result.get("sources", []):
            url = str(raw_source.get("url") or "").strip()
            if (
                not url.startswith(("https://", "http://"))
                or url in seen_urls
            ):
                continue
            seen_urls.add(url)
            title = str(raw_source.get("title") or url).strip()
            snippet = str(raw_source.get("snippet") or "").strip()
            sources.append(
                SourceView(title=title or url, url=url, snippet=snippet)
            )
    return sources


def review_from_state(state: SupervisorState) -> ReviewView | None:
    """Convert the final review without exposing its control-flow action."""
    review = state.get("final_review")
    if review is None:
        return None
    return ReviewView(
        passed=bool(review.get("passed")),
        score=float(review.get("score", 0.0)),
        issues=[str(issue) for issue in review.get("issues", [])],
    )


def workflow_events_from_state(
    state: SupervisorState,
) -> list[WorkflowEventView]:
    """Validate the append-only event feed at the frontend boundary."""
    events: list[WorkflowEventView] = []
    seen_ids: set[str] = set()
    for raw_event in state.get("workflow_events", []):
        event_id = str(raw_event.get("id") or "").strip()
        if not event_id or event_id in seen_ids:
            continue
        try:
            event = WorkflowEventView(
                id=event_id,
                kind=raw_event.get("kind"),
                title=str(raw_event.get("title") or "").strip(),
                created_at=raw_event.get("created_at"),
                content=str(raw_event.get("content") or "").strip(),
                details=[
                    str(item).strip()
                    for item in raw_event.get("details", [])
                    if str(item).strip()
                ],
                decision=str(raw_event.get("decision") or "").strip(),
                subtask_name=str(
                    raw_event.get("subtask_name") or ""
                ).strip(),
                objective=str(raw_event.get("objective") or "").strip(),
                agent=str(raw_event.get("agent") or "").strip(),
                review_criteria=[
                    str(item).strip()
                    for item in raw_event.get("review_criteria", [])
                    if str(item).strip()
                ],
                attempt=raw_event.get("attempt"),
                retry_count=raw_event.get("retry_count"),
                sources=_safe_event_sources(raw_event.get("sources", [])),
            )
        except (TypeError, ValueError):
            continue
        if not event.title:
            continue
        seen_ids.add(event.id)
        events.append(event)
    return events


def _safe_event_sources(raw_sources: list[dict]) -> list[SourceView]:
    """Validate and de-duplicate source links attached to one event."""
    sources: list[SourceView] = []
    seen_urls: set[str] = set()
    for raw_source in raw_sources:
        url = str(raw_source.get("url") or "").strip()
        if not url.startswith(("https://", "http://")) or url in seen_urls:
            continue
        seen_urls.add(url)
        title = str(raw_source.get("title") or url).strip()
        snippet = str(raw_source.get("snippet") or "").strip()
        sources.append(
            SourceView(title=title or url, url=url, snippet=snippet)
        )
    return sources


def projection_from_state(state: SupervisorState) -> TaskProjection:
    """Create the complete persisted UI projection from graph state."""
    workflow_status = state.get("status")
    stage = stage_from_state(state)

    if workflow_status == "completed":
        task_status: TaskStatus = "completed"
        message = "Your reviewed document is ready."
    elif workflow_status == "escalated":
        task_status = "escalated"
        message = (
            state.get("escalation_reason")
            or "This request needs more direction before it can be completed."
        )
    elif workflow_status == "failed":
        task_status = "failed"
        message = "The workflow could not complete. Your request was saved."
    else:
        task_status = "running"
        message = {
            "queued": "Waiting to start…",
            "planning": "Building a plan…",
            "researching": "Researching relevant sources…",
            "analysing": "Analysing the approved research…",
            "writing": "Writing the document…",
            "reviewing": "Reviewing the final document…",
        }.get(stage, "Working on your document…")

    steps = steps_for_stage(stage)
    progress = sum(step.status == "done" for step in steps)
    if stage not in {"queued", "completed"}:
        progress = min(progress + 1, len(STAGE_DEFINITIONS))

    final_answer = (
        state.get("final_answer") if task_status == "completed" else None
    )
    plan_confidence = state.get("plan_confidence")
    return TaskProjection(
        status=task_status,
        stage=stage,
        status_message=message,
        progress_current=progress,
        steps=steps,
        final_answer=final_answer,
        sources=sources_from_state(state),
        plan=state.get("plan") or None,
        plan_confidence=(
            float(plan_confidence) if plan_confidence is not None else None
        ),
        review=review_from_state(state),
        workflow_events=workflow_events_from_state(state),
        can_resume=False,
    )


def jsonable_models(items: list[BaseModel]) -> list[dict[str, Any]]:
    """Serialize a list of Pydantic UI models for JSONB storage."""
    return [item.model_dump(mode="json") for item in items]
