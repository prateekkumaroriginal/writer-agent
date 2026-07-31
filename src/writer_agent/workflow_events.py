"""Meaningful, user-safe artifacts emitted by the agent workflow."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from writer_agent.state import WorkflowEvent, WorkflowEventKind


def workflow_event(
    kind: WorkflowEventKind,
    title: str,
    *,
    content: str = "",
    details: list[str] | None = None,
    decision: str = "",
    subtask_name: str = "",
    objective: str = "",
    agent: str = "",
    review_criteria: list[str] | None = None,
    attempt: int | None = None,
    retry_count: int | None = None,
    sources: list[dict[str, str]] | None = None,
) -> WorkflowEvent:
    """Create one append-only event without prompts or hidden reasoning."""
    event: WorkflowEvent = {
        "id": str(uuid4()),
        "kind": kind,
        "title": title.strip(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    if content.strip():
        event["content"] = content.strip()
    if details:
        event["details"] = [
            item.strip() for item in details if item and item.strip()
        ]
    if decision.strip():
        event["decision"] = decision.strip()
    if subtask_name.strip():
        event["subtask_name"] = subtask_name.strip()
    if objective.strip():
        event["objective"] = objective.strip()
    if agent.strip():
        event["agent"] = agent.strip()
    if review_criteria:
        event["review_criteria"] = [
            item.strip() for item in review_criteria if item and item.strip()
        ]
    if attempt is not None:
        event["attempt"] = max(1, attempt)
    if retry_count is not None:
        event["retry_count"] = max(0, retry_count)
    if sources:
        event["sources"] = [
            {
                "title": str(source.get("title") or "").strip(),
                "url": str(source.get("url") or "").strip(),
                "snippet": str(source.get("snippet") or "").strip(),
            }
            for source in sources
        ]
    return event
