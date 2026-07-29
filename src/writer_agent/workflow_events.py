"""Meaningful, user-safe artifacts emitted by the agent workflow."""

from __future__ import annotations

from uuid import uuid4

from writer_agent.state import WorkflowEvent, WorkflowEventKind


def workflow_event(
    kind: WorkflowEventKind,
    title: str,
    *,
    content: str = "",
    details: list[str] | None = None,
    decision: str = "",
) -> WorkflowEvent:
    """Create one append-only event without prompts or hidden reasoning."""
    event: WorkflowEvent = {
        "id": str(uuid4()),
        "kind": kind,
        "title": title.strip(),
    }
    if content.strip():
        event["content"] = content.strip()
    if details:
        event["details"] = [
            item.strip() for item in details if item and item.strip()
        ]
    if decision.strip():
        event["decision"] = decision.strip()
    return event
