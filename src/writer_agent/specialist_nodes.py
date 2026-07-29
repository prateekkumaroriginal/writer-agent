"""Nodes and routers owned by the specialist execution graph."""

from langchain_core.messages import HumanMessage, SystemMessage

from writer_agent.helpers import (
    format_research_context_for_data,
    format_search_results_for_research,
    format_subtask_brief,
    format_upstream_specialist_brief,
    get_current_subtask,
    update_subtask,
)
from writer_agent.model import llm
from writer_agent.prompts import (
    DATA_SYSTEM_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    SEARCH_QUERY_SYSTEM_PROMPT,
    WRITING_SYSTEM_PROMPT,
)
from writer_agent.provider_errors import raise_if_retryable_provider_error
from writer_agent.schemas import (
    DataResponseSchema,
    ResearchResponseSchema,
    ReviewDecisionSchema,
    SearchQuerySchema,
    WritingResponseSchema,
)
from writer_agent.search import search_web
from writer_agent.state import (
    ReviewReport,
    Subtask,
    SubtaskResult,
    SupervisorState,
)
from writer_agent.workflow_events import workflow_event


def pick_next_subtask(state: SupervisorState) -> SupervisorState:
    """Select and start the first pending subtask."""
    for subtask in state.get("subtasks", []):
        if subtask.get("status") == "pending":
            return {
                "status": "executing",
                "current_subtask_id": subtask["id"],
                "subtasks": update_subtask(
                    state,
                    subtask["id"],
                    status="running",
                ),
            }
    return {"current_subtask_id": None}


def escalate_current_subtask(state: SupervisorState) -> SupervisorState:
    """Escalate the current subtask and stop specialist execution."""
    subtask = get_current_subtask(state)
    if subtask is None:
        return {
            "status": "escalated",
            "current_subtask_id": None,
            "escalation_reason": (
                "Reviewer escalated, but no current subtask was found."
            ),
            "final_answer": None,
        }
    return {
        "status": "escalated",
        "subtasks": update_subtask(state, subtask["id"], status="escalated"),
        "current_subtask_id": None,
        "escalation_reason": (
            f"Subtask {subtask['id']} was escalated by the reviewer."
        ),
        "final_answer": None,
    }


def route_after_pick_next_subtask(state: SupervisorState) -> str:
    """Route the selected subtask to its assigned specialist."""
    subtask = get_current_subtask(state)
    return subtask["agent_type"] if subtask is not None else "done"


def build_search_query(state: SupervisorState, subtask: Subtask) -> str:
    """Generate and normalize a provider-safe research query."""
    search_query_llm = llm.with_structured_output(SearchQuerySchema)
    response = search_query_llm.invoke(
        [
            SystemMessage(content=SEARCH_QUERY_SYSTEM_PROMPT),
            HumanMessage(
                content=f"""
User request:
{state.get("user_request")}

Research objective:
{subtask["objective"]}

Expected output:
{subtask["expected_output"]}

Review criteria:
{chr(10).join(f"- {criterion}" for criterion in subtask.get("review_criteria", []))}
""".strip()
            ),
        ]
    )
    query = " ".join(response.query.split())
    if len(query) > 380:
        raise ValueError(
            f"Generated search query is too long: {len(query)} characters."
        )
    return query


def _review_workflow_event(
    subtask: Subtask | None,
    report: ReviewReport,
):
    """Translate a specialist decision into a user-safe audit event."""
    specialist = (
        {
            "research": "Research",
            "data": "Analysis",
            "writing": "Draft",
        }[subtask["agent_type"]]
        if subtask
        else "Specialist"
    )
    action = report["action"]
    content = {
        "pass": f"{specialist} passed specialist review.",
        "retry": f"The reviewer requested another {specialist.lower()} attempt.",
        "replan": "The reviewer requested a revised workflow plan.",
        "escalate": f"{specialist} could not be approved.",
    }[action]
    return workflow_event(
        "review",
        f"{specialist} review",
        content=content,
        details=[
            f"Score: {report['score']:.0%}",
            *report.get("issues", []),
        ],
        decision=action,
    )


def research_agent(state: SupervisorState) -> SupervisorState:
    """Produce source-grounded research for the current subtask."""
    research_llm = llm.with_structured_output(ResearchResponseSchema)
    subtask = get_current_subtask(state)
    if subtask is None:
        return {
            "error": "research_agent called without a current subtask.",
            "escalation_reason": "Missing current subtask.",
        }

    try:
        query = build_search_query(state, subtask)
    except Exception as exc:
        raise_if_retryable_provider_error(exc)
        raise
    print(f"SEARCH QUERY ({len(query)} chars):")
    print(query)
    events = [
        workflow_event(
            "search",
            "Web search",
            content=query,
        )
    ]

    try:
        search_results = search_web(query)
        search_context = format_search_results_for_research(search_results)
        response = research_llm.invoke(
            [
                SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"""
User request:
{state.get("user_request")}

Research task:
Objective:
{subtask["objective"]}

Expected output:
{subtask["expected_output"]}

Review criteria:
{chr(10).join(f"- {criterion}" for criterion in subtask.get("review_criteria", []))}

Search results:
{search_context}
""".strip()
                ),
            ]
        )
        result: SubtaskResult = {
            "subtask_id": subtask["id"],
            "agent_type": "research",
            "output": {
                "summary": response.summary,
                "findings": response.findings,
                "uncertainties": response.uncertainties,
            },
            "confidence": response.confidence,
            "sources": [
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "snippet": item.get("snippet"),
                }
                for item in search_results
            ],
            "errors": [],
        }
        events.append(
            workflow_event(
                "research",
                "Research response",
                content=response.summary,
                details=[
                    *[f"Finding: {item}" for item in response.findings],
                    *[
                        f"Uncertainty: {item}"
                        for item in response.uncertainties
                    ],
                    f"Sources gathered: {len(search_results)}",
                ],
            )
        )
    except Exception as exc:
        raise_if_retryable_provider_error(exc)
        result = {
            "subtask_id": subtask["id"],
            "agent_type": "research",
            "output": {},
            "confidence": 0.0,
            "sources": [],
            "errors": [f"Research agent failed: {exc}"],
        }

    return {
        "subtask_results": {
            **state.get("subtask_results", {}),
            subtask["id"]: result,
        },
        "workflow_events": events,
    }


def data_agent(state: SupervisorState) -> SupervisorState:
    """Analyze passed research context for the current data subtask."""
    data_llm = llm.with_structured_output(DataResponseSchema)
    subtask = get_current_subtask(state)
    if subtask is None:
        return {
            "error": "data_agent called without a current subtask.",
            "escalation_reason": "Missing current subtask.",
        }

    try:
        response = data_llm.invoke(
            [
                SystemMessage(content=DATA_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"""
User request:
{state.get("user_request")}

Data task:
{format_subtask_brief(subtask)}

Passed research context:
{format_research_context_for_data(state)}
""".strip()
                ),
            ]
        )
        result: SubtaskResult = {
            "subtask_id": subtask["id"],
            "agent_type": "data",
            "output": {"content": response.content},
            "confidence": response.confidence,
            "sources": [],
            "errors": [],
        }
        events = [
            workflow_event(
                "analysis",
                "Analysis response",
                content=response.content,
            )
        ]
    except Exception as exc:
        raise_if_retryable_provider_error(exc)
        result = {
            "subtask_id": subtask["id"],
            "agent_type": "data",
            "output": {},
            "confidence": 0.0,
            "sources": [],
            "errors": [f"Data agent generation failed: {exc}"],
        }
        events = []

    return {
        "subtask_results": {
            **state.get("subtask_results", {}),
            subtask["id"]: result,
        },
        "workflow_events": events,
    }


def writing_agent(state: SupervisorState) -> SupervisorState:
    """Produce writing from approved context and optional revision feedback."""
    writer_llm = llm.with_structured_output(WritingResponseSchema)
    subtask = get_current_subtask(state)
    if subtask is None:
        return {
            "error": "writing_agent called without a current subtask.",
            "escalation_reason": "Missing current subtask.",
        }

    revision_feedback = state.get("revision_feedback", [])
    feedback_text = (
        "\n".join(f"- {issue}" for issue in revision_feedback)
        if revision_feedback
        else ""
    )

    try:
        response = writer_llm.invoke(
            [
                SystemMessage(content=WRITING_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"""
User request:
{state.get("user_request")}

Writing task:
{format_subtask_brief(subtask)}

Upstream specialist context:
{format_upstream_specialist_brief(state)}

Final-review revision feedback:
{feedback_text}
""".strip()
                ),
            ]
        )
        result: SubtaskResult = {
            "subtask_id": subtask["id"],
            "agent_type": "writing",
            "output": {"content": response.content},
            "confidence": response.confidence,
            "sources": [],
            "errors": [],
        }
        events = [
            workflow_event(
                "draft",
                "Revised draft" if revision_feedback else "Draft response",
                content=response.content,
                details=[
                    *[
                        f"Revision requested: {item}"
                        for item in revision_feedback
                    ]
                ],
            )
        ]
    except Exception as exc:
        raise_if_retryable_provider_error(exc)
        result = {
            "subtask_id": subtask["id"],
            "agent_type": "writing",
            "output": {},
            "confidence": 0.0,
            "sources": [],
            "errors": [f"Writing agent generation failed: {exc}"],
        }
        events = []

    return {
        "subtask_results": {
            **state.get("subtask_results", {}),
            subtask["id"]: result,
        },
        "workflow_events": events,
    }


def review_agent(state: SupervisorState) -> SupervisorState:
    """Evaluate the current specialist result and record a recovery action."""
    reviewer_llm = llm.with_structured_output(ReviewDecisionSchema)
    subtask = get_current_subtask(state)
    if subtask is None:
        report: ReviewReport = {
            "subtask_id": None,
            "passed": False,
            "score": 0.0,
            "issues": ["No current subtask found."],
            "action": "escalate",
        }
        return {
            "status": "reviewing",
            "review_reports": [report],
            "workflow_events": [_review_workflow_event(None, report)],
            "escalation_reason": "Reviewer could not find current subtask.",
        }

    result = state.get("subtask_results", {}).get(subtask["id"])
    if result is None:
        report = {
            "subtask_id": subtask["id"],
            "passed": False,
            "score": 0.0,
            "issues": ["No result was produced for this subtask."],
            "action": "retry",
        }
        return {
            "status": "reviewing",
            "review_reports": [report],
            "workflow_events": [
                _review_workflow_event(subtask, report)
            ],
        }

    try:
        decision = reviewer_llm.invoke(
            [
                SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"""
Subtask:
{subtask}

Specialist result:
{result}
""".strip()
                ),
            ]
        )
        report = {
            "subtask_id": subtask["id"],
            "passed": decision.passed,
            "score": decision.score,
            "issues": decision.issues,
            "action": decision.action,
        }
    except Exception as exc:
        raise_if_retryable_provider_error(exc)
        return {
            "status": "reviewing",
            "error": f"Reviewer generation failed: {exc}",
            "escalation_reason": (
                "Reviewer could not produce a valid structured decision."
            ),
        }

    return {
        "status": "reviewing",
        "review_reports": [report],
        "workflow_events": [_review_workflow_event(subtask, report)],
    }


def route_after_subtask_review(state: SupervisorState) -> str:
    """Route a reviewed subtask to pass, retry, replan, failure, or escalation."""
    review_reports = state.get("review_reports", [])
    if not review_reports:
        return "escalate"

    action = review_reports[-1]["action"]
    if action in {"pass", "replan", "escalate"}:
        return action
    if action == "retry":
        subtask = get_current_subtask(state)
        if subtask is None:
            return "escalate"
        if subtask.get("retry_count", 0) < state.get("max_retries", 0):
            return "retry"
        return "fail"
    return "escalate"


def request_replan(state: SupervisorState) -> SupervisorState:
    """Send reviewer issues to the parent graph for supervisor replanning."""
    subtask = get_current_subtask(state)
    review_reports = state.get("review_reports", [])
    feedback = review_reports[-1]["issues"] if review_reports else []

    if state.get("replan_count", 0) >= state.get("max_replans", 0):
        return {
            "status": "escalated",
            "replan_requested": True,
            "replan_feedback": feedback,
            "current_subtask_id": None,
            "escalation_reason": "Workflow replan limit reached.",
            "final_answer": None,
        }

    if subtask is None:
        return {
            "status": "escalated",
            "replan_requested": True,
            "replan_feedback": feedback,
            "current_subtask_id": None,
            "escalation_reason": (
                "Replanning was requested, but no current subtask was found."
            ),
            "final_answer": None,
        }

    return {
        "status": "planning",
        "subtasks": update_subtask(state, subtask["id"], status="escalated"),
        "current_subtask_id": None,
        "replan_requested": True,
        "replan_feedback": feedback,
        "escalation_reason": None,
        "final_answer": None,
    }


def mark_subtask_complete(state: SupervisorState) -> SupervisorState:
    """Mark the current subtask as passed and clear the execution pointer."""
    subtask = get_current_subtask(state)
    if subtask is None:
        return {"escalation_reason": "Tried to complete a missing subtask."}
    return {
        "subtasks": update_subtask(state, subtask["id"], status="passed"),
        "current_subtask_id": None,
    }


def retry_subtask(state: SupervisorState) -> SupervisorState:
    """Return the current subtask to pending and increment its retry count."""
    subtask = get_current_subtask(state)
    if subtask is None:
        return {"escalation_reason": "Tried to retry a missing subtask."}
    return {
        "subtasks": update_subtask(
            state,
            subtask["id"],
            status="pending",
            retry_count=subtask.get("retry_count", 0) + 1,
        ),
        "current_subtask_id": None,
    }


def mark_subtask_failed(state: SupervisorState) -> SupervisorState:
    """Mark the current subtask as failed after its retries are exhausted."""
    subtask = get_current_subtask(state)
    if subtask is None:
        return {
            "status": "escalated",
            "escalation_reason": (
                "Subtask failure occurred, but no current subtask was found."
            ),
            "current_subtask_id": None,
        }
    return {
        "subtasks": update_subtask(state, subtask["id"], status="failed"),
        "current_subtask_id": None,
        "escalation_reason": f"Subtask {subtask['id']} failed after retries.",
    }
