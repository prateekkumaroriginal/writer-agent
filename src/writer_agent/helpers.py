"""Pure state and formatting helpers."""

from typing import Any

from writer_agent.schemas import PlannedSubtaskSchema
from writer_agent.state import AgentType, Subtask, SubtaskResult, SupervisorState

TOOLS_BY_AGENT_TYPE: dict[AgentType, list[str]] = {
    "research": ["web_search"],
    "data": [],
    "writing": [],
}


def get_current_subtask(state: SupervisorState) -> Subtask | None:
    """Return the subtask currently selected for execution."""
    current_id = state.get("current_subtask_id")
    if not current_id:
        return None

    for subtask in state.get("subtasks", []):
        if subtask["id"] == current_id:
            return subtask
    return None


def update_subtask(
    state: SupervisorState,
    subtask_id: str,
    **updates: Any,
) -> list[Subtask]:
    """Return the subtask list with one matching subtask updated."""
    return [
        {**subtask, **updates} if subtask["id"] == subtask_id else subtask
        for subtask in state.get("subtasks", [])
    ]


def all_subtasks_passed(state: SupervisorState) -> bool:
    """Report whether the workflow has at least one subtask and all have passed."""
    subtasks = state.get("subtasks", [])
    return bool(subtasks) and all(
        subtask.get("status") == "passed" for subtask in subtasks
    )


def build_runtime_subtasks(
    planned_subtasks: list[PlannedSubtaskSchema],
    *,
    start_index: int = 1,
) -> list[Subtask]:
    """Convert planned subtasks into executable runtime subtasks."""
    return [
        {
            "id": f"s{index}",
            "agent_type": planned.agent_type,
            "objective": planned.objective,
            "expected_output": planned.expected_output,
            "tools_allowed": TOOLS_BY_AGENT_TYPE[planned.agent_type],
            "review_criteria": planned.review_criteria,
            "status": "pending",
            "retry_count": 0,
        }
        for index, planned in enumerate(planned_subtasks, start=start_index)
    ]


def build_reused_artifacts(
    state: SupervisorState,
    *,
    reuse_research: bool,
    reuse_data: bool,
) -> tuple[list[Subtask], dict[str, SubtaskResult], list[AgentType]]:
    """Copy selected passed parent artifacts into a revision run."""
    allowed: set[AgentType] = set()
    if reuse_research:
        allowed.add("research")
    if reuse_data:
        allowed.add("data")

    previous_results = state.get("previous_subtask_results", {})
    subtasks: list[Subtask] = []
    results: dict[str, SubtaskResult] = {}
    reused_types: list[AgentType] = []
    for index, previous in enumerate(state.get("previous_subtasks", []), start=1):
        agent_type = previous.get("agent_type")
        if previous.get("status") != "passed" or agent_type not in allowed:
            continue
        result = previous_results.get(previous["id"])
        if result is None or result.get("errors"):
            continue
        artifact_id = f"reused-{index}"
        subtasks.append(
            {
                **previous,
                "id": artifact_id,
                "status": "passed",
                "retry_count": 0,
            }
        )
        results[artifact_id] = {**result, "subtask_id": artifact_id}
        if agent_type not in reused_types:
            reused_types.append(agent_type)
    return subtasks, results, reused_types


def effective_request(state: SupervisorState) -> str:
    """Return the standalone request specialists should execute."""
    return state.get("effective_request") or state.get("user_request", "")


def previous_answer_context(state: SupervisorState) -> str:
    """Return the prior reviewed answer only when the supervisor approved reuse."""
    if not state.get("reuse_previous_answer"):
        return "No previous answer was selected for reuse."
    return (
        state.get("previous_final_answer")
        or "No previous answer was available."
    )


def format_previous_artifacts_for_supervisor(state: SupervisorState) -> str:
    """Summarize reusable parent artifacts without exposing graph internals."""
    results = state.get("previous_subtask_results", {})
    sections: list[str] = []
    for subtask in state.get("previous_subtasks", []):
        if (
            subtask.get("status") != "passed"
            or subtask.get("agent_type") == "writing"
        ):
            continue
        result = results.get(subtask["id"])
        if result is None or result.get("errors"):
            continue
        output = format_specialist_output(result.get("output", {}))
        if not output:
            continue
        sections.append(
            f"{subtask['agent_type'].title()} artifact:\n{output[:4000]}"
        )
    return (
        "\n\n".join(sections)
        if sections
        else "No passed specialist artifacts are available."
    )


def format_search_results_for_research(
    search_results: list[dict[str, str]],
) -> str:
    """Format normalized search results as research-agent context."""
    if not search_results:
        return "No search results were found."

    sections = []
    for index, result in enumerate(search_results, start=1):
        sections.append(
            f"""
Result {index}
Title: {result.get("title", "Untitled result")}
URL: {result.get("url", "No URL provided")}
Snippet: {result.get("snippet", "No snippet provided")}
""".strip()
        )
    return "\n\n".join(sections)


def format_research_context_for_data(state: SupervisorState) -> str:
    """Collect passed upstream research as context for the data agent."""
    current_id = state.get("current_subtask_id")
    subtasks = state.get("subtasks", [])
    results = state.get("subtask_results", {})

    current_index = next(
        (
            index
            for index, subtask in enumerate(subtasks)
            if subtask["id"] == current_id
        ),
        None,
    )
    if current_index is None:
        return "No passed research context was available."

    sections: list[str] = []
    for subtask in subtasks[:current_index]:
        if (
            subtask.get("status") != "passed"
            or subtask.get("agent_type") != "research"
        ):
            continue

        result = results.get(subtask["id"])
        if result is None or result.get("errors"):
            continue

        output = result.get("output", {})
        section_parts: list[str] = []

        summary = output.get("summary")
        if summary:
            section_parts.append(f"Summary:\n{summary}")

        findings = output.get("findings", [])
        if findings:
            section_parts.append(
                "Findings:\n" + "\n".join(f"- {finding}" for finding in findings)
            )

        uncertainties = output.get("uncertainties", [])
        if uncertainties:
            section_parts.append(
                "Uncertainties:\n"
                + "\n".join(f"- {uncertainty}" for uncertainty in uncertainties)
            )

        sources = result.get("sources", [])
        if sources:
            source_lines = [
                (
                    f"- {source.get('title', 'Untitled source')}\n"
                    f"  URL: {source.get('url', 'No URL')}\n"
                    f"  Snippet: {source.get('snippet', '')}"
                )
                for source in sources
            ]
            section_parts.append("Sources:\n" + "\n".join(source_lines))

        if section_parts:
            sections.append("\n\n".join(section_parts))

    return (
        "\n\n".join(sections)
        if sections
        else "No passed research context was available."
    )


def format_subtask_brief(subtask: Subtask) -> str:
    """Format a subtask objective, expected output, and review criteria."""
    criteria = subtask.get("review_criteria", [])
    criteria_text = (
        "\n".join(f"- {criterion}" for criterion in criteria)
        if criteria
        else "- No specific review criteria provided."
    )
    return f"""
Objective:
{subtask["objective"]}

Expected output:
{subtask["expected_output"]}

Review criteria:
{criteria_text}
""".strip()


def format_specialist_output(output: Any) -> str:
    """Format a structured specialist result as readable context."""
    if isinstance(output, str):
        return output
    if not isinstance(output, dict):
        return str(output)

    sections: list[str] = []
    summary = output.get("summary")
    if summary:
        sections.append(f"Summary: {summary}")

    findings = output.get("findings")
    if isinstance(findings, list):
        finding_lines = [f"- {finding}" for finding in findings]
        if finding_lines:
            sections.append("Findings:\n" + "\n".join(finding_lines))

    uncertainties = output.get("uncertainties")
    if isinstance(uncertainties, list) and uncertainties:
        sections.append(
            "Uncertainties:\n"
            + "\n".join(f"- {uncertainty}" for uncertainty in uncertainties)
        )

    content = output.get("content")
    if content:
        sections.append(f"Content: {content}")

    return "\n\n".join(sections) if sections else str(output)


def format_upstream_specialist_brief(state: SupervisorState) -> str:
    """Collect passed non-writing outputs that precede the current subtask."""
    current_id = state.get("current_subtask_id")
    subtasks = state.get("subtasks", [])
    results = state.get("subtask_results", {})
    current_index = next(
        (
            index
            for index, subtask in enumerate(subtasks)
            if subtask["id"] == current_id
        ),
        None,
    )
    if current_index is None:
        return "No upstream specialist context was available."

    sections: list[str] = []
    for subtask in subtasks[:current_index]:
        if (
            subtask.get("status") != "passed"
            or subtask.get("agent_type") == "writing"
        ):
            continue

        result = results.get(subtask["id"])
        if result is None or result.get("errors"):
            continue

        sections.append(
            f"""
{result["agent_type"].title()} context:
{format_specialist_output(result.get("output", {}))}

Confidence: {result.get("confidence", 0.0)}
""".strip()
        )

    return (
        "\n\n".join(sections)
        if sections
        else "No upstream specialist context was available."
    )
