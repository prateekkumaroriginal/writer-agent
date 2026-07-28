"""Structured LLM response schemas."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from writer_agent.state import AgentType, FinalReviewAction, ReviewAction


class PlannedSubtaskSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_type: AgentType = Field(
        description="The specialist agent type that should handle this subtask."
    )
    objective: str = Field(
        description=(
            "A clear, specific instruction describing what this specialist must "
            "accomplish for this subtask."
        )
    )
    expected_output: str = Field(
        description=(
            "The expected deliverable from this subtask, including the desired "
            "format or contents where relevant."
        )
    )
    review_criteria: list[str] = Field(
        default_factory=list,
        description=(
            "A checklist the reviewer should use to judge whether this subtask "
            "was completed successfully."
        ),
    )


class SupervisorPlanSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: str = Field(
        description=(
            "A concise high-level execution plan for completing the user request. "
            "Describe the overall approach, not runtime details like ids, retries, "
            "statuses, or tool permissions."
        )
    )
    plan_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence that the plan is suitable for the user request. "
            "Use 0.0 for no confidence and 1.0 for very high confidence."
        ),
    )
    subtasks: list[PlannedSubtaskSchema] = Field(
        min_length=1,
        description=(
            "Ordered subtasks to execute sequentially. "
            "The graph will run these in the given order."
        ),
    )


class ReviewDecisionSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool = Field(
        description=(
            "Whether the specialist output satisfies the subtask objective, "
            "expected output, and review criteria."
        )
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Quality score for the specialist output. "
            "Use 0.0 for unusable output and 1.0 for excellent output."
        ),
    )
    issues: list[str] = Field(
        default_factory=list,
        description=(
            "Specific problems found during review. "
            "Leave empty only if the output passes without concerns."
        ),
    )
    action: ReviewAction = Field(
        description=(
            "The next action for this subtask. Use 'pass' if the result is "
            "acceptable, 'retry' if the same specialist should try again, "
            "'replan' if the workflow needs different or additional subtasks, or "
            "'escalate' if human review is needed."
        )
    )

    @model_validator(mode="after")
    def validate_action_matches_passed(self) -> "ReviewDecisionSchema":
        """Ensure the specialist review action agrees with its pass decision."""
        if self.passed and self.action != "pass":
            raise ValueError("A passed review must use action='pass'.")
        if not self.passed and self.action == "pass":
            raise ValueError("A failed review cannot use action='pass'.")
        return self


class WritingResponseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(
        min_length=20,
        description=(
            "The user-facing content answer produced by the writing agent. "
            "It should satisfy the writing subtask objective and use the approved "
            "previous subtask results."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the quality and completeness of the content. "
            "Use 0.0 for no confidence and 1.0 for very high confidence."
        ),
    )


class ResearchResponseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="Concise summary of the research findings.")
    findings: list[str] = Field(
        description="Important findings relevant to the research subtask."
    )
    uncertainties: list[str] = Field(
        default_factory=list,
        description="Known gaps, caveats, or uncertainty in the research.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the quality and completeness of the research.",
    )


class DataResponseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(
        description=(
            "Structured analysis produced by the data agent. It should satisfy "
            "the current data subtask using the provided research context."
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the quality and completeness of the analysis. "
            "Use 0.0 for no confidence and 1.0 for very high confidence."
        ),
    )


class SearchQuerySchema(BaseModel):
    """LLM output for a provider-safe search query."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=5,
        max_length=380,
        description=(
            "A concise web search query for the research subtask. "
            "It must be specific, provider-safe, and under 380 characters."
        ),
    )


class FinalReviewSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool = Field(
        description="Whether the complete workflow result is ready to return."
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall quality score for the complete workflow result.",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Specific problems with the complete workflow result.",
    )
    action: FinalReviewAction = Field(
        description="Next workflow action. Use return, retry, replan, or escalate."
    )

    @model_validator(mode="after")
    def validate_action_matches_passed(self) -> "FinalReviewSchema":
        """Ensure the final review action agrees with its pass decision."""
        if self.passed and self.action != "return":
            raise ValueError("A passed final review must use action='return'.")
        if not self.passed and self.action == "return":
            raise ValueError("A failed final review cannot use action='return'.")
        return self
