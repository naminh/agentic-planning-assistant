"""Decision output schema for planning applications."""

from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class DecisionType(StrEnum):
    """Enum for decision types."""

    APPROVE = "approve"
    REFUSE = "refuse"


class Decision(BaseModel):
    """Structured planning decision."""

    decision: DecisionType = Field(..., description="The high-level decision outcome")
    reasons: list[str] = Field(
        default_factory=list,
        description="Planning reasons supporting the decision outcome",
    )
    conditions: list[str] = Field(
        default_factory=list,
        description="Planning conditions attached to an approval, if any",
    )

    def to_markdown(
        self,
        batch_name: str | None = None,
        batch_id: str | None = None,
        local_authority: str | None = None,
        created_at: datetime | None = None,
    ) -> str:
        """Generate a markdown representation of the decision.

        Args:
            batch_name: Optional batch name for the planning application.
            batch_id: Optional batch ID.
            local_authority: Optional local authority name.
            created_at: Optional creation timestamp.

        Returns:
            Formatted markdown string with decision and reasons.

        Example:
            >>> decision = Decision(
            ...     decision=DecisionType.APPROVE,
            ...     reasons=[
            ...         "The proposal would not harm neighbouring amenity.",
            ...     ],
            ...     conditions=[
            ...         "The development must begin within three years.",
            ...     ],
            ... )
            >>> print(decision.to_markdown(
            ...     batch_name="Planning Applications Q1 2026",
            ...     batch_id="batch_12345",
            ...     local_authority="Westminster Council",
            ... ))
            # Planning Decision: Approve

            - **Batch Name:** Planning Applications Q1 2026
            - **Batch ID:** batch_12345
            - **Local Authority:** Doncaster Council

            ## Reasons

            - The proposal would not harm neighbouring amenity.

            ## Conditions

            - The development must begin within three years.
        """
        decision_label = self.decision.value.replace("_", " ").title()
        lines = [f"# Planning Decision: {decision_label}", ""]

        metadata_lines = []
        if batch_name:
            metadata_lines.append(f"- **Batch Name:** {batch_name}")
        if batch_id:
            metadata_lines.append(f"- **Batch ID:** {batch_id}")
        if local_authority:
            metadata_lines.append(f"- **Local Authority:** {local_authority}")
        if created_at:
            formatted_date = created_at.strftime("%Y-%m-%d %H:%M:%S")
            metadata_lines.append(f"- **Created:** {formatted_date}")

        if metadata_lines:
            lines.extend(metadata_lines)
            lines.append("")

        if self.reasons:
            if self.decision == DecisionType.REFUSE:
                lines.append("## Reasons for Refusal")
            else:
                lines.append("## Reasons")

            lines.append("")
            for reason in self.reasons:
                lines.append(f"- {reason}")

        if self.conditions:
            if self.reasons:
                lines.append("")

            lines.append("## Conditions")
            lines.append("")
            for condition in self.conditions:
                lines.append(f"- {condition}")

        return "\n".join(lines)

    def to_markdown_file(
        self,
        file_path: Path,
        batch_name: str | None = None,
        batch_id: str | None = None,
        local_authority: str | None = None,
        created_at: datetime | None = None,
    ) -> None:
        """Write the decision as markdown to a file.

        Args:
            file_path: Path where the markdown file will be written.
            batch_name: Optional batch name for the planning application.
            batch_id: Optional batch ID.
            local_authority: Optional local authority name.
            created_at: Optional creation timestamp.
        """
        markdown_content = self.to_markdown(
            batch_name=batch_name,
            batch_id=batch_id,
            local_authority=local_authority,
            created_at=created_at,
        )
        file_path.write_text(markdown_content, encoding="utf-8")
