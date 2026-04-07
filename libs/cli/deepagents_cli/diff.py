"""Session diff generation for the ``/diff`` slash command.

Extracts file modifications from conversation tool calls and produces
unified diffs for display via :class:`~deepagents_cli.widgets.messages.DiffMessage`.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


@dataclass(frozen=True, slots=True)
class FileChange:
    """A single file modification extracted from a tool call."""

    file_path: str
    operation: str  # "edit" or "create"
    old_string: str
    new_string: str


def extract_file_changes(messages: list[BaseMessage]) -> list[FileChange]:
    """Extract successful file changes from conversation messages.

    Walks through the message history, identifies ``edit_file`` and
    ``write_file`` tool calls, and returns only those whose corresponding
    :class:`~langchain_core.messages.ToolMessage` indicates success.

    Args:
        messages: LangChain conversation messages from thread state.

    Returns:
        Ordered list of successful file changes.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    # First pass: collect tool-call metadata keyed by ID.
    tool_calls: dict[str, dict[str, object]] = {}
    call_order: list[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in getattr(msg, "tool_calls", []):
                tc_id = tc.get("id")
                name = tc.get("name", "")
                if tc_id and name in ("edit_file", "write_file"):
                    tool_calls[tc_id] = {"name": name, "args": tc.get("args", {})}
                    call_order.append(tc_id)

    # Second pass: determine which calls succeeded.
    successful: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tc_id = getattr(msg, "tool_call_id", None)
            if tc_id and tc_id in tool_calls:
                status = getattr(msg, "status", "success")
                if status == "error":
                    continue
                content = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
                if content.startswith("Error:"):
                    continue
                successful.add(tc_id)

    # Build change list in chronological order.
    changes: list[FileChange] = []
    for tc_id in call_order:
        if tc_id not in successful:
            continue
        info = tool_calls[tc_id]
        args: dict[str, str] = info["args"]  # type: ignore[assignment]
        if info["name"] == "edit_file":
            changes.append(
                FileChange(
                    file_path=args.get("file_path", "unknown"),
                    operation="edit",
                    old_string=args.get("old_string", ""),
                    new_string=args.get("new_string", ""),
                )
            )
        elif info["name"] == "write_file":
            changes.append(
                FileChange(
                    file_path=args.get("file_path", "unknown"),
                    operation="create",
                    old_string="",
                    new_string=args.get("content", ""),
                )
            )
    return changes


def build_unified_diff(change: FileChange) -> str:
    """Generate a unified diff string for a single file change.

    Args:
        change: The file change to convert.

    Returns:
        Unified diff string compatible with
        :func:`~deepagents_cli.widgets.diff.compose_diff_lines`.
    """
    old_lines = change.old_string.splitlines()
    new_lines = change.new_string.splitlines()

    diff_lines = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=change.file_path,
            tofile=change.file_path,
            lineterm="",
        )
    )
    return "\n".join(diff_lines)


def build_summary(changes: list[FileChange]) -> str:
    """Build a one-line summary of all changes.

    Args:
        changes: List of file changes.

    Returns:
        Human-readable summary string, e.g.
        ``"3 files changed: 2 edits, 1 new file"``.
    """
    files = {c.file_path for c in changes}
    edits = sum(1 for c in changes if c.operation == "edit")
    creates = sum(1 for c in changes if c.operation == "create")
    parts: list[str] = []
    if edits:
        parts.append(f"{edits} edit{'s' if edits != 1 else ''}")
    if creates:
        parts.append(f"{creates} new file{'s' if creates != 1 else ''}")
    return f"{len(files)} file{'s' if len(files) != 1 else ''} changed: {', '.join(parts)}"
