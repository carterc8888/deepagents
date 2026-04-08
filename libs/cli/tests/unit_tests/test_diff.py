"""Unit tests for the /diff and /undo slash command logic."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepagents_cli.diff import (
    FileChange,
    build_summary,
    build_unified_diff,
    extract_file_changes,
    get_last_undoable_change,
)


# ---------------------------------------------------------------------------
# extract_file_changes
# ---------------------------------------------------------------------------


class TestExtractFileChanges:
    """Tests for extracting successful file changes from messages."""

    def test_edit_file_success(self) -> None:
        messages = [
            HumanMessage(content="fix the bug"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc1",
                        "name": "edit_file",
                        "args": {
                            "file_path": "/app.py",
                            "old_string": "old code",
                            "new_string": "new code",
                        },
                    }
                ],
            ),
            ToolMessage(content="Successfully wrote to /app.py", tool_call_id="tc1"),
        ]
        changes = extract_file_changes(messages)
        assert len(changes) == 1
        assert changes[0].file_path == "/app.py"
        assert changes[0].operation == "edit"
        assert changes[0].old_string == "old code"
        assert changes[0].new_string == "new code"

    def test_write_file_success(self) -> None:
        messages = [
            HumanMessage(content="create a file"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc1",
                        "name": "write_file",
                        "args": {
                            "file_path": "/new.py",
                            "content": "print('hello')\n",
                        },
                    }
                ],
            ),
            ToolMessage(content="Successfully wrote to /new.py", tool_call_id="tc1"),
        ]
        changes = extract_file_changes(messages)
        assert len(changes) == 1
        assert changes[0].file_path == "/new.py"
        assert changes[0].operation == "create"
        assert changes[0].new_string == "print('hello')\n"

    def test_failed_edit_filtered_out(self) -> None:
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc1",
                        "name": "edit_file",
                        "args": {
                            "file_path": "/missing.py",
                            "old_string": "x",
                            "new_string": "y",
                        },
                    }
                ],
            ),
            ToolMessage(content="Error: file not found", tool_call_id="tc1"),
        ]
        changes = extract_file_changes(messages)
        assert len(changes) == 0

    def test_error_status_filtered_out(self) -> None:
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc1",
                        "name": "edit_file",
                        "args": {
                            "file_path": "/fail.py",
                            "old_string": "a",
                            "new_string": "b",
                        },
                    }
                ],
            ),
            ToolMessage(
                content="something went wrong",
                tool_call_id="tc1",
                status="error",
            ),
        ]
        changes = extract_file_changes(messages)
        assert len(changes) == 0

    def test_mixed_success_and_failure(self) -> None:
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc1",
                        "name": "edit_file",
                        "args": {
                            "file_path": "/good.py",
                            "old_string": "a",
                            "new_string": "b",
                        },
                    },
                    {
                        "id": "tc2",
                        "name": "edit_file",
                        "args": {
                            "file_path": "/bad.py",
                            "old_string": "x",
                            "new_string": "y",
                        },
                    },
                ],
            ),
            ToolMessage(content="Successfully wrote to /good.py", tool_call_id="tc1"),
            ToolMessage(content="Error: file not found", tool_call_id="tc2"),
        ]
        changes = extract_file_changes(messages)
        assert len(changes) == 1
        assert changes[0].file_path == "/good.py"

    def test_non_file_tools_ignored(self) -> None:
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc1",
                        "name": "execute",
                        "args": {"command": "ls"},
                    },
                    {
                        "id": "tc2",
                        "name": "read_file",
                        "args": {"file_path": "/foo.py"},
                    },
                ],
            ),
            ToolMessage(content="file list", tool_call_id="tc1"),
            ToolMessage(content="file content", tool_call_id="tc2"),
        ]
        changes = extract_file_changes(messages)
        assert len(changes) == 0

    def test_empty_messages(self) -> None:
        assert extract_file_changes([]) == []

    def test_dict_messages_after_conversion(self) -> None:
        """Messages returned as dicts (server mode) work after conversion."""
        from langchain_core.messages.utils import convert_to_messages

        raw = [
            {"type": "human", "content": "edit something"},
            {
                "type": "ai",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "name": "edit_file",
                        "args": {
                            "file_path": "/f.py",
                            "old_string": "a",
                            "new_string": "b",
                        },
                    }
                ],
            },
            {
                "type": "tool",
                "content": "Successfully wrote to /f.py",
                "tool_call_id": "tc1",
            },
        ]
        messages = convert_to_messages(raw)
        changes = extract_file_changes(messages)
        assert len(changes) == 1
        assert changes[0].file_path == "/f.py"

    def test_chronological_order(self) -> None:
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc1",
                        "name": "edit_file",
                        "args": {
                            "file_path": "/first.py",
                            "old_string": "a",
                            "new_string": "b",
                        },
                    }
                ],
            ),
            ToolMessage(content="ok", tool_call_id="tc1"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc2",
                        "name": "write_file",
                        "args": {"file_path": "/second.py", "content": "new"},
                    }
                ],
            ),
            ToolMessage(content="ok", tool_call_id="tc2"),
        ]
        changes = extract_file_changes(messages)
        assert len(changes) == 2
        assert changes[0].file_path == "/first.py"
        assert changes[1].file_path == "/second.py"


# ---------------------------------------------------------------------------
# build_unified_diff
# ---------------------------------------------------------------------------


class TestBuildUnifiedDiff:
    """Tests for unified diff generation."""

    def test_edit_produces_diff(self) -> None:
        change = FileChange(
            file_path="/app.py",
            operation="edit",
            old_string="def hello():\n    return 'hi'",
            new_string="def hello():\n    return 'hello world'",
        )
        diff = build_unified_diff(change)
        assert "---" in diff
        assert "+++" in diff
        assert "-    return 'hi'" in diff
        assert "+    return 'hello world'" in diff

    def test_create_produces_all_additions(self) -> None:
        change = FileChange(
            file_path="/new.py",
            operation="create",
            old_string="",
            new_string="line1\nline2\n",
        )
        diff = build_unified_diff(change)
        assert "+line1" in diff
        assert "+line2" in diff
        # No removal lines
        assert "\n-" not in diff

    def test_no_change_produces_empty(self) -> None:
        change = FileChange(
            file_path="/same.py",
            operation="edit",
            old_string="same content",
            new_string="same content",
        )
        diff = build_unified_diff(change)
        assert diff == ""


# ---------------------------------------------------------------------------
# build_summary
# ---------------------------------------------------------------------------


class TestBuildSummary:
    """Tests for change summary generation."""

    def test_single_edit(self) -> None:
        changes = [
            FileChange("/a.py", "edit", "old", "new"),
        ]
        summary = build_summary(changes)
        assert "1 file changed" in summary
        assert "1 edit" in summary

    def test_single_create(self) -> None:
        changes = [
            FileChange("/a.py", "create", "", "content"),
        ]
        summary = build_summary(changes)
        assert "1 file changed" in summary
        assert "1 new file" in summary

    def test_mixed(self) -> None:
        changes = [
            FileChange("/a.py", "edit", "old", "new"),
            FileChange("/a.py", "edit", "old2", "new2"),
            FileChange("/b.py", "create", "", "content"),
        ]
        summary = build_summary(changes)
        assert "2 files changed" in summary
        assert "2 edits" in summary
        assert "1 new file" in summary

    def test_plural_creates(self) -> None:
        changes = [
            FileChange("/a.py", "create", "", "a"),
            FileChange("/b.py", "create", "", "b"),
        ]
        summary = build_summary(changes)
        assert "2 new files" in summary


# ---------------------------------------------------------------------------
# get_last_undoable_change
# ---------------------------------------------------------------------------


class TestGetLastUndoableChange:
    """Tests for finding the last reversible edit."""

    def test_returns_last_edit(self) -> None:
        changes = [
            FileChange("/a.py", "edit", "old1", "new1"),
            FileChange("/b.py", "edit", "old2", "new2"),
        ]
        result = get_last_undoable_change(changes)
        assert result is not None
        assert result.file_path == "/b.py"

    def test_skips_creates(self) -> None:
        changes = [
            FileChange("/a.py", "edit", "old", "new"),
            FileChange("/b.py", "create", "", "content"),
        ]
        result = get_last_undoable_change(changes)
        assert result is not None
        assert result.file_path == "/a.py"

    def test_only_creates_returns_none(self) -> None:
        changes = [
            FileChange("/a.py", "create", "", "a"),
            FileChange("/b.py", "create", "", "b"),
        ]
        assert get_last_undoable_change(changes) is None

    def test_empty_returns_none(self) -> None:
        assert get_last_undoable_change([]) is None
