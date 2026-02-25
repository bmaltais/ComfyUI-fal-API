## 2026-02-25 - [Python Execution: uv run]
**Learning:** This project uses `uv run` to execute Python commands (including pylint, pytest, etc.) instead of calling Python or tools directly. This is required for consistent environment management and should be used for all Python-related CLI invocations.
**Action:** Always use `uv run` (e.g., `uv run pylint ...`, `uv run pytest ...`) for Python tool execution in this repo. Do not use `python ...` or direct tool calls.
