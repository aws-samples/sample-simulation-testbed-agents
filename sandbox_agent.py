"""
Sandbox Agent - CLI Simulation Environment

An agent that operates within a simulated command-line environment.
It can execute commands, observe outputs, and adapt its strategy
based on results — all within a safe, controlled sandbox.

Prerequisites:
    pip install -r requirements.txt

Learning objectives:
- Understand how agents operate in simulated environments
- See closed-loop reasoning: act → observe → adapt
- Learn how sandboxed execution enables safe experimentation
- Practice building environments that provide feedback to agents
"""

import time
import os
import shlex
import subprocess
import tempfile
import shutil
from shared.model import get_model
from shared.input_utils import get_multiline_input
from shared.streaming import StreamingCallbackHandler
from strands import Agent, tool


# --- Simulated File System Environment ---

class SandboxEnvironment:
    """A sandboxed file system environment for the agent to operate in."""

    def __init__(self):
        """Create a temporary directory as the sandbox."""
        self.root = tempfile.mkdtemp(prefix="agent_sandbox_")
        self.history = []  # Track all actions for learning

    def cleanup(self):
        """Remove the sandbox directory."""
        shutil.rmtree(self.root, ignore_errors=True)

    def log_action(self, action: str, result: str, success: bool):
        """Record an action and its outcome for the agent to learn from."""
        self.history.append({
            "action": action,
            "result": result,
            "success": success,
        })

    def get_history_summary(self) -> str:
        """Summarize past actions for the agent's memory."""
        if not self.history:
            return "No actions taken yet."
        lines = []
        for entry in self.history[-10:]:
            status = "✓" if entry["success"] else "✗"
            lines.append(f"  {status} {entry['action']}")
        return "\n".join(lines)


# Create the global sandbox
sandbox = SandboxEnvironment()


def _resolve_in_sandbox(user_path: str) -> str:
    """Resolve ``user_path`` inside the sandbox, guarding against traversal.

    Joins the path to the sandbox root and fully resolves it with realpath so
    ``..`` segments, absolute paths, and symlinks can't point outside the
    sandbox. Raises ValueError if the result escapes the sandbox root.
    """
    root = os.path.realpath(sandbox.root)
    target = os.path.realpath(os.path.join(root, user_path))
    # Allow the root itself or anything strictly beneath it. The `+ os.sep`
    # avoids a prefix bug where "/tmp/sandbox_evil" looks like it's inside
    # "/tmp/sandbox".
    if target != root and not target.startswith(root + os.sep):
        raise ValueError(f"path {user_path!r} escapes the sandbox root")
    return target


# --- Tools that operate within the sandbox ---

@tool
def list_files(path: str = ".") -> str:
    """List files and directories in the sandbox.

    Args:
        path: Relative path within the sandbox (default: root)
    """
    print(f"          list_files(path={path!r})")
    try:
        target = _resolve_in_sandbox(path)
    except ValueError:
        result = f"Error: Path '{path}' is outside the sandbox"
        sandbox.log_action(f"list_files({path})", result, False)
        print(f"          -> {result}")
        return result
    try:
        entries = os.listdir(target)
        if not entries:
            result = "(empty directory)"
        else:
            result = "\n".join(sorted(entries))
        sandbox.log_action(f"list_files({path})", result, True)
        print(f"          -> {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}")
        return result
    except FileNotFoundError:
        result = f"Error: Path '{path}' does not exist"
        sandbox.log_action(f"list_files({path})", result, False)
        print(f"          -> {result}")
        return result


@tool
def create_file(filename: str, content: str) -> str:
    """Create a file in the sandbox with the given content.

    Args:
        filename: Name of the file to create (relative to sandbox root)
        content: Content to write to the file
    """
    print(f"          create_file(filename={filename!r}, {len(content)} bytes)")
    try:
        target = _resolve_in_sandbox(filename)
    except ValueError:
        result = f"Error: '{filename}' is outside the sandbox"
        sandbox.log_action(f"create_file({filename})", result, False)
        print(f"          -> {result}")
        return result
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True) if os.path.dirname(filename) else None
        with open(target, 'w') as f:
            f.write(content)
        result = f"Created: {filename} ({len(content)} bytes)"
        sandbox.log_action(f"create_file({filename})", result, True)
        print(f"          -> {result}")
        return result
    except Exception as e:
        result = f"Error creating {filename}: {e}"
        sandbox.log_action(f"create_file({filename})", result, False)
        print(f"          -> {result}")
        return result


@tool
def read_file(filename: str) -> str:
    """Read a file from the sandbox.

    Args:
        filename: Name of the file to read (relative to sandbox root)
    """
    print(f"          read_file(filename={filename!r})")
    try:
        target = _resolve_in_sandbox(filename)
    except ValueError:
        result = f"Error: '{filename}' is outside the sandbox"
        sandbox.log_action(f"read_file({filename})", result, False)
        print(f"          -> {result}")
        return result
    try:
        with open(target) as f:
            content = f.read()
        result = content[:2000]  # Limit output size
        sandbox.log_action(f"read_file({filename})", f"Read {len(content)} bytes", True)
        print(f"          -> read {len(content)} bytes")
        return result
    except FileNotFoundError:
        result = f"Error: File '{filename}' not found"
        sandbox.log_action(f"read_file({filename})", result, False)
        print(f"          -> {result}")
        return result


@tool
def delete_file(filename: str) -> str:
    """Delete a file from the sandbox.

    Args:
        filename: Name of the file to delete (relative to sandbox root)
    """
    print(f"          delete_file(filename={filename!r})")
    try:
        target = _resolve_in_sandbox(filename)
    except ValueError:
        result = f"Error: '{filename}' is outside the sandbox"
        sandbox.log_action(f"delete_file({filename})", result, False)
        print(f"          -> {result}")
        return result
    try:
        os.remove(target)
        result = f"Deleted: {filename}"
        sandbox.log_action(f"delete_file({filename})", result, True)
        print(f"          -> {result}")
        return result
    except FileNotFoundError:
        result = f"Error: File '{filename}' not found"
        sandbox.log_action(f"delete_file({filename})", result, False)
        print(f"          -> {result}")
        return result


# ─── Why this lab runs a subprocess at all ────────────────────────────────────
#
# The point of the sandbox agent is that the model acts on a REAL environment
# and reads back real consequences (the act -> observe -> adapt loop). A real
# `wc -l` on a real file is the observation. Faking the commands in Python would
# teach a different, weaker lesson: the agent would be acting on our emulation
# of the world, not the world. So `run_command` has to execute something, and
# `subprocess.run` is the only way to do that from Python.
#
# Security scanners (Semgrep `dangerous-subprocess-use-audit`, Bandit B603) flag
# every `subprocess.run` whose argv is not a string literal, because they cannot
# see data-flow controls. That finding is expected here and is accepted, on the
# basis of the controls below. Each one is enforced in code, not by the model:
#
#   1. argv[0] is never the model's string. The model's command NAME is used
#      only as a lookup key into ALLOWED_COMMANDS; what actually runs is the
#      absolute path resolved at import time. Unknown names are rejected.
#   2. The allowlist is read-only text utilities. `find` is deliberately absent
#      (its -exec/-delete would turn it into an escape hatch); so are `rm`,
#      `bash`, `python`, `sh`, `xargs`, and anything that writes.
#   3. `shell=False` with a token list from `shlex.split`: there is no shell, so
#      `;`, `|`, `$(...)`, backticks and redirects are inert bytes, not syntax.
#   4. Every argument is confined to the sandbox: no absolute paths, no `..`
#      anywhere, and no `/` inside an option (closes `sort -o/tmp/x`,
#      `grep -f/etc/passwd`).
#   5. The process runs in a fresh `mkdtemp` cwd, with stdin closed, a 5s
#      timeout, truncated output, and a scrubbed environment (PATH only) so
#      the child inherits none of the user's secrets or AWS credentials.
#
# If you change the allowlist, keep it to commands that only READ their
# arguments. That is the whole security model of this lab.
#
# Why not remove the call, or replace it with something else?
#
#   - Remove it: the other tools (list/create/read/delete) are plain Python
#     file operations whose outcome the model can fully predict, so there is
#     nothing to observe and adapt to. `run_command` is the one action with
#     real, externally determined results (exit codes, stderr, formatting),
#     which is what APG step 5 ("executes simulated actions") and step 6
#     ("learns and adapts") require. Challenge 2 (Data Processing Pipeline)
#     is built entirely on it.
#   - Re-implement the eight utilities in Python: clears the scanner but the
#     agent then acts on our emulation, whose errors and edge cases differ
#     from the real tools, so "learn from failure" trains on fake failures.
#     Re-implementing grep/sort option parsing is also a far larger bug and
#     attack surface than one guarded subprocess call.
#   - Use `strands_tools.shell`: it runs `/bin/sh -c <command>` (shell=True)
#     and relies on an interactive consent prompt as its only guard. Moving
#     the call into a dependency hides it from the scanner while removing
#     the allowlist and the argument confinement. Strictly worse.
#   - Run it in a container or Amazon Bedrock AgentCore Code Interpreter:
#     the right answer for production, and the README points there. It is
#     out of scope for a local learning lab because it adds a daemon or an
#     AWS resource between the reader and a `python sandbox_agent.py`.
# ──────────────────────────────────────────────────────────────────────────────

_ALLOWED_NAMES = ("echo", "cat", "wc", "sort", "head", "tail", "grep", "ls")

# name -> absolute path, resolved once from the launching user's PATH. A command
# that is not installed is simply absent from the allowlist (fail closed).
ALLOWED_COMMANDS: dict[str, str] = {
    name: path
    for name in _ALLOWED_NAMES
    if (path := shutil.which(name)) is not None
}

# Minimal environment for the child process. Nothing from the parent (AWS
# credentials, tokens, HOME) leaks into the sandboxed command.
_SANDBOX_ENV = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}


@tool
def run_command(command: str) -> str:
    """Execute a safe command in the sandbox environment.

    Args:
        command: Command to run (limited to safe operations)
    """
    print(f"          run_command(command={command!r})")

    # Use shlex.split + shell=False so the allowlist actually constrains what runs.
    # Trade-off: shell features (pipes, redirects, substitution) won't work — the
    # agent runs one command at a time and reasons over the output instead, which
    # fits the closed-loop pattern this lab demonstrates.
    try:
        tokens = shlex.split(command)
    except ValueError as e:
        result = f"Error parsing command: {e}"
        sandbox.log_action(f"run_command({command})", result, False)
        print(f"          -> {result}")
        return result

    cmd_name = tokens[0] if tokens else ""

    # Control 1: the model's string is only a lookup key. The executable that
    # runs is the pre-resolved absolute path from ALLOWED_COMMANDS.
    executable = ALLOWED_COMMANDS.get(cmd_name)
    if executable is None:
        result = f"Error: Command '{cmd_name}' not allowed. Allowed: {', '.join(ALLOWED_COMMANDS)}"
        sandbox.log_action(f"run_command({command})", result, False)
        print(f"          -> {result}")
        return result

    # Confine arguments to the sandbox. Even allowlisted read commands could
    # otherwise reach outside via absolute paths (e.g. `cat /etc/passwd`) or
    # `..` traversal, since cwd only affects relative-path resolution. Paths can
    # also hide inside option values (`sort -o/tmp/x`, `sort --output=/tmp/x`,
    # `grep -f/etc/passwd`), so any option token carrying a `/` is rejected too,
    # and `..` is rejected anywhere in a token, not just as a path segment.
    for arg in tokens[1:]:
        if os.path.isabs(arg) or ".." in arg or (arg.startswith("-") and "/" in arg):
            result = f"Error: Argument '{arg}' escapes the sandbox (no absolute paths, '..', or paths in options)."
            sandbox.log_action(f"run_command({command})", result, False)
            print(f"          -> {result}")
            return result

    try:
        # Accepted scanner finding: see "Why this lab runs a subprocess at all"
        # above for the controls that make this call safe. argv[0] is the
        # resolved allowlisted path, not model input; shell=False; args are
        # sandbox-confined; stdin closed; scrubbed env; 5s timeout; temp cwd.
        proc = subprocess.run(  # nosec B603  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
            [executable, *tokens[1:]],
            capture_output=True, text=True,
            stdin=subprocess.DEVNULL, timeout=5,
            cwd=sandbox.root, env=_SANDBOX_ENV,
        )
        output = proc.stdout[:2000] if proc.stdout else ""
        error = proc.stderr[:500] if proc.stderr else ""
        result = output if output else error if error else "(no output)"
        sandbox.log_action(f"run_command({command})", f"exit={proc.returncode}", proc.returncode == 0)
        print(f"          -> exit={proc.returncode}")
        return result
    except subprocess.TimeoutExpired:
        result = "Error: Command timed out (5s limit)"
        sandbox.log_action(f"run_command({command})", result, False)
        print(f"          -> {result}")
        return result


@tool
def get_action_history() -> str:
    """Review the history of actions taken in this sandbox session.

    Returns a summary of recent actions and their outcomes.
    """
    print("          get_action_history()")
    summary = sandbox.get_history_summary()
    print(f"          -> {len(sandbox.history)} action(s) in history")
    return summary


SYSTEM_PROMPT = """You are an agent operating within a sandboxed file system environment.

You can explore, create, modify, and delete files within your sandbox.
You can also run safe shell commands to process data.

Your environment:
- A temporary directory that exists only for this session
- Tools: list_files, create_file, read_file, delete_file, run_command
- You can review your action history with get_action_history

When given a task:
1. Observe the current state of the environment (list files, read contents)
2. Plan your approach based on what you find
3. Execute actions to accomplish the goal
4. Verify the results by observing the new state
5. Adapt if something didn't work as expected

You learn from your actions — if a command fails, try a different approach.
Always verify your work by checking the results after making changes."""


def main():
    """Run the sandbox agent with predefined challenges."""
    print("Sandbox Agent - Simulated CLI Environment")
    print("=" * 40)
    print(f"Sandbox directory: {sandbox.root}")
    print("The agent operates in a safe sandbox to complete challenges.")
    print("Type 'quit' to exit\n")

    # Predefined challenges for the agent to complete
    challenges = {
        "1": {
            "name": "Project Scaffolding",
            "description": "Create a Python project structure with src/, tests/, docs/ directories, "
                           "a README.md with project description, a requirements.txt with 3 dependencies, "
                           "and a main.py in src/ that imports from a utils.py module.",
            "prompt": "Create a complete Python project structure in this sandbox:\n"
                      "- src/ directory with main.py and utils.py (main.py should import from utils)\n"
                      "- tests/ directory with a test_utils.py file\n"
                      "- docs/ directory with a README.md\n"
                      "- A requirements.txt in the root with at least 3 dependencies\n"
                      "- A .gitignore file with common Python ignores\n\n"
                      "After creating everything, verify the structure by listing all files "
                      "and reading each one to confirm the contents are correct.",
        },
        "2": {
            "name": "Data Processing Pipeline",
            "description": "Create CSV data files, then use shell commands to sort, filter, "
                           "and extract insights from the data.",
            "prompt": "Build a data processing pipeline in this sandbox:\n"
                      "1. Create a file called 'sales.csv' with headers: date,product,quantity,price\n"
                      "   Add at least 10 rows of realistic sales data\n"
                      "2. Use shell commands to:\n"
                      "   - Count the total number of records (wc)\n"
                      "   - Sort by quantity descending (sort)\n"
                      "   - Find all rows where quantity > 5 (grep)\n"
                      "   - Show only the top 3 highest quantity sales (sort + head)\n"
                      "3. Create a 'report.txt' summarizing your findings\n"
                      "4. Verify everything by reading the report and checking your action history",
        },
        "3": {
            "name": "Bug Hunt",
            "description": "The sandbox contains files with intentional errors. "
                           "Find and fix them all.",
            "prompt": "I've set up some files with bugs. Find and fix them all:\n"
                      "1. First, list all files to see what's in the sandbox\n"
                      "2. Read each file and identify the problems\n"
                      "3. Fix each issue by recreating the file with corrected content\n"
                      "4. Verify your fixes by reading the corrected files\n"
                      "5. Summarize what you found and fixed",
        },
    }

    # For challenge 3, pre-populate the sandbox with buggy files
    def setup_bug_hunt():
        """Create files with intentional errors for the agent to find."""
        bugs = {
            "config.json": '{"database": "postgres", "port": 5432, "host": "localhost", "password": }',
            "deploy.sh": '#!/bin/bash\necho "Deploying..."\nrm -rf /\necho "Done"',
            "app.py": 'def calculate_average(numbers):\n    total = sum(numbers)\n    return total / 0\n',
        }
        for filename, content in bugs.items():
            filepath = os.path.join(sandbox.root, filename)
            with open(filepath, 'w') as f:
                f.write(content)

    # Streaming handler so the user can see the agent reason about the
    # sandbox state alongside the tool calls.
    stream_handler = StreamingCallbackHandler()
    agent = Agent(
        model=get_model(),
        system_prompt=SYSTEM_PROMPT,
        tools=[list_files, create_file, read_file, delete_file, run_command, get_action_history],
        callback_handler=stream_handler,
    )

    print("Choose a challenge (pick a number, or type your own):")
    print("  1. Project Scaffolding — Create a complete Python project structure")
    print("  2. Data Processing — Build a CSV pipeline with shell commands")
    print("  3. Bug Hunt — Find and fix errors in pre-existing files\n")

    try:
        while True:
            user_input = get_multiline_input("You: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            # Check if user selected a challenge number
            if user_input in challenges:
                challenge = challenges[user_input]
                print(f"\n    Challenge: {challenge['name']}")
                print(f"    {challenge['description']}\n")

                if user_input == "3":
                    setup_bug_hunt()
                    print("    (Buggy files have been placed in the sandbox)\n")

                user_input = challenge["prompt"]
            # Anything that is not a challenge number is sent to the agent
            # as-is. The sandbox tools (not the prompt) enforce the limits.

            try:
                stream_handler.reset()
                print("\nAgent: ", end="", flush=True)
                start_time = time.time()
                agent(user_input)
                elapsed = time.time() - start_time
                print(f"\n({elapsed:.1f}s)\n")
            except Exception as e:
                print(f"\nError: {e}\n")
    finally:
        sandbox.cleanup()
        print(f"Sandbox cleaned up: {sandbox.root}")


if __name__ == "__main__":
    main()
