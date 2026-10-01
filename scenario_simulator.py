"""
Scenario Simulator - Goal-Oriented Agent in a Simulated World

An agent that pursues goals within a simulated environment, receiving
feedback after each action and adapting its strategy. Demonstrates
the closed-loop pattern: perceive → reason → act → observe → learn.

Prerequisites:
    pip install -r requirements.txt

Learning objectives:
- Understand closed-loop agent behavior in simulated environments
- See how agents adapt strategy based on environmental feedback
- Learn goal-oriented reasoning with state tracking
- Practice building simulation environments with reward signals
"""

import time
import random
from shared.model import get_model
from shared.input_utils import get_multiline_input
from shared.streaming import StreamingCallbackHandler
from strands import Agent, tool


# --- Simulated Environment: Server Troubleshooting ---

class ServerSimulation:
    """Simulates a server environment with issues to diagnose and fix."""

    def __init__(self):
        """Initialize with a random set of issues."""
        self.issues = self._generate_issues()
        self.fixed = set()
        self.observations = []
        self.actions_taken = 0

    def _generate_issues(self) -> dict:
        """Generate a random set of server issues."""
        possible_issues = {
            "high_cpu": {
                "symptom": "CPU usage at 95%",
                "cause": "Runaway process 'data_processor' consuming resources",
                "fix": "kill_process",
            },
            "disk_full": {
                "symptom": "Disk usage at 98%",
                "cause": "Log files in /var/log have grown to 50GB",
                "fix": "clear_logs",
            },
            "memory_leak": {
                "symptom": "Memory usage climbing steadily, now at 87%",
                "cause": "Application 'web_server' has a memory leak",
                "fix": "restart_service",
            },
            "network_timeout": {
                "symptom": "Intermittent connection timeouts to database",
                "cause": "DNS resolution failing for db.internal",
                "fix": "fix_dns",
            },
            "ssl_expired": {
                "symptom": "HTTPS connections failing with certificate error",
                "cause": "SSL certificate expired 2 days ago",
                "fix": "renew_cert",
            },
        }
        # Pick 2-3 random issues for this scenario
        num_issues = random.randint(2, 3)
        selected = random.sample(list(possible_issues.keys()), num_issues)
        return {k: possible_issues[k] for k in selected}

    def get_status(self) -> str:
        """Get current server status."""
        status_lines = ["SERVER STATUS:"]
        for issue_id, issue in self.issues.items():
            if issue_id in self.fixed:
                status_lines.append(f"  ✓ {issue['symptom']} — RESOLVED")
            else:
                status_lines.append(f"  ⚠ {issue['symptom']}")
        status_lines.append(f"\nIssues remaining: {len(self.issues) - len(self.fixed)}/{len(self.issues)}")
        return "\n".join(status_lines)

    def is_resolved(self) -> bool:
        """Check if all issues are fixed."""
        return len(self.fixed) == len(self.issues)


# Global simulation instance
sim = ServerSimulation()


# --- Tools for interacting with the simulation ---

@tool
def check_server_status() -> str:
    """Check the current status of the simulated server.

    Returns the current state of all monitored metrics.
    """
    sim.actions_taken += 1
    status = sim.get_status()
    print("    📊 Server Status:")
    for line in status.split("\n"):
        print(f"       {line}")
    return status


@tool
def diagnose_issue(symptom: str) -> str:
    """Investigate a specific symptom to find the root cause.

    Args:
        symptom: Description of the symptom to investigate
    """
    sim.actions_taken += 1
    symptom_lower = symptom.lower()

    for issue_id, issue in sim.issues.items():
        if issue_id in sim.fixed:
            continue
        # Match on keywords from the symptom
        keywords = issue["symptom"].lower().split()
        if any(kw in symptom_lower for kw in keywords if len(kw) > 3):
            result = f"DIAGNOSIS: {issue['cause']}\nRecommended action: {issue['fix']}"
            print(f"    🔍 {result}")
            return result

    result = "Could not identify the root cause. Try checking server status for current symptoms."
    print(f"    🔍 {result}")
    return result


@tool
def kill_process(process_name: str = "data_processor") -> str:
    """Kill a runaway process.

    Args:
        process_name: Name of the process to kill
    """
    sim.actions_taken += 1
    if "high_cpu" in sim.issues and "high_cpu" not in sim.fixed:
        sim.fixed.add("high_cpu")
        result = f"Process '{process_name}' killed. CPU usage dropping to normal levels."
        print(f"    ✓ {result}")
        return result
    result = f"Process '{process_name}' not found or not causing issues."
    print(f"    ✗ {result}")
    return result


@tool
def clear_logs(path: str = "/var/log") -> str:
    """Clear old log files to free disk space.

    Args:
        path: Log directory to clean
    """
    sim.actions_taken += 1
    if "disk_full" in sim.issues and "disk_full" not in sim.fixed:
        sim.fixed.add("disk_full")
        result = f"Cleared 48GB of old logs from {path}. Disk usage now at 12%."
        print(f"    ✓ {result}")
        return result
    result = f"No significant log files to clear in {path}."
    print(f"    ✗ {result}")
    return result


@tool
def restart_service(service_name: str = "web_server") -> str:
    """Restart a service to clear its state.

    Args:
        service_name: Name of the service to restart
    """
    sim.actions_taken += 1
    if "memory_leak" in sim.issues and "memory_leak" not in sim.fixed:
        sim.fixed.add("memory_leak")
        result = f"Service '{service_name}' restarted. Memory usage returned to normal."
        print(f"    ✓ {result}")
        return result
    result = f"Service '{service_name}' restarted but no improvement observed."
    print(f"    ✗ {result}")
    return result


@tool
def fix_dns(hostname: str = "db.internal") -> str:
    """Fix DNS resolution for a hostname.

    Args:
        hostname: The hostname to fix DNS for
    """
    sim.actions_taken += 1
    if "network_timeout" in sim.issues and "network_timeout" not in sim.fixed:
        sim.fixed.add("network_timeout")
        result = f"DNS entry for '{hostname}' updated. Connection timeouts resolved."
        print(f"    ✓ {result}")
        return result
    result = f"DNS for '{hostname}' appears to be working correctly."
    print(f"    ✗ {result}")
    return result


@tool
def renew_cert(domain: str = "server.example.com") -> str:
    """Renew an SSL certificate.

    Args:
        domain: Domain to renew the certificate for
    """
    sim.actions_taken += 1
    if "ssl_expired" in sim.issues and "ssl_expired" not in sim.fixed:
        sim.fixed.add("ssl_expired")
        result = f"SSL certificate renewed for '{domain}'. HTTPS connections restored."
        print(f"    ✓ {result}")
        return result
    result = f"Certificate for '{domain}' is already valid."
    print(f"    ✗ {result}")
    return result


SYSTEM_PROMPT = """You are a DevOps agent operating in a simulated server environment.

Your goal: Diagnose and fix ALL server issues as efficiently as possible.

Strategy:
1. First, check the server status to see what symptoms exist
2. For each symptom, diagnose the root cause
3. Apply the appropriate fix
4. Verify the fix worked by checking status again
5. Repeat until all issues are resolved

Available tools:
- check_server_status: See current server metrics and issues
- diagnose_issue: Investigate a symptom to find root cause
- kill_process: Kill a runaway process
- clear_logs: Clear old log files
- restart_service: Restart a service
- fix_dns: Fix DNS resolution
- renew_cert: Renew SSL certificate

Be systematic. Check status first, diagnose each issue, fix it, then verify.
Report your progress after each action."""


def main():
    """Run the scenario simulator."""
    global sim
    print("Scenario Simulator - Server Troubleshooting")
    print("=" * 40)
    print("A simulated server has multiple issues. The agent must")
    print("diagnose and fix them all using a closed-loop approach:")
    print("  observe → reason → act → verify → repeat")
    print("Type 'quit' to exit | 'reset' for a new scenario\n")

    print(f"Current scenario: {len(sim.issues)} issues to resolve\n")

    # Streaming handler so the user can watch the agent reason about each
    # symptom alongside the tool prints (📊 🔍 ✓ ✗).
    stream_handler = StreamingCallbackHandler()
    agent = Agent(
        model=get_model(),
        system_prompt=SYSTEM_PROMPT,
        tools=[check_server_status, diagnose_issue, kill_process,
               clear_logs, restart_service, fix_dns, renew_cert],
        callback_handler=stream_handler,
    )

    # Start with an initial prompt to kick off the simulation
    print("Example prompts to try:")
    print("  - Check the server and fix all issues")
    print("  - What's wrong with the server?")
    print("  - Diagnose and resolve all problems systematically\n")

    while True:
        user_input = get_multiline_input("You: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if user_input.lower() == "reset":
            sim = ServerSimulation()
            agent = Agent(
                model=get_model(),
                system_prompt=SYSTEM_PROMPT,
                tools=[check_server_status, diagnose_issue, kill_process,
                       clear_logs, restart_service, fix_dns, renew_cert],
                callback_handler=stream_handler,
            )
            print(f"New scenario generated: {len(sim.issues)} issues to resolve\n")
            continue

        if not user_input:
            continue

        try:
            stream_handler.reset()
            print("\nAgent: ", end="", flush=True)
            start_time = time.time()
            agent(user_input)
            elapsed = time.time() - start_time
            print(f"\n({elapsed:.1f}s)\n")

            # Check if all issues resolved
            if sim.is_resolved():
                print("=" * 40)
                print(f"🎉 ALL ISSUES RESOLVED in {sim.actions_taken} actions!")
                print("=" * 40)
                print("\nType 'reset' for a new scenario or 'quit' to exit.\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
