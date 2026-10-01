"""
Strategy Learner - Agent That Discovers Better Strategies Through Iteration

An agent that plays a number-guessing game 3 times. It starts with no strategy,
reflects on its performance after each round, and develops its own approach
to improve. The user watches the agent reason, fail, reflect, and adapt in real time.

Prerequisites:
    pip install -r requirements.txt

Learning objectives:
- Watch an agent develop strategy through self-reflection
- See how failure drives reasoning improvement
- Understand iterative learning without explicit instruction
- Observe emergent optimization behavior
"""

import time
import random
from shared.input_utils import get_multiline_input
from shared.streaming import StreamingCallbackHandler
from strands import Agent, tool
from shared.model import get_model

# This lab uses the shared model like every other lab. A smaller model (Haiku)
# makes the learning progression more visible because it is less likely to
# apply the optimal strategy on the first try. To try that, set
# STRANDS_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0 in the root .env
# (verify the current id with: aws bedrock list-inference-profiles).
MODEL = get_model()


# --- Simulated Environment: Number Guessing Game ---

class GuessingGame:
    """A game environment where the agent must find a hidden number."""

    def __init__(self, min_val: int = 1, max_val: int = 100):
        self.min_val = min_val
        self.max_val = max_val
        self.target = random.randint(min_val, max_val)
        self.guesses = []
        self.solved = False

    def make_guess(self, number: int) -> str:
        """Process a guess and return feedback."""
        self.guesses.append(number)

        if number == self.target:
            self.solved = True
            return f"CORRECT! The number was {self.target}. Solved in {len(self.guesses)} guesses."
        elif number < self.target:
            return f"TOO LOW. {number} is less than the target."
        else:
            return f"TOO HIGH. {number} is greater than the target."


# Global game instance
game = GuessingGame()


@tool
def guess_number(number: int) -> str:
    """Make a guess in the number guessing game.

    Args:
        number: Your guess (integer between 1 and 100)
    """
    if game.solved:
        return "Game already solved!"
    result = game.make_guess(number)
    status = "✓" if game.solved else "→"
    print(f"      {status} Guess #{len(game.guesses)}: {number} — {result}")
    return result


# --- The Learning Agent ---

SYSTEM_PROMPT = """You are playing a number guessing game. A hidden number between 1 and 100 has been chosen.

Rules:
- You guess numbers using the guess_number tool
- After each guess you receive feedback: TOO HIGH, TOO LOW, or CORRECT
- Your goal is to find the number

For this round, pick numbers randomly. Do NOT use any systematic strategy like
binary search or divide-and-conquer. Just pick numbers that feel right to you —
lucky numbers, favorites, random choices. Use the feedback to avoid obviously
wrong guesses, but don't overthink it.

Think out loud before each guess, in English."""


def run_round(round_num: int, system_prompt: str) -> tuple:
    """Run a single round. Returns (num_guesses, target, guess_history).

    The game agent's reasoning streams to stdout, so the viewer can watch
    each guess justified before it's made.
    """
    global game
    game = GuessingGame()

    handler = StreamingCallbackHandler()
    agent = Agent(
        model=MODEL,
        system_prompt=system_prompt,
        tools=[guess_number],
        callback_handler=handler,
    )

    agent("Start guessing. The number is between 1 and 100.")

    return len(game.guesses), game.target, list(game.guesses)


def _stream_reflection(reflection_agent: Agent, handler: StreamingCallbackHandler, prompt: str) -> str:
    """Stream a reflection step and return the assembled text."""
    handler.reset()
    response = reflection_agent(prompt)
    return str(response)


def main():
    """Run the strategy learner across 3 rounds with reflection."""
    print("Strategy Learner - Watch an Agent Develop Strategy")
    print("=" * 40)
    print("The agent plays 3 rounds of a number-guessing game.")
    print("It starts with NO strategy and must figure out how to improve.")
    print("Watch it reason, fail, reflect, and adapt in real time.")
    print("\nType 'play' to start | 'quit' to exit\n")

    while True:
        user_input = get_multiline_input("You: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if user_input.lower() not in ["play", "start", "go", "run"]:
            print("Type 'play' to start a 3-round learning session or 'quit' to exit.\n")
            continue

        try:
            round_results = []

            # ═══════════════════════════════════════
            # ROUND 1: No strategy given
            # ═══════════════════════════════════════
            print(f"\n{'═' * 40}")
            print("  ROUND 1 — No strategy (baseline)")
            print(f"{'═' * 40}\n")

            start = time.time()
            r1_guesses, r1_target, r1_history = run_round(1, SYSTEM_PROMPT)
            elapsed = time.time() - start
            round_results.append({"guesses": r1_guesses, "target": r1_target, "history": r1_history})
            print(f"\n    Round 1 result: {r1_guesses} guesses ({elapsed:.1f}s)")

            # ═══════════════════════════════════════
            # REFLECTION 1: Agent analyzes its performance
            # ═══════════════════════════════════════
            print(f"\n{'─' * 40}")
            print("  REFLECTION: Agent analyzing Round 1...")
            print(f"{'─' * 40}\n")

            # Reflection streams live — this is the lab's main payoff. The
            # viewer watches strategy emerge token by token, instead of seeing
            # a finished analysis appear all at once.
            reflection_handler = StreamingCallbackHandler()
            reflection_agent = Agent(
                model=MODEL,
                system_prompt="""You are reflecting on your performance in a number guessing game.
You will be given your results from the previous round. Analyze what happened:
- How many guesses did you take?
- Was there a pattern in your guessing?
- What information did you waste or not use effectively?
- What strategy could you try next time to use fewer guesses?

Think out loud, in English. Reason about what went wrong and what you'd do differently.
Be specific about your next strategy.""",
                callback_handler=reflection_handler,
            )

            reflection_1 = _stream_reflection(
                reflection_agent,
                reflection_handler,
                f"Round 1 results:\n"
                f"- Target number: {r1_target}\n"
                f"- Your guesses (in order): {r1_history}\n"
                f"- Total guesses: {r1_guesses}\n\n"
                f"Analyze your performance. What strategy will you try in Round 2?",
            )

            # ═══════════════════════════════════════
            # ROUND 2: Agent applies its own improved strategy
            # ═══════════════════════════════════════
            print(f"\n{'═' * 40}")
            print("  ROUND 2 — Agent's self-developed strategy")
            print(f"{'═' * 40}\n")

            round2_prompt = f"""You are playing a number guessing game. A hidden number between 1 and 100 has been chosen.

Rules:
- You guess numbers using the guess_number tool
- After each guess you receive feedback: TOO HIGH, TOO LOW, or CORRECT
- Your goal is to find the number in as few guesses as possible

Last round, you took {r1_guesses} guesses. Here's what you learned:

{reflection_1}

Apply your improved strategy now. Think out loud before each guess, in English. Start guessing."""

            start = time.time()
            r2_guesses, r2_target, r2_history = run_round(2, round2_prompt)
            elapsed = time.time() - start
            round_results.append({"guesses": r2_guesses, "target": r2_target, "history": r2_history})
            print(f"\n    Round 2 result: {r2_guesses} guesses ({elapsed:.1f}s)")

            # ═══════════════════════════════════════
            # REFLECTION 2: Agent analyzes both rounds
            # ═══════════════════════════════════════
            print(f"\n{'─' * 40}")
            print("  REFLECTION: Agent analyzing Rounds 1 & 2...")
            print(f"{'─' * 40}\n")

            reflection_2 = _stream_reflection(
                reflection_agent,
                reflection_handler,
                f"Round 1 results:\n"
                f"- Target: {r1_target}, Guesses: {r1_history}, Total: {r1_guesses}\n\n"
                f"Round 2 results:\n"
                f"- Target: {r2_target}, Guesses: {r2_history}, Total: {r2_guesses}\n\n"
                f"Your Round 2 strategy was based on this reasoning:\n{reflection_1[:500]}\n\n"
                f"Did your strategy improve? What worked? What didn't?\n"
                f"What is the BEST possible strategy you can think of for Round 3?",
            )

            # ═══════════════════════════════════════
            # ROUND 3: Agent applies its best strategy
            # ═══════════════════════════════════════
            print(f"\n{'═' * 40}")
            print("  ROUND 3 — Agent's optimized strategy")
            print(f"{'═' * 40}\n")

            round3_prompt = f"""You are playing a number guessing game. A hidden number between 1 and 100 has been chosen.

Rules:
- You guess numbers using the guess_number tool
- After each guess you receive feedback: TOO HIGH, TOO LOW, or CORRECT
- Your goal is to find the number in as few guesses as possible

You've played twice before:
- Round 1: {r1_guesses} guesses
- Round 2: {r2_guesses} guesses

Here's your analysis of what works best:

{reflection_2}

This is your final round. Apply your best strategy. Think out loud before each guess, in English. Start guessing."""

            start = time.time()
            r3_guesses, r3_target, r3_history = run_round(3, round3_prompt)
            elapsed = time.time() - start
            round_results.append({"guesses": r3_guesses, "target": r3_target, "history": r3_history})
            print(f"\n    Round 3 result: {r3_guesses} guesses ({elapsed:.1f}s)")

            # ═══════════════════════════════════════
            # FINAL SUMMARY
            # ═══════════════════════════════════════
            print(f"\n{'═' * 40}")
            print("  LEARNING SUMMARY")
            print(f"{'═' * 40}")
            print(f"    Round 1 (no strategy):    {round_results[0]['guesses']:>2} guesses")
            print(f"    Round 2 (self-improved):   {round_results[1]['guesses']:>2} guesses")
            print(f"    Round 3 (optimized):       {round_results[2]['guesses']:>2} guesses")
            print("    Theoretical optimal:       ≤7 guesses (binary search)")
            print()

            # Assess improvement
            if round_results[2]['guesses'] <= round_results[0]['guesses']:
                improvement = round_results[0]['guesses'] - round_results[2]['guesses']
                print(f"    📈 Agent improved by {improvement} guesses from Round 1 to Round 3")
            else:
                print("    📉 Agent did not improve (randomness in targets affects results)")

            if round_results[2]['guesses'] <= 7:
                print("    ✓ Agent discovered an optimal or near-optimal strategy!")
            print("\nType 'play' to run again or 'quit' to exit.\n")

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
