"""
Day 8 — Exercise 1: Manual Reflection Loop (Writer + Critic, no frameworks)
===========================================================================

Goal
----
Rebuild your Day 7 Exercise 1 Writer+Critic reflection team using only
`anthropic` + stdlib. No AutoGen. No LangChain. Just API calls.

Why
---
On Day 7, RoundRobinGroupChat hid the turn-taking logic. Today you write it
yourself and see that "multi-agent reflection" is really just two independent
LLM contexts with orchestrated message-passing between them.

Success criteria
----------------
- Produces an approved haiku (or other short text) in ≤ 6 turns.
- Prints a per-turn log: who spoke, tokens in/out, running total.
- Terminates cleanly on "APPROVE" OR max turns.
- ZERO hidden LLM calls — the count at the end matches the number of turns.

Estimated lines to fill in: ~60 across 5 TODO blocks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

import anthropic

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
# Adjust these to whatever models you have access to. On Day 7 you used Haiku
# for the cheap/fast role (Writer) and Sonnet for the judgment role (Critic).
# Keep that pattern — it will make the cost comparison in Exercise 3 clean.
CHEAP_MODEL = "claude-haiku-4-5"        # for the Writer — fast drafting
SMART_MODEL = "claude-sonnet-4-6"       # for the Critic — stronger judgment

MAX_TURNS = 10           # hard safety net; Day 7 lesson: always have one
APPROVAL_TOKEN = "APPROVE"

# The task. Keep it simple; reflection quality is easier to see on short outputs.
TASK = (
    "Write a haiku about autumn in Warsaw. "
    "Traditional 5-7-5 syllable form. Evocative imagery."
)

# System prompts — copied from Day 7 with minor adjustments for the manual context.
# Notice: we do NOT need to tell these agents "you are part of a multi-agent team"
# the way you had to on Day 7. Why? Because in this manual setup, each agent only
# ever sees messages addressed directly to it — there's no shared message bus
# creating identity confusion. One less thing to contort the system prompt for.
WRITER_SYSTEM = (
    "You are a Writer. Produce a haiku when given a topic. "
    "When given feedback, revise and return ONLY the new haiku — no preamble, "
    "no explanation. Just the three lines."
)

CRITIC_SYSTEM = (
    "You are a Critic of haiku. Given a haiku, either:\n"
    "  (a) reply with concise, specific feedback (1-3 sentences) if it needs work, or\n"
    "  (b) reply with exactly the single word 'APPROVE' if it meets the bar.\n"
    "Criteria: 5-7-5 syllables, concrete imagery, seasonal grounding, emotional resonance.\n"
    "Do not compliment or pad. Be brief."
)


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------
@dataclass
class TurnLog:
    """One row of our token-usage table."""
    turn: int
    speaker: str
    model: str
    input_tokens: int
    output_tokens: int
    preview: str  # first ~60 chars of the response, for the log


@dataclass
class Agent:
    """
    A minimal agent. Holds identity, model choice, and its OWN message history.

    Key insight: each agent has a SEPARATE `messages` list. The outside world
    (our orchestrator loop) is responsible for shuttling text between them.
    From the agent's own perspective, it's just a single-turn conversation
    where the "user" keeps sending it things to respond to.
    """
    name: str
    system: str
    model: str
    messages: list[dict] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Core helper: one agent turn
# ----------------------------------------------------------------------------
def agent_turn(
    client: anthropic.Anthropic,
    agent: Agent,
    incoming_text: str,
) -> tuple[str, TurnLog]:
    """
    Run one turn for an agent: append the incoming text as a user message,
    call the API, append the response to the agent's history, return (text, log).

    Returns
    -------
    (response_text, turn_log)
        response_text: the text the agent produced this turn.
        turn_log:      a TurnLog capturing who/tokens/preview for our table.
    """
    # ------------------------------------------------------------------------
    # TODO 1 — Append the incoming text as a new "user" message on `agent.messages`.
    #
    # Hint: the Anthropic messages format is a list of dicts like
    #   {"role": "user" | "assistant", "content": <string or list of blocks>}
    # For plain text input, content can just be a string.
    # ------------------------------------------------------------------------
    agent.messages.append({"role": "user", "content": incoming_text})

    # ------------------------------------------------------------------------
    # TODO 2 — Call the Anthropic API.
    #
    # Required kwargs: model, max_tokens, system, messages.
    # Recommended max_tokens for haiku work: 300 (plenty, and caps bill if the
    # model ignores instructions).
    #
    # The SDK call shape you want is:
    #   response = client.messages.create(model=..., max_tokens=..., system=..., messages=...)
    # Read the Anthropic docs reading #1 if you need to confirm the shape.
    # ------------------------------------------------------------------------
    response = client.messages.create(
        model = agent.model,
        max_tokens = 300,
        system = agent.system,
        messages = agent.messages
    )


    # ------------------------------------------------------------------------
    # TODO 3 — Extract the text from the response.
    #
    # Hint: `response.content` is a LIST of blocks (not a string!). For a no-
    # tool-use response, there is typically one block with `.type == "text"`
    # and `.text` containing the string. Grab that.
    #
    # If you want to be defensive: join all text blocks with "\n". For this
    # exercise a single block is fine.
    # ------------------------------------------------------------------------
    text = "\n".join(block.text for block in response.content if block.type == "text")


    # ------------------------------------------------------------------------
    # TODO 4 — Append the assistant response to agent.messages so the NEXT
    # call preserves the agent's view of the conversation.
    #
    # You have two choices for the content:
    #   (a) the raw `response.content` list of blocks (SDK-native, works fine)
    #   (b) the extracted text string (simpler, also works fine for text-only)
    # Pick (b) for clarity here. (a) matters more once we add tools in Exercise 2.)
    # ------------------------------------------------------------------------
    agent.messages.append({"role":"assistant", "content":text})

    # Build the log entry. Note: response.usage has .input_tokens and .output_tokens.
    log = TurnLog(
        turn=-1,  # the orchestrator will fill this in
        speaker=agent.name,
        model=agent.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        preview=text.replace("\n", " / ")[:60],
    )
    return text, log


# ----------------------------------------------------------------------------
# Orchestrator: the reflection loop
# ----------------------------------------------------------------------------
def run_reflection(
    client: anthropic.Anthropic,
    writer: Agent,
    critic: Agent,
    task: str,
    max_turns: int = MAX_TURNS,
) -> tuple[str, Literal["approved", "max_turns"], list[TurnLog]]:
    """
    Run the reflection loop.

    Flow:
      Turn 1: Writer receives the task, produces draft 1.
      Turn 2: Critic receives the draft, produces feedback OR "APPROVE".
      Turn 3: Writer receives feedback, produces draft 2.
      Turn 4: Critic again.
      ...

    Returns
    -------
    (final_text, stop_reason, turn_logs)
    """
    logs: list[TurnLog] = []
    turn = 0

    # --------------------------------------------------------------------
    # Turn 1 — Writer sees the task.
    # This is always the same, so not a TODO.
    # --------------------------------------------------------------------
    turn += 1
    draft, log = agent_turn(client, writer, task)
    log.turn = turn
    logs.append(log)
    print(f"\n[Turn {turn}] {writer.name} ({writer.model}):\n{draft}\n")

    # --------------------------------------------------------------------
    # TODO 5 — The reflection loop.
    #
    # Pattern:
    #   while turn < max_turns:
    #       turn += 1
    #       1. Critic turn — pass `draft` to the critic. Get `feedback`.
    #       2. Log it. Print it.
    #       3. If APPROVAL_TOKEN is in feedback (case-insensitive is safest
    #          since models sometimes say "Approve." or "APPROVE!"), the
    #          draft is final. Return (draft, "approved", logs).
    #       4. If we've hit max_turns already (turn >= max_turns), stop.
    #          Return (draft, "max_turns", logs).
    #       5. Writer turn — pass `feedback` to the writer. Get new `draft`.
    #       6. turn += 1. Log it. Print it.
    #
    # Edge-case hint: remember the turn counter — the critic turn AND the
    # writer turn both increment it. If you exit after the critic's approval
    # without incrementing for a writer that never spoke, your total = odd.
    # That's correct and expected.
    #
    # Safety: if the loop exits via max_turns, that's a documented outcome,
    # not a crash. Return it cleanly so Exercise 3 can measure it.
    # --------------------------------------------------------------------
    # --- your code starts here ---
    while turn < max_turns: 
        turn += 1
        feedback, log = agent_turn(client,critic,draft)
        log.turn = turn
        logs.append(log)
        print(f"\n[Turn {turn}] {critic.name} ({critic.model}):\n{feedback}\n")
        
        if feedback.strip().upper().startswith(APPROVAL_TOKEN):
            return(draft, "approved",logs)
        if turn >= max_turns:
            return (draft, "max_turns", logs)
        
        turn +=1
        draft, log = agent_turn(client, writer, feedback)
        log.turn = turn
        logs.append(log)
        print(f"\n[Turn {turn}] {writer.name} ({writer.model}):\n{draft}\n")
    # --- your code ends here ---

    # If we exit the loop naturally (shouldn't happen with the logic above,
    # but belt-and-braces):
    return draft, "max_turns", logs


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def print_usage_table(logs: list[TurnLog]) -> None:
    """Print a per-turn token-usage table + totals. Useful for Exercise 3."""
    print("\n" + "=" * 78)
    print(f"{'Turn':<5} {'Speaker':<8} {'Model':<22} {'In':>7} {'Out':>7}  Preview")
    print("-" * 78)
    total_in = total_out = 0
    for log in logs:
        print(
            f"{log.turn:<5} {log.speaker:<8} {log.model:<22} "
            f"{log.input_tokens:>7} {log.output_tokens:>7}  {log.preview}"
        )
        total_in += log.input_tokens
        total_out += log.output_tokens
    print("-" * 78)
    print(f"{'TOTAL':<5} {'':<8} {'':<22} {total_in:>7} {total_out:>7}")
    print(f"LLM calls: {len(logs)}  (no hidden calls — this is the real count)")
    print("=" * 78)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    # Before implementing: run this file as-is to see the stubs fail loudly.
    # That's the "exploration harness" — you can print-inspect `response.usage`,
    # `response.content`, `response.stop_reason` etc. before filling in the logic.
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise RuntimeError("Set ANTHROPIC_API_KEY env var first.")

    client = anthropic.Anthropic()

    writer = Agent(name="Writer", system=WRITER_SYSTEM, model=CHEAP_MODEL)
    critic = Agent(name="Critic", system=CRITIC_SYSTEM, model=SMART_MODEL)

    final_text, stop_reason, logs = run_reflection(
        client, writer, critic, TASK, max_turns=MAX_TURNS
    )

    print("\n" + "#" * 78)
    print(f"# Final haiku (stop_reason={stop_reason}):")
    print("#" * 78)
    print(final_text)
    print_usage_table(logs)


if __name__ == "__main__":
    main()


# ============================================================================
# Post-exercise reflection questions (answer in your notes)
# ============================================================================
# 1. Compare your total LLM-call count to your Day 7 AutoGen Exercise 1 run.
#    Which was higher? Why?
#
# 2. In the Day 7 Ex 1 summary you noted: "RoundRobin can't skip turns — the
#    Editor can't conditionally act, it always speaks." How would you add an
#    Editor to YOUR manual loop that only acts when needed? Is the code you'd
#    need simpler or more complex than the AutoGen version?
#
# 3. The Day 7 system messages had to say "you are part of a multi-agent team,
#    other messages come from teammate agents, NOT from a human user". Did you
#    need that disclaimer in this version? Why/why not?
#
# 4. Hypothetical: the API returns a 429 rate limit on turn 3. In the AutoGen
#    version, what would happen? In your manual version, what would happen?
#    Which behavior do you want?
