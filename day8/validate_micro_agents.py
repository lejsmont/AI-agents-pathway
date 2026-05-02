import os
import anthropic
from micro_agents import Agent, run_reflection, run_team, TOOL_SCHEMAS

def validate_reflection(client):
    """Smoke test the reflection pattern: stateful agents, no tools."""
    print("\n" + "=" * 60)
    print("VALIDATION 1: Reflection (stateful, no tools)")
    print("=" * 60)

    writer = Agent(
        name="Writer",
        system="Reply with ONE sentence. No preamble.",
        model="claude-haiku-4-5",
    )
    critic = Agent(
        name="Critic",
        system=(
            "You receive a sentence. If it has 5+ words, reply with exactly "
            "'APPROVE'. Otherwise reply with 'Make it longer.' and nothing else."
        ),
        model="claude-haiku-4-5",  # use Haiku for both — this is validation, not quality
    )

    final, stop_reason, logs = run_reflection(
        client, writer, critic,
        task="Write a sentence about coffee.",
        max_turns=4,
        approval_token="APPROVE",
    )

    # Assertions — these are what make this a TEST not just a demo
    assert stop_reason in {"approved", "max_turns"}, f"unexpected stop_reason: {stop_reason}"
    assert len(logs) >= 2, f"expected at least 2 LLM calls, got {len(logs)}"
    assert all(L.input_tokens > 0 for L in logs), "log rows missing input_tokens"
    assert all(L.caller in {"Writer", "Critic"} for L in logs), "unexpected caller in logs"

    print(f"\n✓ Reflection: stop_reason={stop_reason}, calls={len(logs)}, final={final!r}")


def validate_selector_with_tools(client):
    """Smoke test the selector pattern: stateless agents, one with a tool."""
    print("\n" + "=" * 60)
    print("VALIDATION 2: Selector + tools (stateless, tool loop)")
    print("=" * 60)

    # Use just one tool to keep the test minimal
    calc_schema = next(s for s in TOOL_SCHEMAS if s["name"] == "calculate")

    planner = Agent(
        name="PLANNER",
        system=(
            "You coordinate. Ask the MATHER to compute 7 * 8. "
            "After it reports a result, say exactly: TERMINATE on the last line."
        ),
        model="claude-haiku-4-5",
        description="Coordinates and synthesizes. No tools.",
        stateful=False,
    )
    mather = Agent(
        name="MATHER",
        system="Use the calculate tool when asked. Report the numeric result clearly.",
        model="claude-haiku-4-5",
        description="Performs arithmetic via the calculate tool.",
        tools=[calc_schema],
        stateful=False,
    )

    final, stop_reason, logs = run_team(
        client, [planner, mather],
        task="Compute 7 * 8.",
        max_turns=6,
        terminate_token="TERMINATE",
    )

    # Path-coverage assertions
    callers = {L.caller for L in logs}
    assert "SELECTOR" in callers, "selector calls missing from logs — pick_next_speaker bug?"
    assert "PLANNER" in callers or "MATHER" in callers, "no agent calls logged"
    assert stop_reason == "terminated", f"expected 'terminated', got {stop_reason!r}"
    assert "56" in final, f"expected the answer 56 in final output, got: {final!r}"

    # Tool loop sanity: MATHER should have at least 2 entries (initial call + after-tool)
    mather_calls = [L for L in logs if L.caller == "MATHER"]
    assert len(mather_calls) >= 2, (
        f"expected MATHER to make ≥2 calls (tool-use + final), got {len(mather_calls)}"
    )

    print(f"\n✓ Selector: stop_reason={stop_reason}, calls={len(logs)}, "
          f"selector_calls={sum(1 for L in logs if L.caller == 'SELECTOR')}")


if __name__ == "__main__":
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise RuntimeError("Set ANTHROPIC_API_KEY first.")
    client = anthropic.Anthropic()
    validate_reflection(client)
    validate_selector_with_tools(client)
    print("\nAll validations passed.")