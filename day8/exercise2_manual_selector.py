"""
Day 8 — Exercise 2: Manual Selector Team (Planner + Researcher + Analyst, no frameworks)
========================================================================================

Goal
----
Rebuild your Day 7 Exercise 2 SelectorGroupChat team using only `anthropic` +
stdlib. You'll write:
  - a tool-calling loop that handles the `stop_reason == "tool_use"` case,
  - a selector function that picks the next speaker with an LLM call,
  - shared history formatting with speaker tags,
  - the "manual reflection" fix for the bare-number tool-summary bug you hit
    on Day 7 (this replaces AutoGen's broken-on-Anthropic reflect_on_tool_use).

Why this is the core exercise of Day 8
--------------------------------------
Everything frameworks claim to abstract — agents, tools, routing, reflection —
is in this file, in plain sight. Once you finish it, no agent framework will
feel mysterious again.

Success criteria
----------------
- Runs a research+analysis task to completion.
- Terminates cleanly on an explicit "TERMINATE" or at max_turns.
- You can POINT AT every LLM call and its cost — including the selector.
- No 400 errors, no identity confusion, no repetitive loops.

Estimated lines to fill in: ~120 across 8 TODO blocks.
Expected runtime: 45-90 seconds depending on model.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal
import string

import anthropic

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
CHEAP_MODEL = "claude-haiku-4-5"     # selector routing — fast, cheap
SMART_MODEL = "claude-sonnet-4-6"    # agents that think — Planner/Researcher/Analyst

MAX_TURNS = 12                        # safety net
TERMINATE_TOKEN = "TERMINATE"

TASK = (
    "Compare electric vehicle adoption between Poland and Germany for 2025. "
    "Give me adoption rates, recent trends, and a one-sentence conclusion on "
    "which country has stronger momentum."
)


# ============================================================================
# PART 1 — Tools
# ============================================================================
# On Day 7 you learned tools must be `async def` in AutoGen 0.4. Here we own
# the loop, so tools can be plain synchronous functions. Much simpler.
#
# Also recall the Day 7 degenerate runs: mock tools returning IDENTICAL data
# caused repetitive loops. We vary the responses below to avoid that.
# ----------------------------------------------------------------------------

def search_web(query: str) -> str:
    """Mock web search. Returns varied fake results based on the query."""
    # Variation prevents the "Researcher calls search_web 5 times with same
    # answer" degenerate loop you observed on Day 7.
    facts_by_keyword = {
        "poland": (
            "Poland EV adoption 2025: EV market share reached 6.8% of new car "
            "sales in H1 2025, up from 4.2% in 2024. Charging infrastructure "
            "grew 34% YoY. Incentive program 'Mój Elektryk' extended through 2027."
        ),
        "germany": (
            "Germany EV adoption 2025: EV market share 21.4% of new car sales "
            "YTD, a drop from 25.1% in 2024 after subsidy cuts. Total EV stock "
            "crossed 2M vehicles. Fast-charging network densest in EU."
        ),
        "europe": (
            "EU-wide EV share Q1-Q2 2025: 17.8% of new registrations, steady "
            "vs. 2024. Eastern Europe growing faster off smaller base."
        ),
    }
    q = query.lower()
    for kw, txt in facts_by_keyword.items():
        if kw in q:
            return txt
    return f"No strong match for '{query}'. Try queries with country names."


def lookup_documentation(topic: str) -> str:
    """Mock doc lookup. Returns structured stats."""
    stats = {
        "poland_ev_stats_2025": {"market_share_pct": 6.8, "total_stock": 98000, "yoy_growth_pct": 42.0},
        "germany_ev_stats_2025": {"market_share_pct": 21.4, "total_stock": 2100000, "yoy_growth_pct": -14.7},
        "eu_charging_density": {"poland_per_100km": 3.1, "germany_per_100km": 18.9},
    }
    key = topic.lower().replace(" ", "_")
    for k, v in stats.items():
        if key in k or k in key:
            return json.dumps(v)
    return f"No documentation match for '{topic}'."


def calculate(expression: str) -> str:
    """Safe-ish calculator. Returns a string (remember Day 4 bug: return str, not float!)."""
    try:
        # Restrict to math builtins only.
        allowed = {"__builtins__": {}, "abs": abs, "round": round, "min": min, "max": max}
        result = eval(expression, allowed, {})  # noqa: S307 — intentional, sandboxed
        return str(result)  # Day 4 lesson: ALWAYS str(), never the raw value
    except Exception as e:
        # Day 7 bug: returning the exception OBJECT breaks downstream.
        return f"Error: {str(e)}"


def compare_metrics(metric_a: float, metric_b: float, label_a: str, label_b: str) -> str:
    """Compare two metrics and return a verbal summary."""
    if metric_a == metric_b:
        verdict = "equal"
    elif metric_a > metric_b:
        pct = (metric_a - metric_b) / max(abs(metric_b), 1e-9) * 100
        verdict = f"{label_a} is {pct:.1f}% higher than {label_b}"
    else:
        pct = (metric_b - metric_a) / max(abs(metric_a), 1e-9) * 100
        verdict = f"{label_b} is {pct:.1f}% higher than {label_a}"
    return f"{label_a}={metric_a}, {label_b}={metric_b}. {verdict}."


# Registry + schemas ---------------------------------------------------------
# On Day 3 you wrote JSON schemas by hand. Same pattern here.

TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "search_web": search_web,
    "lookup_documentation": lookup_documentation,
    "calculate": calculate,
    "compare_metrics": compare_metrics,
}

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "search_web",
        "description": "Search the web for recent information. Best for news, statistics, trends.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
    {
        "name": "lookup_documentation",
        "description": "Look up structured statistics from a curated knowledge base.",
        "input_schema": {
            "type": "object",
            "properties": {"topic": {"type": "string", "description": "Topic to look up"}},
            "required": ["topic"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a numeric expression, e.g. '(21.4 - 6.8) / 6.8 * 100'.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "compare_metrics",
        "description": "Compare two numeric metrics and return a verbal summary.",
        "input_schema": {
            "type": "object",
            "properties": {
                "metric_a": {"type": "number"},
                "metric_b": {"type": "number"},
                "label_a": {"type": "string"},
                "label_b": {"type": "string"},
            },
            "required": ["metric_a", "metric_b", "label_a", "label_b"],
        },
    },
]


# ============================================================================
# PART 2 — Agents
# ============================================================================

@dataclass
class Agent:
    name: str
    system: str
    model: str
    description: str           # shown to the SELECTOR so it knows when to pick this agent
    tools: list[dict] = field(default_factory=list)  # subset of TOOL_SCHEMAS; empty = no tools


# System messages — lean on lessons from Day 7
# --------------------------------------------
# Day 7 failure: Planner did everyone's job. Fix: explicit prohibitions.
PLANNER_SYSTEM = """You are the Planner.

Your job:
  1. Read the task and break it into 2-4 subtasks, naming which role should handle each.
  2. Assign research subtasks to RESEARCHER, numeric/comparison subtasks to ANALYST.
  3. After teammates report back, synthesize their findings into a final answer.
  4. When the task is FULLY answered, respond with exactly the word: TERMINATE
     on its own line, preceded by your final synthesis.

You MUST NOT:
  - Perform research yourself (no inventing statistics or facts).
  - Perform calculations yourself (no arithmetic in your messages).
  - Call tools. You have no tools.

Keep messages short and directive. No flattery, no meta-commentary.
"""

RESEARCHER_SYSTEM = """You are the Researcher.

Your job: gather facts using your tools (search_web, lookup_documentation)
to answer whatever subtask the Planner assigned to you.

Rules:
  - Use tools. Do not invent facts.
  - After getting tool results, produce ONE short paragraph summarizing what
    you found, clearly stating the numbers and sources. Do not emit bare numbers
    without context — your teammates need to understand the result without
    re-reading tool output.
  - If the task requires calculation, say so and defer to the Analyst.
"""

ANALYST_SYSTEM = """You are the Analyst.

Your job: perform numeric comparisons and calculations using your tools
(calculate, compare_metrics). You work with numbers the Researcher has
already surfaced.

Rules:
  - Do not do research. If you lack data, say so and defer to the Researcher.
  - After getting tool results, produce ONE short paragraph stating the
    computation, the numbers involved, and the conclusion. Never emit a bare
    number — always show the computation it came from.
"""


# ============================================================================
# PART 3 — The tool-calling inner loop
# ============================================================================

@dataclass
class LLMCallLog:
    turn: int
    caller: str     # agent name, or "SELECTOR"
    model: str
    input_tokens: int
    output_tokens: int
    note: str       # e.g. "initial response", "after tool results", "selector pick"


def run_tool(tool_use_block: Any) -> dict:
    """
    Execute one tool_use block. Returns the tool_result block to append
    to `messages` as a user-role content item.

    The Anthropic tool_result format is:
      {"type": "tool_result", "tool_use_id": "<id>", "content": "<string>"}
    """
    # ------------------------------------------------------------------------
    # TODO 1 — Execute the tool.
    #
    # `tool_use_block` has: .name, .input (dict), .id
    # Steps:
    #   1. Look up the function in TOOL_REGISTRY by name.
    #   2. Call it with **tool_use_block.input.
    #   3. Wrap in try/except: on error, return an error STRING (not the
    #      exception object — Day 7 bug #2). Keep the Exception message.
    #   4. Always return str(result) — never the raw object (Day 4 bug).
    #
    # Return shape (REQUIRED):
    #   {"type": "tool_result", "tool_use_id": <block.id>, "content": <str>}
    # ------------------------------------------------------------------------

    fn = TOOL_REGISTRY.get(tool_use_block.name)
    if fn is None:
        result = f"Error: unknown tool '{tool_use_block.name}'"
    else:
        try:
            result = fn(**tool_use_block.input)
        except Exception as e:
            result = f"Error: {str(e)}"

    return {
        "type": "tool_result",
        "tool_use_id": tool_use_block.id,
        "content": str(result),
    }

def agent_step_with_tools(
    client: anthropic.Anthropic,
    agent: Agent,
    incoming_text: str,
    logs: list[LLMCallLog],
    turn_number: int,
) -> str:
    """
    Run ONE agent's turn. Handles the tool-use loop internally: the agent
    may call 0+ tools before producing a final text response.

    After all tools execute, does one more LLM call asking the agent to
    summarize its findings in plain prose (this is the manual replacement
    for AutoGen's broken-on-Anthropic reflect_on_tool_use=True).

    Returns the final text the agent produces.
    """
    # Each agent call starts fresh — the SHARED history (maintained by the
    # orchestrator) is passed in as `incoming_text`. The agent doesn't
    # remember its own past turns across rounds; that's intentional and
    # matches how SelectorGroupChat actually works.
    messages: list[dict] = [{"role": "user", "content": incoming_text}]

    max_inner_iterations = 6   # tool-calling safety net
    inner = 0

    while inner < max_inner_iterations:
        inner += 1

        # --------------------------------------------------------------------
        # TODO 2 — Call the API with tools.
        #
        # Required kwargs: model, max_tokens, system, messages, tools.
        # - tools: pass `agent.tools` directly (it's already a list of dicts).
        # - If the agent has no tools (agent.tools == []), you can either:
        #     (a) call without the tools kwarg, or
        #     (b) pass tools=[] — both work, but (a) is cleaner. Use an if/else.
        # - max_tokens: 1024 is fine.
        #
        # Remember to LOG the call: append an LLMCallLog with caller=agent.name,
        # tokens from response.usage, and a note like "tool-loop iter {inner}".
        # --------------------------------------------------------------------
        kwargs = {
            "model": agent.model,
            "max_tokens": 1024,
            "system": agent.system,
            "messages": messages,
        }
        if agent.tools:
            kwargs["tools"] = agent.tools
        response = client.messages.create(**kwargs)

        # Log it (don't remove this — Exercise 3 depends on complete logs)
        logs.append(LLMCallLog(
            turn=turn_number, caller=agent.name, model=agent.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            note=f"tool-loop iter {inner}",
        ))

        # --------------------------------------------------------------------
        # TODO 3 — Check stop_reason. Branch on the two cases.
        #
        # Case A: response.stop_reason != "tool_use"
        #   The agent is done. Gather text from response.content:
        #     text_parts = [b.text for b in response.content if b.type == "text"]
        #     final_text = "\n".join(text_parts).strip()
        #   Return final_text.
        #
        # Case B: response.stop_reason == "tool_use"
        #   The agent called at least one tool. Continue:
        #     1. Append assistant response to messages:
        #          messages.append({"role": "assistant", "content": response.content})
        #        (note: pass the raw list of blocks here — not a string — so the
        #         tool_use blocks are preserved for the next API call)
        #     2. For every block in response.content with b.type == "tool_use",
        #        execute it via run_tool(b) and collect results.
        #     3. Append all tool_results as ONE user message:
        #          messages.append({"role": "user", "content": [tr1, tr2, ...]})
        #     4. Loop (don't return yet — the agent may want to call more tools
        #        or produce a final summary).
        # --------------------------------------------------------------------
        # --- your code here ---
        if response.stop_reason != "tool_use":
            text_parts = [b.text for b in response.content if b.type == "text"]
            final_text = "\n".join(text_parts).strip()
            return final_text
        
    
        messages.append({"role": "assistant", "content": response.content})
        tool_results = [run_tool(b) for b in response.content if b.type == "tool_use"]
        messages.append({"role":"user", "content": tool_results})


    # If we fall out of the loop without returning, we hit max_inner_iterations.
    # Return whatever text we last got, with a warning.
    return f"[WARNING: agent {agent.name} exceeded tool-loop limit]"


# ============================================================================
# PART 4 — The selector (dynamic routing)
# ============================================================================

SELECTOR_SYSTEM = """You are a routing system. Given a conversation history
and a list of agent roles, pick which agent should speak NEXT.

You MUST respond with EXACTLY ONE WORD: the name of the chosen agent, or
the word TERMINATE if the task is complete.

Valid responses: {valid_names}

Do NOT explain. Do NOT add punctuation. Just the name.
"""


def pick_next_speaker(
    client: anthropic.Anthropic,
    agents: list[Agent],
    history: list[dict],
    logs: list[LLMCallLog],
    turn_number: int,
) -> str:
    """
    Ask an LLM to choose the next speaker (or TERMINATE). Returns a name.

    Notice: this is exactly the call AutoGen's SelectorGroupChat was making
    invisibly on Day 7. Now it's visible in your logs.
    """
    valid_names = [a.name for a in agents] + ["TERMINATE"]
    system = SELECTOR_SYSTEM.format(valid_names=" | ".join(valid_names))

    # Build a compact context for the selector. Don't send the full raw
    # history — just role summaries + a trimmed transcript. The selector
    # needs enough to decide; sending everything wastes tokens.
    role_block = "\n".join(f"  - {a.name}: {a.description}" for a in agents)

    # Flatten the shared history for the selector. Each item in `history` is
    # already a dict like {"role": "user", "content": "[PLANNER]: ..."}.
    # We just want the text in order.
    transcript_lines = []
    for msg in history:
        content = msg["content"]
        if isinstance(content, str):
            transcript_lines.append(content)
    transcript = "\n".join(transcript_lines[-12:])  # last 12 lines is plenty

    user_content = (
        f"Available agents:\n{role_block}\n\n"
        f"Recent conversation:\n{transcript}\n\n"
        f"Who should speak next? (one word)"
    )

    # ------------------------------------------------------------------------
    # TODO 4 — Make the selector LLM call.
    #
    # - Model: CHEAP_MODEL. This call happens ~N times per run; don't waste
    #   Sonnet tokens on routing decisions.
    # - max_tokens: 20 is MORE than enough for one word. Keeping it tight
    #   is also a safety rail — the model can't ramble into an essay.
    # - Pass `system` and messages=[{"role": "user", "content": user_content}].
    # - LOG this call with caller="SELECTOR".
    # ------------------------------------------------------------------------
    response = client.messages.create(
        model = CHEAP_MODEL,
        max_tokens = 20,
        system = system,
        messages = [{"role": "user", "content": user_content}]
    )

    logs.append(LLMCallLog(
        turn=turn_number, caller="SELECTOR", model=CHEAP_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        note="routing decision",
    ))

    # ------------------------------------------------------------------------
    # TODO 5 — Extract and sanitize the selector's answer.
    #
    # Steps:
    #   1. Get the text from response.content (same pattern as Exercise 1).
    #   2. Uppercase it, strip whitespace/punctuation.
    #   3. Check if the cleaned name is in valid_names. If yes, return it.
    #   4. If no (selector returned garbage like "Researcher should go next"),
    #      fallback: pick the FIRST valid name found as a substring of the
    #      response. If still no match, return "TERMINATE" as a safety exit.
    #
    # Robust selector parsing is important — without it, one misformatted
    # reply crashes the run. Frameworks hide this from you; now you see why.
    # ------------------------------------------------------------------------
    text = next((b.text for b in response.content if b.type == "text"), "")
    name = text.upper().strip().translate(str.maketrans("", "", string.punctuation))

    if name in valid_names:
        return name

    for valid in sorted(valid_names, key=len, reverse=True):
        if valid in name:
            return valid

    return "TERMINATE"

# ============================================================================
# PART 5 — The orchestrator
# ============================================================================

def format_for_history(speaker: str, text: str) -> dict:
    """
    Wrap an agent's output as a user-role message in the shared history,
    prefixed with [SPEAKER] so downstream agents know who said what.

    This fixes the Day 7 Ex 2 identity-confusion bug: in AutoGen, everything
    ended up as generic user-role messages and agents lost track of who was
    who. Explicit speaker tags make the shared context unambiguous.
    """
    return {"role": "user", "content": f"[{speaker}]: {text}"}


def run_team(
    client: anthropic.Anthropic,
    agents: list[Agent],
    task: str,
    max_turns: int = MAX_TURNS,
) -> tuple[str, Literal["terminated", "max_turns"], list[LLMCallLog]]:
    """
    Main orchestration loop. Selector picks; chosen agent speaks; output
    goes into shared history; repeat.
    """
    logs: list[LLMCallLog] = []
    agents_by_name = {a.name.upper(): a for a in agents}

    # Shared history seeds with the user task (flat format — strings, not blocks).
    # Each agent receives this history flattened into a single prompt when it's
    # their turn. They don't see it as a multi-turn conversation — they see it
    # as "here's the context, give me your next move".
    history: list[dict] = [{"role": "user", "content": f"[TASK]: {task}"}]

    print(f"[TASK] {task}\n")

    for turn in range(1, max_turns + 1):
        # --------------------------------------------------------------------
        # TODO 6 — Pick the next speaker.
        #
        # Call pick_next_speaker(client, agents, history, logs, turn).
        # Uppercase-safe: name = picked.upper().
        # --------------------------------------------------------------------
        name = pick_next_speaker(client, agents, history, logs, turn)


        print(f"\n--- Turn {turn}: selector picked {name} ---")

        # --------------------------------------------------------------------
        # TODO 7 — Handle TERMINATE.
        #
        # If name == "TERMINATE": break out of the loop. Return the last
        # non-selector message text as the "final answer". Stop reason: "terminated".
        #
        # Hint: the last agent message in `history` is `history[-1]`. Its text
        # is `history[-1]["content"]`, still prefixed with [SPEAKER]. You can
        # strip the prefix for the return value.
        # --------------------------------------------------------------------
        if name == "TERMINATE":
            # --- your code here ---
            last = history[-1]["content"]
            # Strip leading "[SPEAKER]: " if present
            if last.startswith("[") and "]: " in last:
                last = last.split("]: ", 1)[1]
            return last, "terminated", logs


        # --------------------------------------------------------------------
        # TODO 8 — Run the chosen agent and append its output.
        #
        # Steps:
        #   1. agent = agents_by_name[name]
        #   2. Flatten history into a single prompt string. A reasonable format:
        #        prompt = "\n".join(m["content"] for m in history)
        #      (history items are already prefixed with [SPEAKER] or [TASK]).
        #   3. text = agent_step_with_tools(client, agent, prompt, logs, turn)
        #   4. Print a short banner: print(f"\n[{name}]: {text}\n")
        #   5. Append to history using format_for_history(name, text).
        #   6. Also check if text itself contains TERMINATE — the Planner's
        #      final synthesis typically ends with it. If so, break with
        #      stop_reason = "terminated" and return.
        # --------------------------------------------------------------------
        # --- your code here ---
        agent = agents_by_name[name]
        prompt = "\n".join(m["content"] for m in history)
        text = agent_step_with_tools(client, agent, prompt, logs, turn)
        print(f"\n[{name}]: {text}\n")
        history.append(format_for_history(name, text))
        last_line = text.strip().splitlines()[-1].strip().upper() if text.strip() else ""
        if last_line == TERMINATE_TOKEN or last_line.endswith(TERMINATE_TOKEN):
            return text, "terminated", logs

    # Fell through: hit max_turns
    last_msg = history[-1]["content"] if history else ""
    return last_msg, "max_turns", logs


# ============================================================================
# PART 6 — Reporting
# ============================================================================

def print_log_summary(logs: list[LLMCallLog]) -> None:
    """Per-turn, per-caller LLM call summary."""
    print("\n" + "=" * 90)
    print(f"{'Turn':<5} {'Caller':<12} {'Model':<22} {'In':>7} {'Out':>7}  Note")
    print("-" * 90)
    total_in = total_out = 0
    by_caller: dict[str, int] = {}
    for L in logs:
        print(f"{L.turn:<5} {L.caller:<12} {L.model:<22} {L.input_tokens:>7} {L.output_tokens:>7}  {L.note}")
        total_in += L.input_tokens
        total_out += L.output_tokens
        by_caller[L.caller] = by_caller.get(L.caller, 0) + 1
    print("-" * 90)
    print(f"{'TOTAL':<5} {'':<12} {'':<22} {total_in:>7} {total_out:>7}")
    print(f"\nTotal LLM calls: {len(logs)}")
    print("Calls by caller (NOTE THE SELECTOR COUNT — this was invisible in AutoGen):")
    for caller, count in sorted(by_caller.items()):
        print(f"  {caller:<12} {count}")
    print("=" * 90)


# ============================================================================
# PART 7 — Main
# ============================================================================

def build_agents() -> list[Agent]:
    # Descriptions are what the SELECTOR sees — write them like search-engine
    # descriptions for routing (concise, capability-focused), not like system
    # prompts. Day 7 lesson: description ≠ system_message.
    return [
        Agent(
            name="PLANNER",
            description="Decomposes tasks, assigns subtasks, synthesizes final answer. No tools.",
            system=PLANNER_SYSTEM,
            model=SMART_MODEL,
            tools=[],
        ),
        Agent(
            name="RESEARCHER",
            description="Gathers facts using search_web and lookup_documentation. Use for factual lookups.",
            system=RESEARCHER_SYSTEM,
            model=SMART_MODEL,
            tools=[s for s in TOOL_SCHEMAS if s["name"] in {"search_web", "lookup_documentation"}],
        ),
        Agent(
            name="ANALYST",
            description="Runs calculations and numeric comparisons. Use for math, percentages, comparisons.",
            system=ANALYST_SYSTEM,
            model=SMART_MODEL,
            tools=[s for s in TOOL_SCHEMAS if s["name"] in {"calculate", "compare_metrics"}],
        ),
    ]


def main() -> None:
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise RuntimeError("Set ANTHROPIC_API_KEY env var first.")

    client = anthropic.Anthropic()
    agents = build_agents()

    start = time.time()
    final_text, stop_reason, logs = run_team(client, agents, TASK, max_turns=MAX_TURNS)
    elapsed = time.time() - start

    print("\n" + "#" * 90)
    print(f"# FINAL ANSWER (stop_reason={stop_reason}, elapsed={elapsed:.1f}s)")
    print("#" * 90)
    print(final_text)
    print_log_summary(logs)


if __name__ == "__main__":
    main()


# ============================================================================
# Post-exercise reflection questions
# ============================================================================
# 1. On Day 7 Ex 3 you noted: "multi-agent token count is underreported
#    because SelectorGroupChat's routing LLM calls are hidden/internal."
#    What fraction of your TOTAL tokens today were selector calls? Would
#    you have guessed that number before measuring it?
#
# 2. Compare to your Day 7 Exercise 2 cost. Is your raw version cheaper,
#    more expensive, or about the same? (If more expensive: why? If cheaper:
#    is there something AutoGen was doing extra, or differently?)
#
# 3. The Day 7 fix for the "bare number" problem was removing reflect_on_tool_use
#    and rewriting the Analyst's system prompt. Your fix here is different —
#    what is it, and why does it generalize better?
#
# 4. Your selector picks with Haiku. What would change if you used Sonnet?
#    Would the routing decisions improve enough to justify the cost?
#
# 5. Imagine adding a 4th agent (e.g. FACT_CHECKER). How much code changes?
#    How many lines would the equivalent AutoGen change be? Which is clearer?
