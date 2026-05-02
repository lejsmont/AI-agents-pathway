"""
Day 8 — Exercise 3: Comparison + Mini-Framework Extraction
===========================================================

Goal
----
Two parts:
  (A) MEASURE: run your Exercise 2 (raw) and your Day 7 Exercise 2 (AutoGen)
      on the same task. Record LLM calls, tokens, wall-clock, bugs-hit.
  (B) EXTRACT: pull the primitives out of Exercise 2 into a tiny reusable
      module — your own ~100-line "framework". Prove to yourself that you
      now understand what frameworks do by building one.

Success criteria
----------------
- You can run a head-to-head benchmark and print a comparison table.
- You produce `micro_agents.py` with an Agent/ToolAgent/run_team API that
  you actually want to use in Exercises 9-14.
- You write framework_comparison.md with a grounded opinion.

Estimated lines to fill in: ~80 across 5 TODO blocks. Rest is scaffolding.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

# We'll import from Exercise 2 — make sure both files are in the same dir.
# If you renamed Ex2, adjust the import.
try:
    from exercise2_manual_selector import (
        TOOL_SCHEMAS,
        TOOL_REGISTRY,
        build_agents,
        run_team,
        LLMCallLog,
    )
except ImportError as e:
    print(f"Import failed: {e}. Make sure exercise2_manual_selector.py is complete first.")
    sys.exit(1)

import anthropic


# ============================================================================
# PART A — BENCHMARKING
# ============================================================================

# Shared task across both implementations
BENCHMARK_TASK = (
    "Compare electric vehicle adoption between Poland and Germany for 2025. "
    "Give me adoption rates, recent trends, and a one-sentence conclusion on "
    "which country has stronger momentum."
)


@dataclass
class RunRecord:
    """Measurements from one run of one implementation."""
    implementation: str       # "raw" | "autogen"
    wall_clock_s: float
    llm_calls: int
    input_tokens: int
    output_tokens: int
    stop_reason: str
    final_answer_preview: str  # first 200 chars
    # Subjective: filled in by you after running
    bugs_hit: list[str] = field(default_factory=list)
    notes: str = ""


def run_raw(task: str) -> RunRecord:
    """Run the Exercise 2 raw implementation and collect metrics."""
    client = anthropic.Anthropic()
    agents = build_agents()

    start = time.time()
    final, stop_reason, logs = run_team(client, agents, task)
    elapsed = time.time() - start

    # ------------------------------------------------------------------------
    # TODO 1 — Aggregate the LLMCallLog list into a RunRecord.
    #
    # - llm_calls: len(logs)
    # - input_tokens: sum of log.input_tokens
    # - output_tokens: sum of log.output_tokens
    # - final_answer_preview: first 200 chars of `final`, with newlines squashed to spaces.
    # ------------------------------------------------------------------------
    # replace with the RunRecord(...) return
    return RunRecord(
        implementation="raw",
        wall_clock_s=elapsed,
        llm_calls=len(logs),
        input_tokens=sum(log.input_tokens for log in logs),
        output_tokens=sum(log.output_tokens for log in logs),
        stop_reason=stop_reason,
        final_answer_preview=final.replace("\n", " ")[:200]
    )

import asyncio

def run_autogen(task: str) -> RunRecord:
    from day7.exercise2_selector_team import run_selector_team

    start = time.time()
    result = asyncio.run(run_selector_team(task))
    elapsed = time.time() - start

    # AutoGen exposes per-message token usage on .models_usage. It's None for
    # messages that didn't trigger an LLM call (tool result events, etc).
    # CRITICAL: selector LLM calls are NOT in result.messages — they happen
    # inside SelectorGroupChat and never surface as messages. So this total
    # is systematically LOW. That gap IS the Day 8 lesson.
    input_tokens = 0
    output_tokens = 0
    llm_call_count = 0
    for msg in result.messages:
        usage = getattr(msg, "models_usage", None)
        if usage is not None:
            input_tokens += usage.prompt_tokens or 0
            output_tokens += usage.completion_tokens or 0
            llm_call_count += 1

    # Final answer = last text-bearing message (typically Planner's synthesis)
    final = ""
    for msg in result.messages:
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            final = content

    return RunRecord(
        implementation="autogen",
        wall_clock_s=elapsed,
        llm_calls=llm_call_count,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        stop_reason=str(result.stop_reason),
        final_answer_preview=final.replace("\n", " ")[:200],
        notes="Selector routing calls invisible to models_usage — token total is systematically LOW.",
    )


def print_comparison(raw: RunRecord, autogen: RunRecord) -> None:
    """Print a head-to-head comparison table."""
    rows = [
        ("Metric",           "Raw Python",                "AutoGen (Day 7)"),
        ("Wall clock (s)",   f"{raw.wall_clock_s:.1f}",   f"{autogen.wall_clock_s:.1f}"),
        ("LLM calls",        str(raw.llm_calls),          str(autogen.llm_calls)),
        ("Input tokens",     str(raw.input_tokens),       str(autogen.input_tokens)),
        ("Output tokens",    str(raw.output_tokens),      str(autogen.output_tokens)),
        ("Stop reason",      raw.stop_reason,             autogen.stop_reason),
        ("Bugs hit",         str(len(raw.bugs_hit)),      str(len(autogen.bugs_hit))),
    ]
    w1, w2, w3 = 18, 24, 24
    print("\n" + "=" * (w1 + w2 + w3 + 4))
    for i, row in enumerate(rows):
        print(f"{row[0]:<{w1}}  {row[1]:<{w2}}  {row[2]:<{w3}}")
        if i == 0:
            print("-" * (w1 + w2 + w3 + 4))
    print("=" * (w1 + w2 + w3 + 4))

    if raw.notes or autogen.notes:
        print("\nNotes:")
        if raw.notes:
            print(f"  Raw: {raw.notes}")
        if autogen.notes:
            print(f"  AutoGen: {autogen.notes}")

    # Approximate LOC comparison — measured from your actual files
    try:
        raw_loc = _count_loc("exercise2_manual_selector.py")
        # Your Day 7 file path — adjust as needed:
        ag_loc = _count_loc("../day7/exercise2_selector_team.py")
        print(f"\nApproximate LOC: raw={raw_loc}, autogen={ag_loc}, ratio={raw_loc/max(ag_loc,1):.2f}x")
    except Exception as e:
        print(f"\n(LOC counting skipped: {e})")


def _count_loc(path: str) -> int:
    """Count non-blank, non-comment lines of code."""
    p = Path(path)
    if not p.exists():
        return -1
    count = 0
    for line in p.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#") and not s.startswith('"""') and not s.startswith("'''"):
            count += 1
    return count


# ============================================================================
# PART B — MINI-FRAMEWORK EXTRACTION
# ============================================================================
#
# You've now written the raw loop TWICE (Ex 1 and Ex 2). You can see the
# repeated primitives. Let's extract them into a tiny reusable module.
#
# TODO 3 — Design and write `micro_agents.py` in this same directory.
#
# Required API (sketch):
#
#   from micro_agents import Agent, ToolAgent, run_team
#
#   writer = Agent(name="Writer", system="...", model="claude-haiku-4-5")
#   critic = Agent(name="Critic", system="...", model="claude-sonnet-4-6")
#
#   # Reflection pattern — just a function, no framework class:
#   result = run_reflection(writer, critic, task="...", approve_token="APPROVE")
#
#   # Selector pattern with tools:
#   researcher = ToolAgent(
#       name="RESEARCHER", system="...", model="...",
#       tools={"search_web": search_web, "lookup_docs": lookup_docs},
#       tool_schemas=[...],
#   )
#   result = run_team(
#       [planner, researcher, analyst],
#       task="...",
#       terminate_token="TERMINATE",
#       max_turns=12,
#   )
#
# Design goals (non-negotiable):
#   1. No imports besides `anthropic`, stdlib, and your own code.
#   2. Under 200 lines of actual code (this is a MINIMAL framework).
#   3. Every LLM call is logged and accessible via `result.logs`.
#   4. Tools are plain sync functions (no async). No registration magic —
#      you pass the dict in explicitly.
#   5. Termination is explicit: an approve token, a max_turns, or an
#      explicit TERMINATE return from the selector.
#
# Start by copying the relevant bits from Exercise 1 and Exercise 2 into
# micro_agents.py, then refactor. Things to extract:
#   - The Agent dataclass (add optional tools/tool_schemas fields)
#   - agent_turn() from Ex 1 (rename to a method: Agent.turn())
#   - agent_step_with_tools() from Ex 2 (ToolAgent.turn())
#   - pick_next_speaker() from Ex 2 (as a module-level function)
#   - run_reflection() — a new small helper using Ex 1's orchestrator logic
#   - run_team() — from Ex 2, generalized
#
# Make it PRODUCTION-USEFUL-FOR-YOU: the test of a good mini-framework is
# that you'd reach for it in Days 9-14 over raw Python.
# ----------------------------------------------------------------------------


def verify_micro_framework_exists() -> bool:
    """Exercise 3 is complete when micro_agents.py exists and has the right API."""
    p = Path("day8/micro_agents.py")
    if not p.exists():
        return False
    contents = p.read_text()
    required_symbols = ["class Agent", "class ToolAgent", "def run_team"]
    return all(s in contents for s in required_symbols)


# ============================================================================
# PART D — (Bonus) Run your micro_agents.py end-to-end
# ============================================================================

def run_with_micro_framework(task: str) -> RunRecord:
    """
    Re-run the same task using YOUR OWN micro_agents.py framework.

    If your framework is well-designed, this function should be SHORTER than
    run_raw() above. If it isn't, your framework isn't pulling enough weight —
    iterate until it does.
    """
    # ------------------------------------------------------------------------
    # TODO 5 — Use your micro_agents.py to run the same BENCHMARK_TASK.
    #
    # Target: ~15 lines of code or fewer. If it's longer, your framework is
    # underpowered — go back and tighten it.
    #
    #   from micro_agents import ToolAgent, Agent, run_team
    #   planner = Agent(name="PLANNER", system=PLANNER_SYSTEM, model="claude-sonnet-4-6")
    #   researcher = ToolAgent(...)
    #   analyst = ToolAgent(...)
    #   result = run_team([planner, researcher, analyst], task=BENCHMARK_TASK)
    #   return RunRecord(implementation="micro_framework", ...)
    # ------------------------------------------------------------------------
    from micro_agents import Agent, run_team
    
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

    planner = Agent(name="Planner", 
                    system=PLANNER_SYSTEM, 
                    model="claude-sonnet-4-6", 
                    description="Decomposes tasks, assigns subtasks, synthesizes final answer. No tools.",
                    tools= [],
                    stateful=True)
    
    researcher = Agent(name="Researcher", 
                    system=RESEARCHER_SYSTEM, 
                    model="claude-sonnet-4-6", 
                    description="Gathers facts using search_web and lookup_documentation. Use for factual lookups.",
                    tools= [s for s in TOOL_SCHEMAS if s["name"] in {"search_web", "lookup_documentation"}],
                    stateful=True)
    analyst = Agent(name="Analyst", 
                    system=ANALYST_SYSTEM, 
                    model="claude-sonnet-4-6", 
                    description="Runs calculations and numeric comparisons. Use for math, percentages, comparisons.",
                    tools= [s for s in TOOL_SCHEMAS if s["name"] in {"search_web", "lookup_documentation"}],
                    stateful=True)
    
    client = anthropic.Anthropic()
    start = time.time()
    final, stop_reason, logs = run_team(client = client, agents = [planner, researcher, analyst], task=BENCHMARK_TASK, max_turns=15, terminate_token="TERMINATE")
    elapsed = time.time() - start
    return RunRecord(
        implementation="framework",
        wall_clock_s=elapsed,
        llm_calls=len(logs),
        input_tokens=sum(log.input_tokens for log in logs),
        output_tokens=sum(log.output_tokens for log in logs),
        stop_reason=stop_reason,
        final_answer_preview=final.replace("\n", " ")[:200]
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise RuntimeError("Set ANTHROPIC_API_KEY env var first.")

    print("# Day 8 Exercise 3 — Comparison Harness")
    print("#" * 78)

    # # Stage 1 — benchmark raw vs. framework
    # print("\n[1/3] Running raw-Python implementation...")
    # raw = run_raw(BENCHMARK_TASK)

    # print("\n[2/3] Running AutoGen implementation...")
    # autogen = run_autogen(BENCHMARK_TASK)

    # print("\n[3/3] Comparison:")
    # print_comparison(raw, autogen)

    # Stage 2 — did you extract the mini-framework?
    print("\n# Checking mini-framework...")
    
    # Optional: run Bonus Part D
    try:
        micro = run_with_micro_framework(BENCHMARK_TASK)
        print(f"\n[Bonus] micro_agents.py run: {micro.llm_calls} calls, "
                f"{micro.wall_clock_s:.1f}s, "
                f"{micro.input_tokens + micro.output_tokens} tokens")
    except NotImplementedError:
        print("  (Bonus Part D not yet implemented — skip)")
    
    print("\nDone. Commit day8/ to git when all three checks pass.")


if __name__ == "__main__":
    main()


# ============================================================================
# Final reflection (answer in framework_comparison.md, don't just think it)
# ============================================================================
# A. Did the framework actually save you time IN AGGREGATE, factoring in the
#    debugging you did on Day 7? Be honest.
#
# B. For your Day 9 eval harness, which will you use: micro_agents.py,
#    raw Python, or AutoGen/LangChain? Why?
#
# C. One thing you LOST by going raw. (There is something. Name it.)
#
# D. One thing you GAINED that you didn't expect.
#
# E. If you had to teach Day 6-7 to someone else starting tomorrow, would
#    you change the ORDER — raw first, frameworks after? Or the current
#    order — frameworks first, then detox?
