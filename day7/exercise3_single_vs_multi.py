"""
Day 7 — Exercise 3: Single-Agent vs. Multi-Agent Comparison
============================================================

Goal: Run the SAME task through (a) a single agent with all tools, and
(b) the multi-agent team from Exercise 2. Compare output quality, token
usage, API calls, and latency. Write your findings.

This exercise builds your intuition for WHEN to use multi-agent vs. single-agent.
The answer isn't "multi-agent is always better" — it depends on the task.

Concepts practiced:
    - Single agent with multiple tools
    - Measuring and comparing token usage across approaches
    - Timing async operations
    - Critical thinking about architecture trade-offs

Run:
    python exercise3_single_vs_multi.py
"""

import asyncio
import json
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.anthropic import AnthropicChatCompletionClient

# ─── Shared tools ──────────────────────────────────────────────────────────

# TODO 1: Copy (or import) the 4 tool functions from Exercise 2.
# Both the single agent and multi-agent team need the same tools
# so we get a fair comparison.
#
# Tools needed: search_web, lookup_documentation, calculate, compare_metrics
#
# Hint: You can either copy-paste them or put them in a shared module
# and import. For this exercise, copy-paste is fine.

async def search_web(query: str) -> str:
    """Search the web for information on a topic"""
    return "According to recent surveys, Python is used by 70% of ML engineers and Javascript is used by 30% of ML engineers. Python usage is growing 3% YOY over last decade and Javascript usage is flat"

async def lookup_documentation(topic: str) -> str:
    """Look up technical documentation or API refereces."""
    return "Some technical documentation"

async def calculate(expression: str) -> str:
    """Calculate a mathematical expression. Input should be a valid Python math expression."""
    try:
        result = eval(expression)
    except Exception as e:
        print(e)
        return e
    
    return str(result)

async def compare_metrics(metric_a: str, value_a: float, metric_b: str, value_b: float) -> str:
    """Compare two metrics and provide analysis."""
    if value_b == 0 and value_a == 0:
        return f"{metric_a} ({value_a}) and {metric_b} ({value_b}) are both zero."
    
    if value_b == 0:
        return f"Cannot calculate percentage difference: {metric_b} is zero."
    
    pct_diff = ((value_a - value_b) / abs(value_b)) * 100
    higher = metric_a if value_a > value_b else metric_b
    
    return (
        f"Comparison: {metric_a} = {value_a}, {metric_b} = {value_b}. "
        f"{higher} is higher. "
        f"Percentage difference: {pct_diff:+.1f}% "
        f"({metric_a} relative to {metric_b})."
    )

# ─── Single Agent Setup ───────────────────────────────────────────────────

def create_single_agent(model_client):
    """Create a single agent that has ALL tools.

    TODO 2: Create an AssistantAgent with:
    - name: "SingleAgent"
    - model_client: your model client
    - tools: ALL four tools [search_web, lookup_documentation, calculate, compare_metrics]
    - system_message: A comprehensive prompt that tells it to:
        1. Break down complex tasks into steps
        2. Use the right tool for each step
        3. Show its reasoning
        4. Provide a final synthesized answer
        5. Say "TERMINATE" when done

    This single agent does the job of Planner + Researcher + Analyst all in one.
    """
    return AssistantAgent(
        name="SingleAgent",
        model_client=model_client,
        tools=[search_web, lookup_documentation, calculate, compare_metrics],
        system_message = (
            "1. Break down complex tasks into steps\n"
            "2. Use the right tool for each step\n"
            "3. Show its reasoning\n"
            "4. Provide a final synthesized answer\n"
            "5. Say 'TERMINATE' when done\n"
        )
    )


# ─── Multi-Agent Setup ────────────────────────────────────────────────────

def create_model_client():
    """Create the model client."""
    return AnthropicChatCompletionClient(model="claude-sonnet-4-20250514")


def create_planner_agent(model_client):
    """Create the Planner agent.
    """
    return AssistantAgent(
        name= "Planner",
        model_client= model_client,
        description= "Plans and coordinates tasks, breaks complex queries into steps, and determines when the task is complete.",
        system_message=(
            "You are the Planner in a multi-agent team with Researcher and Analyst.\n"
            "Your job is ONLY to coordinate. You must NEVER:\n"
            "- Make up data or statistics\n"
            "- Perform calculations\n"
            "- Write analysis yourself\n\n"
            "What you MUST do:\n"
            "1. Break the task into numbered steps\n"
            "2. Assign each step to Researcher or Analyst by name\n"
            "3. WAIT for their results before proceeding\n"
            "4. After all steps are complete, synthesize ONLY from their actual results\n"
            "Keep your messages short — just the plan and assignments."
        )
    )


def create_researcher_agent(model_client):
    """Create the Researcher agent.

    """
    return AssistantAgent(
        name= "Researcher",
        model_client= model_client,
        description= "Searches for information and looks up documentation.", 
        system_message= (
            "You are a Researcher. Use your tools to find information, report findings clearly and never make up any data"
            "You are part of a multi-agent team. Other messages come from "
            "teammate agents, NOT from a human user. Never ask for clarification "
            "from other agents — just do your assigned work."
        ),
        tools= [search_web, lookup_documentation],
        # reflect_on_tool_use=True
    )


def create_analyst_agent(model_client):
    """Create the Analyst agent.

    """
    return AssistantAgent(
        name= "Analyst",
        model_client=model_client,
        description= "Calculates and compares metrics",
        system_message= (
            "You are a data analyst. Given the tasks you have been assigned, you should analyze the data and provide results using the tools provided. If you have not seen the data, ask for it."
            "You are part of a multi-agent team. Other messages come from "
            "teammate agents, NOT from a human user. Never ask for clarification "
            "from other agents — just do your assigned work."
        ),
        tools= [calculate, compare_metrics],
        # reflect_on_tool_use=True
    )

def create_validator_agent(model_client):
    return AssistantAgent(
        name="Validator",
        model_client=model_client,
        description="Validates the final output for quality, accuracy, and completeness.",
        system_message=(
            "You are a quality validator. When the Planner produces a final summary, "
            "review it against the FULL conversation history and check:\n"
            "1. Completeness: Does it address all parts of the original task?\n"
            "2. Data grounding: Are all claims backed by tool results from the conversation? "
            "Flag any claims that appear fabricated or not supported by tool output.\n"
            "3. Calculation accuracy: Do the numbers and percentages make sense "
            "given the raw data from the Researcher?\n"
            "4. Logic: Are the conclusions reasonable given the evidence?\n\n"
            "If the output passes, say 'TERMINATE'.\n"
            "If it fails, explain what's wrong and send it back to the Planner "
            "for revision. Be specific about what needs fixing."
        )
    ) 

def create_selector_team(planner, researcher, analyst, validator, model_client):
    """Create a SelectorGroupChat team.

    """
    return SelectorGroupChat(
        [planner, researcher, analyst, validator], 
        model_client=model_client,
        termination_condition = TextMentionTermination("TERMINATE") | MaxMessageTermination(15),
        selector_prompt = (
            "Select an agent to perform the next task.\n\n"
            "{roles}\n\n"
            "Current conversation context:\n{history}\n\n"
            "Read the above conversation. Then select an agent from {participants} "
            "to perform the next task.\n"
            "Rules:\n"
            "- The Planner should go first to create a plan.\n"
            "- After the plan, select Researcher or Analyst based on the current step.\n"
            "- Return to Planner after all steps are done to synthesize.\n"
            "- After the Planner produces a final summary, always select the Validator next. Only the Validator can say TERMINATE." 
            "- Only select one agent.\n"
        )
    )



# ─── Measurement Harness ──────────────────────────────────────────────────
import time

def count_tokens(result) -> dict:
    prompt_tokens = 0
    completion_tokens = 0
    for msg in result.messages:
        if hasattr(msg, "models_usage") and msg.models_usage is not None:
            prompt_tokens += msg.models_usage.prompt_tokens
            completion_tokens += msg.models_usage.completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


async def run_single_agent(task: str) -> dict:
    model_client = create_model_client()
    agent = create_single_agent(model_client)

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(10)
    team = RoundRobinGroupChat([agent], termination_condition=termination)

    start_time = time.time()
    result = await Console(team.run_stream(task=task))
    elapsed = time.time() - start_time

    agent_turns = sum(1 for msg in result.messages if msg.source != "user")
    tokens = count_tokens(result)

    await model_client.close()

    return {
        "total_messages": len(result.messages),
        "agent_turns": agent_turns,
        "stop_reason": result.stop_reason,
        "elapsed_time": round(elapsed, 2),
        **tokens,
    }


async def run_multi_agent(task: str) -> dict:
    model_client = create_model_client()
    planner = create_planner_agent(model_client)
    researcher = create_researcher_agent(model_client)
    analyst = create_analyst_agent(model_client)
    validator = create_validator_agent(model_client)
    team = create_selector_team(planner, researcher, analyst, validator, model_client)

    start_time = time.time()
    result = await Console(team.run_stream(task=task))
    elapsed = time.time() - start_time

    routing_order = [msg.source for msg in result.messages]
    agents_used = set(routing_order) - {"user"}
    agent_turns = sum(1 for src in routing_order if src != "user")
    tokens = count_tokens(result)

    await model_client.close()

    return {
        "total_messages": len(result.messages),
        "agent_turns": agent_turns,
        "stop_reason": result.stop_reason,
        "elapsed_time": round(elapsed, 2),
        "agents_used": agents_used,
        "routing_order": " → ".join(routing_order),
        **tokens,
    }


async def compare(task: str):
    print("Running single agent...")
    print("-" * 60)
    single = await run_single_agent(task)

    print("\n\nRunning multi-agent team...")
    print("-" * 60)
    multi = await run_multi_agent(task)

    print("\n")
    print("=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    rows = [
        ("Total messages", single["total_messages"], multi["total_messages"]),
        ("Agent turns", single["agent_turns"], multi["agent_turns"]),
        ("Elapsed time (s)", single["elapsed_time"], multi["elapsed_time"]),
        ("Prompt tokens", single["prompt_tokens"], multi["prompt_tokens"]),
        ("Completion tokens", single["completion_tokens"], multi["completion_tokens"]),
        ("Total tokens", single["total_tokens"], multi["total_tokens"]),
        ("Stop reason", single["stop_reason"], multi["stop_reason"]),
    ]

    print(f"{'Metric':<20} {'Single Agent':<20} {'Multi-Agent Team':<20}")
    print(f"{'-'*20} {'-'*20} {'-'*20}")
    for label, s_val, m_val in rows:
        print(f"{label:<20} {str(s_val):<20} {str(m_val):<20}")

    print(f"\nMulti-agent routing: {multi['routing_order']}")
    print(f"Agents used: {', '.join(multi['agents_used'])}")

# ─── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    task = (
        "Research the adoption rates of Python vs JavaScript for AI/ML projects. "
        "Find relevant statistics, calculate the growth rate difference between them, "
        "and provide a summary with your analysis."
    )

    print(f"\n{'='*60}")
    print("COMPARISON: Single Agent vs. Multi-Agent Team")
    print(f"{'='*60}\n")

    asyncio.run(compare(task))

    # ── After running, fill in single_vs_multi_comparison.md ──────────────
    # Use the template provided to document your observations.
    # Think about:
    # - Which produced better output? Why?
    # - Which was faster? Why?
    # - Which used more messages/tokens? Why?
    # - For what kinds of tasks would each approach be better?

    # ── Bonus challenges ──────────────────────────────────────────────────
    # 1. Try with a SIMPLE task: "What is 2 + 2?"
    #    Multi-agent is overkill here. How does it compare?
    #
    # 2. Try with a COMPLEX task requiring multiple research + calculation steps.
    #    Does multi-agent start to shine?
    #
    # 3. Run each approach 3 times. Do the results vary? How much?
    #    (LLMs are non-deterministic, so results will differ)
