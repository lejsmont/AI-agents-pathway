"""
Day 7 — Exercise 2: Planner + Researcher + Analyst with SelectorGroupChat
=========================================================================

Goal: Build a 3-agent team where an LLM dynamically selects the next speaker.
A Planner breaks tasks into steps, a Researcher searches for info, and an
Analyst does calculations. The SelectorGroupChat manager picks who goes next
based on conversation context.

Concepts practiced:
    - SelectorGroupChat: LLM-driven dynamic agent selection
    - selector_prompt: customizing how the manager picks the next agent
    - Agent descriptions: how the selector uses them to decide routing
    - Tools on agents: giving different agents different capabilities
    - Multi-step task decomposition across agents
    - Comparing structured routing vs. fixed round-robin

Run:
    python exercise2_selector_team.py
"""

import asyncio
import json
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
# ─── Exploration: understand SelectorGroupChat ────────────────────────────
# SelectorGroupChat is fundamentally different from RoundRobinGroupChat:
# - RoundRobin: fixed turn order (A → B → A → B → ...)
# - Selector: an LLM reads the conversation and picks who speaks next
#
# The selector uses two things to decide:
# 1. Each agent's `description` parameter
# 2. A `selector_prompt` template with {roles}, {history}, {participants}

# --- Block: See how selector_prompt variables work ---
# The default selector_prompt uses these variables:
#   {roles}        → "Agent1: description1\nAgent2: description2\n..."
#   {history}      → the full conversation so far
#   {participants} → '["Agent1", "Agent2", "Agent3"]'
# You can customize the prompt to guide selection behavior.


# ─── Mock tools for agents ─────────────────────────────────────────────────

# TODO 1: Create tool functions for the Researcher agent.
# These are async functions that simulate web search and data lookup.
# AutoGen auto-generates the tool schema from the function signature + docstring.
#
# Create two tools:
#
# a) search_web(query: str) -> str
#    - Docstring: "Search the web for information on a topic."
#    - Returns a mock result string with some fake but plausible data
#    - Example: for a query about "Python AI adoption", return a string with
#      fake stats like "According to recent surveys, Python is used by 70% of..."
#
# b) lookup_documentation(topic: str) -> str
#    - Docstring: "Look up technical documentation or API references."
#    - Returns a mock documentation snippet
#
# Hint: Tools MUST be async functions in AutoGen 0.4.
# Hint: Type annotations on ALL parameters and return type are required.
# Hint: The docstring becomes the tool's description for the LLM.

async def search_web(query: str) -> str:
    """Search the web for information on a topic"""
    return "According to recent surveys, Python is used by 70% of ML engineers and Javascript is used by 30% of ML engineers. Python usage is growing 3% YOY over last decade and Javascript usage is flat"

async def lookup_documentation(topic: str) -> str:
    """Look up technical documentation or API refereces."""
    return "Some technical documentation"

# TODO 2: Create tool functions for the Analyst agent.
#
# a) calculate(expression: str) -> str
#    - Docstring: "Calculate a mathematical expression. Input should be a valid Python math expression."
#    - Use eval() on the expression (it's a learning exercise, not production code)
#    - Wrap in try/except and return error message if expression is invalid
#
# b) compare_metrics(metric_a: str, value_a: float, metric_b: str, value_b: float) -> str
#    - Docstring: "Compare two metrics and provide analysis."
#    - Calculate percentage difference between the two values
#    - Return a formatted comparison string
#
# Hint: Return strings, not numbers — AutoGen converts tool output to string messages.

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


# ─── Agent and team creation ──────────────────────────────────────────────

def create_model_client():
    """Create the model client.

    TODO 3: Same as Exercise 1 — create your model client.
    You'll use the same client for all agents AND for the SelectorGroupChat manager.
    """
    return AnthropicChatCompletionClient(model="claude-sonnet-4-20250514")


def create_planner_agent(model_client):
    """Create the Planner agent.

    TODO 4: Create an AssistantAgent with:
    - name: "Planner"
    - model_client: the shared client
    - description: A short description the selector uses to decide when to route to
      this agent. E.g., "Plans and coordinates tasks, breaks complex queries into steps,
      and determines when the task is complete."
    - system_message: Detailed instructions. The planner should:
        1. Break the task into numbered steps
        2. Assign each step to either "Researcher" or "Analyst" by name
        3. After seeing all results, synthesize a final answer
        4. Say "TERMINATE" when the task is fully complete
    - No tools — the planner only thinks and coordinates

    Hint: The `description` is for the SELECTOR (short, role-focused).
    The `system_message` is for the AGENT ITSELF (detailed instructions).
    These serve different purposes — don't confuse them!
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

    TODO 5: Create an AssistantAgent with:
    - name: "Researcher"
    - model_client: the shared client
    - description: Short role description for the selector.
      E.g., "Searches for information and looks up documentation."
    - system_message: Instruct it to use its tools to find information,
      report findings clearly, and never make up data.
    - tools: [search_web, lookup_documentation] — the functions you created in TODO 1
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

    TODO 6: Create an AssistantAgent with:
    - name: "Analyst"
    - description: Short role description for the selector.
    - system_message: Instruct it to perform calculations and comparisons,
      always show its work, and present results clearly.
    - tools: [calculate, compare_metrics] — the functions you created in TODO 2
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

def create_selector_team(planner, researcher, analyst, model_client):
    """Create a SelectorGroupChat team.

    TODO 7: Create a SelectorGroupChat with:
    - participants: [planner, researcher, analyst]
    - model_client: the same client (used by the selector to pick next speaker)
    - termination_condition: Combine:
        a) TextMentionTermination("TERMINATE")
        b) MaxMessageTermination(15)  # more headroom for multi-step tasks
    - selector_prompt: Customize the prompt that guides agent selection.
      The template has access to {roles}, {history}, and {participants}.

    Here's a good selector_prompt to use:

        "Select an agent to perform the next task.\n\n"
        "{roles}\n\n"
        "Current conversation context:\n{history}\n\n"
        "Read the above conversation. Then select an agent from {participants} "
        "to perform the next task.\n"
        "Rules:\n"
        "- The Planner should go first to create a plan.\n"
        "- After the plan, select Researcher or Analyst based on the current step.\n"
        "- Return to Planner after all steps are done to synthesize.\n"
        "- Only select one agent.\n"

    Hint: SelectorGroupChat requires at least 2 participants.
    Hint: The model_client here powers the SELECTOR (the routing decision),
    not the agents themselves (they have their own model_client).
    """
    return SelectorGroupChat(
        [planner, researcher, analyst], 
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


async def run_selector_team(task: str):
    """Run the multi-agent team on a task.

    TODO 8: Wire it all together:
    1. Create model client
    2. Create all three agents
    3. Create the selector team
    4. Run with Console: await Console(team.run_stream(task=task))
    5. After completion, analyze the result:
       - Print which agents spoke and in what order
       - Count messages per agent
       - Print the stop reason
    6. Close the model client

    Hint: Each message in result.messages has a `.source` attribute
    (the agent name) you can use to track who spoke.
    """
    model_client = create_model_client()
    planner = create_planner_agent(model_client)
    researcher = create_researcher_agent(model_client)
    analyst = create_analyst_agent(model_client)
    selector_team = create_selector_team(planner, researcher, analyst, model_client)

    result = await Console(selector_team.run_stream(task=task))

    # Routing order
    routing = [msg.source for msg in result.messages]
    print(f"\nRouting order: {' → '.join(routing)}")

    for msg in result.messages:
        msg_type = type(msg).__name__
        preview = str(msg.content)[:120] if hasattr(msg, 'content') else ""
        print(f"  [{msg.source}] ({msg_type}): {preview}")

    # Messages per agent
    from collections import Counter
    counts = Counter(routing)
    for agent, count in counts.items():
        print(f"  {agent}: {count} messages")

    print(f"Stop reason: {result.stop_reason}")

    await model_client.close()



# ─── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    task = (
        "Research the adoption rates of Python vs JavaScript for AI/ML projects. "
        "Find relevant statistics, calculate the growth rate difference between them, "
        "and provide a summary with your analysis."
    )
    print(f"\n{'='*60}")
    print("SELECTOR PATTERN: Planner + Researcher + Analyst")
    print(f"{'='*60}")
    print(f"Task: {task}\n")

    asyncio.run(run_selector_team(task))

    # ── Bonus challenges ──────────────────────────────────────────────────
    # 1. Add a "Validator" agent that checks the final answer for accuracy.
    #    Update the selector_prompt to route to Validator before TERMINATE.
    #
    # 2. Try setting allow_repeated_speaker=False on SelectorGroupChat.
    #    How does this change the routing behavior?
    #
    # 3. Replace the LLM-based selector with a custom selector_func:
    #    selector_func takes the conversation history and returns the next
    #    agent name. Implement simple rule-based routing.
    #    Example:
    #      def my_selector(messages):
    #          if len(messages) <= 1:
    #              return "Planner"
    #          last = messages[-1]
    #          if "search" in last.content.lower():
    #              return "Researcher"
    #          ...
    #
    # 4. Try a different task that requires more back-and-forth between
    #    Researcher and Analyst. Does the selector route correctly?
