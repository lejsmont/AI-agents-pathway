"""
Day 6 — Exercise 1: LangChain Agent (Rebuilding Day 4)
=======================================================

Goal: Rebuild your Day 4 robust agent using LangChain abstractions.
You'll see how LangChain's @tool, ChatAnthropic, and create_agent
replace the manual plumbing you wrote from scratch.

Mapping to your from-scratch code:
  Day 4 manual tool schema dict      → @tool decorator (auto-generates schema)
  Day 4 anthropic.Anthropic()        → ChatAnthropic() model wrapper
  Day 4 while loop + stop_reason     → create_agent() handles the loop
  Day 4 messages list                → MemorySaver checkpointer
  Day 4 token/cost tracking          → usage_metadata on AIMessage

IMPORTANT: As of LangGraph v1.0 (Oct 2025), `create_react_agent` from
`langgraph.prebuilt` is deprecated. Use `create_agent` from `langchain.agents`.
Docs: https://docs.langchain.com/oss/python/langchain/agents

Setup:
  pip install langchain langchain-anthropic langgraph

5 TODOs — work through them in order. The exploration harness at the top
lets you run and inspect intermediate values at each step.
"""

import os
import math
from typing import Optional

# ── Imports you'll need ──────────────────────────────────────────────────────
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

# ── Verify API key ───────────────────────────────────────────────────────────
assert os.environ.get("ANTHROPIC_API_KEY"), "Set ANTHROPIC_API_KEY env var"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  EXPLORATION HARNESS — Run this file to inspect intermediate values.       ║
# ║  Uncomment sections as you complete each TODO.                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def explore():
    """Run this to see how LangChain objects work before implementing TODOs."""

    print("=" * 60)
    print("EXPLORATION: LangChain Basics")
    print("=" * 60)

    # ── Step A: See how ChatAnthropic wraps the API ──────────────────────────
    # This is the same API you've been calling with anthropic.Anthropic(),
    # but wrapped in a LangChain-compatible interface.
    model = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=1024)

    # Basic invocation — notice it returns a LangChain Message object,
    # not a raw API response dict.
    response = model.invoke([HumanMessage(content="What is 2 + 2? One sentence.")])
    print(f"\n1. Model response type: {type(response)}")
    print(f"   Content: {response.content}")
    print(f"   Usage metadata: {response.usage_metadata}")
    # ^ Compare to Day 4: response.usage.input_tokens / output_tokens

    # ── Step B: See how @tool works ──────────────────────────────────────────
    # This is a FINISHED example tool — inspect its auto-generated schema.
    @tool
    def example_add(a: int, b: int) -> str:
        """Add two integers and return the result."""
        return str(a + b)

    print(f"\n2. Tool name: {example_add.name}")
    print(f"   Tool description: {example_add.description}")
    print(f"   Tool schema: {example_add.args_schema.model_json_schema()}")
    # ^ Compare to Day 3: you hand-wrote this schema dict!

    # ── Step C: See how bind_tools works ─────────────────────────────────────
    # This tells the model about available tools (like passing tools=[] in Day 3)
    model_with_tools = model.bind_tools([example_add])
    response = model_with_tools.invoke([HumanMessage(content="What is 17 + 25?")])
    print(f"\n3. Response with tools bound:")
    print(f"   Content: {response.content}")
    print(f"   Tool calls: {response.tool_calls}")
    # ^ Compare to Day 3: response.content[0].type == "tool_use"

    # ── Step D: See how MemorySaver works ────────────────────────────────────
    memory = MemorySaver()
    print(f"\n4. MemorySaver type: {type(memory)}")
    print(f"   This replaces your Day 4 messages=[] list.")
    print(f"   Pass it to create_agent(checkpointer=memory) and use")
    print(f"   thread_id in config to group messages into conversations.")

    # ── Step E: Uncomment after TODO 1-2 to test your tools ──────────────────
    tools = get_tools()
    for t in tools:
        print(f"\n5. Your tool: {t.name} -> {t.args_schema.model_json_schema()}")

    # ── Step F: Uncomment after TODO 3 to test the agent ─────────────────────
    agent = build_agent()
    result = agent.invoke(
        {"messages": [HumanMessage(content="What is sqrt(144)?")]}
    )
    print(f"\n6. Agent result: {result['messages'][-1].content}")

    # ── Step G: Uncomment after TODO 4-5 to test memory + tracking ───────────
    run_agent_with_memory("What is 15 * 23?", thread_id="test")
    run_agent_with_memory("Now divide that result by 5", thread_id="test")
    # ^ The second query should reference the first — memory at work!


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TODO 1: Define tools using @tool decorator                                ║
# ║                                                                            ║
# ║  Create two tools that mirror your Day 4 agent's capabilities:             ║
# ║  1. calculator(expression: str) -> str — evaluate math expressions         ║
# ║  2. lookup_constant(name: str) -> str — return math/science constants      ║
# ║                                                                            ║
# ║  Key insight: The @tool decorator reads your docstring to create the       ║
# ║  tool description and your type hints to create the parameter schema.      ║
# ║  Compare this to your Day 3 hand-written tool schemas!                     ║
# ║                                                                            ║
# ║  Supported constants: pi, e, golden_ratio, avogadro, speed_of_light       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# YOUR CODE for TODO 1:

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Args:
        expression: A valid Python math expression (e.g. '2 + 2', 'math.sqrt(16)')
    """
    # TODO 1a: Implement this tool.
    # - Use eval() with {"math": math} as allowed namespace
    # - Wrap in try/except and return error string on failure
    # - Return the result as a string
    try:
        return str(eval(expression, {"math": math}, {}))
    except Exception as e:
        return f"Error using calculator: {e}"



@tool
def lookup_constant(name: str) -> str:
    """Look up a mathematical or scientific constant by name.

    Args:
        name: The constant name (pi, e, golden_ratio, avogadro, speed_of_light)
    """
    # TODO 1b: Implement this tool.
    # - Create a dict of constants: pi, e, golden_ratio, avogadro, speed_of_light
    # - Return the value as a string, or an error message if not found
    # - Hint: math.pi, math.e, 1.618033988749895, 6.022e23, 299792458
    constants = {
        "pi": math.pi,
        "e": math.e,
        "golden_ratio": 1.618033988749895,
        "avogardo": 6.02214076,
        "speed_of_light": 299792458, 
    }
    if name in constants:
        return str(constants[name])
    return f"Error: unknown constant '{name}'"

def get_tools() -> list:
    """Return the list of tools for the agent."""
    return [calculator, lookup_constant]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TODO 2: Set up the ChatAnthropic model                                    ║
# ║                                                                            ║
# ║  Create a ChatAnthropic instance. Key parameters:                          ║
# ║  - model: "claude-sonnet-4-20250514"                                       ║
# ║  - max_tokens: 1024                                                        ║
# ║  - temperature: 0    (deterministic for tool-calling agents)               ║
# ║                                                                            ║
# ║  Compare to Day 4: client = anthropic.Anthropic()                          ║
# ║  In LangChain, the model object knows its own config.                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def get_model() -> ChatAnthropic:
    """Create and return the configured ChatAnthropic model."""
    # TODO 2: Create and return a ChatAnthropic instance.
    # Remember: you do NOT need to bind tools here — the agent builder does that.
    model = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens = 1024, temperature = 0)
    return model


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TODO 3: Build the agent using create_agent                                ║
# ║                                                                            ║
# ║  LangChain's create_agent replaces your Day 4 while loop.                 ║
# ║  It builds a graph that:                                                   ║
# ║  1. Calls the model                                                        ║
# ║  2. If model wants to use tools → calls them → loops back to 1             ║
# ║  3. If model is done → returns the result                                  ║
# ║                                                                            ║
# ║  This is exactly your Day 4 loop:                                          ║
# ║    while stop_reason == "tool_use": ...                                    ║
# ║                                                                            ║
# ║  create_agent(model, tools=tools) returns a compiled LangGraph.            ║
# ║  You invoke it with: agent.invoke({"messages": [HumanMessage(...)]})       ║
# ║                                                                            ║
# ║  Note: As of LangGraph v1.0, use `from langchain.agents import            ║
# ║  create_agent` — NOT the old `langgraph.prebuilt.create_react_agent`.      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def build_agent():
    """Build and return an agent using create_agent."""
    model = get_model()
    tools = get_tools()
    # TODO 3: Use create_agent to build the agent.
    #   agent = create_agent(model, tools=tools)
    #   return agent
    # That's it! Compare this to your 50+ line Day 4 agent loop.
    agent = create_agent(model, tools=tools)
    return agent


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TODO 4: Add conversation memory                                           ║
# ║                                                                            ║
# ║  In Day 4, you manually appended to a messages list.                       ║
# ║  create_agent can accept a checkpointer for memory.                        ║
# ║                                                                            ║
# ║  Steps:                                                                    ║
# ║  1. Create: memory = MemorySaver()                                         ║
# ║     (already imported at the top of this file)                             ║
# ║  2. Pass to create_agent: create_agent(model, tools=tools,                 ║
# ║                                   checkpointer=memory)                     ║
# ║  3. When invoking, pass a config with thread_id:                           ║
# ║     agent.invoke(                                                          ║
# ║         {"messages": [HumanMessage(content=query)]},                       ║
# ║         config={"configurable": {"thread_id": "my-thread"}}                ║
# ║     )                                                                      ║
# ║                                                                            ║
# ║  The thread_id groups messages into a conversation.                        ║
# ║  Same thread_id = same conversation history.                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def build_agent_with_memory():
    """Build an agent with conversation memory."""
    model = get_model()
    tools = get_tools()
    # TODO 4: Build the agent with a MemorySaver checkpointer.
    memory = MemorySaver()
    agent = create_agent(model, tools=tools, checkpointer=memory)
    return agent
    

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TODO 5: Add token/cost tracking                                           ║
# ║                                                                            ║
# ║  In Day 4 you tracked tokens manually from response.usage.                 ║
# ║  In LangChain, the usage_metadata is on each AIMessage in the result.      ║
# ║                                                                            ║
# ║  After agent.invoke(), the result["messages"] list contains all messages.  ║
# ║  Each AIMessage has .usage_metadata with input_tokens, output_tokens.      ║
# ║                                                                            ║
# ║  Steps:                                                                    ║
# ║  1. Run the agent and get the result                                       ║
# ║  2. Loop through result["messages"]                                        ║
# ║  3. For each AIMessage, extract usage_metadata                             ║
# ║  4. Sum up total tokens and estimate cost                                  ║
# ║     (Sonnet: ~$3/M input, ~$15/M output tokens)                           ║
# ║                                                                            ║
# ║  Compare to Day 4: you did this in the while loop after each API call.     ║
# ║  Here you do it after the full run — a trade-off of the framework.         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Pricing per million tokens (Sonnet)
INPUT_COST_PER_M = 3.0
OUTPUT_COST_PER_M = 15.0


def run_agent_with_tracking(query: str) -> dict:
    """Run the agent and return result with token/cost tracking."""
    agent = build_agent()
    result = agent.invoke({"messages": [HumanMessage(content=query)]})

    # TODO 5: Extract token usage from the result messages.
    total_input = 0
    total_output = 0
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.usage_metadata:
            total_input += msg.usage_metadata.get("input_tokens", 0)
            total_output += msg.usage_metadata.get("output_tokens", 0)
    
    cost = (total_input / 1_000_000 * INPUT_COST_PER_M +
            total_output / 1_000_000 * OUTPUT_COST_PER_M)
    
    # Print a summary:
    #   - Final answer (last message content)
    #   - Number of messages (i.e. how many steps the agent took)
    #   - Total input/output tokens
    #   - Estimated cost
    #
    return {"answer": result["messages"][-1].content,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "cost": cost,
            "num_steps": len(result["messages"])}
    

def run_agent_with_memory(query: str, thread_id: str = "default-thread"):
    """Run the agent with memory enabled — multi-turn conversations."""
    agent = build_agent_with_memory()
    if agent is None:
        print("⚠️  Complete TODO 4 first.")
        return

    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    print(f"\nQuery: {query}")
    print(f"Answer: {result['messages'][-1].content}")
    return result


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN — run the exploration harness or test your implementations           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    # # Phase 1: Explore LangChain basics (run this first!)
    # explore()

    # # Phase 2: Uncomment after completing TODOs 1-3
    # print("\n" + "=" * 60)
    # print("TESTING: Agent with tools")
    # print("=" * 60)
    # agent = build_agent()
    # result = agent.invoke(
    #     {"messages": [HumanMessage(content="What is pi times the speed of light?")]}
    # )
    # for msg in result["messages"]:
    #     print(f"  {msg.type}: {msg.content[:100] if msg.content else '[tool call]'}...")

    # # # Phase 3: Uncomment after completing TODO 5
    # print("\n" + "=" * 60)
    # print("TESTING: Agent with token tracking")
    # print("=" * 60)
    # stats = run_agent_with_tracking("What is e raised to the power of pi?")
    # print(f"  Stats: {stats}")

    # Phase 4: Uncomment after completing TODO 4
    print("\n" + "=" * 60)
    print("TESTING: Agent with memory (multi-turn)")
    print("=" * 60)
    # run_agent_with_memory("What is 15 * 23?", thread_id="math-session")
    # run_agent_with_memory("Now take the square root of that result", thread_id="math-session")
    agent = build_agent_with_memory()
    query = "What is 15 * 23?"
    result = agent.invoke({"messages": [HumanMessage(content=query)]},config={"configurable": {"thread_id": "test"}})
    print(f"\nQuery: {query}")
    print(f"Answer: {result['messages'][-1].content}")
    query = "Now take the square root of that result"
    result = agent.invoke({"messages": [HumanMessage(content=query)]},config={"configurable": {"thread_id": "test"}})
    print(f"\nQuery: {query}")
    print(f"Answer: {result['messages'][-1].content}")
    
