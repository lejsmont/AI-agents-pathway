"""
Day 6 — Exercise 3: Planner → Executor → Validator Graph
=========================================================

Goal: Build a full LangGraph agent with three distinct nodes:
  1. Planner  — breaks user query into numbered steps
  2. Executor — uses tools (calculator, lookup) to execute the plan
  3. Validator — checks the result quality with structured output

This combines EVERYTHING from Days 2-5 into a LangGraph architecture:
  Day 2 agent loop         → LangGraph state machine
  Day 3 tool calling       → Executor node with bound tools
  Day 4 retry/fallback     → Conditional edges for retry loops
  Day 5 structured output  → Validator node with Pydantic-like scoring

This is also the pattern Databricks uses for production agents:
  LangGraph graph → MLflow logging → serving endpoint

Setup:
  pip install langchain langchain-anthropic langgraph

5 TODOs — each builds one piece of the pipeline.
"""

import os
import re
import math
import json
from typing import TypedDict, Annotated, Literal

from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

assert os.environ.get("ANTHROPIC_API_KEY"), "Set ANTHROPIC_API_KEY env var"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TOOLS — same as Exercise 1 (provided, no TODO)                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Args:
        expression: A valid Python math expression (e.g. '2 + 2', 'math.sqrt(16)')
    """
    try:
        result = eval(expression, {"__builtins__": {}, "math": math})
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def lookup_constant(name: str) -> str:
    """Look up a mathematical or scientific constant by name.

    Args:
        name: The constant name (pi, e, golden_ratio, avogadro, speed_of_light)
    """
    constants = {
        "pi": math.pi,
        "e": math.e,
        "golden_ratio": 1.618033988749895,
        "avogadro": 6.022e23,
        "speed_of_light": 299792458,
    }
    name_lower = name.lower().replace(" ", "_")
    if name_lower in constants:
        return f"{name} = {constants[name_lower]}"
    return f"Unknown constant '{name}'. Available: {', '.join(constants.keys())}"


TOOLS = [calculator, lookup_constant]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODELS                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

planner_model = ChatAnthropic(
    model="claude-sonnet-4-20250514", max_tokens=1024, temperature=0
)
# Executor model has tools bound (like Day 3 tools=[...])
executor_model = ChatAnthropic(
    model="claude-sonnet-4-20250514", max_tokens=1024, temperature=0
).bind_tools(TOOLS)

validator_model = ChatAnthropic(
    model="claude-sonnet-4-20250514", max_tokens=1024, temperature=0
)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  EXPLORATION HARNESS                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def explore():
    """Explore how the pieces work before building the full graph."""

    print("=" * 60)
    print("EXPLORATION: Planner-Executor-Validator Components")
    print("=" * 60)

    # ── Step A: See what the planner produces ────────────────────────────────
    plan_prompt = [
        SystemMessage(content=(
            "You are a planning agent. Given a user query, break it into "
            "numbered steps that can be executed with a calculator tool and "
            "a constant lookup tool. Output ONLY the numbered steps, nothing else."
        )),
        HumanMessage(content="What is pi squared plus e?"),
    ]
    plan_response = planner_model.invoke(plan_prompt)
    print(f"\n1. Planner output:\n{plan_response.content}")

    # ── Step B: See what the executor does with tools ────────────────────────
    exec_prompt = [
        SystemMessage(content=(
            "You are an executor agent. Follow the plan below step by step. "
            "Use the calculator and lookup_constant tools as needed. "
            "After completing all steps, summarize the final answer."
        )),
        HumanMessage(content=f"Plan:\n{plan_response.content}\n\nExecute this plan now."),
    ]
    exec_response = executor_model.invoke(exec_prompt)
    print(f"\n2. Executor response type: {type(exec_response)}")
    print(f"   Has tool calls: {bool(exec_response.tool_calls)}")
    if exec_response.tool_calls:
        for tc in exec_response.tool_calls:
            print(f"   Tool call: {tc['name']}({tc['args']})")

    # ── Step C: See how tool results come back ───────────────────────────────
    # This is LangGraph's tool execution pattern — same concept as Day 3
    # but using LangChain's ToolMessage format.
    if exec_response.tool_calls:
        tc = exec_response.tool_calls[0]
        # Find and call the tool
        tool_map = {t.name: t for t in TOOLS}
        tool_result = tool_map[tc["name"]].invoke(tc["args"])
        print(f"\n3. Tool result: {tool_result}")

        # Create a ToolMessage (like Day 3 tool_result role)
        tool_msg = ToolMessage(content=str(tool_result), tool_call_id=tc["id"])
        print(f"   ToolMessage: {tool_msg}")

    # ── Step D: See what validation looks like ───────────────────────────────
    validation_prompt = [
        SystemMessage(content=(
            "You are a result validator. Given a query and an answer, evaluate:\n"
            "1. Is the answer complete? (addresses all parts of the query)\n"
            "2. Is the answer correct? (math checks out, facts are right)\n"
            "3. Is the answer clear? (well-explained, not confusing)\n\n"
            "Respond in this EXACT format (nothing else):\n"
            "COMPLETE: yes/no\n"
            "CORRECT: yes/no\n"
            "CLEAR: yes/no\n"
            "SCORE: 1-10\n"
            "FEEDBACK: one sentence of feedback"
        )),
        HumanMessage(content=(
            "Query: What is pi squared plus e?\n"
            "Answer: pi^2 + e = 9.8696 + 2.7183 = 12.5879"
        )),
    ]
    val_response = validator_model.invoke(validation_prompt)
    print(f"\n4. Validator output:\n{val_response.content}")

    # ── Step E: Uncomment after TODO 1-5 ─────────────────────────────────────
    pipeline = build_planner_executor_validator()
    result = pipeline.invoke({
        "messages": [HumanMessage(content="What is the golden ratio cubed minus pi?")],
        "plan": "",
        "executor_result": "",
        "validation_score": 0,
        "validation_feedback": "",
        "iteration": 0,
    })
    print(f"\n5. Full pipeline result:")
    print(f"   Plan: {result['plan'][:200]}")
    print(f"   Result: {result['executor_result'][:200]}")
    print(f"   Score: {result['validation_score']}/10")
    print(f"   Iterations: {result['iteration']}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TODO 1: Define the pipeline state                                         ║
# ║                                                                            ║
# ║  Create a TypedDict called PlannerState with:                              ║
# ║  - messages: Annotated[list, add_messages]  (full message history)         ║
# ║  - plan: str                     (the planner's step-by-step plan)         ║
# ║  - executor_result: str          (the executor's final answer)             ║
# ║  - validation_score: int         (0-10 quality score)                      ║
# ║  - validation_feedback: str      (why the score is what it is)             ║
# ║  - iteration: int                (retry counter)                           ║
# ║                                                                            ║
# ║  This state flows through ALL nodes. Each node reads what it needs         ║
# ║  and returns a partial update with the keys it changes.                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# TODO 1: Define PlannerState
# class PlannerState(TypedDict):
#     messages: ...
#     plan: ...
#     executor_result: ...
#     validation_score: ...
#     validation_feedback: ...
#     iteration: ...

class PlannerState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: str
    executor_result: str
    validation_score: int
    validation_feedback: str
    iteration: int


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TODO 2: Build the planner node                                            ║
# ║                                                                            ║
# ║  The planner takes the user query and produces a numbered plan.            ║
# ║                                                                            ║
# ║  Steps:                                                                    ║
# ║  1. Create a system message explaining the planner's role and available    ║
# ║     tools (calculator, lookup_constant)                                    ║
# ║  2. Include the user's messages from state                                 ║
# ║  3. If this is a RETRY (iteration > 0), add the validation feedback       ║
# ║     so the planner can adjust its plan                                     ║
# ║  4. Call planner_model.invoke()                                            ║
# ║  5. Return {"plan": response.content, "iteration": state["iteration"] + 1}║
# ║                                                                            ║
# ║  Design choice: The planner does NOT call tools — it just makes a plan.   ║
# ║  The executor does the actual tool calling. This separation is the key     ║
# ║  insight of the Planner-Executor pattern.                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def planner_node(state: PlannerState) -> dict:
    """Break the user query into executable steps."""
    print(f"\n  [Planner] Creating plan (iteration {state['iteration'] + 1})...")

    # TODO 2: Implement the planner node.
    #
    system_msg = SystemMessage(content=(
        "You are a planning agent. Given a user query, break it into clear "
        "numbered steps. You have two tools available:\n"
        "- calculator: evaluates math expressions (Python syntax, math module available)\n"
        "- lookup_constant: looks up constants (pi, e, golden_ratio, avogadro, speed_of_light)\n\n"
        "Output ONLY the numbered steps. Be specific about what to compute in each step."
    ))
    
    prompt = [system_msg] + list(state["messages"])
    #
    # # On retry, add feedback from the validator
    if state["iteration"] > 0 and state["validation_feedback"]:
        prompt.append(HumanMessage(content=(
            f"Your previous plan led to a score of {state['validation_score']}/10. "
            f"Feedback: {state['validation_feedback']}\n"
            f"Please create an improved plan."
        )))
    
    response = planner_model.invoke(prompt)
    print(f"  [Planner] Plan:\n{response.content}")
    #
    return {
        "plan": response.content,
        "iteration": state["iteration"] + 1,
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TODO 3: Build the executor node                                           ║
# ║                                                                            ║
# ║  The executor takes the plan and executes it using tools.                  ║
# ║  This is a mini agent loop INSIDE a graph node — it calls the LLM,        ║
# ║  handles tool calls, and loops until the LLM is done.                      ║
# ║                                                                            ║
# ║  Steps:                                                                    ║
# ║  1. Create a prompt with the plan and instruction to execute it            ║
# ║  2. Call executor_model.invoke() (tools are already bound)                 ║
# ║  3. While the response has tool_calls:                                     ║
# ║     a. Execute each tool call                                              ║
# ║     b. Create ToolMessage(s) with results                                  ║
# ║     c. Append to messages and call the model again                         ║
# ║  4. Return {"executor_result": final_response.content}                     ║
# ║                                                                            ║
# ║  This is your Day 3-4 tool loop, but inside a LangGraph node.             ║
# ║  Max 10 tool-call iterations to prevent infinite loops (Day 4 lesson).     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

MAX_TOOL_ITERATIONS = 10

def executor_node(state: PlannerState) -> dict:
    """Execute the plan using tools."""
    print(f"\n  [Executor] Executing plan...")

    tool_map = {t.name: t for t in TOOLS}

    # TODO 3: Implement the executor node.
    #
    exec_messages = [
        SystemMessage(content=(
            "You are an executor agent. Follow the plan below step by step. "
            "Use the calculator and lookup_constant tools to complete each step. "
            "After completing ALL steps, provide a clear final answer summarizing "
            "the complete result with all computed values."
        )),
        HumanMessage(content=f"Plan:\n{state['plan']}\n\nExecute this plan now."),
    ]
    #
    for i in range(MAX_TOOL_ITERATIONS):
        response = executor_model.invoke(exec_messages)
        exec_messages.append(response)
    
        # If no tool calls, the executor is done
        if not response.tool_calls:
            print(f"  [Executor] Done after {i + 1} LLM call(s)")
            break
    
        # Execute each tool call and create ToolMessages
        for tc in response.tool_calls:
            print(f"  [Executor] Calling {tc['name']}({tc['args']})")
            tool_fn = tool_map.get(tc["name"])
            if tool_fn:
                result = tool_fn.invoke(tc["args"])
            else:
                result = f"Unknown tool: {tc['name']}"
            exec_messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )
    
    return {"executor_result": response.content}



# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TODO 4: Build the validator node                                          ║
# ║                                                                            ║
# ║  The validator checks the executor's result for quality.                   ║
# ║  It uses structured-output-style prompting (Day 5 concept) to get         ║
# ║  a score and feedback.                                                     ║
# ║                                                                            ║
# ║  Steps:                                                                    ║
# ║  1. Prompt the LLM with the original query and the executor's result      ║
# ║  2. Ask it to evaluate completeness, correctness, clarity                  ║
# ║  3. Parse the SCORE and FEEDBACK from the response                         ║
# ║  4. Return {"validation_score": score, "validation_feedback": feedback}    ║
# ║                                                                            ║
# ║  Response format to request:                                               ║
# ║    SCORE: 1-10                                                             ║
# ║    FEEDBACK: one sentence                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def validator_node(state: PlannerState) -> dict:
    """Validate the executor's result."""
    print(f"\n  [Validator] Checking result quality...")

    # TODO 4: Implement the validator node.
    #
    # Get the original user query from messages
    user_query = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break
    
    validation_prompt = [
        SystemMessage(content=(
            "You are a result validator. Given a query and an answer, evaluate "
            "whether the answer is complete, correct, and clear.\n\n"
            "Respond in this EXACT format (nothing else):\n"
            "SCORE: <1-10>\n"
            "FEEDBACK: <one sentence explaining the score>"
        )),
        HumanMessage(content=(
            f"Original query: {user_query}\n\n"
            f"Answer provided: {state['executor_result']}"
        )),
    ]
    
    response = validator_model.invoke(validation_prompt)
    print(f"  [Validator] Raw response: {response.content}")
    
    # Parse score
    score_match = re.search(r'SCORE:\s*(\d+)', response.content)
    score = int(score_match.group(1)) if score_match else 5
    score = max(1, min(10, score))
    
    # Parse feedback
    feedback_match = re.search(r'FEEDBACK:\s*(.+)', response.content)
    feedback = feedback_match.group(1).strip() if feedback_match else "No feedback"
    
    print(f"  [Validator] Score: {score}/10 | Feedback: {feedback}")
    
    return {
        "validation_score": score,
        "validation_feedback": feedback,
    }



# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TODO 5: Wire up the complete graph                                        ║
# ║                                                                            ║
# ║  Build the full Planner → Executor → Validator pipeline with retry.       ║
# ║                                                                            ║
# ║  Graph structure:                                                          ║
# ║    START → planner → executor → validator → [accept/retry]                 ║
# ║                ↑                                    |                      ║
# ║                └────────────────────────────────────┘ (retry)              ║
# ║                                                                            ║
# ║  Steps:                                                                    ║
# ║  1. Create StateGraph(PlannerState)                                        ║
# ║  2. Add three nodes: planner, executor, validator                          ║
# ║  3. Add edges: START→planner, planner→executor, executor→validator        ║
# ║  4. Create a routing function for after validation:                        ║
# ║     - If score >= 7 or iteration >= 3 → END                               ║
# ║     - Else → back to planner (retry with feedback)                         ║
# ║  5. Add conditional edge from validator using the routing function         ║
# ║  6. Compile and return                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

QUALITY_THRESHOLD = 7
MAX_ITERATIONS = 3


def route_after_validation(state: PlannerState) -> Literal["accept", "retry"]:
    """Decide whether to accept the result or retry the pipeline."""
    # TODO 5a: Implement routing logic (same pattern as Exercise 2 TODO 4)
    # - If validation_score >= QUALITY_THRESHOLD → "accept"
    # - If iteration >= MAX_ITERATIONS → "accept" (graceful degradation)
    # - Otherwise → "retry"
    if state["validation_score"] >= QUALITY_THRESHOLD:
        decision = "accept"
    elif state["iteration"] >= MAX_ITERATIONS:
        decision = "accept"
    else:
        decision = "retry"
    
    return decision


def build_planner_executor_validator():
    """Build the complete Planner → Executor → Validator graph."""
    # TODO 5b: Build the graph.
    
    graph = StateGraph(PlannerState)
    
    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("validator", validator_node)
    
    # Add edges
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "validator")
    
    # Conditional edge: retry or accept
    graph.add_conditional_edges("validator", route_after_validation, {
        "accept": END,
        "retry": "planner",  # Go back to planner with feedback!
    })
    
    return graph.compile()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  COMPARISON HELPER — see the difference from Day 4                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def print_framework_comparison():
    """Print a comparison of from-scratch vs LangGraph approaches."""
    print("\n" + "=" * 60)
    print("COMPARISON: From-Scratch (Day 4) vs LangGraph (Day 6)")
    print("=" * 60)

    comparison = """
    From-Scratch (Day 4)              LangGraph (Day 6)
    ─────────────────────             ─────────────────────
    while loop + stop_reason    →     StateGraph + edges
    messages.append()           →     add_messages reducer
    if/elif routing             →     conditional_edges
    try/except + tenacity       →     retry via graph loop
    manual tool dispatch        →     bind_tools + ToolMessage
    cost tracking in loop       →     usage_metadata on AIMessage
    single file, ~100 lines     →     modular nodes, ~200 lines

    Pros of LangGraph:
    + Clear separation of concerns (planner/executor/validator)
    + Easy to visualize and reason about the flow
    + Built-in state management and message handling
    + Native Databricks integration
    + Easy to add new nodes without rewriting the loop

    Cons of LangGraph:
    - More abstraction to learn
    - Harder to do custom token tracking (framework handles messages)
    - Debugging requires understanding graph execution
    - Overhead for simple single-loop agents (Day 4 is simpler)

    Note: For simple ReAct agents, use `create_agent` from `langchain.agents`
    (high-level). Use raw LangGraph StateGraph when you need custom graph
    topology like this Planner→Executor→Validator pipeline.
    """
    print(comparison)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    # # Phase 1: Explore components
    # explore()

    # # Phase 2: Print comparison
    # print_framework_comparison()

    # Phase 3: Uncomment after all TODOs complete
    print("\n" + "=" * 60)
    print("TESTING: Full Planner → Executor → Validator Pipeline")
    print("=" * 60)
    
    test_queries = [
        "What is pi squared plus the golden ratio?",
        "Calculate e raised to the power of 3, then subtract the speed of light divided by avogadro's number",
        "What is the square root of (pi * e * golden_ratio)?",
    ]
    
    for query in test_queries:
        print(f"\n{'─' * 50}")
        print(f"Query: {query}")
        print('─' * 50)
    
        pipeline = build_planner_executor_validator()
        result = pipeline.invoke({
            "messages": [HumanMessage(content=query)],
            "plan": "",
            "executor_result": "",
            "validation_score": 0,
            "validation_feedback": "",
            "iteration": 0,
        })
    
        print(f"\n  Final answer: {result['executor_result'][:300]}")
        print(f"  Validation: {result['validation_score']}/10")
        print(f"  Iterations: {result['iteration']}")
