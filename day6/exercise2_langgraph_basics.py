"""
Day 6 — Exercise 2: LangGraph Basics
======================================

Goal: Learn LangGraph fundamentals by building a simple research pipeline
as a state machine. No LLM calls yet — just pure graph mechanics so you
understand state, nodes, edges, and conditional routing.

Then you'll add a real LLM call to see how it fits into the graph.

This maps to concepts you already know:
  Day 4 agent loop          → graph with conditional edges (loop back or stop)
  Day 4 messages list       → graph state with add_messages reducer
  Day 4 retry logic         → conditional edge that routes to retry node
  Day 5 validation-retry    → validator node with conditional edge

Setup:
  pip install langchain langchain-anthropic langgraph

5 TODOs — work through them in order.
"""

import os
from typing import TypedDict, Annotated, Literal
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

assert os.environ.get("ANTHROPIC_API_KEY"), "Set ANTHROPIC_API_KEY env var"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  EXPLORATION HARNESS — Run this to understand LangGraph mechanics          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def explore():
    """Explore LangGraph concepts before implementing TODOs."""

    print("=" * 60)
    print("EXPLORATION: LangGraph Fundamentals")
    print("=" * 60)

    # ── Step A: Understand TypedDict state ───────────────────────────────────
    # In LangGraph, state is a TypedDict. Each key is a piece of data that
    # flows through the graph.
    class SimpleState(TypedDict):
        query: str
        result: str
        step_count: int

    # You can create state like a regular dict:
    initial = SimpleState(query="test", result="", step_count=0)
    print(f"\n1. Simple state: {initial}")

    # ── Step B: Understand add_messages reducer ──────────────────────────────
    # The Annotated[list, add_messages] pattern tells LangGraph to APPEND
    # new messages to the list instead of replacing it.
    class MessageState(TypedDict):
        messages: Annotated[list, add_messages]

    # Simulating what add_messages does:
    msgs1 = [HumanMessage(content="Hello")]
    msgs2 = [AIMessage(content="Hi!")]
    combined = add_messages(msgs1, msgs2)
    print(f"\n2. add_messages combines: {[m.content for m in combined]}")
    # Without add_messages, returning {"messages": msgs2} would REPLACE msgs1

    # ── Step C: Build a minimal graph (no LLM) ──────────────────────────────
    class CounterState(TypedDict):
        count: int
        log: list[str]

    def increment(state: CounterState) -> dict:
        new_count = state["count"] + 1
        return {
            "count": new_count,
            "log": state["log"] + [f"incremented to {new_count}"],
        }

    def double(state: CounterState) -> dict:
        new_count = state["count"] * 2
        return {
            "count": new_count,
            "log": state["log"] + [f"doubled to {new_count}"],
        }

    graph = StateGraph(CounterState)
    graph.add_node("increment", increment)
    graph.add_node("double", double)
    graph.add_edge(START, "increment")    # Start → increment
    graph.add_edge("increment", "double")  # increment → double
    graph.add_edge("double", END)          # double → END

    app = graph.compile()
    result = app.invoke({"count": 3, "log": []})
    print(f"\n3. Counter graph result: count={result['count']}, log={result['log']}")
    # 3 → increment → 4 → double → 8

    # ── Step D: Conditional edge ─────────────────────────────────────────────
    def check_threshold(state: CounterState) -> Literal["increment", "done"]:
        """Route based on count value."""
        if state["count"] >= 10:
            return "done"
        return "increment"

    def done_node(state: CounterState) -> dict:
        return {"log": state["log"] + ["finished!"]}

    graph2 = StateGraph(CounterState)
    graph2.add_node("increment", increment)
    graph2.add_node("done", done_node)
    graph2.add_edge(START, "increment")
    # Conditional: after increment, check if we should continue or stop
    graph2.add_conditional_edges("increment", check_threshold, {
        "increment": "increment",   # Loop back
        "done": "done",             # Move to done
    })
    graph2.add_edge("done", END)

    app2 = graph2.compile()
    result2 = app2.invoke({"count": 7, "log": []})
    print(f"\n4. Conditional graph: count={result2['count']}, log={result2['log']}")
    # 7 → 8 → 9 → 10 → done!

    # ── Step E: Visualize graph structure ────────────────────────────────────
    # LangGraph can print the graph as ASCII or Mermaid
    print(f"\n5. Graph nodes: {list(app2.get_graph().nodes.keys())}")
    try:
        print(f"   Graph as Mermaid:\n{app2.get_graph().draw_mermaid()}")
    except Exception:
        print("   (Mermaid rendering not available — that's fine)")

    # ── Step F: Uncomment after TODO 1-3 ─────────────────────────────────────
    pipeline = build_research_pipeline()
    result = pipeline.invoke({
        "messages": [HumanMessage(content="What are the benefits of exercise?")],
        "research_notes": "",
        "quality_score": 0,
        "iteration": 0,
    })
    print(f"\n6. Pipeline result: {result['research_notes'][:200]}...")

    # ── Step G: Uncomment after TODO 4-5 ─────────────────────────────────────
    pipeline = build_research_pipeline_with_validation()
    result = pipeline.invoke({
        "messages": [HumanMessage(content="Explain how transformers work in AI")],
        "research_notes": "",
        "quality_score": 0,
        "iteration": 0,
    })
    print(f"\n7. Validated pipeline: score={result['quality_score']}, "
          f"iterations={result['iteration']}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  STATE DEFINITION — shared across all TODOs                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TODO 1: Define the graph state                                            ║
# ║                                                                            ║
# ║  Create a TypedDict called ResearchState with these fields:                ║
# ║  - messages: Annotated[list, add_messages]   (conversation history)        ║
# ║  - research_notes: str                       (accumulated research)        ║
# ║  - quality_score: int                        (0-10 quality rating)         ║
# ║  - iteration: int                            (retry counter)              ║
# ║                                                                            ║
# ║  The messages field uses the add_messages reducer so new messages          ║
# ║  are appended (not replaced). The other fields use simple replacement.     ║
# ║                                                                            ║
# ║  Compare to Day 4: this is like your messages list + metadata,             ║
# ║  but structured as a typed dict that LangGraph can manage.                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# TODO 1: Define ResearchState here
# class ResearchState(TypedDict):
#     messages: ...
#     research_notes: ...
#     quality_score: ...
#     iteration: ...

# Stub so the rest of the file runs (remove when you implement TODO 1):
class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    research_notes: str
    quality_score: int
    iteration: int


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SHARED MODEL                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

model = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=1024, temperature=0)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TODO 2: Create the researcher node                                        ║
# ║                                                                            ║
# ║  A node is just a function that takes state and returns a partial update.  ║
# ║                                                                            ║
# ║  The researcher node should:                                               ║
# ║  1. Build a prompt: system message asking for detailed research notes,     ║
# ║     plus the user's messages                                               ║
# ║  2. Call model.invoke() with the messages                                  ║
# ║  3. Return a dict updating:                                                ║
# ║     - "messages": [response]   (add the AI response to history)            ║
# ║     - "research_notes": response.content                                   ║
# ║     - "iteration": state["iteration"] + 1                                 ║
# ║                                                                            ║
# ║  Key insight: You return a PARTIAL state update (only the keys you want    ║
# ║  to change). LangGraph merges it into the full state.                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def researcher_node(state: ResearchState) -> dict:
    """Research a topic and produce notes."""
    # TODO 2: Implement the researcher node.
    #
    system_msg = SystemMessage(content=(
        "You are a thorough researcher. Given a topic, produce detailed "
        "research notes covering key concepts, examples, and nuances. "
        "Be specific and cite reasoning. Keep it under 200 words."
    ))
    #
    all_messages = [system_msg] + state["messages"]
    #
    # If there's existing research and this is a retry (iteration > 0),
    # add a HumanMessage saying:
    #   "Your previous research scored below threshold. Improve these notes: {notes}"
    #
    if state["iteration"] > 0:
        all_messages = all_messages + [HumanMessage(content=(f"Your previous research scored below threshold. Improve these notes: {state['research_notes']}"))]

    response = model.invoke(all_messages)
    
    return {
        "messages": [response],
        "research_notes": response.content,
        "iteration": state["iteration"] + 1,
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TODO 3: Create the quality scorer node                                    ║
# ║                                                                            ║
# ║  This node evaluates the research notes and assigns a quality score.       ║
# ║  It uses the LLM as a judge — same concept you'll use in Day 9 evals!     ║
# ║                                                                            ║
# ║  The scorer node should:                                                   ║
# ║  1. Prompt the LLM: "Rate these research notes 1-10 for completeness,     ║
# ║     accuracy, and clarity. Reply with ONLY a number."                      ║
# ║  2. Parse the number from the response                                     ║
# ║  3. Return {"quality_score": parsed_score}                                 ║
# ║                                                                            ║
# ║  Compare to Day 5: This is structured output! But instead of Pydantic,    ║
# ║  we're using a simple "reply with a number" prompt. Sometimes simple       ║
# ║  is enough.                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def scorer_node(state: ResearchState) -> dict:
    """Score the quality of research notes using LLM-as-judge."""
    # TODO 3: Implement the scorer node.
    #
    scoring_prompt = [
        SystemMessage(content=(
            "You are a research quality evaluator. Rate the following "
            "research notes on a scale of 1-10 for completeness, accuracy, "
            "and clarity. Reply with ONLY a single integer number, nothing else."
        )),
        HumanMessage(content=state["research_notes"]),
    ]
    #
    response = model.invoke(scoring_prompt)
    
    # # Parse the score — extract first number found
    import re
    match = re.search(r'\d+', response.content)
    score = int(match.group()) if match else 5  # default to 5 if parsing fails
    score = max(1, min(10, score))  # clamp to 1-10
    
    print(f"  [Scorer] Quality score: {score}/10 (iteration {state['iteration']})")
    
    return {"quality_score": score}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TODO 4: Create the routing function (conditional edge)                    ║
# ║                                                                            ║
# ║  After scoring, decide whether to:                                         ║
# ║  - Accept the research (score >= 7 OR we've done 3 iterations)            ║
# ║  - Retry (score < 7 AND iterations left)                                  ║
# ║                                                                            ║
# ║  Return a string: "accept" or "retry"                                      ║
# ║                                                                            ║
# ║  Compare to Day 4: this is like your max_steps check + quality gate,      ║
# ║  but expressed as a routing decision instead of a while-loop condition.    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

MAX_ITERATIONS = 3
QUALITY_THRESHOLD = 9


def route_after_scoring(state: ResearchState) -> Literal["accept", "retry"]:
    """Decide whether to accept the research or retry."""
    # TODO 4: Implement routing logic.
    # - If quality_score >= QUALITY_THRESHOLD → return "accept"
    # - If iteration >= MAX_ITERATIONS → return "accept" (give up gracefully)
    # - Otherwise → return "retry"
    #
    # Print the routing decision for debugging:
    

    if state["quality_score"] >= QUALITY_THRESHOLD:
        decision = "accept"
    elif state["iteration"] >= MAX_ITERATIONS:
        decision = "accept"
    else:
        decision = "retry"
    
    
    print(f"  [Router] score={state['quality_score']}, "
          f"iteration={state['iteration']}, decision={decision}")

    return decision
    # return "accept"  # Stub: always accept


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TODO 5: Wire up the graph with conditional edges                          ║
# ║                                                                            ║
# ║  Build the full pipeline:                                                  ║
# ║    START → researcher → scorer → [accept: END, retry: researcher]          ║
# ║                                                                            ║
# ║  Steps:                                                                    ║
# ║  1. Create StateGraph(ResearchState)                                       ║
# ║  2. Add nodes: "researcher" and "scorer"                                   ║
# ║  3. Add edge: START → "researcher"                                         ║
# ║  4. Add edge: "researcher" → "scorer"                                      ║
# ║  5. Add conditional edge from "scorer" using route_after_scoring:          ║
# ║     graph.add_conditional_edges("scorer", route_after_scoring, {           ║
# ║         "accept": END,                                                     ║
# ║         "retry": "researcher",                                             ║
# ║     })                                                                     ║
# ║  6. Compile and return                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def build_research_pipeline():
    """Build a simple research pipeline (no validation loop)."""
    graph = StateGraph(ResearchState)
    graph.add_node("researcher", researcher_node)
    graph.add_node("scorer", scorer_node)
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "scorer")
    graph.add_edge("scorer", END)
    return graph.compile()


def build_research_pipeline_with_validation():
    """Build the full pipeline with retry loop."""
    # TODO 5: Build the graph with conditional edges.
    #
    graph = StateGraph(ResearchState)
    graph.add_node("researcher", researcher_node)
    graph.add_node("scorer", scorer_node)
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "scorer")
    graph.add_conditional_edges("scorer", route_after_scoring, {
        "accept": END,
        "retry": "researcher",
    })
    return graph.compile()

    # Stub: return the simple pipeline
    # return build_research_pipeline()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN — run exploration, then test your pipeline                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    # Phase 1: Explore LangGraph basics
    # explore()

    # Phase 2: Uncomment after TODOs 1-3 (simple pipeline, no retry)
    # print("\n" + "=" * 60)
    # print("TESTING: Simple research pipeline")
    # print("=" * 60)
    # pipeline = build_research_pipeline()
    # result = pipeline.invoke({
    #     "messages": [HumanMessage(content="Explain the CAP theorem in distributed systems")],
    #     "research_notes": "",
    #     "quality_score": 0,
    #     "iteration": 0,
    # })
    # print(f"\nResearch notes:\n{result['research_notes'][:500]}")
    # print(f"\nQuality score: {result['quality_score']}/10")
    # print(f"Iterations used: {result['iteration']}")

    # Phase 3: Uncomment after TODOs 4-5 (with validation/retry loop)
    print("\n" + "=" * 60)
    print("TESTING: Research pipeline with validation loop")
    print("=" * 60)
    pipeline = build_research_pipeline_with_validation()
    result = pipeline.invoke({
        "messages": [HumanMessage(content="Explain the differences between SQL and NoSQL databases")],
        "research_notes": "",
        "quality_score": 0,
        "iteration": 0,
    })
    print(f"\nFinal research notes:\n{result['research_notes'][:500]}")
    print(f"\nFinal quality score: {result['quality_score']}/10")
    print(f"Total iterations: {result['iteration']}")
