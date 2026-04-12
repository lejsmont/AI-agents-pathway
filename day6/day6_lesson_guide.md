# Day 6: LangChain + LangGraph Essentials

## Overview

You've spent Days 2–5 building agents from scratch: raw API calls, manual tool
dispatch, retry logic, structured outputs. Today you learn what **frameworks**
do with all that plumbing — and, critically, when they help vs. when they hide
too much.

By the end of today you will:

1. Rebuild your Day 4 robust agent using **LangChain** (tools + memory)
2. Understand **LangGraph** state machines and build a Planner → Executor → Validator graph
3. Know exactly which framework abstractions map to your from-scratch code
4. See how this connects to Databricks, which uses LangGraph natively for agents

---

## Part 1: LangChain — Agent Abstractions

### What LangChain Actually Does

Remember your Day 4 `robust_agent_loop.py`? You wrote:
- A `while` loop checking `stop_reason`
- A tool registry dict mapping names → functions
- Manual `tool_use` / `tool_result` message construction
- Retry decorators, token counting, cost tracking

LangChain wraps all of that into abstractions:

| Your from-scratch code | LangChain equivalent |
|---|---|
| `anthropic.Anthropic()` | `ChatAnthropic()` model wrapper |
| Tool function + JSON schema | `@tool` decorator (auto-generates schema) |
| `while stop_reason == "tool_use"` loop | `create_agent()` handles the loop |
| Appending messages manually | Handled internally by the agent |
| Conversation history list | Checkpointer-based memory |

### Key Concepts

**Chat Models** — LangChain wraps every LLM provider into a common interface.
`ChatAnthropic`, `ChatOpenAI`, etc. all have `.invoke()`, `.stream()`, and
`.bind_tools()`. This means you can swap providers without rewriting your agent.

**Tools** — The `@tool` decorator inspects your function's docstring and type
hints to auto-generate the JSON schema. No more hand-writing `input_schema`
dicts. Compare:

```python
# Day 3: Manual schema
calculator_tool = {
    "name": "calculator",
    "description": "Evaluate a math expression",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression"}
        },
        "required": ["expression"]
    }
}

# Day 6: LangChain @tool
from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression and return the result."""
    return str(eval(expression))
```

**Memory** — LangChain uses checkpointer-based memory. Pass a `MemorySaver()`
to `create_agent()` and use `thread_id` in the config to group messages
into conversations (like your Day 4 `messages` list, but managed
automatically).

### Targeted Reading

Read these **specific sections** (not the full docs):

1. **LangChain Agents — create_agent**
   https://docs.langchain.com/oss/python/langchain/agents
   → Read the "Core components" and "Model" sections. See how `create_agent`
   wraps the loop you built manually on Day 4.

2. **LangChain Tools — Defining Tools**
   https://docs.langchain.com/oss/python/langchain/tools
   → Read the "Create tools" section. Notice how the `@tool` decorator reads
   your docstring and type hints to auto-generate the JSON schema.

3. **LangChain Models — Chat Models**
   https://docs.langchain.com/oss/python/langchain/models
   → Skim the model configuration. Notice how `ChatAnthropic` wraps the API
   you've been using directly.

4. **LangChain Short-term Memory**
   https://docs.langchain.com/oss/python/langchain/short-term-memory
   → Skim how conversation memory works with checkpointers. Note: this is
   the same concept as your Day 4 `messages` list, just managed by the
   framework.

---

## Part 2: LangGraph — State Machines for Agents

### Why LangGraph?

LangChain's `create_agent()` is a simple loop: call LLM → maybe call tool → repeat.
But real agent workflows are **graphs**, not loops:

- A planner decides what to do
- An executor runs tools
- A validator checks the output
- If validation fails, route back to the executor (or the planner)
- If it passes, return the result

LangGraph lets you build these as **state machines**: nodes are functions,
edges define the flow, and a shared **state** object passes between nodes.

### Key Concepts

**State** — A `TypedDict` (or Pydantic model) that every node can read and
write. This is the "memory" of the graph — it accumulates data as nodes run.

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Chat history (appends automatically)
    plan: str                                 # Current plan
    result: str                               # Final output
    retry_count: int                          # How many retries so far
```

The `Annotated[list, add_messages]` pattern is special: it tells LangGraph to
**append** new messages instead of replacing the list. This is called a
"reducer" — same concept as Redux if you've seen that.

**Nodes** — Python functions that take state and return a partial state update.
Each node does one job: plan, execute, validate, etc.

```python
def planner_node(state: AgentState) -> dict:
    # Call LLM to make a plan
    response = model.invoke(state["messages"])
    return {"plan": response.content}  # Partial state update
```

**Edges** — Define the flow between nodes. Can be static (always go A → B) or
**conditional** (go to B if condition X, else go to C).

```python
# Conditional edge: route based on validation result
def should_retry(state: AgentState) -> str:
    if state["retry_count"] >= 3:
        return "end"
    if state["validation_passed"]:
        return "end"
    return "executor"  # Try again
```

**Compile & Run** — After building the graph, you `.compile()` it into a
runnable. Then call `.invoke()` with initial state.

### The Planner → Executor → Validator Pattern

This is a classic agent architecture:

```
START → planner → executor → validator → END
                     ↑            |
                     └────────────┘  (retry if validation fails)
```

1. **Planner**: Takes the user query, breaks it into steps
2. **Executor**: Runs tools to accomplish each step
3. **Validator**: Checks the result quality (using structured output!)
4. **Conditional edge**: If validation fails, route back to executor with
   feedback; if passes, route to END

You already know all the building blocks: tool calling (Day 3), retry logic
(Day 4), structured validation (Day 5). LangGraph just gives you a clean way
to wire them together.

### Targeted Reading

5. **LangGraph Overview + Quickstart**
   https://docs.langchain.com/oss/python/langgraph/overview
   → Read the overview to understand what LangGraph provides. Then follow
   the quickstart to see the graph-building mechanics.

6. **LangGraph — Graph API**
   https://docs.langchain.com/oss/python/langgraph/graph-api
   → Read how to define state, nodes, and edges using StateGraph. Understand
   `Annotated` reducers and why `add_messages` exists.

7. **LangGraph Guides — All Topics**
   https://docs.langchain.com/oss/python/langgraph/guides
   → Skim the guides index. Focus on "Persistence" and "Human-in-the-loop"
   sections — these show patterns you'll use in production.

---

## Part 3: Framework Trade-offs — When to Use What

### What Frameworks Give You
- **Less boilerplate**: Tool schemas auto-generated, message routing handled
- **Swappable components**: Change LLM provider in one line
- **Built-in patterns**: Memory, streaming, callbacks all wired up
- **Ecosystem**: Pre-built tool integrations, vector store connectors, etc.

### What Frameworks Hide
- **Token costs**: Easy to burn tokens without realizing (your Day 4 cost
  tracker would be harder to build inside LangChain)
- **Error handling**: Framework catches errors for you, but sometimes you need
  to handle them differently (your Day 4 "feed error back to LLM" pattern)
- **Message format**: You can't easily inspect the raw API messages (your Day 5
  `tool_result` with `is_error=True` pattern is harder to express)
- **Debugging**: When something goes wrong, you're debugging framework internals
  instead of your own code

### The Pragmatic Rule
- **Prototyping / standard patterns** → Use frameworks (faster to build)
- **Production / custom behavior** → Start from scratch or use LangGraph
  (more control)
- **Databricks** → LangGraph is the recommended approach (native support)

---

## Part 4: Databricks Connection

Databricks uses LangGraph as its primary agent authoring framework. When you
deploy agents on Databricks, you:

1. Define your agent as a LangGraph graph
2. Log it with MLflow
3. Deploy to a serving endpoint

The `ChatDatabricks` model wrapper in `langchain-databricks` connects to
Databricks Foundation Model APIs — same concept as `ChatAnthropic` but pointing
to your Databricks workspace.

**Read** (skim for awareness — you'll do this hands-on on Day 13):
https://docs.databricks.com/aws/en/generative-ai/agent-framework/create-agent
→ Look at the "Author an agent with LangGraph" section. Notice how the graph
structure is the same as what you'll build today.

Also check:
https://docs.databricks.com/aws/en/generative-ai/tutorials/agent-framework-notebook
→ This is the full tutorial you'll follow on Day 13. Skim the agent
definition to see LangGraph in a Databricks context.

---

## Setup

```bash
pip install langchain langchain-anthropic langgraph
```

You'll keep using your `ANTHROPIC_API_KEY` environment variable.

---

## Exercises

| Exercise | File | What You Build |
|---|---|---|
| 1 | `exercise1_langchain_agent.py` | Rebuild Day 4 agent with LangChain `create_agent` + tools + memory |
| 2 | `exercise2_langgraph_basics.py` | LangGraph fundamentals — state, nodes, conditional edges |
| 3 | `exercise3_langgraph_planner_executor.py` | Full Planner → Executor → Validator graph with tools |

Each exercise has 5 guided TODOs and an exploration harness at the top so you
can run the code and inspect intermediate values before implementing.

---

## Concepts Checklist

After completing today, you should understand:

- [ ] How `ChatAnthropic` wraps the raw Anthropic API
- [ ] How `@tool` auto-generates JSON schemas from docstrings + type hints
- [ ] How `create_agent` replaces your manual agent loop
- [ ] How checkpointer-based memory works vs. your manual `messages` list
- [ ] What LangGraph state is and how reducers (`add_messages`) work
- [ ] How to build nodes (functions), edges (flow), and conditional routing
- [ ] The Planner → Executor → Validator pattern
- [ ] When frameworks help vs. when from-scratch is better
- [ ] How LangGraph connects to Databricks agent deployment
