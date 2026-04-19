# Day 7: Multi-Agent Basics

## Overview

Today you move from single-agent systems to **multi-agent** systems — where multiple
specialized agents collaborate to solve tasks that would be awkward or brittle for a
single agent. You'll learn the key patterns (delegation, reflection, dynamic routing),
build them with AutoGen's AgentChat API, and then compare against the single-agent
approach so you know when multi-agent actually helps.

---

## Key Concepts

### Why Multi-Agent?

A single agent with many tools and a long system prompt eventually hits limits:
- **Cognitive overload**: one LLM juggling planning, execution, and quality checking
  produces worse results than specialized agents doing each part.
- **Separation of concerns**: a "planner" agent and an "executor" agent can use
  different models, prompts, or temperature settings.
- **Reflection / critique loops**: a second agent reviewing the first agent's output
  catches errors the first agent is blind to (same reason code review works).

### When Multi-Agent Helps vs. Hurts

| Helps | Hurts |
|-------|-------|
| Complex tasks with distinct subtask types | Simple tasks a single agent handles fine |
| Tasks needing self-critique or verification | When agent coordination overhead > task complexity |
| When different subtasks need different tools | When shared context gets so large it blows context windows |
| When you want modular, reusable agent roles | When latency matters (each hop = more API calls) |

### Core Multi-Agent Patterns

1. **Reflection (Writer + Critic)**: Agent A produces output, Agent B critiques it,
   Agent A revises. Simple round-robin coordination.

2. **Planner + Executor**: A planning agent decomposes the task into subtasks.
   An executor agent (with tools) handles each subtask. Optional: a validator checks results.

3. **Dynamic Routing (Selector)**: An LLM-based "manager" reads conversation context
   and picks the best-suited agent for each turn. More flexible than fixed ordering.

4. **Swarm / Handoff**: Agents decide locally when to hand off to another agent.
   Decentralized — no central manager.

### Message Passing

In multi-agent systems, agents communicate by passing messages through shared
conversation history. Key design decisions:
- **Shared context**: all agents see all messages (group chat style)
- **Filtered context**: agents only see messages relevant to them
- **Broadcast vs. directed**: messages go to everyone vs. a specific agent

---

## AutoGen AgentChat — The Framework for Today

AutoGen (by Microsoft) provides high-level abstractions for multi-agent systems.
The **AgentChat** API (v0.4+) is what we'll use.

> **Note**: AutoGen is now in maintenance mode — Microsoft shipped Agent Framework 1.0
> as its successor (see section above). But AutoGen remains the most widely-used
> educational multi-agent framework, the concepts transfer directly, and the community
> knowledge base is vastly deeper. Learn the patterns here; migrate the API later.

### Key Classes

```
AssistantAgent     — An agent backed by an LLM, optionally with tools
RoundRobinGroupChat — Agents take turns in fixed order (reflection pattern)
SelectorGroupChat   — An LLM picks the next speaker dynamically
TextMentionTermination — Stop when a keyword appears in output
MaxMessageTermination  — Stop after N messages
```

### Installation

```bash
pip install "autogen-agentchat" "autogen-ext[openai,anthropic]"
```

### Using Anthropic Models

AutoGen supports Claude via `AnthropicChatCompletionClient`:

```python
from autogen_ext.models.anthropic import AnthropicChatCompletionClient

model_client = AnthropicChatCompletionClient(
    model="claude-sonnet-4-20250514",
    # api_key="..." or set ANTHROPIC_API_KEY env var
)
```

### Using OpenAI Models

```python
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    # api_key="..." or set OPENAI_API_KEY env var
)
```

---

## What About Microsoft Agent Framework?

As of April 3, 2026, AutoGen is officially in **maintenance mode**. Microsoft shipped
**Microsoft Agent Framework 1.0** as its production successor, unifying AutoGen's
multi-agent orchestration with Semantic Kernel's enterprise features (middleware,
sessions, telemetry) into a single SDK.

Agent Framework supports the same orchestration patterns — sequential, concurrent,
handoff, group chat — and has first-party connectors for Anthropic Claude, OpenAI,
Azure OpenAI, Bedrock, Gemini, and Ollama.

**Why we're still using AutoGen for this lesson:**
- The multi-agent *concepts* (agent roles, delegation, message passing, routing) are
  identical across both frameworks. Learning them in AutoGen transfers directly.
- AutoGen has 2+ years of community examples, blog posts, and Stack Overflow answers.
  Agent Framework 1.0 is 9 days old — docs are solid but community content is thin.
- Day 8 ("Framework Detox") will have you rebuild everything in raw Python, which is
  more future-proof than any specific framework.
- The Anthropic connector for Agent Framework is still beta (`pip install agent-framework-anthropic --pre`).

**When you're ready to explore Agent Framework**, here's the quick-start equivalent:

```python
from agent_framework.anthropic import AnthropicClient

client = AnthropicClient()  # reads ANTHROPIC_API_KEY from env
agent = client.as_agent(
    name="MyAgent",
    instructions="You are a helpful assistant.",
)
result = await agent.run("Hello!")
print(result.text)
```

Key links:
- Overview: `https://learn.microsoft.com/en-us/agent-framework/overview/`
- GitHub: `https://github.com/microsoft/agent-framework`
- AutoGen migration guide: `https://learn.microsoft.com/en-us/agent-framework/migrate/autogen`
- Anthropic setup: `https://learn.microsoft.com/en-us/agent-framework/agents/providers/anthropic`

---

## Targeted Reading

Read these **specific sections** before doing the exercises:

1. **AutoGen AgentChat Tutorial — Agents**
   `https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/agents.html`
   → Read: how `AssistantAgent` works, `tools` parameter, `reflect_on_tool_use`, `run()` vs `run_stream()`

2. **AutoGen AgentChat Tutorial — Teams**
   `https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/teams.html`
   → Read: `RoundRobinGroupChat` setup, the reflection pattern example, termination conditions

3. **AutoGen Selector Group Chat**
   `https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/selector-group-chat.html`
   → Read: `SelectorGroupChat` with planning agent example, `selector_prompt` template, how dynamic routing works

4. **AutoGen Models — Anthropic**
   `https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/models.html`
   → Read: `AnthropicChatCompletionClient` setup (scroll to Anthropic section)

---

## Exercises

### Exercise 1: Reflection Pattern with RoundRobinGroupChat
Build a **Writer + Critic** team. The writer drafts content, the critic reviews it,
the writer revises. Uses `RoundRobinGroupChat` with `TextMentionTermination`.
This is the simplest multi-agent pattern — great for understanding how AutoGen coordinates agents.

### Exercise 2: Planner + Executor with SelectorGroupChat
Build a 3-agent team: **Planner** (decomposes tasks), **Researcher** (has search tools),
**Analyst** (has calculation tools). Uses `SelectorGroupChat` so the LLM dynamically
picks which agent should act next. This demonstrates the power of dynamic routing.

### Exercise 3: Single-Agent vs. Multi-Agent Comparison
Run the **same task** through a single agent (with all tools) and the multi-agent team
from Exercise 2. Compare: output quality, token usage, number of API calls, and latency.
Write your findings in `single_vs_multi_comparison.md`.

---

## What You'll Commit

```
day7/
├── day7_lesson_guide.md
├── exercise1_reflection.py          # Writer + Critic (RoundRobinGroupChat)
├── exercise2_selector_team.py       # Planner + Researcher + Analyst (SelectorGroupChat)
├── exercise3_single_vs_multi.py     # Comparison harness
└── single_vs_multi_comparison.md    # Your written comparison (fill in after running)
```

---

## Databricks Connection

Databricks recently introduced **Agent Bricks** with a Multi-Agent Supervisor (MAS)
pattern. This is conceptually similar to `SelectorGroupChat` — a supervisor agent
routes tasks to specialized sub-agents. When you get to Day 13 (Databricks deep dive),
you'll see how these patterns map to Databricks' managed agent infrastructure.

Reference: `https://www.databricks.com/blog/introducing-agent-bricks`
