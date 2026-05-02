# Day 8 — Framework Detox: Rebuild Without Frameworks

**Phase:** Frameworks (the introspective bookend)
**Prereqs:** Days 2–7 complete. You have a working AutoGen multi-agent system from Day 7.
**Goal:** Reimplement your Day 7 multi-agent system using **only** `anthropic` + standard library. Feel the abstractions. Decide for yourself when a framework earns its seat.

---

## Why this day matters

You just spent Days 6–7 fighting frameworks as much as using them:

- `reflect_on_tool_use=True` crashed silently with the Anthropic client (400 error on empty text blocks).
- `RoundRobinGroupChat` couldn't skip a turn — you had to contort the Editor's system message.
- `SelectorGroupChat`'s routing LLM call was *invisible* in your token accounting (Day 7 Exercise 3: multi-agent token count was underreported).
- Tools had to be `async def` in AutoGen 0.4 — a footgun with no compile-time check.
- The Planner did everyone's job because its `system_message` didn't prohibit it loudly enough.

Some of this is AutoGen-specific. But the *pattern* is universal: **frameworks trade visibility for convenience.** When convenience outweighs visibility, great. When it doesn't, you're debugging someone else's abstraction instead of your problem.

Today you'll find out that the whole thing — reflection, tool calling, dynamic routing — is ~250 lines of Python. Once you've written it, frameworks stop feeling magical. You'll pick them (or not) on evidence instead of default.

---

## Targeted readings (do these first, ~30 min)

1. **Anthropic tool use — the *implementation* section, not the overview.**
   https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview
   → Skim the page, then read specifically: "How tool use works" and "Handling tool use and tool result content blocks". The key mental model is: `stop_reason == "tool_use"` means the loop continues; `stop_reason == "end_turn"` means you're done.

2. **Anthropic Messages API — content block types.**
   https://docs.anthropic.com/en/api/messages
   → You care about one thing: `response.content` is a *list of blocks*, each with `.type` ∈ `{"text", "tool_use", "thinking"}`. There is no single `.text` field on a response with tools. That's the gotcha that breaks first-time tool code.

3. **Lilian Weng's agent post — revisit the "Planning" and "Memory" sections you skimmed on Day 2.**
   https://lilianweng.github.io/posts/2023-06-23-agent/
   → With Days 3–7 under your belt these sections read completely differently. "Memory" here is just "what you append to `messages`". "Planning" is just "the Planner's first LLM call". The vocabulary demystifies fast.

4. **Anthropic cookbook — customer_service_agent.ipynb (tool-calling loop from scratch).**
   https://github.com/anthropics/anthropic-cookbook/tree/main/tool_use
   → Skim the `while response.stop_reason == "tool_use"` loop in any notebook. That eight-line loop is the whole "agent framework" for a single agent.

5. *(Optional)* **Databricks Foundation Model APIs — OpenAI-compatible client.**
   https://docs.databricks.com/aws/en/machine-learning/model-serving/score-foundation-models
   → You don't need this today, but notice: Databricks exposes `databricks-meta-llama-3-3-70b-instruct` and similar via an OpenAI-compatible endpoint. Everything you write today transfers directly — just swap `base_url` and model name.

---

## The core agent loop, in five lines

Strip everything away. An agent is:

```python
messages = [{"role": "user", "content": task}]
while True:
    response = client.messages.create(model=M, max_tokens=1024, tools=TOOLS, messages=messages, system=SYSTEM)
    messages.append({"role": "assistant", "content": response.content})
    if response.stop_reason != "tool_use":
        break
    tool_results = [run_tool(b) for b in response.content if b.type == "tool_use"]
    messages.append({"role": "user", "content": tool_results})
```

That's it. That's the thing. LangChain's `AgentExecutor`, AutoGen's `AssistantAgent`, CrewAI's `Agent`, OpenAI's `Assistants API` — they all wrap this loop. Everything else is ergonomics: retry logic (you built that Day 4), logging (Day 4), structured outputs (Day 5), orchestration (today).

**When a framework hides this loop from you, debugging is 10x harder.** When you see the raw loop, a 400 error from Anthropic is 30 seconds of `print(messages)`.

---

## What each framework abstraction actually was

Let's map Day 6–7 concepts to their raw-Python equivalents:

| Framework concept                       | Raw Python equivalent                                                               | LoC   |
|-----------------------------------------|-------------------------------------------------------------------------------------|-------|
| `AssistantAgent(system_message=...)`    | A `system` kwarg + a `messages` list                                                | 0     |
| `RoundRobinGroupChat`                   | A `for agent in cycle(agents)` loop                                                 | ~5    |
| `SelectorGroupChat` (LLM router)        | An extra `client.messages.create(...)` that returns a name                          | ~15   |
| `TextMentionTermination("APPROVE")`     | `if "APPROVE" in last_message: break`                                               | 1     |
| `MaxMessageTermination(20)`             | `if turn >= 20: break`                                                              | 1     |
| `reflect_on_tool_use=True`              | One extra `client.messages.create(...)` after tool results, asking for a summary    | ~5    |
| `Console(team.run_stream(...))`         | `print()` and/or `rich.print()`                                                     | 1     |
| `models_usage` token tracking           | Read `response.usage.input_tokens` and `.output_tokens` yourself                    | 2     |
| LangGraph `StateGraph` with nodes/edges | A dictionary of functions + a `current_node` variable + a `while` loop              | ~30   |
| Tool schema auto-generation from `def`  | Write a `list[dict]` yourself (you did this Day 3)                                  | ~10/tool |

None of this is complicated. It's just *tedious* — and that tedium is exactly what frameworks sell. Sometimes worth it; sometimes not.

---

## Two multi-agent patterns, rebuilt by hand

### Pattern 1 — Reflection (Writer ↔ Critic)

**Framework version (Day 7 Ex 1):** `RoundRobinGroupChat([writer, critic], termination=...)`.

**Raw version:** Two separate `messages` lists, one per agent. Each agent's turn:
1. Append the *other agent's last output* as a `"user"` message in your own history.
2. Call the API with your system prompt + your history.
3. Append your response to your history, return the text.

The key mental shift: **multi-agent is an illusion.** There's no "agents talking to each other". There are N independent LLM contexts, and the orchestrator (you, or the framework) decides which context gets fed which text, in which order.

```python
# Pseudocode — you'll write the real thing in Exercise 1
writer_hist = []
critic_hist = []
draft = writer.turn(writer_hist, task)
for turn in range(MAX_TURNS):
    feedback = critic.turn(critic_hist, draft)
    if "APPROVE" in feedback:
        break
    draft = writer.turn(writer_hist, feedback)
```

Notice what's *gone*: no scheduler, no termination class hierarchy, no `RoundRobinGroupChat` object, no teardown. And critically — **no hidden LLM calls.** Your token count is exact.

### Pattern 2 — Dynamic routing (Planner → Researcher ↔ Analyst → …)

**Framework version (Day 7 Ex 2):** `SelectorGroupChat` with a selector prompt, `{roles}/{history}/{participants}` templating.

**Raw version:** Before each turn, ask an LLM "given this history and these roles, who speaks next?" The answer is a name. You dispatch to that agent. It speaks. Repeat.

```python
# Pseudocode — Exercise 2
shared_history = [{"role": "user", "content": task}]
for turn in range(MAX_TURNS):
    next_agent_name = pick_next_speaker(shared_history, AGENT_ROLES)
    if next_agent_name == "TERMINATE":
        break
    agent = agents_by_name[next_agent_name]
    response_text = agent.turn(shared_history)
    shared_history.append({"role": "user", "content": f"[{next_agent_name}]: {response_text}"})
```

Two things to notice:

1. **The selector is visibly a separate LLM call.** On Day 7 Exercise 3 you flagged "multi-agent token count is underreported because SelectorGroupChat's routing LLM calls are hidden/internal". Today, those calls are *right there*, impossible to miss.

2. **Agent messages become `"user"` messages in the shared history, prefixed with the speaker's name.** This solves the identity-confusion bug from Day 7 Exercise 2 — your agents lost track of "who is the human" because AutoGen put everything into the same `role="user"` bucket with no speaker tags. When you own the formatting, you add the tag and the confusion goes away.

---

## The tool-calling context gap (and the manual fix)

On Day 7 Exercise 2 you hit a real bug: the Analyst returned a bare `ToolCallSummaryMessage` like `"18.666..."` or `"3"`, and the Planner had no idea what number that was. Root cause: tool results, when passed between agents, lose their *question*. `reflect_on_tool_use=True` was AutoGen's fix — and it crashed on Anthropic.

The manual fix is embarrassingly simple. After an agent runs tools, ask *that same agent* to write a one-sentence summary before passing the message along:

```python
# After tool execution, run a "reflection" pass:
messages.append({"role": "user", "content": tool_results})
summary_response = client.messages.create(
    model=M, system=agent.system + "\n\nSummarize the tool result in one sentence.",
    messages=messages, max_tokens=200
)
# Now summary_response.content[0].text is a natural-language wrapper — safe to send to other agents.
```

That's what `reflect_on_tool_use` was doing under the hood. Five lines. No 400 errors. Always works.

---

## When to use frameworks vs. raw Python (decision matrix)

After today you'll have earned the right to this opinion. For now, here's a starter:

| If your situation is…                                       | Lean…          | Why                                                                 |
|-------------------------------------------------------------|-----------------|---------------------------------------------------------------------|
| Prototype, single agent, 1–3 tools                          | **Raw**        | Framework overhead exceeds the problem                              |
| Complex state graph with retries/fallbacks/branches         | LangGraph      | The graph abstraction genuinely reduces cognitive load              |
| Production, need tracing, eval, deploy                      | LangGraph / Databricks Agent Framework | MLflow integration, OpenInference, Unity Catalog tools      |
| Multi-agent with well-known pattern (swarm, supervisor)     | AutoGen / MAF  | Don't reinvent `HandoffMessage` logic                               |
| Team onboarding, docs matter more than control              | LangChain      | Largest community, most Stack Overflow hits                         |
| Weird provider (Anthropic tool-use quirks, custom endpoints)| **Raw**        | You'll debug the abstraction otherwise (you already did, Day 7)     |
| Research / evals where output must be reproducible          | **Raw**        | Full control over every API call, every token                       |

**Heuristic:** if you can't explain what a framework call does in ~5 lines of pseudocode, you don't understand it well enough to debug it in production.

---

## Databricks angle (today's parallel track)

The curriculum has you building a pure-Python agent with Databricks Foundation Model APIs today. The Foundation Model APIs are **OpenAI-compatible** — meaning every line of code you write with the `anthropic` SDK has a near-identical `openai` SDK equivalent hitting a Databricks endpoint:

```python
from openai import OpenAI
client = OpenAI(
    base_url="https://<workspace>.cloud.databricks.com/serving-endpoints",
    api_key=DATABRICKS_TOKEN,
)
response = client.chat.completions.create(
    model="databricks-meta-llama-3-3-70b-instruct",
    messages=[...],
    tools=[...],
)
```

Same loop. Same `tool_calls`. Same stop-reason logic (it's `"tool_calls"` in OpenAI-style, `"tool_use"` in Anthropic-style — trivially different).

Practically: if you ever deploy your raw-Python agent on Databricks, the only change is the client instantiation. This is the payoff for NOT picking a framework that abstracts away the HTTP layer.

Relevant doc for today: **Build a custom agent on Databricks without LangChain**
https://docs.databricks.com/aws/en/generative-ai/agent-framework/create-agent

---

## AutoGen → Microsoft Agent Framework note (carried over from Day 7)

You already documented this on Day 7: AutoGen is in maintenance mode, MAF 1.0 shipped April 3, 2026. Today's exercise reinforces why that migration matters less than you'd think: **if you understand the raw loop, you can adopt or drop any framework in a day.** The conceptual debt is low when the implementation is transparent.

---

## Exercises overview

Three exercises, ~3–4 hours total. Each is a runnable stub with TODO hints — fill them in, run, observe, debug.

### Exercise 1 — `exercise1_manual_reflection.py`
**Rebuild the Day 7 Exercise 1 Writer+Critic reflection team in raw Python.**
~150 lines. You'll:
- Write a tiny `Agent` helper that wraps a single `client.messages.create(...)` call.
- Run a manual turn-taking loop.
- Detect `"APPROVE"` termination yourself.
- Track per-turn token usage yourself (`response.usage`).
- Compare totals to your Day 7 AutoGen version.

**Success criteria:** produces an approved haiku, ≤ 6 total turns, prints a token-usage table at the end.

### Exercise 2 — `exercise2_manual_selector.py`
**Rebuild the Day 7 Exercise 2 Planner+Researcher+Analyst team with dynamic routing, in raw Python.**
~300 lines. You'll:
- Define 4 tool schemas as dicts (reuse your Day 7 mock tool logic).
- Implement the `tool_use` loop for a tool-enabled agent.
- Implement a selector: an LLM call that returns the next speaker's name.
- Apply the "manual reflection" fix: every tool-using agent summarizes results before its output goes into shared history.
- Terminate on an explicit `"TERMINATE"` string or max turns.

**Success criteria:** same task as Day 7 runs to completion; you can see every single LLM call and its token cost (including the selector); no 400 errors.

### Exercise 3 — `exercise3_comparison_and_mini_framework.py`
**Measure raw vs. framework. Then extract your own tiny framework.**
~200 lines. You'll:
- Run your Exercise 2 manual version and your Day 7 AutoGen version on the *same* task.
- Record: total LLM calls, total tokens, wall-clock time, lines of code, number of framework-specific bugs hit.
- Extract reusable primitives from Exercise 2 into a `micro_agents.py`-style module: `Agent`, `ToolAgent`, `run_team(agents, selector, task)`.
- Write a 3-paragraph `framework_comparison.md` with your conclusion: "when will I reach for a framework, and when won't I".

**Success criteria:** you end the day with (a) a working raw multi-agent system, (b) your own ~100-line "framework" extracted from it, and (c) a written opinion grounded in your own measurements.

---

## Expected commits (per tracker)

- `day8/manual_multi_agent.py` — use your Exercise 2 as this file, or rename
- `day8/framework_comparison.md` — from Exercise 3
- `day8/micro_agents.py` — your extracted mini-framework (bonus)

---

## Common mistakes to watch for

Based on real failure modes when rebuilding from scratch:

1. **Forgetting `stop_reason` check.** The loop continues while `stop_reason == "tool_use"`. Miss this and your agent silently drops tool calls or loops forever.
2. **Feeding the wrong role for tool_results.** Tool results go back as a **`"user"` message**, not `"assistant"` or `"tool"`. The content is a list of `{"type": "tool_result", "tool_use_id": ..., "content": str(...)}`.
3. **Serializing tool results as objects instead of strings.** Same Day 4 bug you caught: `return str(result)`, not `return result`. JSON-serializable strings only.
4. **Treating `response.content` as a string.** It's a list of blocks. Text is `response.content[0].text` *only if* the first block is a text block. After tool use, block 0 is often the `tool_use` block and block 1 is a trailing text block, or vice versa.
5. **Using the wrong `messages` list in multi-agent reflection.** Each agent has its own history. If you shove everything into one shared list with alternating roles, Anthropic will reject the sequence (consecutive `"assistant"` messages aren't allowed).
6. **Selector returning prose instead of a name.** Constrain the selector with a tight system prompt: "Respond with ONLY one of: PLANNER, RESEARCHER, ANALYST, TERMINATE. No other text." Without this, you'll parse "I think the Researcher should go next" and fail.
7. **No max-turns safety net.** Even with clean termination logic, keep `MAX_TURNS = 20` as an emergency brake. The Day 7 lesson "always use MaxMessageTermination as safety net" still applies.
8. **Not logging each LLM call.** Without a `log_call(agent_name, usage, cost)` helper, you lose the cost visibility that was the whole point of going raw. Print a table at the end of each run.

---

## How today connects to what's next

- **Day 9 (Evaluation):** you'll write an eval harness. Raw-Python agents are dramatically easier to eval because you control every input and output. Frameworks often make eval harder, not easier.
- **Day 10–11 (RAG):** the core loop doesn't change. You just add `retrieve(query)` as another tool. The mental model stays the same.
- **Day 12 (MCP):** MCP exposes tools over a protocol, but your agent side is still the same tool-use loop. MCP clients just populate the `tools=[...]` list dynamically at startup.
- **Day 13 (Databricks):** you'll use Databricks' managed agent framework, but you'll know exactly what it's doing underneath.

The deeper point: **after today, every framework you encounter becomes a translation exercise.** "Oh, their `Chain` is my `Agent.step`. Their `Memory` is my `messages` list. Their `Router` is my `pick_next_speaker`." You stop learning frameworks and start reading their abstractions.

That's the detox.

---

Good luck. Open `exercise1_manual_reflection.py` when ready.
