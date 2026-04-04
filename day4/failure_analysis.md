# Failure Analysis — Day 4

Document at least 5 experiments where you intentionally break your agent.
For each, fill in: what you did, what happened, and what you learned.

---

## Experiment 1: Step Limit Too Low

**What I did:** Set `max_steps=1` and asked a query needing multiple tool calls:
```python
run_agent("Weather in Tokyo plus calculate 22*9/5+32", max_steps=1)
```

**What happened:**
<!-- TODO: Run it and describe what the agent returned -->
08:01:32 | WARNING  | Hit max_steps (1) without final answer!
08:01:32 | INFO     | ─── Cost Summary ───
08:01:32 | INFO     |   Step        Input     Output
08:01:32 | INFO     |   1             499        109
08:01:32 | INFO     |   TOTAL         499        109
08:01:32 | INFO     |   Model: claude-sonnet-4-20250514
08:01:32 | INFO     |   Estimated cost: $0.003132


**What I learned:**
<!-- TODO: Did it call one tool and give a partial answer? Did it crash? -->
called 2 tools
---

## Experiment 2: Empty Tool Description

**What I did:** Changed a tool's description to `""` or `"does stuff"`:
```python
# In TOOL_SCHEMAS, changed calculator description to:
"description": ""
```

**What happened:**
<!-- TODO: Did the LLM still pick the right tool? Did it hallucinate? -->
Yes, it picked the right tool.
**What I learned:**
<!-- TODO: How important are descriptions for tool selection? -->
I guess quite important but name of the tool is also important
---

## Experiment 3: Tool Throws Exception

**What I did:** Made calculator raise an error for any input:
```python
def calculator(expression: str) -> str:
    raise RuntimeError("Service unavailable!")
```

**What happened:**
<!-- TODO: Before your try/except fix — did the whole agent crash? -->
<!-- TODO: After your try/except fix — did the LLM adapt? How? -->
yes, LLM tried to solve the problem by itself

**What I learned:**
<!-- TODO: What's the difference between crashing vs. feeding the error back? -->

---

## Experiment 4: Infinite Loop Potential

**What I did:** Asked a query that the agent can't fully answer with available tools,
to see if it loops trying different approaches:
```python
run_agent("Search the internet for today's stock price of Apple", max_steps=10)
```

**What happened:**
<!-- TODO: Did it loop? How many steps before it gave up? Did max_steps save you? -->
it said that it didn't have tools to search through internet.
**What I learned:**
<!-- TODO: How does the LLM behave when tools can't satisfy the request? -->

---

## Experiment 5: Tiny Cost Budget

**What I did:** Set `max_cost_usd=0.001` (fraction of a cent):
```python
run_agent("Weather in London, Tokyo, and Warsaw. Average the temps.", max_cost_usd=0.001)
```

**What happened:**
<!-- TODO: At which step did the budget guard kick in? -->
<!-- TODO: How many tokens was one step? How much did it cost? -->

**What I learned:**
<!-- TODO: What's a realistic budget for a simple agent query? -->

---

## Experiment 6 (Bonus): Model Fallback

**What I did:** Set the primary model to a non-existent model name:
```python
run_agent("Hello", model="claude-nonexistent-model")
```

**What happened:**
<!-- TODO: Did the fallback to Haiku work? What was logged? -->

**What I learned:**
<!-- TODO: Is falling back to a cheaper model a good strategy? When would it not be? -->

---

## Experiment 7 (Bonus): Conversation Growth

**What I did:** Logged `len(str(messages))` at each step to watch memory grow:

**What happened:**
<!-- TODO: How fast did the message list grow? Plot it if you can. -->

**What I learned:**
<!-- TODO: At what point would this become a cost or context window problem? -->

---

## Key Takeaways

<!-- TODO: Summarize your top 3-5 lessons from these experiments -->

1. ...
2. ...
3. ...
