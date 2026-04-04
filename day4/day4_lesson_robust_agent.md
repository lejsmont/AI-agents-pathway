# Day 4: Robust Agent Loop + Error Handling
## Your Guided Lesson

---

## Why This Day Matters

On Day 2 you built a working agent loop. On Day 3 you added tools and structured outputs.
But if you deployed either of those to production, they'd break within hours. Why?

- APIs go down, return 500 errors, or throttle you with rate limits
- LLMs occasionally return garbage — malformed JSON, infinite tool loops, hallucinated tool names
- A single long conversation can blow through your context window (and your budget)
- Without logging, you have no idea *why* something failed

Today you make your agent **survive the real world**. This is the difference between
a demo and something you can actually rely on.

---

## Part 1: What Goes Wrong (The Failure Catalog)

Before building solutions, you need to know the failure modes. There are 4 categories:

### 1a. API Failures (the network layer)
These come from the API itself, not the LLM's intelligence:

| Error | What It Means | How Often |
|-------|--------------|-----------|
| `429 Too Many Requests` | Rate limited — you're calling too fast | Very common |
| `500 Internal Server Error` | Provider is having issues | Occasional |
| `529 Overloaded` | API is at capacity (Anthropic-specific) | During peak times |
| `timeout` | Request took too long | With complex prompts |
| `APIConnectionError` | Network issue between you and API | Rare but happens |

Key insight: **429 and 529 are expected, not exceptional.** Your code MUST handle them.

### 1b. LLM Failures (the intelligence layer)
The API works fine, but the LLM does something unhelpful:

- **Infinite tool loops** — keeps calling tools without converging on an answer
- **Hallucinated tool names** — tries to call a tool that doesn't exist
- **Malformed arguments** — calls a real tool with wrong argument types
- **Refusal** — "I can't help with that" when it actually could
- **Context overflow** — conversation grows until it exceeds the context window

### 1c. Tool Failures (the execution layer)
Your tool code itself fails:

- External API the tool calls is down
- Tool returns unexpected data format
- Tool throws an unhandled exception
- Tool hangs / takes too long

### 1d. Cost Failures (the budget layer)
Everything works, but too expensively:

- Agent uses 10 steps when 2 would suffice
- Large conversation history gets re-sent every step (tokens add up fast)
- Using Opus when Haiku would suffice for simple routing

📖 **Read this for more on agent failure modes:**
https://www.promptingguide.ai/techniques/reflexion
→ Focus on: understanding how agents can self-reflect on failures
→ Skip: the academic paper details, just get the concept

---

## Part 2: Retry Logic with Tenacity

### Why not just try/except + time.sleep?

You could write retry logic by hand, but you'd need to handle:
- Exponential backoff (wait 1s, then 2s, then 4s, then 8s...)
- Max retry count
- Which exceptions to retry vs. which to raise immediately
- Jitter (random delay so multiple clients don't all retry at the same instant)

**Tenacity** handles all of this in a single decorator.

📖 **Read:**
https://tenacity.readthedocs.io/en/latest/
→ Focus on: "Basic Usage" and "Retry on exception" sections
→ Key concepts: `@retry`, `wait_exponential`, `stop_after_attempt`, `retry_if_exception_type`

### The Pattern

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import anthropic

# This decorator says:
#   - Retry up to 4 times
#   - Wait 1s, then 2s, then 4s between retries (exponential backoff)
#   - Only retry on these specific error types (not on every error!)
@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type((
        anthropic.RateLimitError,       # 429 — too many requests
        anthropic.InternalServerError,  # 500 — server error
        anthropic.APIConnectionError,   # network issues
    )),
    before_sleep=lambda retry_state: print(
        f"  ⏳ Retry {retry_state.attempt_number} after error: {retry_state.outcome.exception()}"
    ),
)
def call_llm(client, messages, tools, system_prompt):
    """Make an LLM call with automatic retry on transient errors."""
    return client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        tools=tools,
        messages=messages,
    )
```

### What NOT to retry

Not every error should be retried:
- `400 Bad Request` → your request is malformed, retrying won't help
- `401 Unauthorized` → wrong API key, retrying won't help  
- `ValidationError` → Pydantic says the data is wrong, retrying the *same* request gets the *same* bad data

Rule of thumb: **retry transient errors (network, rate limits, server errors), fail fast on client errors (bad request, auth, validation).**

---

## Part 3: Token Counting and Cost Tracking

### Why Tokens Matter

Every API call costs money based on tokens sent (input) and received (output).
A token is roughly ¾ of a word. The cost adds up because in an agent loop,
**you re-send the entire conversation history on every step.**

```
Step 1: Send 500 tokens  → Pay for 500 input tokens
Step 2: Send 1,200 tokens → Pay for 1,200 input (includes step 1's history!)
Step 3: Send 2,100 tokens → Pay for 2,100 input (includes steps 1+2!)
```

This is why agents can get expensive fast — it's not linear, it's cumulative.

### Current Pricing (March 2026)

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Best For |
|-------|-----------------------|------------------------|----------|
| Haiku 4.5 | $1.00 | $5.00 | Simple routing, classification |
| Sonnet 4.6 | $3.00 | $15.00 | Most agent tasks, coding |
| Opus 4.6 | $5.00 | $25.00 | Complex reasoning, large context |

### How to Count Tokens

Anthropic returns token counts directly in the response — no extra library needed:

```python
response = client.messages.create(...)

# These are available on every response:
input_tokens = response.usage.input_tokens    # tokens you sent
output_tokens = response.usage.output_tokens  # tokens the LLM generated
```

For *estimating* tokens before you send (useful for cost projection),
use `tiktoken` (OpenAI's tokenizer). It's not exact for Claude but close enough
for estimation:

```python
import tiktoken

# cl100k_base is the closest publicly available tokenizer
enc = tiktoken.get_encoding("cl100k_base")
token_count = len(enc.encode("Your text here"))
```

📖 **Read:**
https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
→ Focus on: "How strings are tokenized" and the basic encode/decode examples
→ Key takeaway: different text tokenizes very differently
   (code uses more tokens than English prose, non-Latin scripts use even more)

### The Cost Tracker Pattern

```python
class CostTracker:
    """Track token usage and costs across agent steps."""
    
    # Pricing per million tokens (update if prices change)
    PRICING = {
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    }

    def __init__(self, model: str):
        self.model = model
        self.steps = []          # list of per-step usage
        self.total_input = 0
        self.total_output = 0

    def record(self, response, step_num: int):
        """Record usage from an API response."""
        inp = response.usage.input_tokens
        out = response.usage.output_tokens
        self.total_input += inp
        self.total_output += out
        self.steps.append({"step": step_num, "input": inp, "output": out})

    @property
    def total_cost(self) -> float:
        prices = self.PRICING.get(self.model, {"input": 3.0, "output": 15.0})
        return (
            (self.total_input / 1_000_000) * prices["input"]
            + (self.total_output / 1_000_000) * prices["output"]
        )

    def summary(self) -> str:
        lines = [f"{'Step':<6} {'Input':>8} {'Output':>8}"]
        for s in self.steps:
            lines.append(f"{s['step']:<6} {s['input']:>8,} {s['output']:>8,}")
        lines.append(f"{'TOTAL':<6} {self.total_input:>8,} {self.total_output:>8,}")
        lines.append(f"Estimated cost: ${self.total_cost:.4f}")
        return "\n".join(lines)
```

---

## Part 4: Context Window Limits

### The Problem

Every LLM has a maximum context window — the total number of tokens it can process
in a single request (input + output combined):

| Model | Context Window |
|-------|---------------|
| Sonnet 4.6 | 1,000,000 tokens |
| Haiku 4.5 | 200,000 tokens |

This sounds enormous, but in an agent loop your messages grow every step.
After 20 steps of tool calling, you could easily have 50,000+ tokens of history.
And you're *paying for all of it every step*.

### Solutions

**1. Max steps** (you already have this from Day 2):
```python
for step in range(1, max_steps + 1):
    # ...
```

**2. Token budget** — stop if accumulated cost exceeds a threshold:
```python
if cost_tracker.total_cost > max_cost_usd:
    return "Stopping: cost budget exceeded."
```

**3. Conversation summarization** — when history gets long, summarize it:
```python
if total_tokens > 10_000:
    # Ask the LLM to summarize the conversation so far
    # Replace the full history with the summary
    summary = summarize_conversation(messages)
    messages = [{"role": "user", "content": f"Previous context: {summary}\n\n{original_query}"}]
```

**4. Sliding window** — keep only the last N messages:
```python
if len(messages) > max_messages:
    # Keep system context + first user message + last N messages
    messages = [messages[0]] + messages[-max_messages:]
```

---

## Part 5: Structured Logging with Loguru

Python's built-in `logging` (from Day 2) works, but **Loguru** is more ergonomic
for development — colored output, easy file logging, and structured data.

📖 **Read:**
https://loguru.readthedocs.io/en/stable/overview.html
→ Focus on: "Take the tour" section — the first 5 examples are all you need
→ Key concept: `logger.info()`, `logger.warning()`, `logger.error()`, and `logger.bind()`

### Loguru vs logging (quick comparison)

```python
# ── Built-in logging (Day 2 style) ──
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("agent")
log.info("Starting step 3")
# Output: 2026-03-31 14:32:07 [INFO] Starting step 3

# ── Loguru (Day 4 upgrade) ──
from loguru import logger
logger.info("Starting step 3")
# Output: 2026-03-31 14:32:07.123 | INFO | __main__:run_agent:42 - Starting step 3
#         ^^^ colored, includes file+line number, no setup needed

# Log to file AND console simultaneously:
logger.add("agent_run.log", rotation="1 MB")

# Structured data in logs (great for debugging agents):
logger.info("Tool call", tool="calculator", args={"expression": "125000000/47"}, step=3)
```

### What to Log in an Agent

Every agent run should produce a log that lets you reconstruct what happened:

| Event | What to Log | Why |
|-------|------------|-----|
| Query start | User query, max_steps, model | Know what was asked |
| Each LLM call | Step number, input token count | Track growth |
| Tool decision | Tool name, arguments | Debug wrong tool choices |
| Tool result | Result (truncated), duration | Find slow/broken tools |
| Retries | Error type, attempt number, wait time | Diagnose API issues |
| Final answer | Output, total steps, total cost | Measure efficiency |
| Failures | Error type, stack trace, conversation state | Debug crashes |

---

## Part 6: Graceful Degradation

This is a production concept: when something fails, don't crash — fall back to
something less ideal but still useful.

### Examples:

```python
# Fallback 1: If the primary model fails, try a cheaper one
def call_with_fallback(messages, tools, system_prompt):
    try:
        return call_llm(client, messages, tools, system_prompt, model="claude-sonnet-4-20250514")
    except Exception:
        logger.warning("Sonnet failed, falling back to Haiku")
        return call_llm(client, messages, tools, system_prompt, model="claude-haiku-4-5-20251001")

# Fallback 2: If a tool fails, tell the LLM (don't crash the loop)
try:
    result = TOOL_FUNCTIONS[tool_name](**tool_input)
except Exception as e:
    result = f"Tool error: {e}. Please try a different approach."
    # The LLM sees this error and can adapt!

# Fallback 3: If cost budget is exceeded, give a partial answer
if cost_tracker.total_cost > max_cost:
    return f"I've gathered partial information so far: {partial_answer}. " \
           f"Stopping due to cost limits (${cost_tracker.total_cost:.4f})."
```

The key insight: **feed errors back to the LLM as information, not as crashes.**
LLMs can often self-correct when told "that tool failed, try something else."

---

## Part 7: Putting It All Together — The Architecture

Here's how all of Day 4's concepts combine into one robust loop:

```
User Query
    │
    ▼
┌──────────────────────────────┐
│  Initialize:                 │
│  - CostTracker               │
│  - Logger                    │
│  - Step counter              │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  GUARD CHECKS:               │◄────────────────────┐
│  - max_steps exceeded?       │                      │
│  - cost budget exceeded?     │                      │
│  - context window near limit?│                      │
└──────────┬───────────────────┘                      │
           │ (all ok)                                 │
           ▼                                          │
┌──────────────────────────────┐                      │
│  CALL LLM (with retry)      │                      │
│  - tenacity handles 429/500  │                      │
│  - record tokens in tracker  │                      │
└──────────┬───────────────────┘                      │
           │                                          │
      stop_reason?                                    │
       │        │                                     │
  "tool_use"  "end_turn"                              │
       │        │                                     │
       ▼        ▼                                     │
  ┌──────────┐  Return final                          │
  │EXECUTE   │  answer + cost                         │
  │TOOL      │  summary                               │
  │(try/     │                                        │
  │ except)  │                                        │
  └────┬─────┘                                        │
       │                                              │
       ▼                                              │
  Append results                                      │
  to messages ────────────────────────────────────────┘
```

---

## Part 8: Your Hands-On Tasks

### Task 1: robust_agent_loop.py
Extend your Day 2 `minimal_agent_loop.py` with ALL of these features:
- Tenacity retry wrapper for LLM calls (handle 429, 500, 529, connection errors)
- CostTracker that records every step's token usage
- A cost budget (`max_cost_usd`) that stops the agent if exceeded
- Loguru logging for every key event (see the table in Part 5)
- Graceful tool error handling (catch exceptions, feed error message back to LLM)
- A model fallback (try Sonnet, fall back to Haiku if Sonnet fails)

### Task 2: failure_analysis.md  
Break your agent intentionally in at least 5 ways and document what happens:

1. Set `max_steps=1` and give it a query needing 3 tool calls — what does it say?
2. Give it a tool schema with an empty description — does it still choose the right tool?
3. Make a tool raise an exception — does the agent recover or crash?
4. Send a query that causes an infinite tool-calling loop — does max_steps save you?
5. Set `max_cost_usd=0.001` — at which step does it stop?

For each, document: what you did, what happened, and what you learned.

### Task 3: cost_tracker.py
Build a standalone CostTracker class that:
- Records per-step input/output tokens
- Calculates running cost based on model pricing
- Prints a formatted summary table at the end
- Can export results to JSON (for later analysis)
- Supports multiple models (Haiku, Sonnet, Opus)

Test it by running the same query through different models and comparing costs.

---

## Quick Reference: What to Install

```bash
pip install anthropic tiktoken tenacity loguru
```

## Key Readings (Targeted)

| Resource | What to Read | Time |
|----------|-------------|------|
| [Tenacity docs](https://tenacity.readthedocs.io/en/latest/) | "Basic Usage" through "Retry on exception" | 15 min |
| [Tiktoken cookbook](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken) | "How strings are tokenized" + basic examples | 15 min |
| [Loguru overview](https://loguru.readthedocs.io/en/stable/overview.html) | "Take the tour" — first 5 examples | 10 min |
| [Reflexion paper summary](https://www.promptingguide.ai/techniques/reflexion) | Just the concept, not the full paper | 10 min |
| [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) | Token costs + tool use token overhead | 5 min |

## Files to Commit

| File | What It Demonstrates |
|------|---------------------|
| `robust_agent_loop.py` | Production-ready agent with retry, cost tracking, logging, fallbacks |
| `failure_analysis.md` | 5+ documented failure experiments with learnings |
| `cost_tracker.py` | Reusable cost tracking module with multi-model support |
