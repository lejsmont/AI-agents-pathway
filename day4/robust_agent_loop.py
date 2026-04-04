"""
Day 4 — Task 1: Robust Agent Loop
===================================
Take your Day 2 agent and make it production-ready.

Features to add:
  ✅ Retry logic with tenacity (429, 500, 529, connection errors)
  ✅ Token counting and cost tracking per step
  ✅ Cost budget (stop if exceeded)
  ✅ Structured logging with loguru
  ✅ Graceful tool error handling
  ✅ Model fallback (Sonnet → Haiku)

Run:    python robust_agent_loop.py
Install: pip install anthropic tenacity loguru tiktoken

YOUR TASKS (search for "TODO"):
  1. Complete the tenacity retry decorator
  2. Wire up the CostTracker
  3. Add the cost budget guard
  4. Handle tool execution errors gracefully
  5. Implement model fallback
"""

import json
import time
import anthropic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from loguru import logger

# ──────────────────────────────────────────────
# LOGGING SETUP
# Loguru works out of the box — no basicConfig needed.
# Let's also log to a file so we can review later.
# ──────────────────────────────────────────────

# Remove default handler (optional — keeps console output clean)
logger.remove()
# Add console with color + abbreviated format
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level="INFO",
    colorize=True,
)
# Add file logger — captures everything, rotates at 5MB
logger.add("agent_runs.log", rotation="5 MB", level="DEBUG")


# ──────────────────────────────────────────────
# COST TRACKER
# Study this — you'll build your own standalone version in Task 3
# ──────────────────────────────────────────────

class CostTracker:
    """Track token usage and estimated costs across agent steps."""

    PRICING = {
        # Model name → (input $/1M tokens, output $/1M tokens)
        "claude-sonnet-4-20250514":  {"input": 3.00, "output": 15.00},
        "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    }

    def __init__(self, model: str):
        self.model = model
        self.steps: list[dict] = []
        self.total_input = 0
        self.total_output = 0

    def record(self, response, step_num: int):
        """Record token usage from an API response."""
        inp = response.usage.input_tokens
        out = response.usage.output_tokens
        self.total_input += inp
        self.total_output += out
        step_data = {"step": step_num, "input_tokens": inp, "output_tokens": out}
        self.steps.append(step_data)
        logger.debug(f"Step {step_num} tokens: {inp:,} in / {out:,} out")

    @property
    def total_cost(self) -> float:
        prices = self.PRICING.get(self.model, {"input": 3.0, "output": 15.0})
        return (
            (self.total_input / 1_000_000) * prices["input"]
            + (self.total_output / 1_000_000) * prices["output"]
        )

    def print_summary(self):
        """Print a formatted cost summary table."""
        logger.info("─── Cost Summary ───")
        logger.info(f"  {'Step':<6} {'Input':>10} {'Output':>10}")
        for s in self.steps:
            logger.info(f"  {s['step']:<6} {s['input_tokens']:>10,} {s['output_tokens']:>10,}")
        logger.info(f"  {'TOTAL':<6} {self.total_input:>10,} {self.total_output:>10,}")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  Estimated cost: ${self.total_cost:.6f}")


# ──────────────────────────────────────────────
# TOOLS (same as Day 2, but with intentional error potential)
# ──────────────────────────────────────────────

def calculator(expression: str) -> str:
    """Safely evaluate a math expression."""
    allowed = set("0123456789+-*/().% ")
    if not all(ch in allowed for ch in expression):
        raise ValueError(f"Unsafe characters in expression: '{expression}'")
    return str(eval(expression))

def get_weather(city: str) -> str:
    """Mock weather — sometimes fails to simulate real-world issues."""
    mock_data = {
        "london": "Cloudy, 14°C, 70% humidity",
        "tokyo": "Rainy, 22°C, 85% humidity",
        "warsaw": "Partly cloudy, 18°C, 55% humidity",
    }
    result = mock_data.get(city.lower())
    if result is None:
        # Simulate a tool failure for unknown cities
        raise KeyError(f"Weather service has no data for '{city}'")
    return result


def slow_tool(query: str) -> str:
    """A deliberately slow tool — for testing timeout patience."""
    time.sleep(3)
    return f"Slow result for: {query}"


TOOL_REGISTRY = {
    "calculator": calculator,
    "get_weather": get_weather,
    "slow_tool": slow_tool,
}

TOOL_SCHEMAS = [
    {
        "name": "calculator",
        "description": "",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression, e.g. '(10 + 5) * 3'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'London'",
                }
            },
            "required": ["city"],
        },
    },
]


# ──────────────────────────────────────────────
# LLM CALL WITH RETRY
#
# TODO 1: Complete the retry decorator.
#
# The decorator should:
#   - Stop after 4 attempts
#   - Wait exponentially: 1s → 2s → 4s (use min=1, max=30)
#   - Only retry on these Anthropic error types:
#       anthropic.RateLimitError      (429)
#       anthropic.InternalServerError (500)
#       anthropic.APIConnectionError  (network)
#   - Log each retry attempt using logger.warning
#
# Hints:
#   - Use @retry(...) decorator from tenacity
#   - stop=stop_after_attempt(4)
#   - wait=wait_exponential(multiplier=1, min=1, max=30)
#   - retry=retry_if_exception_type((...))
# ──────────────────────────────────────────────

# TODO: Add the @retry decorator here
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
def call_llm(client, messages, tools, system_prompt, model="claude-sonnet-4-20250514"):
    """Make an LLM call with automatic retry on transient errors."""
    logger.debug(f"Calling {model}...")
    return client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_prompt,
        tools=tools,
        messages=messages,
    )


# ──────────────────────────────────────────────
# THE ROBUST AGENT LOOP
# ──────────────────────────────────────────────

def run_agent(
    user_query: str,
    max_steps: int = 5,
    max_cost_usd: float = 0.05,
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """
    Production-ready agent loop with retry, cost tracking, and graceful degradation.

    Args:
        user_query:   What the user wants.
        max_steps:    Safety limit on iterations.
        max_cost_usd: Maximum dollar cost before stopping.
        model:        Which Claude model to use.
    """
    client = anthropic.Anthropic()
    cost_tracker = CostTracker(model)
    messages = [{"role": "user", "content": user_query}]
    system_prompt = "You are a helpful assistant with access to tools. Use them when needed."

    logger.info(f"{'=' * 50}")
    logger.info(f"QUERY: {user_query}")
    logger.info(f"Model: {model} | Max steps: {max_steps} | Budget: ${max_cost_usd}")
    logger.info(f"{'=' * 50}")

    for step in range(1, max_steps + 1):
        logger.info(f"--- Step {step}/{max_steps} ---")

        # ── GUARD: Cost budget check ──
        # TODO 3: Check if cost_tracker.total_cost exceeds max_cost_usd
        # If it does:
        #   - Log a warning
        #   - Print the cost summary
        #   - Return a message explaining the agent stopped due to budget
        # This prevents runaway costs in production!
        if cost_tracker.total_cost > max_cost_usd:
            logger.warning("Token Cost exceeded budget")
            cost_tracker.print_summary()
            return "Stop: cost exceeded max budget"
        
        # TODO: Your cost budget guard here

        # ── CALL LLM (with retry) ──
        try:
            response = call_llm(client, messages, TOOL_SCHEMAS, system_prompt, model)
        except Exception as e:
            # TODO 5: Implement model fallback
            # If the primary model fails after all retries, try a cheaper model.
            # Steps:
            #   1. Log a warning: f"Primary model {model} failed: {e}"
            #   2. If model isn't already Haiku, try again with Haiku
            #   3. If Haiku also fails, raise the error
            #
            # Hint: call call_llm again with model="claude-haiku-4-5-20251001"
            logger.warning(f"Primary model {model} failed: {e}")
            if model != "claude-haiku-4-5-20251001":
                try:
                    model = "claude-haiku-4-5-20251001"
                    response = call_llm(client, messages, TOOL_SCHEMAS, system_prompt, model)
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    raise RuntimeError(f"Both primary and fallback models failed. No more fallback models available.") from fallback_error
            else:
                raise RuntimeError(
                    f"Model {model} failed and it is already the last fallback option. "
                    f"No more fallback models available. Original error: {e}"
                ) from e
        # TODO: Replace this with fallback logic

        # ── RECORD COST ──
        # TODO 2: Record this step's token usage in the cost tracker
        # Hint: cost_tracker.record(response, step)
        cost_tracker.record(response,step)

        logger.info(f"Stop reason: {response.stop_reason}")

        # ── TOOL USE ──
        if response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type == "text" and block.text:
                    logger.info(f"LLM: {block.text}")

                elif block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    logger.info(f"TOOL: {tool_name}({json.dumps(tool_input)})")

                    # TODO 4: Handle tool execution errors gracefully
                    #
                    # Right now, if the tool crashes, the whole agent crashes.
                    # Wrap the tool execution in try/except:
                    #   - If it succeeds: use the result
                    #   - If it fails: set result to an error message string
                    #     like f"Tool '{tool_name}' failed: {error}. Try a different approach."
                    #
                    # Why? The LLM can READ this error message and adapt!
                    # It might try different arguments, a different tool, or answer without tools.

                    # Current (fragile) version:
                    try:
                        fn = TOOL_REGISTRY.get(tool_name)
                        if fn is None:
                            result = f"Tool '{tool_name}' does not exist. Available tools: {list(TOOL_REGISTRY.keys())}"
                        else:
                            result = fn(**tool_input)
                    except Exception as error:
                        result = f"Tool '{tool_name}' failed: {error}. Try a different approach."

                    # TODO: Wrap the above in try/except

                    logger.info(f"RESULT: {str(result)[:200]}")  # truncate long results

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            logger.info(f"Current message length: {len(str(messages))}")

        # ── FINAL ANSWER ──
        elif response.stop_reason == "end_turn":
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text

            logger.info(f"DONE in {step} step(s)")
            cost_tracker.print_summary()
            return final_text

    # ── HIT MAX STEPS ──
    logger.warning(f"Hit max_steps ({max_steps}) without final answer!")
    cost_tracker.print_summary()
    return "I couldn't complete this task within the step limit."


# ──────────────────────────────────────────────
# TEST SCENARIOS
# Run these and observe the logs carefully!
# ──────────────────────────────────────────────

if __name__ == "__main__":

    # # ── Test 1: Normal operation (should work fine) ──
    # print("\n" + "=" * 60)
    # print("TEST 1: Normal operation")
    # print("=" * 60)
    # answer = run_agent("What's the weather in Warsaw?")
    # print(f"\n💬 {answer}")

    # # ── Test 2: Multi-step (watch the costs accumulate) ──
    # print("\n" + "=" * 60)
    # print("TEST 2: Multi-step with cost tracking")
    # print("=" * 60)
    # answer = run_agent(
    #     "What's the weather in Tokyo? Also, if the temperature is 22°C, "
    #     "what is that in Fahrenheit? (formula: C * 9/5 + 32)"
    # )
    # print(f"\n💬 {answer}")

    # # ── Test 3: Tool failure (should recover gracefully) ──
    # print("\n" + "=" * 60)
    # print("TEST 3: Tool failure recovery")
    # print("=" * 60)
    # answer = run_agent("What's the weather in Atlantis?")  # city doesn't exist
    # print(f"\n💬 {answer}")

    # # ── Test 4: Tight budget (should stop early) ──
    # print("\n" + "=" * 60)
    # print("TEST 4: Cost budget limit")
    # print("=" * 60)
    # answer = run_agent(
    #     "Tell me the weather in London, Tokyo, and Warsaw, "
    #     "then calculate the average temperature.",
    #     max_cost_usd=0.001,  # very tight budget!
    # )
    # print(f"\n💬 {answer}")

    # # ── Test 5: Max steps = 1 (can it handle a multi-tool query?) ──
    # print("\n" + "=" * 60)
    # print("TEST 5: Step limit")
    # print("=" * 60)
    # answer = run_agent(
    #     "Weather in Tokyo plus calculate 22 * 9/5 + 32",
    #     max_steps=1,
    # )
    # print(f"\n💬 {answer}")
    answer = run_agent("Weather in London, Tokyo, and Warsaw. Average the temps.")
    print(f"\n💬 {answer}")