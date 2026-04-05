"""
Day 5 — Exercise 2: Robust Parsing Pipeline (with exploration harness)
======================================================================

Run this file AS-IS to see what the API returns at each step.
Then implement the TODOs one at a time, re-running after each change.

Usage:
    python robust_parser.py

5 TODOs below — but first, run it and read the output!
"""

import json
import anthropic
from pydantic import BaseModel, Field, ValidationError, field_validator
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)


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

# ---------------------------------------------------------------------------
# Pydantic model with strict constraints (will trigger validation errors)
# ---------------------------------------------------------------------------

class BugReport(BaseModel):
    """Structured bug report extracted from user messages."""
    title: str = Field(description="Concise bug title, max 80 chars", max_length=80)
    severity: str = Field(description="One of: critical, high, medium, low")
    component: str = Field(description="Affected system component")
    steps_to_reproduce: list[str] = Field(
        description="Ordered steps to reproduce the bug",
        min_length=1,
    )
    expected_vs_actual: str = Field(
        description="What should happen vs what actually happens"
    )

    @field_validator("severity")
    @classmethod
    def severity_must_be_valid(cls, v):
        allowed = {"critical", "high", "medium", "low"}
        if v.lower() not in allowed:
            raise ValueError(f"severity must be one of {allowed}, got '{v}'")
        return v.lower()


# ---------------------------------------------------------------------------
# Helpers (from Exercise 1)
# ---------------------------------------------------------------------------

def pydantic_to_tool(model_class: type[BaseModel]) -> dict:
    return {
        "name": model_class.__name__.lower(),
        "description": model_class.__doc__ or "",
        "input_schema": model_class.model_json_schema(),
    }


# ---------------------------------------------------------------------------
# EXPLORATION: Run this to see what the API actually returns
# ---------------------------------------------------------------------------

def explore_api_response():
    """Call the API once and print EVERYTHING so you can see the shapes."""
    client = anthropic.Anthropic()
    tool_def = pydantic_to_tool(BugReport)

    print("=" * 60)
    print("STEP 1: What does the tool definition look like?")
    print("=" * 60)
    print(json.dumps(tool_def, indent=2))

    prompt = """
    Something weird happened in the dashboard yesterday. The charts were
    kind of glitchy and some of the numbers looked off. Not sure if it's
    a big deal or not. Might be related to the recent deploy.
    """

    messages = [{"role": "user", "content": prompt}]

    print("\n" + "=" * 60)
    print("STEP 2: Calling the API with forced tool use...")
    print("=" * 60)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=[tool_def],
        tool_choice={"type": "tool", "name": tool_def["name"]},
        messages=messages,
    )

    print(f"\nstop_reason: {response.stop_reason}")
    print(f"Number of content blocks: {len(response.content)}")

    # Find the tool_use block
    tool_block = None
    for i, block in enumerate(response.content):
        print(f"\n--- Block {i} ---")
        print(f"  type: {block.type}")
        if block.type == "tool_use":
            tool_block = block
            print(f"  id: {block.id}")
            print(f"  name: {block.name}")
            print(f"  input: {json.dumps(block.input, indent=4)}")
            print(f"  type of input: {type(block.input)}")

    if not tool_block:
        print("ERROR: No tool_use block found!")
        return

    raw_data = tool_block.input

    print("\n" + "=" * 60)
    print("STEP 3: Can we validate this with Pydantic?")
    print("=" * 60)

    try:
        result = BugReport(**raw_data)
        print(f"\nValidation PASSED!")
        print(f"  title: {result.title}")
        print(f"  severity: {result.severity}")
        print(f"  component: {result.component}")
        print(f"  steps: {result.steps_to_reproduce}")
    except ValidationError as e:
        print(f"\nValidation FAILED! This is what you'd feed back to the LLM:")
        print(f"  {str(e)}")

    print("\n" + "=" * 60)
    print("STEP 4: What would a retry message look like?")
    print("=" * 60)
    print("To retry, you'd append these two messages:\n")

    # Message 1: the assistant's failed attempt (echo back what it said)
    assistant_msg = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_block.id,
                "name": tool_block.name,
                "input": tool_block.input,
            }
        ],
    }
    print("Assistant message (echo back the tool call):")
    print(json.dumps(assistant_msg, indent=2))

    # Message 2: your error feedback
    error_msg = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "is_error": True,
                "content": "Validation failed: severity must be one of {'critical', 'high', 'medium', 'low'}. Please fix and try again.",
            }
        ],
    }
    print("\nUser message (error feedback):")
    print(json.dumps(error_msg, indent=2))

    print("\n" + "=" * 60)
    print("STEP 5: Full messages list for a retry would be:")
    print("=" * 60)
    retry_messages = messages + [assistant_msg, error_msg]
    print(f"  {len(retry_messages)} messages total:")
    for i, msg in enumerate(retry_messages):
        role = msg["role"]
        if isinstance(msg["content"], str):
            preview = msg["content"][:50] + "..."
        else:
            preview = f"[{len(msg['content'])} content blocks]"
        print(f"    [{i}] {role}: {preview}")


# ---------------------------------------------------------------------------
# TODO 1: Build the validation-retry loop
# ---------------------------------------------------------------------------
#   1. Starts with messages = [{"role": "user", "content": prompt}]
#   2. Loops up to (max_retries + 1) times (first attempt + retries)
#   3. Calls extract_with_block() to get (raw_data, tool_block)
#   4. Tries to validate with output_model(**raw_data)
#   5. If validation succeeds → return the model instance
#   6. If ValidationError is raised:
#       a. Log the validation errors
#       b. Append the failed attempt to messages as an assistant message:
#          {"role": "assistant", "content": [
#              {"type": "tool_use", "id": tool_block.id,
#               "name": tool_block.name, "input": tool_block.input}
#          ]}
#       c. Append error feedback as a user message:
#          {"role": "user", "content": [
#              {"type": "tool_result", "tool_use_id": tool_block.id,
#               "is_error": True,
#               "content": f"Validation failed: {str(e)}. Please fix and try again."}
#          ]}
#       d. Continue the loop (LLM will see the error and try again)
#   7. If all retries exhausted → raise the last ValidationError
#
# Run explore_api_response() first to see what all these pieces look like!
#
# The function needs a helper that returns BOTH raw_data AND the tool_block.
# Here it is:

def extract_with_block(
    client: anthropic.Anthropic,
    messages: list[dict],
    output_model: type[BaseModel],
) -> tuple[dict, object]:
    """Call Claude with forced tool use. Returns (raw_data, tool_block)."""
    tool_def = pydantic_to_tool(output_model)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=[tool_def],
        tool_choice={"type": "tool", "name": tool_def["name"]},
        messages=messages,
    )
    for block in response.content:
        if block.type == "tool_use":
            return block.input, block
    raise RuntimeError("No tool_use block in response")





# ---------------------------------------------------------------------------
# TODO 2: Add tenacity for API-level retries (outer layer)
# ---------------------------------------------------------------------------
# Decorate get_structured_with_retries with tenacity.
# Retry on anthropic.APIStatusError, NOT on ValidationError.

# Your tenacity-wrapped function here

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
def get_structured_with_retries(
    client: anthropic.Anthropic,
    prompt: str,
    output_model: type[BaseModel],
    max_retries: int = 2,
) -> BaseModel:
    """Extract structured output with validation retry loop."""
    messages = [{"role": "user", "content": prompt}]
    last_error = None
    
    logger.info(f"{'=' * 50}")
    logger.info(f"PROMPT: {prompt}")
    logger.info(f"{'=' * 50}")

    for attempt in range(max_retries + 1):
        try: 
            (raw_data, tool_block) = extract_with_block(client, messages, output_model)
            return output_model(**raw_data)
        except ValidationError as e:
            last_error = e
            logger.error(f"Validation failed on attempt {attempt + 1}: {e}")
            messages.append({"role": "assistant", "content": [
                {"type": "tool_use", "id": tool_block.id,
                "name": tool_block.name, "input": tool_block.input}
            ]})
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tool_block.id,
                "is_error": True,
                "content": f"Validation failed: {str(e)}. Please fix and try again."}
            ]})
    raise last_error

# ---------------------------------------------------------------------------
# TODO 3: Add logging throughout the pipeline
# ---------------------------------------------------------------------------
# Go back through your code and add loguru logging.

# ---------------------------------------------------------------------------
# TODO 4: Test with tricky inputs
# ---------------------------------------------------------------------------
def test_tricky_inputs():
    client = anthropic.Anthropic()

    vague_text = """
    Something weird happened in the dashboard yesterday. The charts were
    kind of glitchy and some of the numbers looked off. Not sure if it's
    a big deal or not. Might be related to the recent deploy.
    """

    # Uncomment once TODO 1 is done:
    result = get_structured_with_retries(client, vague_text, BugReport)
    print(f"Title: {result.title}")
    print(f"Severity: {result.severity}")
    print(f"Steps: {result.steps_to_reproduce}")


# ---------------------------------------------------------------------------
# TODO 5: Handle the "unfixable" case gracefully
# ---------------------------------------------------------------------------
def safe_extract(
    client: anthropic.Anthropic,
    text: str,
    output_model: type[BaseModel],
) -> BaseModel | str:
    """Try to extract structured output. Return model instance or error string."""
    try:
        return get_structured_with_retries(client, text, output_model)
    except Exception as e:
        print(f"Could not obtain structured output: {e}")



# ---------------------------------------------------------------------------
# Main — run exploration first, then switch to your implementations
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # print("\n>>> EXPLORATION MODE — see what the API returns\n")
    # explore_api_response()

    # Uncomment these as you implement each TODO:
    # print("\n\n>>> TESTING TODO 1 — validation retries\n")
    # test_tricky_inputs()

    # print("\n\n>>> TESTING TODO 5 — safe extract\n")
    client = anthropic.Anthropic()
    result = safe_extract(
        client,
        "What's the weather like today?",
        BugReport,
    )
    print(f"Non-bug-report result: {result}")