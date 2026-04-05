"""
Day 5 — Exercise 1: Structured Output Basics
=============================================

Learn the core pattern: force the Anthropic API to return structured data
by (ab)using tool definitions as output schemas.

You'll build:
1. A helper that converts any Pydantic model → Anthropic tool definition
2. A function that sends a prompt and gets back a validated Pydantic object
3. A practical example: extracting structured info from messy text

Prerequisites:
    pip install anthropic pydantic

5 TODOs below. Work through them in order.
"""

import json
import anthropic
from pydantic import BaseModel, Field, ValidationError


# ---------------------------------------------------------------------------
# Pydantic models — these define the SHAPE of data you want back from the LLM
# ---------------------------------------------------------------------------

class MeetingInfo(BaseModel):
    """Structured representation of a meeting extracted from natural language."""
    title: str = Field(description="Short title for the meeting")
    date: str = Field(description="Date in YYYY-MM-DD format")
    time: str = Field(description="Time in HH:MM 24-hour format")
    attendees: list[str] = Field(description="List of attendee names")
    action_items: list[str] = Field(
        description="Key action items or decisions from the meeting",
        default_factory=list,
    )


class SentimentResult(BaseModel):
    """Sentiment analysis output."""
    sentiment: str = Field(description="One of: positive, negative, neutral, mixed")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)
    reasoning: str = Field(description="Brief explanation of the sentiment assessment")


# ---------------------------------------------------------------------------
# TODO 1: Convert a Pydantic model into an Anthropic tool definition
# ---------------------------------------------------------------------------
# The Anthropic API expects tools in this shape:
#
# {
#     "name": "tool_name",
#     "description": "What this tool does",
#     "input_schema": { ... JSON Schema ... }
# }
#
# Pydantic gives you a JSON Schema via: MyModel.model_json_schema()
#
# Write this function:
#   - Use the model's class name (lowercased) as the tool name
#   - Use the model's docstring as the description
#   - Use model_json_schema() as the input_schema
#
# Hint: MyModel.__name__ gives the class name, MyModel.__doc__ gives docstring.

def pydantic_to_tool(model_class: type[BaseModel]) -> dict:
    """Convert a Pydantic model class into an Anthropic tool definition."""
    return {
        "name": model_class.__name__.lower(),
        "description": model_class.__doc__,
        "input_schema": model_class.model_json_schema()
    }


# ---------------------------------------------------------------------------
# TODO 2: Call the Anthropic API with forced tool use
# ---------------------------------------------------------------------------
# Write this function:
#   - Create the tool definition using your pydantic_to_tool() helper
#   - Call client.messages.create() with:
#       * model="claude-sonnet-4-20250514"
#       * max_tokens=1024
#       * tools=[your_tool_def]
#       * tool_choice={"type": "tool", "name": <tool_name>}  ← FORCE it
#       * messages=[{"role": "user", "content": prompt}]
#   - Find the tool_use block in the response content
#   - Return the tool_use block's "input" dict (the structured data)
#
# Remember from Day 3: response.content is a list of blocks.
# When you force tool use, one block will have type == "tool_use".

def extract_structured(
    client: anthropic.Anthropic,
    prompt: str,
    output_model: type[BaseModel],
) -> dict:
    """Send prompt to Claude and force a structured response matching output_model."""
    p_tool = pydantic_to_tool(output_model)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=[p_tool],
        tool_choice={"type": "tool", "name": p_tool["name"]},
        messages=[{"role": "user", "content": prompt}]
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input


# ---------------------------------------------------------------------------
# TODO 3: Validate the raw dict against the Pydantic model
# ---------------------------------------------------------------------------
# Write this function:
#   - Take the raw dict from extract_structured()
#   - Validate it by constructing the Pydantic model: output_model(**raw_data)
#   - If validation succeeds, return the Pydantic model instance
#   - If ValidationError is raised, print the errors and re-raise
#
# Why separate step? In Exercise 2, you'll add retry logic between
# extraction and validation. Keep them decoupled.

def validate_output(raw_data: dict, output_model: type[BaseModel]) -> BaseModel:
    """Validate raw dict against a Pydantic model. Returns model instance or raises."""
    try:
        return output_model(**raw_data)
    except ValidationError as e:
        print(f"Validation Error: {e}")
        raise

    

# ---------------------------------------------------------------------------
# TODO 4: Put it all together — extract + validate in one call
# ---------------------------------------------------------------------------
# Write this function:
#   - Call extract_structured() to get raw data
#   - Call validate_output() to validate it
#   - Return the validated Pydantic model instance
#
# Keep it simple — no retry logic yet (that's Exercise 2).

def get_structured_output(
    client: anthropic.Anthropic,
    prompt: str,
    output_model: type[BaseModel],
) -> BaseModel:
    """Extract and validate structured output from the LLM."""
    raw_data = extract_structured(client, prompt, output_model)
    result = validate_output(raw_data, output_model)
    return result


# ---------------------------------------------------------------------------
# TODO 5: Test with both models on real-ish text
# ---------------------------------------------------------------------------
# Uncomment and complete the test code below.
# Run it and inspect the output:
#   - Are the fields populated correctly?
#   - Does the date come back in YYYY-MM-DD format as specified?
#   - What happens if you give it ambiguous text?

def main():
    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

    # --- Test 1: Meeting extraction ---
    meeting_text = """
    Hey team, just a quick recap of our standup from March 15th at 2:30pm.
    Present were Alice, Bob, and Charlie. We decided to move the deadline
    for the data pipeline to next Friday. Bob will handle the schema migration,
    and Alice is going to set up monitoring dashboards. Charlie mentioned
    he'll be OOO next week.
    """

    # Uncomment and complete:
    # meeting = get_structured_output(client, meeting_text, MeetingInfo)
    # print(f"Title: {meeting.title}")
    # print(f"Date: {meeting.date}")
    # print(f"Time: {meeting.time}")
    # print(f"Attendees: {meeting.attendees}")
    # print(f"Action items: {meeting.action_items}")

    print("\n" + "=" * 50 + "\n")

    # --- Test 2: Sentiment analysis ---
    review_text = """
    The new feature works but the onboarding is confusing. I spent 20 minutes
    just figuring out where to click. Once I got past that, it was actually
    pretty powerful. Mixed feelings overall.
    """

    # Uncomment and complete:
    sentiment = get_structured_output(client, review_text, SentimentResult)
    print(f"Sentiment: {sentiment.sentiment}")
    print(f"Confidence: {sentiment.confidence}")
    print(f"Reasoning: {sentiment.reasoning}")

    # --- Bonus: try it with your own text! ---


if __name__ == "__main__":
    main()
