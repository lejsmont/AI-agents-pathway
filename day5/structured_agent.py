"""
Day 5 — Exercise 3: Structured Agent Loop
==========================================

Integrate structured outputs into your agent loop from Days 2–4.

Real agents need BOTH:
  - Tool calling: LLM decides to use a tool → you execute it → return results
  - Structured extraction: you force the LLM to return data in a specific shape

The challenge: these are two different interaction patterns, and you need to
know when to use which. In this exercise, the agent:

  1. Has real tools (search, calculator) it can call freely
  2. After the conversation is done, you extract a structured summary

Think of it as:
  - DURING the loop: tools help the agent gather information
  - AFTER the loop: structured output extracts the answer in a usable format

Prerequisites:
    pip install anthropic pydantic loguru

5 TODOs below.
"""

import json
import anthropic
from pydantic import BaseModel, Field, ValidationError
from loguru import logger

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
# Pydantic model for the final structured output
# ---------------------------------------------------------------------------

class ResearchResult(BaseModel):
    """Structured output from a research agent."""
    question: str = Field(description="The original question that was researched")
    answer: str = Field(description="Concise answer in 1-3 sentences")
    sources_used: list[str] = Field(
        description="Which tools/sources were consulted",
        default_factory=list,
    )
    confidence: str = Field(description="One of: high, medium, low")
    follow_up_questions: list[str] = Field(
        description="2-3 suggested follow-up questions",
        default_factory=list,
    )


# ---------------------------------------------------------------------------
# Simulated tools (same pattern as Day 3, simplified for focus)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": "Search an internal knowledge base for information. Returns relevant text snippets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": "Perform mathematical calculations. Accepts a math expression string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate, e.g. '(100 * 0.15) + 50'",
                }
            },
            "required": ["expression"],
        },
    },
]

# Simulated knowledge base
KNOWLEDGE_BASE = {
    "pricing": "Our standard plan is $49/month. Enterprise plan is $199/month. Both include API access. Annual billing saves 20%.",
    "api limits": "Standard plan: 10,000 API calls/month. Enterprise: 100,000 API calls/month. Rate limit: 100 calls/minute.",
    "features": "Standard includes: dashboard, API access, email support. Enterprise adds: SSO, priority support, custom integrations, SLA.",
    "sla": "Enterprise SLA: 99.9% uptime guarantee. Standard plan has no SLA. Downtime credits available for Enterprise.",
}


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool call and return the result as a string."""
    if tool_name == "search_knowledge_base":
        query = tool_input["query"].lower()
        results = []
        for key, value in KNOWLEDGE_BASE.items():
            if any(word in query for word in key.split()):
                results.append(value)
        if results:
            return "\n".join(results)
        return "No relevant results found. Try different search terms."

    elif tool_name == "calculator":
        try:
            # Safe eval for simple math (in production, use a proper parser)
            allowed_chars = set("0123456789+-*/().% ")
            expr = tool_input["expression"]
            if all(c in allowed_chars for c in expr):
                return str(eval(expr))
            return "Error: expression contains invalid characters"
        except Exception as e:
            return f"Calculation error: {e}"

    return f"Unknown tool: {tool_name}"


# ---------------------------------------------------------------------------
# TODO 1: Build the agent loop (from Days 2-4, streamlined)
# ---------------------------------------------------------------------------
# Write the agent loop that:
#   1. Takes a user question and sends it to Claude with the TOOLS available
#   2. If stop_reason == "tool_use": execute the tools, append results, continue
#   3. If stop_reason == "end_turn": the agent is done thinking, break
#   4. Safety: max 10 iterations
#   5. Returns the full message history (you'll need it for TODO 3)
#
# This is a review/consolidation of Days 2-4. Keep it clean:
#   - Use loguru for logging tool calls and results
#   - Handle unknown tools gracefully (return error message to LLM)
#   - No need for tenacity/cost tracking here (keep focus on structured output)

def run_agent_loop(
    client: anthropic.Anthropic,
    question: str,
    max_iterations: int = 10,
) -> list[dict]:
    """Run the agent loop. Returns the full message history."""
    messages = [{"role": "user", "content": question}]
    model = "claude-sonnet-4-20250514"

    for step in range(1, max_iterations + 1):
        logger.info(f"--- Step {step}/{max_iterations} ---")

        # ── CALL LLM (with retry) ──
        
        logger.debug(f"Calling ...{model}")
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )
    
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
                    try:
                        result = execute_tool(tool_name, tool_input)
                    except Exception as e:
                        result = f"Tool '{tool_name}' failed: {e}"
                    
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
            logger.info(f"DONE in {step} step(s)")
            return messages

    # ── HIT MAX STEPS ──
    logger.warning(f"Hit max_steps ({max_iterations}) without final answer!")
    return messages
            


# ---------------------------------------------------------------------------
# TODO 2: Extract structured output from the conversation
# ---------------------------------------------------------------------------
# After the agent loop finishes, you have a conversation history where the
# agent gathered information using tools. Now extract a structured summary.
#
# Write a function that:
#   1. Takes the message history from the agent loop
#   2. Builds a NEW API call that:
#       - Includes the conversation history for context
#       - Appends a new user message: "Based on your research above,
#         provide a structured summary."
#       - Uses the ResearchResult tool schema (forced tool choice)
#       - Does NOT include the real tools (search, calculator) —
#         only the structured output "tool"
#   3. Validates the output with the Pydantic model
#   4. Returns the ResearchResult
#
# Key insight: this is a SEPARATE API call from the agent loop.
# The agent loop uses real tools freely. The extraction step forces
# a specific output shape.

def pydantic_to_tool(model_class: type[BaseModel]) -> dict:
    """Convert a Pydantic model class into an Anthropic tool definition."""
    return {
        "name": model_class.__name__.lower(),
        "description": model_class.__doc__,
        "input_schema": model_class.model_json_schema()
    }

def extract_research_result(
    client: anthropic.Anthropic,
    conversation_history: list[dict],
) -> ResearchResult:
    """Extract structured ResearchResult from agent conversation."""
    # Step 1: Take the existing conversation and add one more message
    messages = conversation_history + [
        {"role": "user", "content": "Based on your research above, provide a structured summary."}
    ]

    # Step 2: Build the ResearchResult tool definition
    # (same pattern as Exercise 1 — pydantic_to_tool)
    research_tool = pydantic_to_tool(ResearchResult)

    # Step 3: Call the API with ONLY the structured output tool
    # (no TOOLS list — just your fake tool, with forced tool_choice)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=[research_tool],
        tool_choice={"type": "tool", "name": research_tool["name"]},
        messages=messages
    )

    # Step 4: Find the tool_use block, validate with ResearchResult(**raw_data)
    for block in response.content:
        if block.type == "tool_use":
            try:
                return ResearchResult(**block.input)
            except ValidationError as e:
                print(f"Validation Error: {e}")
                raise

    raise RuntimeError("No tool_use block in response")
    # Step 5: Return the validated ResearchResult
  

# ---------------------------------------------------------------------------
# TODO 3: Wire it together — agent loop → structured extraction
# ---------------------------------------------------------------------------
# Write the main pipeline function:
#   1. Run the agent loop to research the question
#   2. Extract structured output from the conversation
#   3. Return the validated ResearchResult
#
# Think about error handling:
#   - What if the agent loop produces no useful info?
#   - What if structured extraction fails validation?

def research(client: anthropic.Anthropic, question: str) -> ResearchResult:
    """Full research pipeline: agent loop → structured extraction."""
    try:
        messages = run_agent_loop(client, question)
    except Exception as e:
        logger.error(f"Agent loop failed: {e}")
        return f"Agent loop failed: {e}"

    try:
        return extract_research_result(client, messages)
    except ValidationError as e:
        logger.error(f"Structured extraction failed validation: {e}")
        return f"Extraction failed validation: {e}"
    except RuntimeError as e:
        logger.error(f"No structured output returned: {e}")
        return f"Extraction failed: {e}"


# ---------------------------------------------------------------------------
# TODO 4: Test with questions that exercise different tools
# ---------------------------------------------------------------------------
# Run your pipeline with these questions and verify the structured output
# makes sense:
#
#   a) "What's the price difference between standard and enterprise annually?"
#      (Should use both search AND calculator)
#
#   b) "Does the standard plan include SSO?"
#      (Should use search, answer is 'no')
#
#   c) "What's the uptime guarantee and how many minutes of downtime per year
#       does 99.9% allow?"
#      (Should use search AND calculator: 365 * 24 * 60 * 0.001)
#
# For each, print the full ResearchResult and check:
#   - Does sources_used accurately reflect which tools were called?
#   - Is the confidence level reasonable?
#   - Are follow_up_questions relevant?

def main():
    client = anthropic.Anthropic()

    questions = [
        "What's the price difference between standard and enterprise annually?",
        "Does the standard plan include SSO?",
        "What's the uptime guarantee and how many minutes of downtime per year does 99.9% allow?",
    ]

    for q in questions:
        print(f"\n{'=' * 60}")
        print(f"Question: {q}")
        print("=" * 60)

        # Uncomment once you've implemented the pipeline:
        result = research(client, q)
        print(f"Answer: {result.answer}")
        print(f"Sources: {result.sources_used}")
        print(f"Confidence: {result.confidence}")
        print(f"Follow-ups: {result.follow_up_questions}")


# ---------------------------------------------------------------------------
# TODO 5: Think about the design tradeoffs (written reflection)
# ---------------------------------------------------------------------------
# After running the tests, add your thoughts as comments below.
# No code needed — this is about understanding the pattern.
#
# Questions to consider:
#
# 1. Why is structured extraction a SEPARATE API call instead of forcing
#    tool_choice during the agent loop? What would break if you forced it?
#
# 2. The agent might call search 3 times and calculator twice in the loop.
#    The structured extraction then makes one more API call. That's 6+ API
#    calls for one question. How does this affect cost? When is it worth it?
#
# 3. In your Databricks work, imagine an agent that reads support tickets
#    and extracts structured data. Would you use this two-phase pattern
#    (agent loop → extraction), or just a single forced-tool-use call?
#    When would each approach make sense?

# Your reflections here:
# 1. It would force using one tool, instead of using a tool that is best designed for the task. 
# Structure extraction is not really "tool pattern" but rather hacking "tool pattern" for a different purpose.
# 2. Each API call will cost and costs grows as with each API call the whole conversation is send which grows
# cumulatively. It make sense if there is big value out of it, e.g. a table or dashboard used by 100s people.
# It does not make sense for ad-hoc, one-off queries. 
# 3. I would use 2 phase pattern if an agent requires extra steps to analyze the problem. Use 1 phase approach
# if the query contains all info and only requires structuring.


if __name__ == "__main__":
    main()
