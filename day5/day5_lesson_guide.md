# Day 5: Structured Outputs & Parsing

## Why This Matters for Agents

In Days 2–4, your agent loop received free-text responses from the LLM and used tool calls.
But what happens when you need the LLM to return **data**, not prose?

Examples in real agent systems:
- "Extract these 5 fields from the user's request"
- "Return a plan as a list of steps with priorities"
- "Classify this input into one of 4 categories with a confidence score"

Free-text is unreliable for this. You need **structured outputs** — guaranteed-shape responses
you can parse programmatically without fragile regex.

---

## The Spectrum of Approaches (Least → Most Reliable)

### 1. Prompt Engineering ("Please return JSON")
You just ask nicely. Sometimes you get JSON, sometimes you get ```json``` fences,
sometimes you get "Sure! Here's the JSON:" followed by JSON. Fragile.

### 2. JSON Mode (OpenAI-specific)
OpenAI's `response_format={"type": "json_object"}` guarantees valid JSON,
but does NOT enforce a schema. You could get any valid JSON back.

### 3. Tool Use as Structured Output (Anthropic pattern)
This is the key technique for today. You already know tool calling from Day 3.
The insight: **you can define a "tool" that isn't a real tool — it's just a schema
you want the LLM to fill in.** When the model "calls" this fake tool, you get
structured data matching your schema.

How it works:
- Define a tool with the JSON schema of your desired output
- Send your prompt with this tool available
- Set `tool_choice={"type": "tool", "name": "your_tool_name"}` to FORCE the model to use it
- The model's `tool_use` response contains your structured data

### 4. Instructor Library
A popular Python library that wraps the above patterns with a clean Pydantic interface.
Works with Anthropic, OpenAI, and others. Handles retries on validation failure automatically.

---

## Today's Reading (targeted sections)

1. **Anthropic — Force tool use for structured output**
   https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview#forcing-tool-use
   Read the "Forcing tool use" section and the example. This is the core pattern.

2. **Anthropic — Extracting structured JSON**
   https://docs.anthropic.com/en/docs/test-and-evaluate/strengthen-guardrails/increase-consistency#structured-output-json-mode
   Read the "Structured output (JSON mode)" section — shows the tool_use trick explicitly.

3. **Pydantic docs — Model fundamentals**
   https://docs.pydantic.dev/latest/concepts/models/
   Skim "Basic model usage" and "Model methods and properties" sections.
   You touched Pydantic in Day 3 — today you'll use it for validation, not just schemas.

4. **Instructor library — Getting started (Anthropic)**
   https://python.useinstructor.com/integrations/anthropic/
   Read the quick-start example. Notice how it maps Pydantic models → API calls → validated objects.

5. **Instructor — Retry logic on validation failure**
   https://python.useinstructor.com/concepts/retrying/
   Skim this — notice how it feeds Pydantic validation errors back to the LLM (sound familiar from Day 4?).

---

## Key Concepts

### Tool Use as Structured Output — The Mental Model

```
Normal tool calling (Day 3):
  LLM decides to call a tool → you EXECUTE the tool → return result to LLM

Structured output trick (today):
  You FORCE the LLM to call a "tool" → you KEEP the arguments as your data → done
```

The tool's input_schema IS your output schema. The model fills it in like a form.

### Pydantic as Your Validation Layer

Pydantic isn't just for defining schemas — it validates, coerces types, and gives
structured error messages. When the LLM returns data that doesn't match:

```python
from pydantic import BaseModel, Field, field_validator

class TaskPlan(BaseModel):
    goal: str = Field(description="What the agent should accomplish")
    steps: list[str] = Field(description="Ordered list of steps", min_length=1)
    priority: int = Field(description="1-5 priority scale", ge=1, le=5)

    @field_validator("steps")
    @classmethod
    def steps_not_empty(cls, v):
        return [s for s in v if s.strip()]  # filter blank steps
```

If the LLM returns `{"priority": 7}`, Pydantic raises a `ValidationError` with a
clear message you can feed back to the LLM for a retry (the Instructor pattern).

### Converting Pydantic Models to Tool Schemas

You'll write a helper that converts a Pydantic model → Anthropic tool schema.
This is what Instructor does under the hood:

```
Pydantic model → .model_json_schema() → Anthropic tool definition → force tool call → validate response
```

---

## Exercises

### Exercise 1: `structured_output_basics.py`
Learn the core pattern: force tool use to get structured data from the Anthropic API.
Build a helper that converts Pydantic models to tool schemas.
5 TODOs.

### Exercise 2: `robust_parser.py`
Build a parsing pipeline with fallbacks: try structured output, handle validation errors,
retry with error feedback. Integrates Day 4's retry and error-handling patterns.
5 TODOs.

### Exercise 3: `structured_agent.py`
Integrate structured outputs into your agent loop. The agent uses BOTH real tools
AND structured output extraction — knowing when to use each.
5 TODOs.

---

## How This Connects to Your Databricks Work

Structured outputs are essential for:
- **Data extraction pipelines**: LLM reads unstructured text → returns structured rows for Delta tables
- **Classification tasks**: Consistent labels/scores for ML pipelines
- **Agent orchestration**: One agent returns a structured plan, another executes it
- **Guardrails**: Validate LLM outputs before they hit production systems

In Databricks, you'll often chain: raw text → LLM structured extraction → Pydantic validation → DataFrame → Delta table.
