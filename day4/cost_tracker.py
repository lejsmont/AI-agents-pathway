"""
Day 4 — Task 3: Standalone Cost Tracker
=========================================
A reusable module for tracking LLM token usage and costs.

This is a utility you'll keep using in every future lesson.

Run tests: python cost_tracker.py
Requires:  pip install anthropic

YOUR TASKS (search for "TODO"):
  1. Add Opus 4.6 to the pricing table
  2. Implement the export_to_json method
  3. Implement the compare_models class method
  4. Add a "cost per step" breakdown to the summary
"""

import json
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class StepUsage:
    """Token usage for a single agent step."""
    step: int
    input_tokens: int
    output_tokens: int
    model: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class CostTracker:
    """
    Track token usage and costs across agent steps.

    Supports multiple models, JSON export, and comparison analysis.

    Usage:
        tracker = CostTracker(model="claude-sonnet-4-20250514")
        # After each LLM call:
        tracker.record_from_response(response, step_num=1)
        # At the end:
        tracker.print_summary()
        tracker.export_to_json("run_costs.json")
    """

    # Pricing per 1 million tokens (USD)
    # Source: https://platform.claude.com/docs/en/about-claude/pricing
    # Last updated: March 2026
    PRICING = {
        "claude-opus-4-6":  {"input": 5.00, "output": 25.00},
        "claude-sonnet-4-20250514":  {"input": 3.00, "output": 15.00},
        "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
        # TODO 1: Add Opus 4.6 pricing
        # Opus 4.6 costs $5.00 per 1M input tokens and $25.00 per 1M output tokens
        # The model string is: "claude-opus-4-6"
        # Add it here following the same pattern as above
    }

    def __init__(self, model: str):
        self.model = model
        self.steps: list[StepUsage] = []
        self.start_time = datetime.now()

    # ── Recording ──

    def record(self, input_tokens: int, output_tokens: int, step_num: int, model: str = None):
        """Manually record token usage for a step."""
        self.steps.append(StepUsage(
            step=step_num,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model or self.model,
        ))

    def record_from_response(self, response, step_num: int):
        """
        Record token usage directly from an Anthropic API response.

        The response object has a .usage attribute with:
          - response.usage.input_tokens
          - response.usage.output_tokens
        """
        self.record(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            step_num=step_num,
        )

    # ── Calculations ──

    @property
    def total_input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.steps)

    @property
    def total_output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def cost_for_tokens(self, input_tokens: int, output_tokens: int, model: str = None) -> float:
        """Calculate cost for a given number of tokens."""
        m = model or self.model
        prices = self.PRICING.get(m, {"input": 3.0, "output": 15.0})
        return (
            (input_tokens / 1_000_000) * prices["input"]
            + (output_tokens / 1_000_000) * prices["output"]
        )

    @property
    def total_cost(self) -> float:
        """Total estimated cost across all steps."""
        return self.cost_for_tokens(self.total_input_tokens, self.total_output_tokens)

    def step_cost(self, step: StepUsage) -> float:
        """Cost for a single step."""
        return self.cost_for_tokens(step.input_tokens, step.output_tokens, step.model)

    # ── Output ──

    def print_summary(self):
        """Print a formatted summary table."""
        print("\n┌─── Cost Summary ───────────────────────────────────────┐")
        print(f"│  {'Step':<6} {'Input':>10} {'Output':>10} {'Total':>10} {'Cost':>10} │")
        print(f"│  {'─' * 50} │")

        for s in self.steps:
            cost = self.step_cost(s)
            # TODO 4: This currently doesn't show per-step cost
            # Replace the line below to include cost in the output
            # Use f"${cost:.6f}" to format the cost
            print(f"│  {s.step:<6} {s.input_tokens:>10,} {s.output_tokens:>10,} {s.total_tokens:>10,} {cost:>10,.6f} │")

        print(f"│  {'─' * 50} │")
        print(f"│  {'TOTAL':<6} {self.total_input_tokens:>10,} {self.total_output_tokens:>10,} {self.total_tokens:>10,} ${self.total_cost:>8.6f} │")
        print(f"│  Model: {self.model:<43} │")
        print(f"│  Steps: {len(self.steps):<43} │")
        print(f"└────────────────────────────────────────────────────────┘")

    def export_to_json(self, filepath: str):
        """
        Export the full cost report to a JSON file.

        TODO 2: Implement this method.

        

        The JSON should include:
        {
            "model": "claude-sonnet-4-20250514",
            "start_time": "2026-03-31T14:32:07",
            "total_input_tokens": 5432,
            "total_output_tokens": 891,
            "total_cost_usd": 0.029793,
            "num_steps": 3,
            "steps": [
                {
                    "step": 1,
                    "input_tokens": 1200,
                    "output_tokens": 300,
                    "cost_usd": 0.008100,
                    "timestamp": "2026-03-31T14:32:07"
                },
                ...
            ]
        }

        Hints:
          - Use json.dump(data, f, indent=2) to write pretty JSON
          - Access step data from self.steps (each is a StepUsage dataclass)
          - Use self.step_cost(s) to get cost per step
        """
        data = {}
        data["model"] = self.model
        data["start_time"] = self.start_time.isoformat()
        data["total_input_tokens"] = self.total_input_tokens
        data["total_output_tokens"] = self.total_output_tokens
        data["total_cost_usd"] = self.total_cost
        data["num_steps"] = len(self.steps)
        for s in self.steps:
            data["steps"] = {
                "step": s.step,
                "input_tokens": s.input_tokens,
                "output_tokens": s.output_tokens,
                "cost_usd": self.step_cost(s),
                "timestamp": s.timestamp
        }
            
        
        with open(filepath, "w") as f:
            json.dump(data,f, indent=2)
       
    @classmethod
    def compare_models(cls, input_tokens: int, output_tokens: int) -> None:
        """
        Compare costs across all supported models for the same token usage.

        TODO 3: Implement this class method.

        Example output:
          Model Comparison for 5,000 input + 800 output tokens:
            claude-haiku-4-5-20251001:  $0.009000
            claude-sonnet-4-20250514:   $0.027000
            claude-opus-4-6:            $0.045000

        Hints:
          - Loop through cls.PRICING.items()
          - Calculate cost for each model
          - Print formatted output
          - Sort by cost (cheapest first)
        """
        pricing_list = []
        
        for item in cls.PRICING.items():
            model_name = item[0]
            model_price = (item[1]["input"] * input_tokens + item[1]["output"] * output_tokens) / 1000000
            pricing_list.append((model_name, model_price))
        
        pricing_list.sort(key=lambda x: x[1])


        print(f"Model Comparison for {input_tokens} input + {output_tokens} output tokens:")
        for item in pricing_list:
            print(f"  {item[0]:<43}: ${item[1]:<43}")


# ──────────────────────────────────────────────
# TESTS — Run this file to verify your implementation
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Basic tracking")
    print("=" * 60)

    tracker = CostTracker(model="claude-sonnet-4-20250514")

    # Simulate a 3-step agent run
    # (In real code, you'd use record_from_response with actual API responses)
    tracker.record(input_tokens=1200, output_tokens=300, step_num=1)
    tracker.record(input_tokens=2100, output_tokens=250, step_num=2)  # notice: input grew!
    tracker.record(input_tokens=2800, output_tokens=400, step_num=3)  # and again!

    tracker.print_summary()

    # Observation: Input tokens grow each step because we re-send the full
    # conversation history. Step 3 sends ~2.3x more input than Step 1!

    print("\n" + "=" * 60)
    print("Test 2: Compare models")
    print("=" * 60)

    # How much would the same workload cost on different models?
    try:
        CostTracker.compare_models(
            input_tokens=tracker.total_input_tokens,
            output_tokens=tracker.total_output_tokens,
        )
    except NotImplementedError:
        print("  (TODO: Implement compare_models first)")

    print("\n" + "=" * 60)
    print("Test 3: Export to JSON")
    print("=" * 60)

    try:
        tracker.export_to_json("test_cost_report.json")
        print("  Exported to test_cost_report.json")
        # Read it back to verify
        with open("test_cost_report.json") as f:
            print(json.dumps(json.load(f), indent=2)[:500])
    except NotImplementedError:
        print("  (TODO: Implement export_to_json first)")

    print("\n" + "=" * 60)
    print("Test 4: Cumulative cost growth")
    print("=" * 60)

    # This demonstrates WHY cost tracking matters for agents.
    # Simulate a long-running agent with growing context:
    long_tracker = CostTracker(model="claude-sonnet-4-20250514")
    base_input = 800
    for step in range(1, 11):
        # Each step, input grows by ~500 tokens (accumulated history)
        input_toks = base_input + (step - 1) * 500
        long_tracker.record(input_tokens=input_toks, output_tokens=200, step_num=step)

    long_tracker.print_summary()
    print(f"\n  Notice: Step 1 input = 800, Step 10 input = {800 + 9 * 500:,}")
    print(f"  That's {(800 + 9 * 500) / 800:.1f}x growth! This is why agents get expensive.")
