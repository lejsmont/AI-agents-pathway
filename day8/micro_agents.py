from dataclasses import dataclass, field
from typing import Literal
import anthropic
from typing import Any, Callable, Literal
import string
import json

def search_web(query: str) -> str:
    """Mock web search. Returns varied fake results based on the query."""
    # Variation prevents the "Researcher calls search_web 5 times with same
    # answer" degenerate loop you observed on Day 7.
    facts_by_keyword = {
        "poland": (
            "Poland EV adoption 2025: EV market share reached 6.8% of new car "
            "sales in H1 2025, up from 4.2% in 2024. Charging infrastructure "
            "grew 34% YoY. Incentive program 'Mój Elektryk' extended through 2027."
        ),
        "germany": (
            "Germany EV adoption 2025: EV market share 21.4% of new car sales "
            "YTD, a drop from 25.1% in 2024 after subsidy cuts. Total EV stock "
            "crossed 2M vehicles. Fast-charging network densest in EU."
        ),
        "europe": (
            "EU-wide EV share Q1-Q2 2025: 17.8% of new registrations, steady "
            "vs. 2024. Eastern Europe growing faster off smaller base."
        ),
    }
    q = query.lower()
    for kw, txt in facts_by_keyword.items():
        if kw in q:
            return txt
    return f"No strong match for '{query}'. Try queries with country names."


def lookup_documentation(topic: str) -> str:
    """Mock doc lookup. Returns structured stats."""
    stats = {
        "poland_ev_stats_2025": {"market_share_pct": 6.8, "total_stock": 98000, "yoy_growth_pct": 42.0},
        "germany_ev_stats_2025": {"market_share_pct": 21.4, "total_stock": 2100000, "yoy_growth_pct": -14.7},
        "eu_charging_density": {"poland_per_100km": 3.1, "germany_per_100km": 18.9},
    }
    key = topic.lower().replace(" ", "_")
    for k, v in stats.items():
        if key in k or k in key:
            return json.dumps(v)
    return f"No documentation match for '{topic}'."


def calculate(expression: str) -> str:
    """Safe-ish calculator. Returns a string (remember Day 4 bug: return str, not float!)."""
    try:
        # Restrict to math builtins only.
        allowed = {"__builtins__": {}, "abs": abs, "round": round, "min": min, "max": max}
        result = eval(expression, allowed, {})  # noqa: S307 — intentional, sandboxed
        return str(result)  # Day 4 lesson: ALWAYS str(), never the raw value
    except Exception as e:
        # Day 7 bug: returning the exception OBJECT breaks downstream.
        return f"Error: {str(e)}"


def compare_metrics(metric_a: float, metric_b: float, label_a: str, label_b: str) -> str:
    """Compare two metrics and return a verbal summary."""
    if metric_a == metric_b:
        verdict = "equal"
    elif metric_a > metric_b:
        pct = (metric_a - metric_b) / max(abs(metric_b), 1e-9) * 100
        verdict = f"{label_a} is {pct:.1f}% higher than {label_b}"
    else:
        pct = (metric_b - metric_a) / max(abs(metric_a), 1e-9) * 100
        verdict = f"{label_b} is {pct:.1f}% higher than {label_a}"
    return f"{label_a}={metric_a}, {label_b}={metric_b}. {verdict}."


def format_for_history(speaker: str, text: str) -> dict:
    """
    Wrap an agent's output as a user-role message in the shared history,
    prefixed with [SPEAKER] so downstream agents know who said what.

    This fixes the Day 7 Ex 2 identity-confusion bug: in AutoGen, everything
    ended up as generic user-role messages and agents lost track of who was
    who. Explicit speaker tags make the shared context unambiguous.
    """
    return {"role": "user", "content": f"[{speaker}]: {text}"}

TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "search_web": search_web,
    "lookup_documentation": lookup_documentation,
    "calculate": calculate,
    "compare_metrics": compare_metrics,
}

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "search_web",
        "description": "Search the web for recent information. Best for news, statistics, trends.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
    {
        "name": "lookup_documentation",
        "description": "Look up structured statistics from a curated knowledge base.",
        "input_schema": {
            "type": "object",
            "properties": {"topic": {"type": "string", "description": "Topic to look up"}},
            "required": ["topic"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a numeric expression, e.g. '(21.4 - 6.8) / 6.8 * 100'.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "compare_metrics",
        "description": "Compare two numeric metrics and return a verbal summary.",
        "input_schema": {
            "type": "object",
            "properties": {
                "metric_a": {"type": "number"},
                "metric_b": {"type": "number"},
                "label_a": {"type": "string"},
                "label_b": {"type": "string"},
            },
            "required": ["metric_a", "metric_b", "label_a", "label_b"],
        },
    },
]

@dataclass
class LLMCallLog:
    turn: int
    caller: str     # agent name, or "SELECTOR"
    model: str
    input_tokens: int
    output_tokens: int
    note: str       # e.g. "initial response", "after tool results", "selector pick"


@dataclass
class Agent:
    """
    A single agent. One class handles both reflection-style and selector-style
    patterns — the difference is just whether tools are configured and whether
    the message history persists across turns.

    Stateful mode (default): the agent accumulates its own messages list across
    turns. Use this for reflection loops where context should carry forward.
    Call .reset() between independent runs.

    Stateless mode (stateful=False): each turn() starts fresh from the
    incoming_text. Use this for selector loops where the orchestrator owns the
    shared history and rebuilds the prompt for each call.
    """
    name: str
    system: str
    model: str
    description: str = ""
    tools: list[dict] = field(default_factory=list)
    stateful: bool = True
    messages: list[dict] = field(default_factory=list)

    def reset(self) -> None:
        """Clear conversation history. Call between independent runs."""
        self.messages.clear()

    def turn(
        self,
        client: anthropic.Anthropic,
        incoming_text: str,
        logs: list[LLMCallLog],
        turn_number: int,
    ) -> str:
        """
        Run one turn. If tools are configured, runs the tool-use loop until
        the agent produces a final text response. Returns the text.

        Logging: appends one LLMCallLog per API call. The orchestrator owns
        the logs list — Agent just adds rows to it.
        """
        # State management: stateful agents continue their conversation;
        # stateless ones start fresh each turn from just the incoming text.
        if self.stateful:
            self.messages.append({"role": "user", "content": incoming_text})
            messages = self.messages
        else:
            messages = [{"role": "user", "content": incoming_text}]

        max_inner_iterations = 6
        for inner in range(1, max_inner_iterations + 1):
            kwargs = {
                "model": self.model,
                "max_tokens": 1024,
                "system": self.system,
                "messages": messages,
            }
            if self.tools:
                kwargs["tools"] = self.tools

            response = client.messages.create(**kwargs)

            logs.append(LLMCallLog(
                turn=turn_number,
                caller=self.name,
                model=self.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                note=f"iter {inner}" if self.tools else "turn",
            ))

            if response.stop_reason != "tool_use":
                # Done — extract text and return.
                text_parts = [b.text for b in response.content if b.type == "text"]
                final_text = "\n".join(text_parts).strip()
                if self.stateful:
                    # Persist the final text back to our history (not the raw
                    # blocks — keeps the history clean for the next turn).
                    self.messages.append({"role": "assistant", "content": final_text})
                return final_text

            # tool_use branch — execute tools, append results, loop.
            messages.append({"role": "assistant", "content": response.content})
            tool_results = [self.run_tool(b) for b in response.content if b.type == "tool_use"]
            messages.append({"role": "user", "content": tool_results})

        return f"[WARNING: agent {self.name} exceeded tool-loop limit]"

    def run_tool(self, tool_use_block: Any) -> dict:
        fn = TOOL_REGISTRY.get(tool_use_block.name)
        if fn is None:
            result = f"Error: unknown tool '{tool_use_block.name}'"
        else:
            try:
                result = fn(**tool_use_block.input)
            except Exception as e:
                result = f"Error: {str(e)}"
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_block.id,
            "content": str(result),
        }


SELECTOR_SYSTEM = """You are a routing system. Given a conversation history
and a list of agent roles, pick which agent should speak NEXT.

You MUST respond with EXACTLY ONE WORD: the name of the chosen agent, or
the word TERMINATE if the task is complete.

Valid responses: {valid_names}

Do NOT explain. Do NOT add punctuation. Just the name.
"""

def pick_next_speaker(
    client: anthropic.Anthropic,
    agents: list[Agent],
    history: list[dict],
    logs: list[LLMCallLog],
    turn_number: int,
    model: str = "claude-haiku-4-5"
) -> str:
    """
    Ask an LLM to choose the next speaker (or TERMINATE). Returns a name.

    Notice: this is exactly the call AutoGen's SelectorGroupChat was making
    invisibly on Day 7. Now it's visible in your logs.
    """
    valid_names = [a.name for a in agents] + ["TERMINATE"]
    valid_upper = {v.upper() for v in valid_names}

    system = SELECTOR_SYSTEM.format(valid_names=" | ".join(valid_names))

    # Build a compact context for the selector. Don't send the full raw
    # history — just role summaries + a trimmed transcript. The selector
    # needs enough to decide; sending everything wastes tokens.
    role_block = "\n".join(f"  - {a.name}: {a.description}" for a in agents)

    # Flatten the shared history for the selector. Each item in `history` is
    # already a dict like {"role": "user", "content": "[PLANNER]: ..."}.
    # We just want the text in order.
    transcript_lines = []
    for msg in history:
        content = msg["content"]
        if isinstance(content, str):
            transcript_lines.append(content)
    transcript = "\n".join(transcript_lines[-12:])  # last 12 lines is plenty

    user_content = (
        f"Available agents:\n{role_block}\n\n"
        f"Recent conversation:\n{transcript}\n\n"
        f"Who should speak next? (one word)"
    )

    response = client.messages.create(
        model = model,
        max_tokens = 20,
        system = system,
        messages = [{"role": "user", "content": user_content}]
    )

    logs.append(LLMCallLog(
        turn=turn_number, caller="SELECTOR", model=model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        note="routing decision",
    ))

   
    text = next((b.text for b in response.content if b.type == "text"), "")
    name = text.upper().strip().translate(str.maketrans("", "", string.punctuation))
   
    if name in valid_upper:
        return name

    for upper_name in sorted(valid_names, key=len, reverse=True):
        if upper_name in name:
            return upper_name

    return "TERMINATE"

def run_reflection(
    client: anthropic.Anthropic,
    writer: Agent,
    critic: Agent,
    task: str,
    max_turns: int,
    approval_token: str,
) -> tuple[str, Literal["approved", "max_turns"], list[LLMCallLog]]:
    logs: list[LLMCallLog] = []
    turn = 1

    draft = writer.turn(client, task, logs, turn)
    print(f"\n[Turn {turn}] {writer.name}:\n{draft}\n")

    while turn < max_turns:
        turn += 1
        feedback = critic.turn(client, draft, logs, turn)
        print(f"\n[Turn {turn}] {critic.name}:\n{feedback}\n")

        if feedback.strip().upper().startswith(approval_token):
            return draft, "approved", logs
        if turn >= max_turns:
            return draft, "max_turns", logs

        turn += 1
        draft = writer.turn(client, feedback, logs, turn)
        print(f"\n[Turn {turn}] {writer.name}:\n{draft}\n")

    return draft, "max_turns", logs

def run_team(
    client: anthropic.Anthropic,
    agents: list[Agent],
    task: str,
    max_turns: int,
    terminate_token: str,
    selector_model: str = "claude-haiku-4-5",
) -> tuple[str, Literal["terminated", "max_turns"], list[LLMCallLog]]:
    """
    Main orchestration loop. Selector picks; chosen agent speaks; output
    goes into shared history; repeat.
    """
    logs: list[LLMCallLog] = []
    agents_by_name = {a.name.upper(): a for a in agents}

    # Shared history seeds with the user task (flat format — strings, not blocks).
    # Each agent receives this history flattened into a single prompt when it's
    # their turn. They don't see it as a multi-turn conversation — they see it
    # as "here's the context, give me your next move".
    history: list[dict] = [{"role": "user", "content": f"[TASK]: {task}"}]

    print(f"[TASK] {task}\n")

    for turn in range(1, max_turns + 1):
        name = pick_next_speaker(client, agents, history, logs, turn, selector_model)


        print(f"\n--- Turn {turn}: selector picked {name} ---")
        
        if name == "TERMINATE":
            # --- your code here ---
            last = history[-1]["content"]
            # Strip leading "[SPEAKER]: " if present
            if last.startswith("[") and "]: " in last:
                last = last.split("]: ", 1)[1]
            return last, "terminated", logs
        
        agent = agents_by_name[name]
        prompt = "\n".join(m["content"] for m in history)
        text = agent.turn(client, prompt, logs, turn)
        print(f"\n[{name}]: {text}\n")
        history.append(format_for_history(name, text))
        last_line = text.strip().splitlines()[-1].strip().upper() if text.strip() else ""
        if last_line == terminate_token or last_line.endswith(terminate_token):
            return text, "terminated", logs

    # Fell through: hit max_turns
    last_msg = history[-1]["content"] if history else ""
    return last_msg, "max_turns", logs


