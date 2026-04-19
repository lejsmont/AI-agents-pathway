"""
Day 7 — Exercise 1: Reflection Pattern with RoundRobinGroupChat
================================================================

Goal: Build a Writer + Critic multi-agent team using AutoGen's AgentChat API.
The writer drafts content, the critic reviews it, the writer revises.
This is the "reflection" pattern — the simplest multi-agent coordination.

Install first:
    pip install "autogen-agentchat" "autogen-ext[openai,anthropic]"

Concepts practiced:
    - AssistantAgent: LLM-backed agent with system_message
    - RoundRobinGroupChat: fixed turn-taking coordination
    - TextMentionTermination: stop when a keyword appears
    - MaxMessageTermination: safety limit on total messages
    - Combining termination conditions with | (OR)
    - run_stream() + Console for observing agent communication
    - async/await (AutoGen 0.4 is fully async)

Run:
    python exercise1_reflection.py
"""

import asyncio
import os

# ─── Exploration: inspect what AutoGen gives us ────────────────────────────
# Before implementing, let's explore the key classes.
# Uncomment and run each block to understand the API.

# --- Block 1: Check your installation ---
import autogen_agentchat
print(f"autogen-agentchat version: {autogen_agentchat.__version__}")
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
print("All imports successful!")

# --- Block 2: Create a model client and test it ---
# from autogen_ext.models.anthropic import AnthropicChatCompletionClient
# from autogen_core.models import UserMessage
# async def test_model():
#     client = AnthropicChatCompletionClient(model="claude-sonnet-4-20250514")
#     result = await client.create([UserMessage(content="Say hello in 5 words", source="user")])
#     print(f"Response: {result.content}")
#     print(f"Tokens used: {result.usage}")
#     await client.close()
# asyncio.run(test_model())

# # --- Block 3: Create a single agent and run it ---
# from autogen_agentchat.agents import AssistantAgent
# from autogen_ext.models.anthropic import AnthropicChatCompletionClient
# async def test_single_agent():
#     client = AnthropicChatCompletionClient(model="claude-sonnet-4-20250514")
#     agent = AssistantAgent(
#         name="test_agent",
#         model_client=client,
#         system_message="You are a helpful assistant. Be concise.",
#     )
#     result = await agent.run(task="What is the capital of Poland?")
#     for msg in result.messages:
#         print(f"[{msg.source}]: {msg.content[:100]}")
#     await client.close()
# asyncio.run(test_single_agent())


# ─── Your implementation starts here ───────────────────────────────────────

# TODO 1: Import the required classes
# You need:
#   - AssistantAgent from autogen_agentchat.agents
#   - RoundRobinGroupChat from autogen_agentchat.teams
#   - TextMentionTermination, MaxMessageTermination from autogen_agentchat.conditions
#   - Console from autogen_agentchat.ui  (for pretty-printing the stream)
#   - AnthropicChatCompletionClient from autogen_ext.models.anthropic
#     (or OpenAIChatCompletionClient from autogen_ext.models.openai if you prefer)
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.anthropic import AnthropicChatCompletionClient


def create_model_client(model_name):
    """Create and return a model client.

    TODO 2: Create an AnthropicChatCompletionClient (or OpenAI).
    Use model="claude-sonnet-4-20250514" (or your preferred model).
    The API key is read from ANTHROPIC_API_KEY env var automatically.
    """
    return AnthropicChatCompletionClient(model=model_name)
    



def create_writer_agent(model_client):
    """Create the Writer agent.

    TODO 3: Create an AssistantAgent with:
    - name: "Writer" (must be a valid Python identifier, no spaces)
    - model_client: the client you created
    - system_message: Tell it to write high-quality content based on the task.
      Instruct it that when it receives critique, it should revise its work.
      Tell it NOT to include "APPROVE" in its responses — only the critic does that.

    Hint: The system_message is the most important part. Be specific about what
    the writer should do when it gets feedback from the critic.
    """
    return AssistantAgent(
        name="Writer",
        model_client=model_client,
        system_message=(
            "You are a skilled blog post writer. Your job:\n"
            "1. On your FIRST turn, write a draft based on the task (150-200 words).\n"
            "2. On LATER turns, you will see feedback from the Critic. Revise your "
            "draft to address their feedback. Keep the word count at 150-200 words.\n"
            "3. Never include the word 'APPROVE' in your response.\n"
            "4. Output ONLY the blog post text — no meta-commentary about your process."
        )
    )


def create_critic_agent(model_client):
    """Create the Critic agent.

    TODO 4: Create an AssistantAgent with:
    - name: "Critic"
    - model_client: the client you created
    - system_message: Tell it to review writing and provide constructive feedback.
      Be specific: it should check for clarity, accuracy, structure, and engagement.
      IMPORTANT: Tell it to respond with "APPROVE" when the writing is satisfactory.
      This keyword triggers the termination condition.

    Hint: Without clear instructions about when to say "APPROVE", the critic
    might never stop critiquing, or might approve too quickly.
    """
    return AssistantAgent(
        name="Critic",
        model_client=model_client,
        system_message=(
            "You are a sharp but constructive blog post reviewer. Your job:\n"
            "1. Review the Writer's latest draft for: clarity, accuracy, structure, "
            "engagement, and audience fit.\n"
            "2. If the draft needs improvement, provide 2-3 specific, actionable "
            "suggestions. Be concise — no more than 3-4 sentences per suggestion.\n"
            "3. If the draft is good enough (only minor issues remain), say exactly: "
            "'The draft is ready for final editing.' and list any minor polish items.\n"
            "4. Never include the word 'APPROVE' in your response.\n"
            "5. Do NOT rewrite the draft yourself — that's the Writer's job."
            )    
    )

def create_editor_agent(model_client):
    return AssistantAgent(
        name="Editor",
        model_client=model_client,
        system_message=(
            "You are a final-pass editor. Your job:\n"
            "1. Read the Critic's latest feedback and the Writer's latest draft.\n"
            "2. If the Critic says 'ready for final editing' or similar, take the "
            "Writer's latest draft and apply final polish: fix grammar, tighten "
            "phrasing, improve flow. Then output the final version and end with "
            "'APPROVE' on its own line.\n"
            "3. If the Critic is still requesting significant revisions, respond "
            "with a single sentence: 'Revisions still in progress — back to the Writer.' "
            "Do NOT attempt to edit or rewrite the draft yourself in this case.\n"
            "4. Only say 'APPROVE' when you are outputting a finished, polished post."
        )    
)

def create_team(writer, critic, editor):
    """Create a RoundRobinGroupChat team.

    TODO 5: Set up the team with:
    - participants: [writer, critic] — the order matters! Writer goes first.
    - termination_condition: Combine two conditions with | (OR):
        a) TextMentionTermination("APPROVE") — stops when critic approves
        b) MaxMessageTermination(6) — safety limit to prevent infinite loops
      The combined condition triggers when EITHER is met.

    Hint: The | operator creates an OR condition. You can also use & for AND.
    A MaxMessageTermination is essential — without it, agents could loop forever
    if the critic never says "APPROVE".
    """
    termination_condition = TextMentionTermination("APPROVE") | MaxMessageTermination(20)

    return RoundRobinGroupChat([writer,critic,editor], termination_condition = termination_condition)


async def run_reflection(task: str):
    """Run the Writer + Critic reflection loop.

    TODO 6: Wire it all together:
    1. Create the model client
    2. Create writer and critic agents
    3. Create the team
    4. Run the team with run_stream(task=task) and wrap in Console() for display
       - Usage: await Console(team.run_stream(task=task))
       - Console() prints each message as it arrives, showing which agent said what
    5. The return value of Console() is a TaskResult — print summary info:
       - result.messages: list of all messages exchanged
       - Count how many turns it took
       - Check result.stop_reason to see why it stopped
    6. Close the model client when done: await model_client.close()
    """
    haiku = create_model_client("claude-haiku-4-5")
    sonnet = create_model_client("claude-sonnet-4-6")
    writer = create_writer_agent(haiku)
    critic = create_critic_agent(sonnet)
    editor = create_editor_agent(sonnet)
    team = create_team(writer, critic, editor)
    result = await Console(team.run_stream(task=task))
    for msg in result.messages:
        print(f"[{msg.source}]: {msg.content[:300]}")
    print(f"Stop Reason: {result.stop_reason}")
    await haiku.close()
    await sonnet.close()


# ─── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    task = (
        "Write a short blog post (150-200 words) explaining why Python is popular "
        "for AI/ML development. Target audience: developers who know Java but not Python."
    )
    print(f"\n{'='*60}")
    print("REFLECTION PATTERN: Writer + Critic")
    print(f"{'='*60}")
    print(f"Task: {task}\n")

    asyncio.run(run_reflection(task))

    # ── Bonus challenges ──────────────────────────────────────────────────
    # 1. Add a third agent — an "Editor" that polishes the final approved draft.
    #    Put it after the Critic in the participants list. How does the flow change?
    #
    # 2. Try different termination strategies:
    #    - What happens with MaxMessageTermination(3)? Does the critic get to respond?
    #    - What happens with MaxMessageTermination(20)? More revision rounds?
    #
    # 3. Use a cheaper/faster model for the Critic and a stronger model for the Writer.
    #    Does this work well? (Hint: critic doesn't need to write, just evaluate)
