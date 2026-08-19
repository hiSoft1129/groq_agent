import os
import json
from dotenv import load_dotenv
from groq import Groq

from tools import search_web, open_url, calculate
from tool_schemas import TOOLS

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY was not found")

client = Groq(api_key=api_key)

TOOL_FUNCTIONS = {
    "search_web": search_web,
    "open_url": open_url,
    "calculate": calculate,
}

# ============================================================
# GROQ TPM PROTECTION
# ============================================================
# The previous version fixed the oversized history problem, but
# the agent could still spend all 6 iterations calling tools.
#
# This version uses a TWO-PHASE design:
#
#   PHASE 1: one small research/tool request
#   PHASE 2: one tool-free answer request
#
# Therefore the model cannot get stuck in an endless tool loop.
#
# We also intentionally keep every prompt/result very small.
# ============================================================

MAX_SEARCH_RESULTS = 5
MAX_TOOL_RESULT_CHARS = 2500
MAX_RESEARCH_CHARS = 5000
MAX_TOOL_TURNS = 2
MAX_OUTPUT_TOKENS = 800

SYSTEM_PROMPT = """You are a concise research assistant.
Use the available tools only when needed.
After enough evidence is collected, stop researching and answer.
Do not invent facts or tools.
Keep the final answer concise and directly answer the user's question.
"""

FINAL_SYSTEM_PROMPT = """You are a concise answer writer.
Answer the user's question using ONLY the research notes provided.
Do not call tools.
Do not discuss the research process.
If the notes are incomplete, clearly say what is uncertain.
Give a direct, useful answer with brief reasons.
"""


def trim_text(text, limit):
    if text is None:
        return ""

    text = str(text)

    if len(text) <= limit:
        return text

    return text[:limit] + "\n...[TRUNCATED]..."


def validate_tool_arguments(tool_name, args):
    if not isinstance(args, dict):
        return False, "Arguments must be a JSON object."

    if tool_name == "search_web":
        if not isinstance(args.get("query"), str):
            return False, "search_web requires string 'query'."

        # Never allow the model to request a huge result set.
        args["max_results"] = min(
            int(args.get("max_results", MAX_SEARCH_RESULTS)),
            MAX_SEARCH_RESULTS
        )

        allowed = {"query", "max_results"}
        extra = set(args) - allowed

        if extra:
            return False, f"Unsupported parameters: {sorted(extra)}"

    elif tool_name == "open_url":
        url = args.get("url")

        if not isinstance(url, str):
            return False, "open_url requires string 'url'."

        if not url.startswith(("http://", "https://")):
            return False, "URL must start with http:// or https://"

        extra = set(args) - {"url"}

        if extra:
            return False, f"Unsupported parameters: {sorted(extra)}"

    elif tool_name == "calculate":
        if not isinstance(args.get("expression"), str):
            return False, "calculate requires string 'expression'."

    else:
        return False, f"Unknown tool: {tool_name}"

    return True, ""


def compact_research(notes):
    parts = []

    for item in notes:
        parts.append(
            f"TOOL: {item['tool']}\n"
            f"RESULT:\n{trim_text(item['result'], 1800)}"
        )

    return trim_text(
        "\n\n".join(parts),
        MAX_RESEARCH_CHARS
    )


def call_model(messages, use_tools=True):
    kwargs = {
        "model": "openai/gpt-oss-120b",
        "messages": messages,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }

    if use_tools:
        kwargs["tools"] = TOOLS
        kwargs["tool_choice"] = "auto"

    return client.chat.completions.create(**kwargs)


def execute_tool(tool_name, args):
    valid, error = validate_tool_arguments(tool_name, args)

    if not valid:
        return f"ERROR: {error}"

    try:
        result = TOOL_FUNCTIONS[tool_name](**args)
    except Exception as e:
        return (
            f"ERROR executing {tool_name}: "
            f"{type(e).__name__}: {e}"
        )

    if not isinstance(result, str):
        result = json.dumps(result, ensure_ascii=False)

    return trim_text(result, MAX_TOOL_RESULT_CHARS)


def final_answer(question, notes):
    """Tool-free final call. This prevents endless tool loops."""

    research = compact_research(notes)

    messages = [
        {
            "role": "system",
            "content": FINAL_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": (
                f"QUESTION:\n{trim_text(question, 1800)}\n\n"
                f"RESEARCH NOTES:\n{research}\n\n"
                "Now provide the final answer."
            )
        }
    ]

    response = call_model(
        messages,
        use_tools=False
    )

    return response.choices[0].message.content or (
        "No final answer was returned."
    )


def run_agent(user_message):
    print(
        f"\n🤖 Agent starting on: '{user_message}'\n"
    )
    print("=" * 60)

    notes = []

    # ========================================================
    # PHASE 1: RESEARCH
    # Maximum TWO tool turns.
    # ========================================================

    for iteration in range(1, MAX_TOOL_TURNS + 1):

        print(f"\n📍 Research iteration {iteration}")

        research_context = compact_research(notes)

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n"
                    f"{trim_text(user_message, 1800)}\n\n"
                    f"RESEARCH ALREADY COLLECTED:\n"
                    f"{research_context or 'None.'}\n\n"
                    "If another source is genuinely needed, use ONE "
                    "tool call. Otherwise give your answer."
                )
            }
        ]

        try:
            response = call_model(
                messages,
                use_tools=True
            )

        except Exception as e:
            print("\n❌ Groq API error:")
            print(e)

            # If research failed, still make a tiny tool-free
            # request rather than crashing.
            try:
                return final_answer(user_message, notes)
            except Exception:
                return f"Agent API error: {e}"

        choice = response.choices[0]
        message = choice.message

        print(
            f"   Finish reason: {choice.finish_reason}"
        )

        # ----------------------------------------------------
        # Model already answered.
        # ----------------------------------------------------

        if choice.finish_reason == "stop":
            if message.content:
                return message.content

            return final_answer(
                user_message,
                notes
            )

        # ----------------------------------------------------
        # Execute requested tools.
        # ----------------------------------------------------

        if choice.finish_reason != "tool_calls":
            return final_answer(
                user_message,
                notes
            )

        tool_calls = message.tool_calls or []

        if not tool_calls:
            return final_answer(
                user_message,
                notes
            )

        # Only execute the FIRST tool call.
        # This keeps both latency and context small.
        tool_call = tool_calls[0]

        tool_name = tool_call.function.name

        print(
            f"   🔧 Calling tool: {tool_name}"
        )

        try:
            args = json.loads(
                tool_call.function.arguments
            )
        except json.JSONDecodeError:
            result = "ERROR: invalid JSON tool arguments."
            args = {}

        if args:
            print("   📥 Input:")
            print(json.dumps(args, indent=6))

            result = execute_tool(
                tool_name,
                args
            )
        else:
            result = "ERROR: invalid JSON tool arguments."

        print(
            "   📤 Result preview: "
            f"{result[:300]}..."
        )

        notes.append(
            {
                "tool": tool_name,
                "result": result
            }
        )

    # ========================================================
    # PHASE 2: FORCE FINAL ANSWER
    #
    # No tools are supplied here. The model MUST answer.
    # ========================================================

    print(
        "\n📝 Research limit reached; generating final answer..."
    )

    try:
        return final_answer(
            user_message,
            notes
        )

    except Exception as e:
        print("\n❌ Final-answer API error:")
        print(e)

        # Last-resort local response.
        if notes:
            return (
                "Research results were collected, but the final "
                "AI synthesis failed.\n\n"
                + compact_research(notes)
            )

        return f"Agent API error: {e}"


if __name__ == "__main__":

    question = (
        "What are the top 3 most popular programming "
        "languages in 2025 and why?"
    )

    answer = run_agent(question)

    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(answer)