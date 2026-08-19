import os
import json

from dotenv import load_dotenv
from groq import Groq

from tools import (
    search_web,
    open_url,
    calculate
)

from tool_schemas import TOOLS


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY was not found"
    )


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(
    api_key=api_key
)


# ============================================================
# TOOL FUNCTIONS
# ============================================================

TOOL_FUNCTIONS = {

    "search_web": search_web,

    "open_url": open_url,

    "calculate": calculate,

}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a helpful research assistant.

You have access to three tools.

============================================================
1. search_web
============================================================

Search the internet for current information.

Parameters:

query
max_results


============================================================
2. open_url
============================================================

Open and read the contents of a web page.

Parameters:

url


============================================================
3. calculate
============================================================

Perform mathematical calculations.

Parameter:

expression


============================================================
RESEARCH WORKFLOW
============================================================

When answering a research question:

1. Determine whether current information is needed.

2. If current information is needed, use search_web.

3. Examine the search results.

4. If a search result contains a useful URL and you need
   more detailed information, use open_url.

5. You may perform additional searches if necessary.

6. Use calculate for mathematical calculations.

7. Synthesize the information.

8. Provide a clear final answer.

============================================================
IMPORTANT TOOL RULES
============================================================

ONLY use these tools:

- search_web
- open_url
- calculate

NEVER invent another tool.

NEVER call:

- open_file
- browse
- fetch_url
- web_search
- fetch
- read_url

When using search_web:

- Always provide "query".
- Do not use cursor.
- Do not use id.
- Do not use page.
- Do not use offset.
- Do not use pagination parameters.

If you need another search, call search_web again with
a new query.

When you want to inspect a web page, use open_url.

open_url requires:

{
    "url": "https://example.com"
}

Do not use open_file for URLs.

Stop searching once you have enough reliable information
to answer the user's question.
"""


# ============================================================
# TOOL ARGUMENT VALIDATION
# ============================================================

def validate_tool_arguments(
    tool_name,
    tool_input
):

    if not isinstance(
        tool_input,
        dict
    ):
        return (
            False,
            "Tool arguments must be a JSON object."
        )


    # --------------------------------------------------------
    # search_web
    # --------------------------------------------------------

    if tool_name == "search_web":

        if "query" not in tool_input:

            return (
                False,
                (
                    "search_web requires a 'query' "
                    "parameter."
                )
            )

        if not isinstance(
            tool_input["query"],
            str
        ):

            return (
                False,
                "'query' must be a string."
            )

        if not tool_input["query"].strip():

            return (
                False,
                "'query' cannot be empty."
            )

        allowed = {
            "query",
            "max_results"
        }

        unexpected = (
            set(tool_input.keys())
            - allowed
        )

        if unexpected:

            return (
                False,
                (
                    "Unsupported search_web parameters: "
                    f"{sorted(unexpected)}"
                )
            )


    # --------------------------------------------------------
    # open_url
    # --------------------------------------------------------

    elif tool_name == "open_url":

        if "url" not in tool_input:

            return (
                False,
                (
                    "open_url requires a 'url' "
                    "parameter."
                )
            )

        if not isinstance(
            tool_input["url"],
            str
        ):

            return (
                False,
                "'url' must be a string."
            )

        url = tool_input["url"].strip()

        if not url.startswith(
            (
                "http://",
                "https://"
            )
        ):

            return (
                False,
                (
                    "URL must start with "
                    "http:// or https://"
                )
            )

        allowed = {
            "url"
        }

        unexpected = (
            set(tool_input.keys())
            - allowed
        )

        if unexpected:

            return (
                False,
                (
                    "Unsupported open_url "
                    f"parameters: {sorted(unexpected)}"
                )
            )


    # --------------------------------------------------------
    # calculate
    # --------------------------------------------------------

    elif tool_name == "calculate":

        if "expression" not in tool_input:

            return (
                False,
                (
                    "calculate requires an "
                    "'expression' parameter."
                )
            )

        if not isinstance(
            tool_input["expression"],
            str
        ):

            return (
                False,
                "'expression' must be a string."
            )


    return True, ""


# ============================================================
# AGENT LOOP
# ============================================================

def run_agent(
    user_message: str,
    max_iterations: int = 8
) -> str:

    messages = [

        {
            "role": "user",
            "content": user_message
        }

    ]

    iteration = 0


    print(
        f"\n🤖 Agent starting on: "
        f"'{user_message}'\n"
    )

    print("=" * 60)


    while iteration < max_iterations:

        iteration += 1

        print(
            f"\n📍 Iteration {iteration}"
        )


        # ====================================================
        # CALL GROQ
        # ====================================================

        try:

            response = (
                client.chat.completions.create(

                    model="openai/gpt-oss-120b",

                    max_tokens=4096,

                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT
                        },
                        *messages
                    ],

                    tools=TOOLS,

                    tool_choice="auto",

                )
            )

        except Exception as e:

            print(
                "\n❌ Groq API error:"
            )

            print(e)

            return (
                f"Agent API error: {e}"
            )


        # ====================================================
        # ASSISTANT MESSAGE
        # ====================================================

        assistant_message = (
            response.choices[0].message
        )

        finish_reason = (
            response.choices[0].finish_reason
        )

        print(
            f"   Finish reason: "
            f"{finish_reason}"
        )


        # ====================================================
        # FINAL ANSWER
        # ====================================================

        if finish_reason == "stop":

            final_answer = (
                assistant_message.content
                or ""
            )

            print(
                "\n✅ Final answer reached "
                f"after {iteration} iterations\n"
            )

            return final_answer


        # ====================================================
        # TOOL CALLS
        # ====================================================

        if finish_reason == "tool_calls":

            # ------------------------------------------------
            # Save assistant tool-call message
            # ------------------------------------------------

            messages.append(
                {
                    "role": "assistant",

                    "content":
                        assistant_message.content,

                    "tool_calls": [

                        {
                            "id": tool_call.id,

                            "type": "function",

                            "function": {

                                "name":
                                    tool_call.function.name,

                                "arguments":
                                    tool_call.function.arguments

                            }

                        }

                        for tool_call
                        in assistant_message.tool_calls

                    ]
                }
            )


            # ------------------------------------------------
            # Execute every tool call
            # ------------------------------------------------

            for tool_call in (
                assistant_message.tool_calls
            ):

                tool_name = (
                    tool_call.function.name
                )


                print(
                    f"   🔧 Calling tool: "
                    f"{tool_name}"
                )


                # ============================================
                # PARSE JSON
                # ============================================

                try:

                    tool_input = json.loads(
                        tool_call.function.arguments
                    )

                except json.JSONDecodeError:

                    result = (
                        "ERROR: Invalid JSON tool "
                        "arguments. Please retry."
                    )

                    messages.append(
                        {
                            "role": "tool",

                            "tool_call_id":
                                tool_call.id,

                            "content":
                                result
                        }
                    )

                    continue


                print(
                    "   📥 Input:"
                )

                print(
                    json.dumps(
                        tool_input,
                        indent=6
                    )
                )


                # ============================================
                # UNKNOWN TOOL
                # ============================================

                if tool_name not in TOOL_FUNCTIONS:

                    result = (
                        f"ERROR: Unknown tool "
                        f"'{tool_name}'. "
                        f"Available tools: "
                        f"{list(TOOL_FUNCTIONS.keys())}"
                    )

                    messages.append(
                        {
                            "role": "tool",

                            "tool_call_id":
                                tool_call.id,

                            "content":
                                result
                        }
                    )

                    continue


                # ============================================
                # VALIDATE
                # ============================================

                valid, error_message = (
                    validate_tool_arguments(
                        tool_name,
                        tool_input
                    )
                )


                if not valid:

                    result = (
                        f"ERROR: {error_message}"
                    )

                    print(
                        f"   ❌ {result}"
                    )

                    messages.append(
                        {
                            "role": "tool",

                            "tool_call_id":
                                tool_call.id,

                            "content":
                                result
                        }
                    )

                    continue


                # ============================================
                # EXECUTE
                # ============================================

                try:

                    result = (
                        TOOL_FUNCTIONS[tool_name](
                            **tool_input
                        )
                    )

                except Exception as e:

                    result = (
                        f"ERROR executing "
                        f"{tool_name}: "
                        f"{type(e).__name__}: {e}"
                    )


                # ============================================
                # NORMALIZE RESULT
                # ============================================

                if not isinstance(
                    result,
                    str
                ):

                    result = json.dumps(
                        result,
                        ensure_ascii=False
                    )


                print(
                    "   📤 Result preview: "
                    f"{result[:300]}..."
                )


                # ============================================
                # ADD TOOL RESULT
                # ============================================

                messages.append(
                    {
                        "role": "tool",

                        "tool_call_id":
                            tool_call.id,

                        "content":
                            result
                    }
                )


            # ------------------------------------------------
            # Continue agent loop
            # ------------------------------------------------

            continue


        # ====================================================
        # UNEXPECTED
        # ====================================================

        print(
            f"   ⚠️ Unexpected finish reason: "
            f"{finish_reason}"
        )

        return (
            "Agent stopped unexpectedly. "
            f"Finish reason: {finish_reason}"
        )


    # ========================================================
    # MAX ITERATIONS
    # ========================================================

    return (
        f"Max iterations "
        f"({max_iterations}) reached "
        f"without a final answer."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    questions = [

        "What are the top 3 most popular "
        "programming languages in 2025 and why?"

    ]


    for question in questions:

        answer = run_agent(
            question
        )


        print(
            "\n" + "=" * 60
        )

        print(
            "FINAL ANSWER:"
        )

        print(
            "=" * 60
        )

        print(
            answer
        )

        print("\n")