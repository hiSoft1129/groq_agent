import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain_groq import ChatGroq


from tools import search_web

# ============================================================
# CONFIG
# ============================================================

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY is missing. Check your .env file.")


# ============================================================
# WEB SEARCH TOOL
# ============================================================


@tool(description="""
Search the internet for information.

IMPORTANT:
The only argument is `query`.
Example:
{"query": "top programming languages 2025"}

Do not provide cursor, id, URL, page number, or any other
parameters.
""")
def web_search(query: str) -> str:

    print(f"\n🔎 SEARCHING: {query}")

    try:
        result = search_web(query=query, max_results=3)

        result = str(result)

        # Prevent huge tool results from being sent to Groq.
        if len(result) > 3000:
            result = result[:3000] + "\n...[TRUNCATED]..."

        print(f"   Result size: {len(result):,} characters")

        return result

    except Exception as e:

        error = f"Search failed: " f"{type(e).__name__}: {e}"

        print(f"   ❌ {error}")

        return error


# ============================================================
# MODEL
# ============================================================

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    timeout=30,
    max_retries=0,
    max_tokens=300,
)


# ============================================================
# AGENT
# ============================================================
checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[web_search],
    checkpointer=checkpointer,
    system_prompt="""
You are a concise web research assistant.

You have ONE tool:

web_search(query)

The tool accepts ONLY a string called `query`.

When calling it, the arguments MUST look exactly like:

{
    "query": "your search query"
}

Never use:
- cursor
- id
- url
- page
- offset
- max_results

The tool already controls the number of results.

Research rules:

1. Use web_search when current information is needed.
2. You may search more than once if necessary.
3. Do not search the same thing repeatedly.
4. Once you have enough information, stop calling the tool.
5. Give the user a concise final answer.
6. Do not describe internal tool calls.
""",
)
config = {"configurable": {"thread_id": "conversation-1"}}


# ============================================================
# RUN
# ============================================================


def run_agent():

    while True:

        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        print("\n" + "=" * 60)
        print("🤖 AGENT STARTING")
        print("=" * 60)

        try:

            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input,
                        }
                    ]
                },
                config=config,
            )

        except Exception as e:

            print("\n❌ AGENT ERROR")
            print(type(e).__name__)
            print(str(e))

            continue

        messages = result.get("messages", [])

        final_content = None

        # Find the last message containing actual text
        for message in reversed(messages):

            content = getattr(message, "content", None)

            if not content:
                continue

            # Handle normal string content
            if isinstance(content, str):

                final_content = content

            # Handle structured content
            elif isinstance(content, list):

                text_parts = []

                for block in content:

                    if isinstance(block, dict):

                        text = block.get("text")

                        if text:
                            text_parts.append(text)

                    elif isinstance(block, str):

                        text_parts.append(block)

                if text_parts:
                    final_content = "\n".join(text_parts)

            if final_content:
                break

        # --------------------------------------------------
        # DISPLAY + SAVE
        # --------------------------------------------------

        if final_content:

            print("\n" + "=" * 60)
            print("✅ FINAL ANSWER")
            print("=" * 60)

            print(final_content)

            with open(
                "response.md",
                "a",
                encoding="utf-8"
            ) as f:

                f.write("# User\n\n")
                f.write(user_input)
                f.write("\n\n")

                f.write("# Assistant\n\n")
                f.write(final_content)
                f.write("\n\n---\n\n")

            print("\n💾 Saved to response.md")

        else:

            print("\n⚠️ No final answer returned.")

            # Useful for debugging
            print("\nMessages returned:")

            for message in messages:
                print(
                    type(message).__name__,
                    getattr(message, "type", None),
                    repr(getattr(message, "content", None))
                )

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_agent()
