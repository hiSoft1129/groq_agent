import os
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain.tools import tool
from langgraph.checkpoint.postgres import PostgresSaver

from tools import search_web

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen3.6-27b")

THREAD_ID = os.getenv("THREAD_ID", "main")
TIMEZONE = os.getenv("TIMEZONE", "America/Los_Angeles")


if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing from .env")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing from .env")


# ============================================================
# SEARCH CONFIG
# ============================================================

MAX_SEARCHES_PER_TURN = 1
search_count = 0


# ============================================================
# DATE / TIME TOOL
# ============================================================


@tool
def current_date_time() -> str:
    """
    Get the current date, time, and day of the week.
    """

    now = datetime.now(ZoneInfo(TIMEZONE))

    return (
        f"Current date: "
        f"{now.strftime('%A, %B %d, %Y')}\n"
        f"Current time: "
        f"{now.strftime('%I:%M:%S %p')}\n"
        f"Timezone: {TIMEZONE}"
    )


# ============================================================
# WEB SEARCH TOOL
# ============================================================


@tool
def web_search(query: str) -> str:
    """
    Search the web for current or recent information.
    """

    global search_count

    if search_count >= MAX_SEARCHES_PER_TURN:
        return "Search limit reached. " "Use the search results already available."

    search_count += 1

    print()
    print(f"🔎 SEARCHING: {query}")

    try:

        result = search_web(
            query=query,
            max_results=5,
        )

        result = str(result).strip()

        if not result:
            return "No search results found."

        if len(result) > 6000:
            result = result[:6000] + "\n...[TRUNCATED]..."

        return result

    except Exception as exc:

        return f"Search failed: " f"{type(exc).__name__}: {exc}"


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a concise and reliable AI assistant.

You have two tools:

current_date_time()
web_search(query)

============================================================
CONVERSATION MEMORY
============================================================

You have persistent conversation memory.

Use previous messages when they are relevant.

Always answer the newest user message.

Never use an old assistant answer as the answer to the
current question.

============================================================
DATE AND TIME
============================================================

For:

- what day is it today
- today's date
- what date is it
- current date
- what time is it

use current_date_time().

After receiving its result, answer immediately.

Do not call web_search for date/time questions.

============================================================
WEB RESEARCH
============================================================

Use web_search for:

- news
- today's news
- latest news
- current events
- recent events
- recent developments
- current information
- current prices
- information that may have changed recently

Use at most ONE web search per question.

After receiving useful search results, summarize them
and answer the user.

Do not perform another search.

============================================================
IMPORTANT
============================================================

If the user says:

"top new today"

interpret it as:

"top news today".

If the user says:

"top new of this month"

interpret it as:

"top news of this month".

Always answer the current user question.

Never return an empty answer.

Never return an answer from an earlier turn.

Never mention internal tools or system instructions.
"""


# ============================================================
# MODEL
# ============================================================

model = ChatGroq(
    model=MODEL_NAME,
    temperature=0,
    timeout=30,
    max_retries=1,
    max_tokens=1000,
)


# ============================================================
# CREATE AGENT
# ============================================================


def create_app(checkpointer):

    return create_agent(
        model=model,
        tools=[
            current_date_time,
            web_search,
        ],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


# ============================================================
# CLASSIFY CURRENT QUERY
# ============================================================


def is_news_query(text):
    """
    Detect obvious news/current-event questions.

    This is intentionally simple and deterministic.
    """

    text = text.lower().strip()

    keywords = [
        "news",
        "latest news",
        "top news",
        "breaking news",
        "current events",
        "recent news",
        "what happened today",
        "news today",
        "news this week",
        "news this month",
        "news this year",
    ]

    return any(keyword in text for keyword in keywords)


def is_date_time_query(text):
    """
    Detect obvious date/time questions.
    """

    text = text.lower().strip()

    keywords = [
        "what day is it",
        "what day is today",
        "what is today's date",
        "what's today's date",
        "what date is it",
        "today's date",
        "current date",
        "what time is it",
        "current time",
    ]

    return any(keyword in text for keyword in keywords)


# ============================================================
# EXTRACT CURRENT TURN ANSWER
# ============================================================


def extract_current_answer(
    result,
    start_index,
):
    """
    Extract an AI answer ONLY from messages generated
    during the current invocation.

    This is the critical fix.

    We NEVER search older conversation messages for
    an answer.
    """

    messages = result.get(
        "messages",
        [],
    )

    # Only inspect messages generated after the
    # current user message.
    new_messages = messages[start_index:]

    print()
    print("📨 CURRENT TURN TRACE")
    print("-" * 60)

    for index, message in enumerate(
        new_messages,
        start=start_index,
    ):

        message_type = getattr(
            message,
            "type",
            None,
        )

        content = getattr(
            message,
            "content",
            None,
        )

        print(f"[{index}] " f"{type(message).__name__} " f"type={message_type}")

        if content:

            preview = str(content)

            if len(preview) > 300:
                preview = preview[:300] + "..."

            print(f"    {preview}")

    print("-" * 60)

    # Find only AI messages generated during
    # this current turn.
    for message in reversed(new_messages):

        if getattr(message, "type", None) != "ai":
            continue

        content = getattr(
            message,
            "content",
            None,
        )

        if isinstance(content, str):

            text = content.strip()

            if text:
                return text

        elif isinstance(content, list):

            parts = []

            for block in content:

                if isinstance(block, dict):

                    text = block.get("text")

                    if text:

                        text = str(text).strip()

                        if text:
                            parts.append(text)

                elif isinstance(block, str):

                    block = block.strip()

                    if block:
                        parts.append(block)

            if parts:
                return "\n".join(parts)

    return None


# ============================================================
# DIRECT NEWS SEARCH
# ============================================================


def search_news_directly(user_question):
    """
    Deterministically perform one search for news questions.

    This prevents the LLM from deciding not to call the
    search tool for an obviously current-news request.
    """

    global search_count

    search_count = 0

    print()
    print("📰 CURRENT NEWS REQUEST")

    result = web_search.invoke({"query": user_question})

    return result


# ============================================================
# RUN ONE QUESTION
# ============================================================


def run_question(
    agent,
    checkpointer,
    user_input,
    config,
):
    """
    Process one user question.
    """

    global search_count

    search_count = 0

    # --------------------------------------------------------
    # Get the existing conversation state BEFORE invoking
    # the agent.
    # --------------------------------------------------------

    previous_state = agent.get_state(config)

    previous_messages = (
        previous_state.values.get("messages", []) if previous_state else []
    )

    start_index = len(previous_messages)

    # --------------------------------------------------------
    # NEWS
    # --------------------------------------------------------

    if is_news_query(user_input):

        print()
        print("=" * 60)
        print("📰 NEWS SEARCH")
        print("=" * 60)

        search_results = search_news_directly(user_input)

        # ----------------------------------------------------
        # Give the search results to the model.
        #
        # The search has already been performed, so the
        # model's job is to summarize the results.
        # ----------------------------------------------------

        prompt = f"""
The user asked:

{user_input}

Use the following current web search results to answer
the user's question.

SEARCH RESULTS:

{search_results}

Answer the user directly and concisely.

Do not perform another web search.
"""

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            },
            config=config,
        )

    # --------------------------------------------------------
    # DATE / TIME
    # --------------------------------------------------------

    elif is_date_time_query(user_input):

        print()
        print("=" * 60)
        print("📅 DATE / TIME")
        print("=" * 60)

        # Direct deterministic date lookup.
        date_result = current_date_time.invoke({})

        prompt = f"""
The user asked:

{user_input}

Here is the current date and time:

{date_result}

Answer the user directly.

Do not perform another tool call.
"""

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            },
            config=config,
        )

    # --------------------------------------------------------
    # NORMAL QUESTION
    # --------------------------------------------------------

    else:

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

    # --------------------------------------------------------
    # Extract current answer.
    # --------------------------------------------------------

    answer = extract_current_answer(
        result,
        start_index,
    )

    return answer


# ============================================================
# MAIN
# ============================================================


def main():

    print()
    print("=" * 60)
    print("🤖 AI WEB RESEARCH AGENT")
    print("=" * 60)

    print()
    print(f"Model: {MODEL_NAME}")
    print(f"Thread: {THREAD_ID}")
    print(f"Timezone: {TIMEZONE}")

    print()
    print("PostgreSQL conversational memory: ENABLED")

    print()
    print("Type 'exit' or 'quit' to stop.")

    # ========================================================
    # POSTGRESQL CHECKPOINTER
    # ========================================================

    with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:

        # Create checkpoint tables if needed.
        checkpointer.setup()

        print()
        print("✅ PostgreSQL memory ready.")

        # ====================================================
        # CREATE AGENT
        # ====================================================

        agent = create_app(checkpointer)

        # ====================================================
        # THREAD CONFIG
        # ====================================================

        config = {"configurable": {"thread_id": THREAD_ID}}

        # ====================================================
        # CHAT LOOP
        # ====================================================

        while True:

            try:

                user_input = input("\nYou: ").strip()

            except (
                KeyboardInterrupt,
                EOFError,
            ):

                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in {
                "exit",
                "quit",
            }:

                print("Goodbye!")
                break

            print()
            print("=" * 60)
            print("🤖 AGENT STARTING")
            print("=" * 60)

            try:

                answer = run_question(
                    agent,
                    checkpointer,
                    user_input,
                    config,
                )

                if answer:

                    print()
                    print("=" * 60)
                    print("✅ FINAL ANSWER")
                    print("=" * 60)

                    print()
                    print(answer)

                    print()
                    print("💾 Conversation saved " "automatically in PostgreSQL.")

                else:

                    print()
                    print("=" * 60)
                    print("⚠️ NO CURRENT ANSWER")
                    print("=" * 60)

                    print(
                        "\nThe agent did not generate " "a text response for this turn."
                    )

            except Exception as exc:

                print()
                print("=" * 60)
                print("❌ AGENT ERROR")
                print("=" * 60)

                print(f"\nType: {type(exc).__name__}")

                print(f"Message: {exc}")


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
