import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise RuntimeError("GOOGLE_API_KEY is not set")


# --------------------------------------------------
# Tool
# --------------------------------------------------

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""

    try:
        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception as e:
        return f"Calculation error: {e}"


# --------------------------------------------------
# Gemini model
# --------------------------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)


# --------------------------------------------------
# Agent
# --------------------------------------------------

agent = create_agent(
    model=model,
    tools=[calculate],
    system_prompt=(
        "You are a helpful AI assistant. "
        "Use the calculator tool when mathematical calculations "
        "are required."
    )
)


# --------------------------------------------------
# Run agent
# --------------------------------------------------

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is 125 * 48?"
            }
        ]
    }
)


# --------------------------------------------------
# Print final answer
# --------------------------------------------------

print(result["messages"][-1].content)