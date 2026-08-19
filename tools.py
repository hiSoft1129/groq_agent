import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for current information."""

    try:
        with DDGS(timeout=10) as ddgs:
            results = list(
                ddgs.text(
                    query,
                    max_results=max_results
                )
            )

        if not results:
            return "No results found."

        formatted = []

        for i, result in enumerate(results, 1):
            formatted.append(
                f"[{i}] {result.get('title', 'No title')}\n"
                f"URL: {result.get('href', '')}\n"
                f"Summary: {result.get('body', '')}\n"
            )

        return "\n".join(formatted)

    except Exception as e:
        return f"SEARCH_FAILED: {type(e).__name__}: {e}"


def open_url(url: str) -> str:
    """Open a web page and extract readable text."""

    try:

        if not url.startswith(("http://", "https://")):
            return "OPEN_URL_FAILED: URL must start with http:// or https://"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/151.0.0.0 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=15,
            allow_redirects=True
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove elements that don't contain useful article text
        for element in soup(
            [
                "script",
                "style",
                "noscript",
                "nav",
                "footer",
                "header",
                "aside"
            ]
        ):
            element.decompose()

        text = soup.get_text(
            separator="\n",
            strip=True
        )

        # Clean excessive blank lines
        lines = []

        for line in text.splitlines():

            line = line.strip()

            if line:
                lines.append(line)

        text = "\n".join(lines)

        if not text:
            return "OPEN_URL_FAILED: No readable text found."

        # Prevent huge pages from consuming the model context
        max_chars = 20000

        if len(text) > max_chars:
            text = text[:max_chars] + (
                "\n\n[Page truncated because it was too large.]"
            )

        return (
            f"URL: {response.url}\n"
            f"Status: {response.status_code}\n\n"
            f"{text}"
        )

    except requests.exceptions.Timeout:
        return "OPEN_URL_FAILED: Request timed out."

    except requests.exceptions.RequestException as e:
        return f"OPEN_URL_FAILED: {type(e).__name__}: {e}"

    except Exception as e:
        return f"OPEN_URL_FAILED: {type(e).__name__}: {e}"


def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression."""

    try:

        allowed_chars = set(
            "0123456789+-*/()., "
        )

        if not all(
            c in allowed_chars
            for c in expression
        ):
            return "Error: Invalid characters in expression"

        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"