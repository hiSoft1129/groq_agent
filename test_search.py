from ddgs import DDGS

with DDGS(timeout=10) as ddgs:
    results = ddgs.text(
        "Python programming language",
        backend="google,bing",
        max_results=5,
    )

for result in results:
    print(result)