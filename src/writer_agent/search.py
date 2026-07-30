"""External search adapter."""

import os

from tavily import TavilyClient


def search_web(query: str) -> list[dict[str, str]]:
    """Search Tavily and normalize provider results for the research agent."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("Missing TAVILY_API_KEY environment variable.")

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        search_depth="basic",
        max_results=8,
        include_answer=False,
        include_raw_content=False,
    )

    return [
        {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "snippet": result.get("content", ""),
        }
        for result in response.get("results", [])
    ]
