"""
Web Search Functionality for Rationality Checks

Provides web search capabilities for fact-checking claims using Serper API.
"""

import json
import logging
import os
import requests
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class WebSearchTool:
    """Web search tool using Serper API"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize web search tool

        Args:
            api_key: Serper API key (defaults to SERPER_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        if not self.api_key:
            logger.warning("SERPER_API_KEY not set - web search will be disabled")

    def search(self, query: str, num_results: int = 3) -> Dict[str, Any]:
        """
        Perform a single search query

        Args:
            query: Search query string
            num_results: Number of results to return

        Returns:
            Dictionary with search results
        """
        try:
            if not self.api_key:
                return {
                    "error": "API key not available",
                    "query": query,
                    "results": []
                }

            # Add filter to exclude PDFs and docs
            search_query = "-filetype:pdf -filetype:docx -filetype:doc " + query
            url = "https://google.serper.dev/search"
            payload = json.dumps({"q": search_query, "num": num_results})
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }

            response = requests.post(url, headers=headers, data=payload, timeout=15)
            response.raise_for_status()
            data = response.json()

            results = []
            if "organic" in data:
                for result in data["organic"][:num_results]:
                    results.append({
                        "title": result.get("title", "No Title"),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", "")
                    })

            return {
                "query": query,
                "results": results,
                "total_results": len(results)
            }

        except Exception as e:
            logger.error(f"Error in search for query '{query}': {str(e)}")
            return {
                "error": str(e),
                "query": query,
                "results": []
            }

    def search_multiple(self, queries: List[str], num_results: int = 3) -> List[Dict[str, Any]]:
        """
        Perform multiple search queries

        Args:
            queries: List of search query strings
            num_results: Number of results per query

        Returns:
            List of search result dictionaries
        """
        results = []
        for query in queries:
            logger.info(f"Searching: {query}")
            result = self.search(query, num_results)
            results.append(result)
        return results

    def format_results(self, search_results: List[Dict[str, Any]]) -> str:
        """
        Format search results as text for LLM consumption

        Args:
            search_results: List of search result dictionaries

        Returns:
            Formatted text string
        """
        if not search_results:
            return "No search results available."

        formatted = "=" * 50 + "\n"
        formatted += "WEB SEARCH RESULTS\n"
        formatted += "=" * 50 + "\n\n"

        for i, result in enumerate(search_results, 1):
            query = result.get("query", "Unknown")
            formatted += f"Query {i}: {query}\n"
            formatted += "-" * 30 + "\n"

            if "error" in result:
                formatted += f"Error: {result['error']}\n\n"
                continue

            results_list = result.get("results", [])
            if not results_list:
                formatted += "No results found.\n\n"
                continue

            for j, item in enumerate(results_list, 1):
                formatted += f"  Result {j}: {item.get('title', 'No Title')}\n"
                formatted += f"  URL: {item.get('link', '')}\n"
                formatted += f"  Snippet: {item.get('snippet', '')}\n"
                formatted += "  " + "-" * 26 + "\n"

            formatted += "\n"

        return formatted

    def get_tool_definition(self) -> Dict[str, Any]:
        """
        Get Bedrock tool definition for this web search tool

        Returns:
            Dictionary containing tool specification for Bedrock
        """
        return {
            "toolSpec": {
                "name": "web_search",
                "description": "Search the web for information to verify factual claims. Returns relevant search results with titles, URLs, and snippets.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "required": ["queries"],
                        "properties": {
                            "queries": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of search queries to execute"
                            },
                            "num_results": {
                                "type": "integer",
                                "description": "Number of search results to return per query (default: 3)",
                                "default": 3
                            }
                        }
                    }
                }
            }
        }

    def execute_from_tool_use(self, tool_input: Dict[str, Any]) -> str:
        """
        Execute web search from Bedrock tool use input

        Args:
            tool_input: Dictionary containing 'queries' and optionally 'num_results'

        Returns:
            Formatted search results as string
        """
        queries = tool_input.get("queries", [])
        num_results = tool_input.get("num_results", 3)

        if not queries:
            return "No search queries provided."

        logger.info(f"Executing web search for {len(queries)} queries")
        search_results = self.search_multiple(queries, num_results)
        return self.format_results(search_results)


# Create a global instance for easy import
_web_search_tool = None

def get_web_search_tool() -> WebSearchTool:
    """Get or create the global web search tool instance"""
    global _web_search_tool
    if _web_search_tool is None:
        _web_search_tool = WebSearchTool()
    return _web_search_tool
