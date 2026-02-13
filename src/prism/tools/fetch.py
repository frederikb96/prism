"""
Fetch MCP tool.

Direct Tavily extract wrapper for single-URL content extraction.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"
TIMEOUT_SECONDS = 30


async def execute_fetch(url: str) -> dict[str, Any]:
    """
    Extract content from a URL via Tavily extract API.

    Uses extract_depth="advanced" for thorough extraction.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {
            "success": False,
            "error": "TAVILY_API_KEY environment variable not set",
        }

    payload = {
        "api_key": api_key,
        "urls": [url],
        "extract_depth": "advanced",
    }

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            resp = await client.post(TAVILY_EXTRACT_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        return {
            "success": False,
            "error": f"Tavily API error: {e.response.status_code}",
            "url": url,
        }
    except httpx.RequestError as e:
        return {
            "success": False,
            "error": f"Request failed: {e}",
            "url": url,
        }

    results = data.get("results", [])
    failed = data.get("failed_results", [])

    if not results:
        error_detail = failed[0] if failed else "No content extracted"
        return {
            "success": False,
            "error": str(error_detail),
            "url": url,
        }

    result = results[0]
    response: dict[str, Any] = {"success": True}
    response.update(result)
    return response
