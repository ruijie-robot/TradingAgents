"""Huatai (华泰智研) research MCP vendor.

Speaks the server's streamable HTTP transport: one JSON-RPC POST, with the
response delivered as an SSE ``data:`` line. The app key stays in the
``HTSC_APP_KEY`` header and is stripped from any error this module raises.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import requests

from tradingagents.dataflows.date_window import as_of, as_of_window
from tradingagents.dataflows.errors import VendorNotConfiguredError, VendorUnavailableError

DEFAULT_URL = "https://inst.htsc.com/mcp/v1/ris/htsc_research_mcp"
SOURCE = "华泰智研MCP数据服务"
REQUEST_TIMEOUT = 30
# A month of daily readings is enough for a market report. Omitting both dates
# makes the server return its latest print, which is after a backtest's trade date.
DEFAULT_LOOKBACK_DAYS = 30


class HtscNotConfiguredError(VendorNotConfiguredError):
    """Raised when the Huatai MCP is selected but ``HTSC_APP_KEY`` is unset."""


def clamp_window(
    start_date: str | None, end_date: str | None, trade_date: str
) -> tuple[str | None, str | None]:
    """The window sent to the server, never ending after the run's trade date.

    Both dates omitted on a graph run become the ``DEFAULT_LOOKBACK_DAYS`` ending
    that day. A direct call with no dates and no trade date leaves them empty so
    the server can return its latest print. An unreadable or reversed window
    raises ``ValueError``, matching the other dated tools.
    """
    start_date = start_date or None
    end_date = end_date or None
    if start_date is None and end_date is None:
        if not trade_date:
            return None, None
        end = as_of(None, trade_date)
        return _lookback(end), end
    if start_date is None:
        end = as_of(end_date, trade_date)
        return _lookback(end), end
    if end_date is None and not trade_date:
        start, _ = as_of_window(start_date, start_date, "")
        return start, None
    return as_of_window(start_date, end_date or trade_date, trade_date)


def _lookback(end: str) -> str:
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    return f"{end_dt - timedelta(days=DEFAULT_LOOKBACK_DAYS):%Y-%m-%d}"


def get_market_emotion(
    market: str = "A",
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """A-share (``A``) or Hong Kong (``HK``) sentiment index, in percent."""
    arguments: dict[str, str] = {"market": _market(market)}
    if start_date:
        arguments["start_date"] = start_date
    if end_date:
        arguments["end_date"] = end_date
    return _unwrap(call_tool("get_market_emotion", arguments))


def call_tool(name: str, arguments: dict) -> str:
    """Call one MCP tool and return the text content the server sent back."""
    key = os.getenv("HTSC_APP_KEY")
    if not key:
        raise HtscNotConfiguredError(
            "HTSC_APP_KEY environment variable is not set."
        )
    url = os.getenv("HTSC_MCP_URL", DEFAULT_URL)
    try:
        response = requests.post(
            url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
            headers={
                "Accept": "application/json, text/event-stream",
                "HTSC_APP_KEY": key,
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise VendorUnavailableError(str(exc).replace(key, "***")) from None
    return _content_text(_rpc(response.text))


def _market(market: str) -> str:
    value = (market or "A").strip().upper()
    if value not in {"A", "HK"}:
        raise ValueError("market must be 'A' (A-shares) or 'HK' (Hong Kong)")
    return value


def _rpc(raw: str) -> dict:
    """The JSON-RPC message, whether the server sent SSE or a bare JSON body."""
    lines = [line[5:].strip() for line in raw.splitlines() if line.startswith("data:")]
    body = "\n".join(lines) if lines else raw.strip()
    try:
        message = json.loads(body)
    except json.JSONDecodeError:
        raise VendorUnavailableError("HTSC MCP returned a response that is not JSON") from None
    if not isinstance(message, dict):
        raise VendorUnavailableError("HTSC MCP returned a response that is not JSON")
    if "error" in message:
        error = message["error"]
        detail = error.get("message") if isinstance(error, dict) else str(error)
        raise VendorUnavailableError(detail or "HTSC MCP tool call failed")
    result = message.get("result")
    if not isinstance(result, dict):
        raise VendorUnavailableError("HTSC MCP returned no tool result")
    if result.get("isError"):
        raise VendorUnavailableError(_blocks(result) or "HTSC MCP tool call failed")
    return result


def _content_text(result: dict) -> str:
    text = _blocks(result)
    if not text:
        raise VendorUnavailableError("HTSC MCP returned an empty tool result")
    return text


def _blocks(result: dict) -> str:
    parts = []
    for block in result.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(part for part in parts if part).strip()


def _unwrap(text: str) -> str:
    """The markdown table inside the server's ``{status, message, data}`` envelope."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict) and "data" in parsed:
        if parsed.get("status") not in (None, "success"):
            detail = parsed.get("message") or parsed.get("data") or text
            return str(detail)
        data = parsed["data"]
        text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    if SOURCE not in text:
        text = f"{text.rstrip()}\n\n数据来源: {SOURCE}"
    return text
