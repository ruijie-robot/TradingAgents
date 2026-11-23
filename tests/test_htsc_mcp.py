"""Huatai market-sentiment tool: window clamping, MCP parsing, and a missing key."""

from __future__ import annotations

import inspect
import json
from unittest import mock

import pytest
import requests
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import tools
from tradingagents.agents.analysts import market_analyst
from tradingagents.dataflows import router
from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.vendors import htsc_mcp

TRADE_DATE = "2026-08-14"
TABLE = "# 市场情绪指数\n\n| 日期 | 数值 |\n| --- | --- |\n| 2026-08-14 | 48.20 |\n"


def _sse(payload: dict) -> str:
    return "event: message\ndata: " + json.dumps(payload) + "\n\n"


def _success(data: str, status: str = "success") -> str:
    text = json.dumps(
        {"status": status, "message": "查询成功", "data": data}, ensure_ascii=False
    )
    return _sse({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [{"type": "text", "text": text}], "isError": False},
    })


@pytest.mark.unit
def test_omitted_dates_on_a_run_stay_on_or_before_the_trade_date():
    assert htsc_mcp.clamp_window(None, None, TRADE_DATE) == ("2026-07-15", TRADE_DATE)


@pytest.mark.unit
def test_a_future_end_is_clamped_and_a_window_after_the_run_moves_back():
    assert htsc_mcp.clamp_window("2026-08-01", "2026-09-14", TRADE_DATE) == (
        "2026-08-01", TRADE_DATE,
    )
    assert htsc_mcp.clamp_window("2026-09-01", "2026-09-08", TRADE_DATE) == (
        "2026-08-07", TRADE_DATE,
    )


@pytest.mark.unit
def test_a_direct_call_with_no_dates_asks_the_server_for_its_latest_print():
    assert htsc_mcp.clamp_window(None, None, "") == (None, None)


@pytest.mark.unit
def test_a_reversed_window_is_refused():
    with pytest.raises(ValueError, match="start_date"):
        htsc_mcp.clamp_window("2026-08-12", "2026-08-01", TRADE_DATE)


@pytest.mark.unit
def test_the_envelope_returns_the_table_and_keeps_the_required_source_line():
    raw = _success(TABLE + "\n**数据来源**: 华泰智研MCP数据服务\n")
    text = htsc_mcp._unwrap(htsc_mcp._content_text(htsc_mcp._rpc(raw)))
    assert "48.20" in text
    assert "华泰智研MCP数据服务" in text
    bare = htsc_mcp._unwrap("just a number")
    assert bare.endswith("数据来源: 华泰智研MCP数据服务")


@pytest.mark.unit
def test_a_failed_query_returns_the_server_message():
    # status is not success, so the message is what the model should read.
    payload = json.dumps(
        {"status": "error", "message": "无数据", "data": ""}, ensure_ascii=False
    )
    raw = _sse({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [{"type": "text", "text": payload}], "isError": False},
    })
    assert htsc_mcp._unwrap(htsc_mcp._content_text(htsc_mcp._rpc(raw))) == "无数据"


def _http_response(body: bytes, content_type: str) -> requests.Response:
    response = requests.Response()
    response.status_code = 200
    response._content = body
    response.headers["Content-Type"] = content_type
    response.encoding = requests.utils.get_encoding_from_headers(response.headers)
    return response


@pytest.mark.unit
def test_an_event_stream_without_charset_is_read_as_utf8(monkeypatch):
    # 「情」is UTF-8 E6 83 85. Decoded as ISO-8859-1, 0x85 becomes a line break
    # and the JSON-RPC data line is cut in half.
    monkeypatch.setenv("HTSC_APP_KEY", "sk_test_key")
    text = json.dumps(
        {"status": "success", "message": "查询成功", "data": TABLE},
        ensure_ascii=False,
    )
    payload = {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": text}], "isError": False}}
    raw = ("event: message\r\ndata: " + json.dumps(payload, ensure_ascii=False) + "\r\n\r\n").encode()
    assert b"\x85" in raw
    response = _http_response(raw, "text/event-stream")
    assert response.encoding == "ISO-8859-1"

    monkeypatch.setattr(htsc_mcp.requests, "post", lambda *a, **k: response)
    out = htsc_mcp.call_tool("get_market_emotion", {"market": "HK"})
    assert "48.20" in out
    assert "市场情绪指数" in out


@pytest.mark.unit
def test_an_http_error_does_not_carry_the_app_key(monkeypatch):
    monkeypatch.setenv("HTSC_APP_KEY", "sk_test_key")

    def _post(*args, **kwargs):
        assert kwargs["headers"]["HTSC_APP_KEY"] == "sk_test_key"
        raise requests.ConnectionError("boom sk_test_key")

    monkeypatch.setattr(htsc_mcp.requests, "post", _post)
    with pytest.raises(htsc_mcp.VendorUnavailableError, match="boom") as raised:
        htsc_mcp.call_tool("get_market_emotion", {"market": "A"})
    assert "sk_test_key" not in str(raised.value)


@pytest.mark.unit
def test_a_missing_key_never_calls_the_server_and_does_not_abort(monkeypatch):
    monkeypatch.delenv("HTSC_APP_KEY", raising=False)
    with (
        mock.patch.object(htsc_mcp.requests, "post") as posted,
        pytest.raises(htsc_mcp.HtscNotConfiguredError),
    ):
        htsc_mcp.get_market_emotion("A", "2026-08-01", TRADE_DATE)
    assert not posted.called

    with run_config({"data_vendors": {"market_emotion": "htsc"}}):
        out = router.route_to_vendor("get_market_emotion", "A", "2026-08-01", TRADE_DATE)
    assert out.startswith("DATA_UNAVAILABLE")


@pytest.mark.unit
def test_the_tool_sends_the_clamped_window():
    with mock.patch.object(tools, "route_to_vendor", return_value="ok") as routed:
        out = tools.get_market_emotion.func(
            "HK", "2026-08-01", "2026-09-14", trade_date=TRADE_DATE
        )
    assert out == "ok"
    assert routed.call_args.args == ("get_market_emotion", "HK", "2026-08-01", TRADE_DATE)


class _State(MessagesState):
    trade_date: str
    company_of_interest: str


@pytest.mark.unit
def test_the_run_date_is_injected_and_hidden_from_the_model():
    schema = tools.get_market_emotion.tool_call_schema.model_json_schema()["properties"]
    assert "trade_date" not in schema
    assert schema["market"]["enum"] == ["A", "HK"]

    graph = StateGraph(_State)
    graph.add_node("tools", ToolNode([tools.get_market_emotion]))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    with mock.patch.object(tools, "route_to_vendor", return_value="ok") as routed:
        graph.compile().invoke({
            "messages": [AIMessage("", tool_calls=[{
                "name": "get_market_emotion", "args": {"market": "A"}, "id": "1",
            }])],
            "trade_date": TRADE_DATE,
            "company_of_interest": "600519.SS",
        })
    assert routed.call_args.args == ("get_market_emotion", "A", "2026-07-15", TRADE_DATE)


@pytest.mark.unit
def test_the_market_analyst_offers_the_tool_and_the_source_line():
    assert tools.get_market_emotion in market_analyst.TOOLS
    text = inspect.getsource(market_analyst)
    assert "get_market_emotion" in text
    assert "华泰智研MCP数据服务" in text
