"""The hand-written MCP server: protocol, validation, lifecycle, and that it cannot write."""

import json
import os
import subprocess
import sys

import pytest

from esis_mcp import protocol
from esis_mcp.server import Server, ToolError, validate
from esis_mcp.protocol import RpcError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _srv():
    s = Server(name="t", version="1")

    @s.tool("echo", "echo", {"text": {"type": "string", "maxLength": 5},
                             "n": {"type": "integer", "minimum": 1}}, required=["text"])
    def echo(text, n=1):
        return text * n

    @s.tool("boom", "fails")
    def boom():
        raise ToolError("no data")
    return s


def _call(s, method, params=None, id_=1):
    msg = {"jsonrpc": "2.0", "id": id_, "method": method}
    if params is not None:
        msg["params"] = params
    return protocol.handle_line(json.dumps(msg).encode(), s.dispatch)


def test_version_negotiation():
    s = _srv()
    assert _call(s, "initialize", {"protocolVersion": "2025-06-18"})["result"]["protocolVersion"] == "2025-06-18"
    r = _call(s, "initialize", {"protocolVersion": "1999-01-01"})["result"]
    assert r["protocolVersion"] == "2025-11-25" and "tools" in r["capabilities"]


def test_notifications_get_no_reply():
    s = _srv()
    msg = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode()
    assert protocol.handle_line(msg, s.dispatch) is None and s.initialized
    assert protocol.handle_line(json.dumps({"jsonrpc": "2.0", "method": "x/unknown"}).encode(), s.dispatch) is None


def test_tool_call_validation_and_errors():
    s = _srv()
    ok = _call(s, "tools/call", {"name": "echo", "arguments": {"text": "ab", "n": 2}})["result"]
    assert ok == {"content": [{"type": "text", "text": "abab"}], "isError": False}
    for args, msg in [({}, "missing"), ({"text": 3}, "type"), ({"text": "toolong"}, "longer"),
                      ({"text": "a", "n": 0}, ">="), ({"text": "a", "z": 1}, "unknown"),
                      ({"text": "a", "n": True}, "type")]:
        e = _call(s, "tools/call", {"name": "echo", "arguments": args})["error"]
        assert e["code"] == -32602 and msg in e["message"]
    assert _call(s, "tools/call", {"name": "nope"})["error"]["code"] == -32602
    bad = _call(s, "tools/call", {"name": "boom", "arguments": {}})["result"]
    assert bad["isError"] and bad["content"][0]["text"] == "no data"         # a tool failure is not a protocol error


def test_protocol_errors():
    s = _srv()
    assert protocol.handle_line(b"{not json", s.dispatch)["error"]["code"] == -32700
    assert protocol.handle_line(b'{"id":1,"method":"ping"}', s.dispatch)["error"]["code"] == -32600
    assert _call(s, "orders/place")["error"]["code"] == -32601
    batch = protocol.handle_line(json.dumps([{"jsonrpc": "2.0", "id": 1, "method": "ping"},
                                             {"jsonrpc": "2.0", "method": "notifications/initialized"}]).encode(),
                                 s.dispatch)
    assert batch == [{"jsonrpc": "2.0", "id": 1, "result": {}}]


def test_real_server_has_no_trading_tool_and_every_write_is_flagged():
    from esis_mcp.tools import server
    names = set(server.tools)
    assert not any(w in n for n in names for w in ("order", "buy", "sell", "trade", "ticket"))
    writes = {n for n, t in server.tools.items() if not t.read_only}
    assert writes == {"refresh_watchlist", "watch_add", "research_note"}
    for t in server.tools.values():
        assert t.schema["additionalProperties"] is False


def test_stdio_session_end_to_end():
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=ROOT)
    import config
    env["ESIS_TEST_DB"] = config.DB_PATH
    p = subprocess.Popen([sys.executable, "-c",
                          "import os,config; config.DB_PATH=os.environ['ESIS_TEST_DB'];"
                          "from esis_mcp.__main__ import main; main()"],
                         cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = [{"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "pytest"}}},
             {"jsonrpc": "2.0", "method": "notifications/initialized"},
             {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
             {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
              "params": {"name": "position_size_arithmetic", "arguments": {"ticker": "COMI"}}}]
    out, _ = p.communicate("".join(json.dumps(m) + "\n" for m in lines).encode(), timeout=120)
    resp = [json.loads(x) for x in out.decode("utf-8").splitlines()]
    assert [r["id"] for r in resp] == [1, 2, 3]                       # nothing else ever reaches stdout
    assert resp[0]["result"]["serverInfo"]["name"] == "esis-quant"
    assert len(resp[1]["result"]["tools"]) >= 10
    assert "shares" in resp[2]["result"]["content"][0]["text"]


def test_readonly_connection_refuses_writes(monkeypatch):
    monkeypatch.setenv("ESIS_DB_READONLY", "1")
    from data.database import get_connection
    with pytest.raises(Exception, match="readonly"):
        with get_connection() as conn:
            conn.execute("UPDATE stocks SET sector = sector")
