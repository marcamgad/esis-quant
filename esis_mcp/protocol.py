"""
JSON-RPC 2.0 over stdio -- the transport MCP uses for local servers.

Written from the specification, no SDK. The rules that matter:

  * one JSON message per line, UTF-8, on stdin (client -> server) and stdout
    (server -> client); a message never contains a raw newline;
  * stdout carries PROTOCOL ONLY. Anything else printed there corrupts the
    stream, so main() swaps sys.stdout for stderr and keeps the real stdout
    handle for messages. Logs go to stderr / logs/mcp_server.log;
  * a request has an "id" and gets exactly one response with the same id; a
    notification has no "id" and never gets a response;
  * errors use the standard codes below. A TOOL failing is not a protocol
    error: it is a normal result with isError=true (see server.py).
"""

import json
import logging
import sys

logger = logging.getLogger("esis_mcp.protocol")

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class RpcError(Exception):
    """Raised by a handler to send a JSON-RPC error response."""

    def __init__(self, code: int, message: str, data=None):
        super().__init__(message)
        self.code, self.message, self.data = code, message, data


def _error(id_, code, message, data=None) -> dict:
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": id_, "error": err}


def handle_message(msg, dispatch):
    """One parsed message -> a response dict, or None for a notification.

    `dispatch(method, params) -> result` raises RpcError for protocol errors."""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
        if isinstance(msg, dict) and ("result" in msg or "error" in msg):
            return None                      # a response to a server->client request; we send none
        return _error(msg.get("id") if isinstance(msg, dict) else None, INVALID_REQUEST, "invalid request")
    is_request = "id" in msg
    params = msg.get("params") or {}
    if not isinstance(params, (dict, list)):
        return _error(msg.get("id"), INVALID_PARAMS, "params must be an object") if is_request else None
    try:
        result = dispatch(msg["method"], params, is_request)
    except RpcError as e:
        return _error(msg.get("id"), e.code, e.message, e.data) if is_request else None
    except Exception as e:  # noqa: BLE001 -- never let a bug kill the server
        logger.exception("unhandled error in %s", msg["method"])
        return _error(msg.get("id"), INTERNAL_ERROR, f"internal error: {type(e).__name__}") if is_request else None
    if not is_request:
        return None
    return {"jsonrpc": "2.0", "id": msg["id"], "result": result if result is not None else {}}


def handle_line(line: bytes, dispatch):
    """One raw line -> the response object to write (dict or list), or None."""
    line = line.strip()
    if not line:
        return None
    try:
        msg = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        return _error(None, PARSE_ERROR, f"parse error: {e}")
    if isinstance(msg, list):                # JSON-RPC batch (older MCP revisions allowed it)
        if not msg:
            return _error(None, INVALID_REQUEST, "empty batch")
        out = [r for r in (handle_message(m, dispatch) for m in msg) if r is not None]
        return out or None
    return handle_message(msg, dispatch)


def serve(dispatch, stdin=None, stdout=None) -> None:
    """Read requests until stdin closes; write one line per response."""
    stdin = stdin or sys.stdin.buffer
    stdout = stdout or sys.stdout.buffer
    for line in iter(stdin.readline, b""):
        resp = handle_line(line, dispatch)
        if resp is None:
            continue
        stdout.write((json.dumps(resp, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
                     .encode("utf-8"))
        stdout.flush()
