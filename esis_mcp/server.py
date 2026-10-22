"""
The MCP layer: lifecycle, tools, prompts, resources. Written from the spec.

Methods implemented:
  initialize                 version negotiation + capabilities + instructions
  notifications/initialized  (notification; no reply)
  ping                       -> {}
  tools/list, tools/call
  prompts/list, prompts/get
  resources/list, resources/read
  notifications/cancelled    accepted and ignored (calls here are short)

VERSION NEGOTIATION. The client proposes a protocol version; if this server
supports it, it answers with the same one, otherwise with the newest it
supports, and the client decides whether to continue.

TOOL ERRORS vs PROTOCOL ERRORS. An unknown tool or bad arguments is a protocol
error (-32602) -- the request itself was wrong. A tool that runs and fails
(no such ticker, no data) returns a normal result with isError=true and the
reason as text, so the model can read it and recover.

ARGUMENTS are checked against each tool's JSON Schema subset (type, required,
enum, minimum/maximum, no unknown keys) before the handler runs.
"""

import logging
from dataclasses import dataclass, field

from esis_mcp.protocol import INVALID_PARAMS, METHOD_NOT_FOUND, RpcError

logger = logging.getLogger("esis_mcp.server")

SUPPORTED_VERSIONS = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"]
_PY_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
             "object": dict, "array": list}


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    handler: callable
    read_only: bool = True
    title: str = ""


@dataclass
class Prompt:
    name: str
    description: str
    arguments: list            # [{"name", "description", "required"}]
    render: callable           # (**args) -> str


@dataclass
class Resource:
    uri: str
    name: str
    description: str
    read: callable             # () -> str
    mime_type: str = "text/markdown"


def validate(schema: dict, args: dict) -> None:
    """Raise RpcError(-32602) if args do not satisfy the (subset) schema."""
    if not isinstance(args, dict):
        raise RpcError(INVALID_PARAMS, "arguments must be an object")
    props = schema.get("properties", {})
    for req in schema.get("required", []):
        if req not in args:
            raise RpcError(INVALID_PARAMS, f"missing required argument: {req}")
    if schema.get("additionalProperties") is False:
        extra = set(args) - set(props)
        if extra:
            raise RpcError(INVALID_PARAMS, f"unknown argument(s): {', '.join(sorted(extra))}")
    for k, v in args.items():
        p = props.get(k)
        if not p or v is None:
            continue
        t = p.get("type")
        if t and not isinstance(v, _PY_TYPES[t]) or (t in ("integer", "number") and isinstance(v, bool)):
            raise RpcError(INVALID_PARAMS, f"{k} must be of type {t}")
        if "enum" in p and v not in p["enum"]:
            raise RpcError(INVALID_PARAMS, f"{k} must be one of {p['enum']}")
        if "minimum" in p and v < p["minimum"]:
            raise RpcError(INVALID_PARAMS, f"{k} must be >= {p['minimum']}")
        if "maximum" in p and v > p["maximum"]:
            raise RpcError(INVALID_PARAMS, f"{k} must be <= {p['maximum']}")
        if t == "string" and "maxLength" in p and len(v) > p["maxLength"]:
            raise RpcError(INVALID_PARAMS, f"{k} is longer than {p['maxLength']} characters")


@dataclass
class Server:
    name: str
    version: str
    instructions: str = ""
    tools: dict = field(default_factory=dict)
    prompts: dict = field(default_factory=dict)
    resources: dict = field(default_factory=dict)
    initialized: bool = False
    protocol_version: str = ""

    # ---- registration -------------------------------------------------
    def tool(self, name: str, description: str, properties: dict = None, required=(), read_only=True, title=""):
        """Decorator: register `fn(**arguments) -> str` as a tool."""
        schema = {"type": "object", "properties": properties or {}, "required": list(required),
                  "additionalProperties": False}

        def deco(fn):
            self.tools[name] = Tool(name, description, schema, fn, read_only, title or name)
            return fn
        return deco

    def prompt(self, name: str, description: str, arguments=()):
        def deco(fn):
            self.prompts[name] = Prompt(name, description, list(arguments), fn)
            return fn
        return deco

    def resource(self, uri: str, name: str, description: str, mime_type: str = "text/markdown"):
        def deco(fn):
            self.resources[uri] = Resource(uri, name, description, fn, mime_type)
            return fn
        return deco

    # ---- dispatch -----------------------------------------------------
    def dispatch(self, method: str, params, is_request: bool):
        if not isinstance(params, dict):
            params = {}
        handler = {
            "initialize": self._initialize,
            "notifications/initialized": self._initialized,
            "notifications/cancelled": lambda p: None,
            "ping": lambda p: {},
            "tools/list": self._tools_list,
            "tools/call": self._tools_call,
            "prompts/list": self._prompts_list,
            "prompts/get": self._prompts_get,
            "resources/list": self._resources_list,
            "resources/read": self._resources_read,
        }.get(method)
        if handler is None:
            if not is_request:
                return None                  # unknown notifications are ignored
            raise RpcError(METHOD_NOT_FOUND, f"method not found: {method}")
        return handler(params)

    def _initialize(self, p):
        asked = p.get("protocolVersion")
        self.protocol_version = asked if asked in SUPPORTED_VERSIONS else SUPPORTED_VERSIONS[0]
        client = (p.get("clientInfo") or {}).get("name", "?")
        logger.info("initialize from %s, protocol %s (asked %s)", client, self.protocol_version, asked)
        return {"protocolVersion": self.protocol_version,
                "capabilities": {"tools": {"listChanged": False}, "prompts": {"listChanged": False},
                                 "resources": {"listChanged": False, "subscribe": False}},
                "serverInfo": {"name": self.name, "version": self.version},
                "instructions": self.instructions}

    def _initialized(self, p):
        self.initialized = True

    def _tools_list(self, p):
        return {"tools": [{"name": t.name, "title": t.title, "description": t.description,
                           "inputSchema": t.schema,
                           "annotations": {"title": t.title, "readOnlyHint": t.read_only,
                                           "destructiveHint": False, "idempotentHint": t.read_only,
                                           "openWorldHint": False}}
                          for t in self.tools.values()]}

    def _tools_call(self, p):
        name, args = p.get("name"), p.get("arguments") or {}
        tool = self.tools.get(name)
        if tool is None:
            raise RpcError(INVALID_PARAMS, f"unknown tool: {name}")
        validate(tool.schema, args)
        logger.info("tools/call %s %s", name, args)
        try:
            text = tool.handler(**args)
            return {"content": [{"type": "text", "text": str(text)}], "isError": False}
        except ToolError as e:
            return {"content": [{"type": "text", "text": str(e)}], "isError": True}
        except Exception as e:  # noqa: BLE001 -- the model gets the reason, the server lives
            logger.exception("tool %s failed", name)
            return {"content": [{"type": "text", "text": f"{name} failed: {type(e).__name__}: {e}"}],
                    "isError": True}

    def _prompts_list(self, p):
        return {"prompts": [{"name": x.name, "description": x.description, "arguments": x.arguments}
                            for x in self.prompts.values()]}

    def _prompts_get(self, p):
        x = self.prompts.get(p.get("name"))
        if x is None:
            raise RpcError(INVALID_PARAMS, f"unknown prompt: {p.get('name')}")
        args = p.get("arguments") or {}
        for a in x.arguments:
            if a.get("required") and a["name"] not in args:
                raise RpcError(INVALID_PARAMS, f"missing prompt argument: {a['name']}")
        return {"description": x.description,
                "messages": [{"role": "user", "content": {"type": "text", "text": x.render(**args)}}]}

    def _resources_list(self, p):
        return {"resources": [{"uri": r.uri, "name": r.name, "description": r.description,
                               "mimeType": r.mime_type} for r in self.resources.values()]}

    def _resources_read(self, p):
        r = self.resources.get(p.get("uri"))
        if r is None:
            raise RpcError(-32002, f"resource not found: {p.get('uri')}")
        return {"contents": [{"uri": r.uri, "mimeType": r.mime_type, "text": r.read()}]}


class ToolError(Exception):
    """Raised by a tool for an expected failure (bad ticker, no data)."""
