"""
Run the ESIS-QUANT MCP server on stdio:

    py -3.11 -m esis_mcp

Claude Code / Claude Desktop start it themselves from their MCP configuration
(.mcp.json in this project). Order matters below: the read-only flag must be set
BEFORE any module opens the database, and stdout must be reserved for protocol
messages before anything can print to it.
"""

import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    os.environ["ESIS_DB_READONLY"] = "1"
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    out = sys.stdout.buffer             # the protocol channel
    sys.stdout = sys.stderr             # any stray print() goes to stderr, not into the stream
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=os.path.join(ROOT, "logs", "mcp_server.log"),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from esis_mcp.protocol import serve
    from esis_mcp.tools import server
    logging.getLogger("esis_mcp").info("starting, %d tools", len(server.tools))
    serve(server.dispatch, stdin=sys.stdin.buffer, stdout=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
