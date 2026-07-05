"""
Entry point for running the ResearchMind MCP server over stdio.

Not meant to be run interactively by a human directly — an MCP host
(Claude Desktop, or the test client in scripts/test_mcp_client.py)
spawns this as a subprocess and communicates with it over stdin/stdout.

Running it directly will just sit waiting for stdio input (Ctrl+C to exit) —
that's expected, not a hang.
"""

from researchmind.mcp_server import mcp

if __name__ == "__main__":
    mcp.run()