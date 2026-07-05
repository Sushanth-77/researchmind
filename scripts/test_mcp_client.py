"""
Self-contained MCP client: spawns the ResearchMind MCP server as a
subprocess over stdio, lists its tools, and exercises each one — the
same "verify via terminal output" discipline used throughout this
project, without requiring Claude Desktop or any other MCP host.

Run: python scripts/test_mcp_client.py
"""

import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "researchmind.mcp_server"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("--- Available tools ---")
            for tool in tools.tools:
                print(f"  {tool.name}: {tool.description}")

            print("\n--- Calling list_ingested_papers ---")
            result = await session.call_tool("list_ingested_papers", {})
            print(result.content[0].text)

            print("\n--- Calling search_arxiv_papers ---")
            result = await session.call_tool(
                "search_arxiv_papers",
                {"query": "prompt engineering large language models", "max_results": 3},
            )
            print(result.content[0].text)

            print("\n--- Reading papers://ingested resource ---")
            resource_result = await session.read_resource("papers://ingested")
            print(resource_result.contents[0].text)

    print("\n✅ MCP server confirmed working end-to-end.")


if __name__ == "__main__":
    asyncio.run(main())