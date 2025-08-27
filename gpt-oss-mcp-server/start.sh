#!/bin/bash

if [ "$MCP_SERVICE" = "python" ]; then
    echo "Starting Python MCP Server..."
    exec mcp run -t sse python_server.py:mcp
else
    echo "Starting Browser MCP Server..."
    exec mcp run -t sse browser_server.py:mcp
fi
