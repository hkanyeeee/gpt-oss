#!/bin/bash

if [ "$MCP_SERVICE" = "python" ]; then
    echo "Starting Python MCP Server..."
    exec python python_server.py
else
    echo "Starting Browser MCP Server..."
    exec python browser_server.py
fi
