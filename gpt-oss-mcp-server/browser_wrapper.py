#!/usr/bin/env python3
"""
Browser MCP Server Wrapper - 确保环境变量正确设置
"""

import os
import sys

# 设置默认环境变量（如果未设置）
DEFAULT_ENV = {
    "SEARXNG_QUERY_URL": "http://192.168.31.125:8080/search",
    "EMBEDDING_SERVICE_URL": "http://localhost:1234/v1", 
    "QDRANT_HOST": "192.168.31.125",
    "QDRANT_PORT": "6333",
    "QDRANT_COLLECTION_NAME": "browser_mcp",
    "DATABASE_URL": "sqlite+aiosqlite:///./data/mcp_browser.db",
    "WEB_LOADER_ENGINE": "safe_web",
    "CHUNK_SIZE": "800",
    "CHUNK_OVERLAP": "80",
    "RAG_TOP_K": "12"
}

# 设置环境变量
for key, value in DEFAULT_ENV.items():
    if key not in os.environ:
        os.environ[key] = value

# 导入并运行主服务器
if __name__ == "__main__":
    # 确保当前目录在Python路径中
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # 导入并运行主服务器模块
    from browser_server import mcp
    print("Starting Browser MCP Server...")
    mcp.run()
