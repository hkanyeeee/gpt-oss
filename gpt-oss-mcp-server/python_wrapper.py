#!/usr/bin/env python3
"""
Python MCP Server Wrapper - 确保正确的模块路径
"""

import os
import sys

# 确保gpt-oss项目在Python路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入并运行主服务器
if __name__ == "__main__":
    from python_server import mcp
    print("Starting Python MCP Server...")
    mcp.run()
