#!/usr/bin/env python3
"""
Browser MCP Server Wrapper - 确保环境变量正确设置
"""

import os
import sys

def load_env_file(env_file_path: str = ".env"):
    """加载.env文件中的环境变量"""
    if not os.path.exists(env_file_path):
        print(f"警告: 未找到.env文件: {env_file_path}")
        return
    
    try:
        with open(env_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过空行和注释行
                if not line or line.startswith('#'):
                    continue
                
                # 解析键值对
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # 移除引号
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    # 只设置尚未设置的环境变量
                    if key not in os.environ:
                        os.environ[key] = value
                        print(f"从.env加载: {key}={value}")
        
        print("✅ .env文件加载完成")
    except Exception as e:
        print(f"警告: 加载.env文件失败: {e}")

# 首先尝试加载.env文件
load_env_file()

# 设置默认环境变量（如果未设置）
DEFAULT_ENV = {
    "SEARXNG_QUERY_URL": "http://192.168.31.125:8080/search",
    "EMBEDDING_SERVICE_URL": "http://192.168.31.125:7998/v1", 
    "QDRANT_HOST": "192.168.31.125",
    "QDRANT_PORT": "6333",
    "QDRANT_COLLECTION_NAME": "browser_mcp",
    "DATABASE_URL": "sqlite+aiosqlite:///./data/mcp_browser.db",
    "WEB_LOADER_ENGINE": "playwright",
    "CHUNK_SIZE": "600",
    "CHUNK_OVERLAP": "60",
    "RAG_TOP_K": "5"
}

# 设置默认环境变量（仅在环境变量不存在时）
for key, value in DEFAULT_ENV.items():
    if key not in os.environ:
        os.environ[key] = value
        print(f"使用默认值: {key}={value}")

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
