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
