#!/usr/bin/env python3
"""
测试Browser MCP Server的各个依赖服务
"""

import asyncio
import httpx
from typing import Dict, Tuple

# 服务配置
SERVICES = {
    "SearxNG": "http://192.168.31.125:8080/search?q=test&format=json",
    "Embedding Service": "http://192.168.31.125:8001/v1/models",  
    "Qdrant": "http://192.168.31.125:6333/collections"
}

async def test_service(name: str, url: str) -> Tuple[str, bool, str]:
    """测试单个服务"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            if response.status_code == 200:
                return name, True, f"✅ 服务正常 (状态码: {response.status_code})"
            else:
                return name, False, f"❌ 服务异常 (状态码: {response.status_code})"
    except httpx.ConnectError:
        return name, False, "❌ 连接失败 - 服务可能未启动"
    except httpx.TimeoutException:
        return name, False, "❌ 连接超时"
    except Exception as e:
        return name, False, f"❌ 错误: {str(e)}"

async def main():
    """主测试函数"""
    print("正在测试Browser MCP Server的依赖服务...")
    print("=" * 50)
    
    tasks = [test_service(name, url) for name, url in SERVICES.items()]
    results = await asyncio.gather(*tasks)
    
    all_ok = True
    for name, status, message in results:
        print(f"{name:20}: {message}")
        if not status:
            all_ok = False
    
    print("=" * 50)
    if all_ok:
        print("✅ 所有服务正常运行")
    else:
        print("❌ 有服务无法访问，请检查这些服务是否启动：")
        print("1. SearxNG: docker run -d -p 8080:8080 searxng/searxng")
        print("2. 嵌入服务: 需要运行兼容OpenAI API的嵌入服务")
        print("3. Qdrant: docker run -d -p 6333:6333 qdrant/qdrant")

if __name__ == "__main__":
    asyncio.run(main())



