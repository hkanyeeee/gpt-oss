#!/usr/bin/env python3
"""
调试搜索功能的测试脚本
"""

import asyncio
import httpx
from browser_wrapper import DEFAULT_ENV
import os

# 设置环境变量
for key, value in DEFAULT_ENV.items():
    if key not in os.environ:
        os.environ[key] = value

async def test_searxng_search():
    """测试SearxNG搜索"""
    searxng_url = os.getenv("SEARXNG_QUERY_URL")
    print(f"测试SearxNG搜索: {searxng_url}")
    
    params = {
        "q": "重庆今天天气",
        "format": "json",
        "pageno": 1,
        "safesearch": "1",
        "time_range": "",
        "image_proxy": 0,
    }
    
    headers = {
        "User-Agent": "Open WebUI (https://github.com/open-webui/open-webui) RAG Bot",
        "Accept": "text/html",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
    }
    
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(searxng_url, params=params, headers=headers)
            print(f"状态码: {resp.status_code}")
            
            if resp.status_code == 200:
                payload = resp.json()
                results = payload.get("results", [])
                print(f"搜索结果数量: {len(results)}")
                
                for i, result in enumerate(results[:3]):  # 显示前3个结果
                    print(f"\n结果 {i+1}:")
                    print(f"  标题: {result.get('title', 'N/A')}")
                    print(f"  URL: {result.get('url', 'N/A')}")
                    print(f"  得分: {result.get('score', 'N/A')}")
                    print(f"  内容: {result.get('content', 'N/A')[:100]}...")
                
                return len(results) > 0
            else:
                print(f"搜索失败，状态码: {resp.status_code}")
                print(f"响应内容: {resp.text[:200]}...")
                return False
                
    except Exception as e:
        print(f"搜索异常: {e}")
        return False

async def test_embedding_service():
    """测试嵌入服务"""
    embedding_url = os.getenv("EMBEDDING_SERVICE_URL")
    print(f"\n测试嵌入服务: {embedding_url}")
    
    payload = {
        "input": ["重庆今天天气"],
        "model": "text-embedding-3-small"
    }
    
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{embedding_url.rstrip('/')}/embeddings",
                json=payload
            )
            print(f"状态码: {resp.status_code}")
            
            if resp.status_code == 200:
                data = resp.json()
                embeddings = [item["embedding"] for item in data["data"]]
                print(f"嵌入向量数量: {len(embeddings)}")
                print(f"向量维度: {len(embeddings[0]) if embeddings else 0}")
                return True
            else:
                print(f"嵌入服务失败，状态码: {resp.status_code}")
                print(f"响应内容: {resp.text[:200]}...")
                return False
                
    except Exception as e:
        print(f"嵌入服务异常: {e}")
        return False

async def test_web_fetch():
    """测试网页抓取"""
    print(f"\n测试网页抓取...")
    test_url = "https://www.baidu.com"
    
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    
    try:
        async with httpx.AsyncClient(trust_env=True, headers=headers) as client:
            resp = await client.get(test_url, timeout=15.0)
            print(f"状态码: {resp.status_code}")
            print(f"内容长度: {len(resp.text)}")
            return resp.status_code == 200
    except Exception as e:
        print(f"网页抓取异常: {e}")
        return False

async def main():
    """主测试函数"""
    print("开始调试搜索功能...")
    print("=" * 50)
    
    # 打印当前配置
    print("当前配置:")
    for key, value in DEFAULT_ENV.items():
        print(f"  {key}: {os.getenv(key, 'NOT SET')}")
    
    print("\n" + "=" * 50)
    
    # 测试各个组件
    searxng_ok = await test_searxng_search()
    embedding_ok = await test_embedding_service()
    web_fetch_ok = await test_web_fetch()
    
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print(f"SearxNG搜索: {'✅' if searxng_ok else '❌'}")
    print(f"嵌入服务: {'✅' if embedding_ok else '❌'}")
    print(f"网页抓取: {'✅' if web_fetch_ok else '❌'}")
    
    if not searxng_ok:
        print("\n❌ SearxNG搜索失败是主要问题！")
    elif not embedding_ok:
        print("\n❌ 嵌入服务失败会导致向量化失败！")
    elif not web_fetch_ok:
        print("\n❌ 网页抓取失败会导致内容获取失败！")
    else:
        print("\n✅ 所有组件测试正常，可能是其他问题")

if __name__ == "__main__":
    asyncio.run(main())
