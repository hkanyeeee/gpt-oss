"""
Browser MCP Server - 独立的网页搜索和检索服务

环境变量配置：
- SEARXNG_QUERY_URL: SearxNG搜索服务URL (必需)
- EMBEDDING_SERVICE_URL: 嵌入向量服务URL (必需)
- QDRANT_HOST: Qdrant向量数据库主机 (默认: localhost)
- QDRANT_PORT: Qdrant向量数据库端口 (默认: 6333)
- QDRANT_API_KEY: Qdrant API密钥 (可选)
- QDRANT_COLLECTION_NAME: Qdrant集合名称 (默认: browser_mcp)
- DATABASE_URL: SQLite数据库URL (默认: sqlite+aiosqlite:///./data/mcp_browser.db)
- PROXY_URL: 代理服务器URL (可选)
- WEB_LOADER_ENGINE: 网页加载引擎 (safe_web/playwright, 默认: safe_web)
- CHUNK_SIZE: 文本分块大小 (默认: 600)
- CHUNK_OVERLAP: 分块重叠大小 (默认: 60)
- RAG_TOP_K: 检索返回数量 (默认: 12)
- EMBEDDING_BATCH_SIZE: 嵌入批处理大小 (默认: 4)
- WEB_SEARCH_TIMEOUT: 网页抓取超时时间 (默认: 15.0)
- PLAYWRIGHT_TIMEOUT: Playwright超时时间 (默认: 15.0)
"""

import os
import re
import uuid
import asyncio
import json
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from urllib.parse import urlparse
from datetime import datetime

import httpx
from bs4 import BeautifulSoup
from mcp.server.fastmcp import Context, FastMCP
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, create_engine, select, text
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import relationship, sessionmaker
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

# 配置
def get_env_config(key: str, default: str = None) -> str:
    """从环境变量获取配置值"""
    return os.getenv(key, default)

# 必需的环境变量
SEARXNG_QUERY_URL = get_env_config("SEARXNG_QUERY_URL")
EMBEDDING_SERVICE_URL = get_env_config("EMBEDDING_SERVICE_URL")

if not SEARXNG_QUERY_URL:
    raise EnvironmentError("SEARXNG_QUERY_URL environment variable is required")
if not EMBEDDING_SERVICE_URL:
    raise EnvironmentError("EMBEDDING_SERVICE_URL environment variable is required")

# 可选环境变量
QDRANT_HOST = get_env_config("QDRANT_HOST", "localhost")
QDRANT_PORT = get_env_config("QDRANT_PORT", "6333")
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
QDRANT_API_KEY = get_env_config("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = get_env_config("QDRANT_COLLECTION_NAME", "browser_mcp")
DATABASE_URL = get_env_config("DATABASE_URL", "sqlite+aiosqlite:///./data/mcp_browser.db")
PROXY_URL = get_env_config("PROXY_URL")
WEB_LOADER_ENGINE = get_env_config("WEB_LOADER_ENGINE", "safe_web")
CHUNK_SIZE = int(get_env_config("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(get_env_config("CHUNK_OVERLAP", "60"))
RAG_TOP_K = int(get_env_config("RAG_TOP_K", "12"))
EMBEDDING_BATCH_SIZE = int(get_env_config("EMBEDDING_BATCH_SIZE", "4"))
WEB_SEARCH_TIMEOUT = float(get_env_config("WEB_SEARCH_TIMEOUT", "15.0"))
PLAYWRIGHT_TIMEOUT = float(get_env_config("PLAYWRIGHT_TIMEOUT", "15.0"))

print("=== Browser MCP Server Configuration ===")
print(f"SEARXNG_QUERY_URL: {SEARXNG_QUERY_URL}")
print(f"EMBEDDING_SERVICE_URL: {EMBEDDING_SERVICE_URL}")
print(f"QDRANT_URL: {QDRANT_URL}")
print(f"QDRANT_COLLECTION_NAME: {QDRANT_COLLECTION_NAME}")
print(f"DATABASE_URL: {DATABASE_URL}")
print(f"WEB_LOADER_ENGINE: {WEB_LOADER_ENGINE}")
print(f"CHUNK_SIZE: {CHUNK_SIZE}")
print(f"RAG_TOP_K: {RAG_TOP_K}")
print("==========================================")

# 数据库模型
Base = declarative_base()

class Source(Base):
    __tablename__ = "sources"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    url = Column(String, nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Chunk(Base):
    __tablename__ = "chunks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    source_id = Column(Integer, nullable=False)
    chunk_id = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class BrowserMCPContext:
    """Browser MCP 上下文管理器"""
    
    def __init__(self):
        self.db_engine = None
        self.db_session_factory = None
        self.qdrant_client = None
        self.session_id = str(uuid.uuid4())
        
    async def initialize(self):
        """初始化数据库连接"""
        try:
            # 初始化SQLite数据库
            self.db_engine = create_async_engine(DATABASE_URL, echo=False)
            self.db_session_factory = async_sessionmaker(
                self.db_engine, class_=AsyncSession, expire_on_commit=False
            )
            
            # 创建表
            async with self.db_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            # 初始化Qdrant客户端
            try:
                self.qdrant_client = QdrantClient(
                    url=QDRANT_URL, 
                    api_key=QDRANT_API_KEY, 
                    timeout=300
                )
                print("Qdrant client initialized successfully")
            except Exception as e:
                print(f"Failed to initialize Qdrant client: {e}")
                self.qdrant_client = None
            
            print("Browser MCP Context initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Browser MCP Context: {e}")
            raise
    
    async def cleanup(self):
        """清理资源"""
        if self.db_engine:
            await self.db_engine.dispose()


# 全局上下文实例
browser_context = BrowserMCPContext()


@asynccontextmanager
async def app_lifespan(_server: FastMCP) -> AsyncIterator[BrowserMCPContext]:
    """应用生命周期管理"""
    await browser_context.initialize()
    try:
        yield browser_context
    finally:
        await browser_context.cleanup()


# 创建 MCP 服务器
mcp = FastMCP(
    name="browser",
    instructions=r"""
Browser tool for web search and retrieval.
- search: Search the web using SearxNG and process results (fetch, embed, store in vector DB)
- find: Retrieve relevant content from vector database using hybrid search
""".strip(),
    lifespan=app_lifespan,
    port=8001,
)


# 工具函数
async def search_searxng(query: str, count: int = 4) -> List[Dict[str, str]]:
    """使用 SearxNG 搜索"""
    params = {
        "q": query,
        "format": "json",
        "pageno": 1,
        "safesearch": "1",
        "time_range": "",
        "image_proxy": 0,
    }
    
    headers = {
        "User-Agent": "Open WebUI (https://github.com/open-webui/open-webui) RAG Bot",
        "Accept": "application/json, text/html",  # 改为支持JSON
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
    }
    
    print(f"[SearxNG] 搜索查询: {query}")
    
    try:
        proxy = PROXY_URL if PROXY_URL else None
        client_kwargs = {"timeout": 60}
        if proxy:
            client_kwargs["proxy"] = proxy
        async with httpx.AsyncClient(**client_kwargs) as client:
            resp = await client.get(SEARXNG_QUERY_URL, params=params, headers=headers)
            resp.raise_for_status()
            
            payload = resp.json()
            results = payload.get("results", [])
            print(f"[SearxNG] 获取到 {len(results)} 个原始结果")
            
            # 按 score 降序排列并限制数量
            results_sorted = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
            items = []
            for r in results_sorted[:count]:
                title = r.get("title") or r.get("name") or "Untitled"
                url = r.get("url") or r.get("link")
                snippet = r.get("content") or r.get("snippet") or ""
                if url:
                    items.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet
                    })
            
            print(f"[SearxNG] 返回 {len(items)} 个有效结果")
            return items
    except Exception as e:
        print(f"[SearxNG] 搜索失败: {type(e).__name__}: {e}")
        import traceback
        print(f"[SearxNG] 错误堆栈: {traceback.format_exc()}")
        return []


async def fetch_html(url: str, timeout: float = 15.0) -> str:
    """使用 httpx 获取网页HTML"""
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    
    try:
        proxy = PROXY_URL if PROXY_URL else None
        client_kwargs = {"trust_env": True, "headers": headers}
        if proxy:
            client_kwargs["proxy"] = proxy
        async with httpx.AsyncClient(**client_kwargs) as client:
            resp = await client.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.text
    except Exception as e:
        print(f"Failed to fetch HTML from {url}: {e}")
        return ""


def extract_text_from_html(html: str, selector: str = "article") -> str:
    """从HTML中提取文本内容"""
    if not html:
        return ""
    
    soup = BeautifulSoup(html, "html.parser")
    
    # 移除脚本和样式标签
    for script in soup(["script", "style"]):
        script.decompose()
    
    def get_text_from_node(node):
        """从节点获取文本"""
        if not node:
            return ""
        text = node.get_text(separator="\n", strip=True)
        # 清理多余的空白
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()
    
    MIN_LEN = 200
    
    # 1. 尝试指定的选择器
    if selector:
        target = soup.select_one(selector)
        if target:
            text = get_text_from_node(target)
            if len(text) >= MIN_LEN:
                return text
    
    # 2. 尝试常见的内容标签
    for tag in ["article", "main", ".content", ".post", ".entry"]:
        element = soup.select_one(tag)
        if element:
            text = get_text_from_node(element)
            if len(text) >= MIN_LEN:
                return text
    
    # 3. 尝试ID或class包含content关键词的div
    for div in soup.find_all(["div", "section"]):
        id_class = " ".join(filter(None, [div.get("id", ""), *div.get("class", [])]))
        if re.search(r"\b(content|article|post|main|body|entry)\b", id_class, re.I):
            text = get_text_from_node(div)
            if len(text) >= MIN_LEN:
                return text
    
    # 4. 获取所有段落文本
    paragraphs = [get_text_from_node(p) for p in soup.find_all("p")]
    if paragraphs:
        text = "\n\n".join(p for p in paragraphs if p)
        if text:
            return text
    
    # 5. 最后兜底：获取body的所有文本
    body = soup.find("body")
    if body:
        return get_text_from_node(body)
    
    return get_text_from_node(soup)


async def fetch_web_content(url: str) -> Optional[str]:
    """抓取网页内容"""
    try:
        html = await fetch_html(url, timeout=WEB_SEARCH_TIMEOUT)
        if html:
            content = extract_text_from_html(html, selector="article")
            return content if content else None
        return None
    except Exception as e:
        print(f"抓取网页内容失败 {url}: {e}")
        return None


def chunk_text(text: str, chunk_size: int = 600, chunk_overlap: int = 60) -> List[str]:
    """将文本分块"""
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        
        # 如果不是最后一块，尝试在句号或换行符处分割
        if end < text_len:
            # 寻找最近的句号或换行符
            for i in range(end, max(start + chunk_size // 2, end - 100), -1):
                if text[i] in '.。\n':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - chunk_overlap if end < text_len else end
    
    return chunks


async def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """生成文本嵌入向量"""
    if not texts:
        return []
    
    payload = {
        "input": texts,
        "model": model
    }
    
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{EMBEDDING_SERVICE_URL.rstrip('/')}/embeddings",
                json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]
    except Exception as e:
        print(f"嵌入向量生成失败: {e}")
        return []


async def ensure_qdrant_collection(vector_size: int):
    """确保Qdrant集合存在"""
    if not browser_context.qdrant_client:
        return
    
    try:
        browser_context.qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
    except Exception:
        print(f"创建Qdrant集合: {QDRANT_COLLECTION_NAME}")
        browser_context.qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


async def add_embeddings_to_qdrant(source_id: int, chunks: List[Chunk], embeddings: List[List[float]]):
    """将嵌入向量添加到Qdrant"""
    if not browser_context.qdrant_client or not chunks or not embeddings:
        return
    
    vector_size = len(embeddings[0])
    await ensure_qdrant_collection(vector_size)
    
    points = [
        models.PointStruct(
            id=chunk.id,
            vector=embedding,
            payload={
                "session_id": chunk.session_id,
                "source_id": chunk.source_id,
                "content": chunk.content,
                "chunk_id": chunk.chunk_id
            }
        )
        for chunk, embedding in zip(chunks, embeddings) if chunk.id is not None
    ]
    
    if points:
        browser_context.qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=points,
            wait=True
        )


async def query_qdrant(query_embedding: List[float], top_k: int = 12, session_id: str = None) -> List[Tuple[Dict, float]]:
    """查询Qdrant向量数据库"""
    if not browser_context.qdrant_client:
        return []
    
    try:
        filter_conditions = None
        if session_id:
            filter_conditions = models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=session_id)
                    )
                ]
            )
        
        search_result = browser_context.qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=filter_conditions,
            limit=top_k,
            with_payload=True
        )
        
        results = []
        for scored_point in search_result:
            payload = scored_point.payload
            score = scored_point.score
            results.append((payload, score))
        
        return results
    except Exception as e:
        print(f"Qdrant查询失败: {e}")
        return []



# MCP工具实现
@mcp.tool(
    name="search",
    title="Search web and process results",
    description="Search the web using SearxNG, fetch content, generate embeddings, and store in vector database. Returns 4 search results.",
)
async def search(ctx: Context, query: str) -> str:
    """搜索网页并处理结果"""
    try:
        print(f"[Search] 开始搜索: {query}")
        
        # 1. SearxNG搜索
        search_results = await search_searxng(query, count=4)
        if not search_results:
            return f"搜索查询 '{query}' 没有找到任何结果。"
        
        print(f"[Search] 找到 {len(search_results)} 个搜索结果")
        
        # 2. 并发抓取网页内容
        print(f"[Search] 开始抓取网页内容...")
        fetch_tasks = [fetch_web_content(result["url"]) for result in search_results]
        contents = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        
        # 3. 过滤有效内容
        valid_documents = []
        for i, (result, content) in enumerate(zip(search_results, contents)):
            if isinstance(content, str) and content.strip() and len(content) > 100:
                valid_documents.append({
                    "title": result["title"],
                    "url": result["url"],
                    "content": content.strip(),
                    "snippet": result.get("snippet", "")
                })
        
        if not valid_documents:
            return f"搜索到 {len(search_results)} 个结果，但无法成功抓取任何网页内容。"
        
        print(f"[Search] 成功抓取 {len(valid_documents)} 个文档")
        
        # 4. 存储到数据库并生成嵌入
        session_id = browser_context.session_id
        async with browser_context.db_session_factory() as db:
            all_chunks = []
            source_ids = []
            
            for doc in valid_documents:
                # 创建Source记录
                source = Source(
                    session_id=session_id,
                    url=doc["url"],
                    title=doc["title"],
                    content=doc["content"][:1000] + "..." if len(doc["content"]) > 1000 else doc["content"]
                )
                db.add(source)
                await db.flush()
                source_ids.append(source.id)
                
                # 分块
                chunks = chunk_text(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)
                for i, chunk_content in enumerate(chunks):
                    chunk = Chunk(
                        session_id=session_id,
                        source_id=source.id,
                        chunk_id=i,
                        content=chunk_content
                    )
                    db.add(chunk)
                    all_chunks.append(chunk)
            
            await db.commit()
            
            # 刷新获取ID
            for chunk in all_chunks:
                await db.refresh(chunk)
        
        # 5. 生成嵌入向量并存储到Qdrant
        if all_chunks:
            print(f"[Search] 开始生成 {len(all_chunks)} 个块的嵌入向量...")
            texts = [chunk.content for chunk in all_chunks]
            embeddings = await embed_texts(texts)
            
            if embeddings:
                await add_embeddings_to_qdrant(0, all_chunks, embeddings)
                print("[Search] 嵌入向量存储完成")
        
        # 6. 格式化返回结果
        result_lines = [f"搜索查询 '{query}' 完成，共处理了 {len(valid_documents)} 个网页：\n"]
        
        for i, doc in enumerate(valid_documents, 1):
            result_lines.append(f"{i}. {doc['title']}")
            result_lines.append(f"   URL: {doc['url']}")
            if doc.get('snippet'):
                result_lines.append(f"   摘要: {doc['snippet'][:200]}...")
            result_lines.append("")
        
        result_lines.append("所有内容已成功索引到向量数据库，可使用 find 工具进行检索。")
        
        return "\n".join(result_lines)
        
    except Exception as e:
        error_msg = f"搜索过程中发生错误: {str(e)}"
        print(f"[Search] {error_msg}")
        return error_msg


@mcp.tool(
    name="find",
    title="Find relevant content",
    description="Retrieve relevant content from vector database. Returns top 5 results.",
)
async def find_content(ctx: Context, pattern: str) -> str:
    """从向量数据库中检索相关内容"""
    try:
        print(f"[Find] 开始检索: {pattern}")
        
        # 1. 生成查询向量
        query_embeddings = await embed_texts([pattern])
        if not query_embeddings:
            return f"无法生成查询向量。"
        
        query_embedding = query_embeddings[0]
        
        # 2. 向量检索
        session_id = browser_context.session_id
        results = await query_qdrant(query_embedding, top_k=RAG_TOP_K, session_id=session_id)
        
        if not results:
            return f"未找到与 '{pattern}' 相关的内容。请先使用 search 工具搜索相关信息。"
        
        # 取前5个结果
        top_results = results[:5]
        
        # 4. 获取源信息
        async with browser_context.db_session_factory() as db:
            result_lines = [f"找到 {len(top_results)} 个与 '{pattern}' 相关的内容片段：\n"]
            
            for i, (payload, score) in enumerate(top_results, 1):
                source_id = payload.get("source_id")
                content = payload.get("content", "")
                
                # 查询源信息
                source_query = select(Source).where(Source.id == source_id)
                result = await db.execute(source_query)
                source = result.scalar_one_or_none()
                
                result_lines.append(f"{i}. 相关度: {score:.3f}")
                if source:
                    result_lines.append(f"   来源: {source.title}")
                    result_lines.append(f"   URL: {source.url}")
                else:
                    result_lines.append(f"   来源: Unknown")
                result_lines.append(f"   内容: {content[:300]}...")
                result_lines.append("")
        
        return "\n".join(result_lines)
        
    except Exception as e:
        error_msg = f"检索过程中发生错误: {str(e)}"
        print(f"[Find] {error_msg}")
        return error_msg


if __name__ == "__main__":
    print("Starting Browser MCP Server...")
    mcp.run()