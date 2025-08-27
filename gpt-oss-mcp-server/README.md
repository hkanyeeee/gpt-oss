# Browser MCP Server

独立的网页搜索和检索MCP服务器，提供完整的网页搜索、内容抓取、向量化存储和智能检索功能。

## 功能特性

- **search**: 使用SearxNG搜索网页，抓取内容，生成嵌入向量并存储到向量数据库
- **find**: 从向量数据库中检索相关内容

## 环境变量配置

### 必需配置
- `SEARXNG_QUERY_URL`: SearxNG搜索服务URL
- `EMBEDDING_SERVICE_URL`: 嵌入向量服务URL (支持OpenAI API格式)

### 可选配置
- `QDRANT_HOST`: Qdrant向量数据库主机 (默认: localhost)
- `QDRANT_PORT`: Qdrant向量数据库端口 (默认: 6333)
- `QDRANT_API_KEY`: Qdrant API密钥 (可选)
- `QDRANT_COLLECTION_NAME`: Qdrant集合名称 (默认: browser_mcp)
- `DATABASE_URL`: SQLite数据库URL (默认: sqlite+aiosqlite:///./data/mcp_browser.db)
- `PROXY_URL`: 代理服务器URL (可选)
- `WEB_LOADER_ENGINE`: 网页加载引擎 (safe_web/playwright, 默认: safe_web)
- `CHUNK_SIZE`: 文本分块大小 (默认: 600)
- `CHUNK_OVERLAP`: 分块重叠大小 (默认: 60)
- `RAG_TOP_K`: 检索返回数量 (默认: 12)
- `EMBEDDING_BATCH_SIZE`: 嵌入批处理大小 (默认: 4)
- `WEB_SEARCH_TIMEOUT`: 网页抓取超时时间 (默认: 15.0)
- `PLAYWRIGHT_TIMEOUT`: Playwright超时时间 (默认: 15.0)

## 安装和启动

### 1. 安装依赖

```bash
cd gpt-oss-mcp-server
pip install -e .
```

### 2. 配置环境变量

创建 `.env` 文件或设置环境变量：

```bash
export SEARXNG_QUERY_URL="http://localhost:8080/search"
export EMBEDDING_SERVICE_URL="http://localhost:1234/v1"
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"
```

### 3. 启动服务

```bash
mcp run -t sse browser_server.py:mcp
```

服务将在端口8001上启动。

### 4. 测试服务

使用MCP Inspector测试工具：
- 设置SSE到 `http://localhost:8001/sse`

## 工具说明

### search工具

搜索网页并处理结果，包括：
1. 使用SearxNG搜索（固定返回4个结果）
2. 并发抓取网页内容
3. 文本分块处理
4. 生成嵌入向量
5. 存储到向量数据库

**参数:**
- `query`: 搜索查询字符串

**示例:**
```
search("人工智能最新发展")
```

### find工具

从向量数据库检索相关内容：
1. 生成查询向量
2. 向量相似度搜索
3. 返回top 5结果

**参数:**
- `pattern`: 检索模式/查询

**示例:**
```
find("机器学习算法")
```

## 依赖服务

### SearxNG搜索引擎
```bash
docker run -d -p 8080:8080 searxng/searxng
```

### Qdrant向量数据库
```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

### 嵌入服务
支持OpenAI API格式的嵌入服务，如LM Studio、Ollama等。

## 架构说明

1. **数据存储**: 使用SQLite存储元数据，Qdrant存储向量
2. **文本处理**: 智能分块，保持语义完整性
3. **向量检索**: 基于余弦相似度的向量搜索

## 故障排除

1. **导入错误**: 确保所有依赖都已正确安装
2. **连接失败**: 检查各服务的URL配置和网络连通性
3. **搜索无结果**: 验证SearxNG服务是否正常运行
4. **向量存储失败**: 检查Qdrant服务状态和权限
