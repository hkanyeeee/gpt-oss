# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WEB_LOADER_ENGINE=safe_web

WORKDIR /app

# 安装 Docker 客户端与运行期必要工具
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl ca-certificates \
       docker.io \
    && rm -rf /var/lib/apt/lists/*

# 安装 MCP 与其他依赖
RUN pip install --no-cache-dir \
      "mcp[cli]>=1.12.2" \
      "httpx>=0.24.0" \
      "beautifulsoup4>=4.12.0" \
      "sqlalchemy>=2.0.0" \
      "aiosqlite>=0.19.0" \
      "qdrant-client>=1.7.0" \
      "playwright>=1.40.0" \
      "docker>=6.0.0" \
      "openai-harmony"

# 复制 gpt_oss 包源码（python_server 需要依赖它）
COPY ../gpt_oss /app/gpt_oss

# 复制 MCP 服务器源码
COPY browser_server.py python_server.py start.sh ./
RUN chmod +x start.sh

# 数据目录（SQLite 默认写入 ./data/mcp_browser.db）
RUN mkdir -p /app/data
VOLUME ["/app/data"]

# 暴露两个服务端口
EXPOSE 8001 8002

# 默认启动 browser_server，可通过环境变量 MCP_SERVICE 切换
ENV MCP_SERVICE=browser
CMD ["./start.sh"]


