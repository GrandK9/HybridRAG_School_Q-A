"""路径和模型名。密钥只从环境变量或 .env 读取。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

CORPUS_DIR = ROOT / "data" / "campus"
INDEX_DIR = ROOT / "data" / "index"
FAISS_PATH = INDEX_DIR / "vectors.faiss"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "campus_chunks")

# 与商城项目同一套硅基流动接口。Spring AI 的 base-url 不带 /v1，这里要带上。
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# BGE 检索模型的官方用法：查询加指令，文档不加。文档加了指令，相似度会整体抬高、区分度下降。
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# 每一路先多召回一些，融合和重排才有选择空间。最终给生成模型的是 TOP_N 条。
RECALL_K = 20
TOP_N = 5
RRF_K = 60

STRATEGIES = ("vector", "bm25", "hybrid", "hybrid_rerank")
STRATEGY_LABELS = {
    "vector": "纯向量",
    "bm25": "纯 BM25",
    "hybrid": "混合（RRF）",
    "hybrid_rerank": "混合 + 重排",
}
