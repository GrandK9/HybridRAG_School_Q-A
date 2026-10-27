"""索引、四路检索和带引用的回答。

LangChain 负责切片、远程向量接口和对话模型。
BM25 走 Elasticsearch，向量结果存在 FAISS，融合在本文件用 RRF 完成。
ES 自带的 rank.rrf 在 Basic 许可下不可用，所以融合放在应用层。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hybridrag.config import (
    CORPUS_DIR,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    ES_INDEX,
    ES_URL,
    FAISS_PATH,
    INDEX_DIR,
    QUERY_INSTRUCTION,
    RECALL_K,
    RERANKER_MODEL,
    RRF_K,
    STRATEGIES,
    TOP_N,
    VECTOR_ID_WORKER,
)
from hybridrag.fusion import reciprocal_rank_fusion
from hybridrag.snowflake import Snowflake

SYSTEM_PROMPT = """你是示范大学的校园制度问答助手。
只能根据给定资料回答，不要使用资料以外的数字、日期或部门名称。
资料不够时，直接回答：现有校园知识库里没有足够依据，建议咨询相关部门。
使用简体中文。引用资料时在句末标出编号，例如 [1]。"""


@dataclass
class Chunk:
    vector_id: int
    chunk_id: str
    doc_id: str
    title: str
    source: str
    content: str

    def to_dict(self) -> dict:
        return {
            "vector_id": self.vector_id,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "source": self.source,
            "content": self.content,
        }

    @classmethod
    def from_source(cls, source: dict, vector_id: int) -> "Chunk":
        # 以 ES 的 _id 为准。JSON 数字超过 2^53 时有的客户端会丢掉低位，字符串 _id 不会。
        return cls(
            vector_id=vector_id,
            chunk_id=source["chunk_id"],
            doc_id=source["doc_id"],
            title=source["title"],
            source=source["source"],
            content=source["content"],
        )


@dataclass
class Hit:
    vector_id: int
    chunk_id: str
    title: str
    source: str
    content: str
    score: float
    rank: int


@dataclass
class Answer:
    text: str
    strategy: str
    hits: list[Hit]
    sparse_backend: str
    generated: bool


_embeddings: OpenAIEmbeddings | None = None


def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        if not EMBEDDING_API_KEY:
            raise RuntimeError("请在 .env 里填写 EMBEDDING_API_KEY")
        # 不传 dimensions：bge 系列不接受这个参数，输出维度由模型自己决定。
        # 关闭 tiktoken：那是 OpenAI 模型的分词器，用来截断中文会算错长度。
        _embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=EMBEDDING_BASE_URL,
            api_key=EMBEDDING_API_KEY,
            check_embedding_ctx_length=False,
            tiktoken_enabled=False,
            chunk_size=16,
        )
    return _embeddings


def rerank_scores(query: str, documents: list[str]) -> list[float]:
    """调用远程重排。返回值与 documents 对齐，缺省分数为 0。"""
    if not EMBEDDING_API_KEY:
        raise RuntimeError("请在 .env 里填写 EMBEDDING_API_KEY")
    url = EMBEDDING_BASE_URL.rstrip("/") + "/rerank"
    payload = json.dumps(
        {
            "model": RERANKER_MODEL,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "return_documents": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {EMBEDDING_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"重排接口失败：HTTP {exc.code} {detail}") from exc
    scores = [0.0] * len(documents)
    for item in body.get("results", []):
        index = int(item["index"])
        if 0 <= index < len(scores):
            scores[index] = float(item["relevance_score"])
    return scores


def load_corpus() -> list[Document]:
    if not CORPUS_DIR.exists():
        raise FileNotFoundError(f"找不到语料目录：{CORPUS_DIR}")
    documents: list[Document] = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        title = path.stem
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                title = stripped.lstrip("#").strip()
                break
        documents.append(
            Document(
                page_content=text,
                metadata={"doc_id": path.stem, "title": title, "source": path.name},
            )
        )
    if not documents:
        raise FileNotFoundError(f"{CORPUS_DIR} 里没有 markdown 文档")
    return documents


def split_documents(documents: list[Document], ids: Snowflake | None = None) -> list[Chunk]:
    # 中文制度文本按标题和句号切。块太长时，一个向量会混进多条规定，检索会漂。
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=280,
        chunk_overlap=40,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", " "],
    )
    generator = ids or Snowflake(worker_id=VECTOR_ID_WORKER)
    chunks: list[Chunk] = []
    for document in documents:
        for index, piece in enumerate(splitter.split_documents([document])):
            content = piece.page_content.strip()
            if len(content) < 20:
                continue
            meta = document.metadata
            chunks.append(
                Chunk(
                    vector_id=generator.next_id(),
                    chunk_id=f"{meta['doc_id']}-{index:03d}",
                    doc_id=meta["doc_id"],
                    title=meta["title"],
                    source=meta["source"],
                    content=content,
                )
            )
    return chunks


def embed_documents(texts: list[str]) -> np.ndarray:
    vectors = get_embeddings().embed_documents(texts)
    array = np.ascontiguousarray(np.array(vectors, dtype="float32"))
    import faiss

    # 归一化后内积等于余弦相似度，FAISS 用 IndexFlatIP 即可。
    faiss.normalize_L2(array)
    return array


def embed_query(text: str) -> np.ndarray:
    vector = get_embeddings().embed_query(QUERY_INSTRUCTION + text)
    array = np.ascontiguousarray(np.array([vector], dtype="float32"))
    import faiss

    faiss.normalize_L2(array)
    return array


def connect_es() -> Elasticsearch | None:
    client = Elasticsearch(ES_URL, request_timeout=3, retry_on_timeout=False)
    try:
        if not client.ping():
            return None
    except Exception:
        return None
    return client


def ik_installed(client: Elasticsearch) -> bool:
    try:
        rows = client.cat.plugins(format="json")
    except Exception:
        return False
    for row in rows:
        blob = f"{row.get('component', '')} {row.get('name', '')}".lower()
        if "ik" in blob:
            return True
    return False


def sync_elasticsearch(client: Elasticsearch, chunks: list[Chunk]) -> str:
    """按 vector_id 重建索引。正文只放在这里，向量检索命中后再按这个编号取回。"""
    use_ik = ik_installed(client)
    text_field = {"type": "text"}
    if use_ik:
        text_field = {
            "type": "text",
            "analyzer": "ik_max_word",
            "search_analyzer": "ik_smart",
        }
    if client.indices.exists(index=ES_INDEX):
        client.indices.delete(index=ES_INDEX)
    client.indices.create(
        index=ES_INDEX,
        settings={"number_of_shards": 1, "number_of_replicas": 0},
        mappings={
            "properties": {
                "vector_id": {"type": "long"},
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "title": text_field,
                "source": {"type": "keyword"},
                "content": text_field,
            }
        },
    )
    actions = [
        {"_index": ES_INDEX, "_id": str(chunk.vector_id), "_source": chunk.to_dict()}
        for chunk in chunks
    ]
    success, errors = bulk(client, actions, refresh="wait_for")
    if errors or success != len(chunks):
        raise RuntimeError(f"写入 Elasticsearch 失败：success={success}, errors={errors}")
    return "ik" if use_ik else "standard"


def save_vector_index(chunks: list[Chunk]) -> None:
    import faiss

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectors = embed_documents([chunk.content for chunk in chunks])
    # IndexIDMap2 把调用方给出的 int64 和向量存在同一个文件里，查询直接返回该编号。
    base = faiss.IndexFlatIP(vectors.shape[1])
    index = faiss.IndexIDMap2(base)
    vector_ids = np.ascontiguousarray(
        np.array([chunk.vector_id for chunk in chunks], dtype=np.int64)
    )
    index.add_with_ids(vectors, vector_ids)
    faiss.write_index(index, str(FAISS_PATH))


def ingest(preload_reranker: bool = True) -> dict:
    documents = load_corpus()
    chunks = split_documents(documents)
    client = connect_es()
    if client is None:
        raise RuntimeError(f"Elasticsearch 未连接（{ES_URL}），正文无法入库")
    print(f"文档 {len(documents)} 篇，切片 {len(chunks)} 条，远程向量化（{EMBEDDING_MODEL}）")
    analyzer = sync_elasticsearch(client, chunks)
    save_vector_index(chunks)
    print(f"已写入 Elasticsearch 索引 {ES_INDEX}，分词：{analyzer}")
    if preload_reranker:
        print(f"重排接口：{RERANKER_MODEL}")
    return {"documents": len(documents), "chunks": len(chunks), "analyzer": analyzer}


class CampusQA:
    def __init__(self) -> None:
        if not FAISS_PATH.exists():
            raise FileNotFoundError("还没有向量索引。请先运行 python -m hybridrag.ingest")
        import faiss

        self.faiss_index = faiss.read_index(str(FAISS_PATH))
        if not isinstance(self.faiss_index, faiss.IndexIDMap2):
            raise RuntimeError("向量索引没有身份编号，请重新执行 ingest")
        self.es = connect_es()
        if self.es is None:
            raise RuntimeError(f"Elasticsearch 未连接：{ES_URL}")
        if not self.es.indices.exists(index=ES_INDEX):
            raise RuntimeError(f"Elasticsearch 中没有索引 {ES_INDEX}，请先执行 ingest")
        count = self.es.count(index=ES_INDEX)["count"]
        if count != self.faiss_index.ntotal:
            raise RuntimeError("ES 文档数和 FAISS 向量数不一致，请重新执行 ingest")
        kind = "IK" if ik_installed(self.es) else "标准分词"
        self.sparse_backend = f"Elasticsearch BM25（{kind}）"

    def search(self, query: str, strategy: str, top_n: int = TOP_N) -> list[Hit]:
        query = query.strip()
        if not query:
            return []
        if strategy not in STRATEGIES:
            raise ValueError(f"未知策略：{strategy}")
        if strategy == "vector":
            return self._search_vector(query, top_n)
        if strategy == "bm25":
            return self._search_bm25(query, top_n)
        fused = self._fuse(query, RECALL_K if strategy == "hybrid_rerank" else max(top_n, RECALL_K))
        if strategy == "hybrid":
            return fused[:top_n]
        return self._rerank(query, fused[:RECALL_K], top_n)

    def compare(self, query: str, top_n: int = 3) -> dict[str, list[Hit]]:
        return {strategy: self.search(query, strategy, top_n=top_n) for strategy in STRATEGIES}

    def answer(self, query: str, strategy: str) -> Answer:
        hits = self.search(query, strategy)
        if not hits:
            return Answer(
                text="没有检索到相关制度条款。",
                strategy=strategy,
                hits=[],
                sparse_backend=self.sparse_backend,
                generated=False,
            )
        if not DEEPSEEK_API_KEY:
            excerpt = hits[0].content
            return Answer(
                text=(
                    "未配置 DEEPSEEK_API_KEY，下面是检索到的制度原文，还没有经过大模型生成。\n\n"
                    f"{excerpt}"
                ),
                strategy=strategy,
                hits=hits,
                sparse_backend=self.sparse_backend,
                generated=False,
            )
        text = self._generate(query, hits)
        return Answer(
            text=text,
            strategy=strategy,
            hits=hits,
            sparse_backend=self.sparse_backend,
            generated=True,
        )

    def _search_vector(self, query: str, top_k: int) -> list[Hit]:
        scores, vector_ids = self.faiss_index.search(embed_query(query), top_k)
        ordered = [
            (int(vector_id), float(score))
            for score, vector_id in zip(scores[0], vector_ids[0])
            if int(vector_id) >= 0
        ]
        chunks = self._load_chunks([vector_id for vector_id, _ in ordered])
        hits: list[Hit] = []
        for rank, (vector_id, score) in enumerate(ordered, start=1):
            chunk = chunks.get(vector_id)
            if chunk is None:
                continue
            hits.append(self._hit(chunk, score, rank))
        return hits

    def _search_bm25(self, query: str, top_k: int) -> list[Hit]:
        return self._search_es(query, top_k)

    def _search_es(self, query: str, top_k: int) -> list[Hit]:
        response = self.es.search(
            index=ES_INDEX,
            size=top_k,
            query={
                "multi_match": {
                    "query": query,
                    "fields": ["title^2", "content"],
                    "type": "best_fields",
                }
            },
        )
        hits: list[Hit] = []
        for rank, item in enumerate(response["hits"]["hits"], start=1):
            chunk = self._chunk_from_hit(item)
            hits.append(self._hit(chunk, float(item["_score"]), rank))
        return hits

    def _load_chunks(self, vector_ids: list[int]) -> dict[int, Chunk]:
        if not vector_ids:
            return {}
        response = self.es.mget(index=ES_INDEX, ids=[str(vector_id) for vector_id in vector_ids])
        loaded: dict[int, Chunk] = {}
        for doc in response["docs"]:
            if not doc.get("found"):
                continue
            vector_id = int(doc["_id"])
            loaded[vector_id] = Chunk.from_source(doc["_source"], vector_id)
        return loaded

    @staticmethod
    def _chunk_from_hit(item: dict) -> Chunk:
        return Chunk.from_source(item["_source"], int(item["_id"]))

    def _fuse(self, query: str, recall_k: int) -> list[Hit]:
        vector_hits = self._search_vector(query, recall_k)
        bm25_hits = self._search_bm25(query, recall_k)
        fused = reciprocal_rank_fusion(
            [
                [str(hit.vector_id) for hit in vector_hits],
                [str(hit.vector_id) for hit in bm25_hits],
            ],
            k=RRF_K,
        )
        lookup = {str(hit.vector_id): hit for hit in vector_hits + bm25_hits}
        hits: list[Hit] = []
        for rank, (vector_id, score) in enumerate(fused, start=1):
            found = lookup.get(vector_id)
            if found is None:
                continue
            hits.append(self._copy_hit(found, score, rank))
        return hits

    def _rerank(self, query: str, candidates: list[Hit], top_n: int) -> list[Hit]:
        if not candidates:
            return []
        scores = rerank_scores(query, [hit.content for hit in candidates])
        order = np.argsort(np.array(scores))[::-1][:top_n]
        return [
            self._copy_hit(candidates[int(index)], float(scores[int(index)]), rank)
            for rank, index in enumerate(order, start=1)
        ]

    def _generate(self, query: str, hits: list[Hit]) -> str:
        from langchain_openai import ChatOpenAI

        context = "\n\n".join(
            f"[{hit.rank}] 标题：{hit.title}｜来源：{hit.source}\n{hit.content}"
            for hit in hits
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "资料：\n{context}\n\n问题：{question}"),
            ]
        )
        llm = ChatOpenAI(
            base_url=DEEPSEEK_BASE_URL,
            api_key=DEEPSEEK_API_KEY,
            model=DEEPSEEK_MODEL,
            temperature=0.1,
        )
        message = (prompt | llm).invoke({"context": context, "question": query})
        return str(message.content)

    @staticmethod
    def _hit(chunk: Chunk, score: float, rank: int) -> Hit:
        return Hit(
            vector_id=chunk.vector_id,
            chunk_id=chunk.chunk_id,
            title=chunk.title,
            source=chunk.source,
            content=chunk.content,
            score=score,
            rank=rank,
        )

    @staticmethod
    def _copy_hit(hit: Hit, score: float, rank: int) -> Hit:
        return Hit(
            vector_id=hit.vector_id,
            chunk_id=hit.chunk_id,
            title=hit.title,
            source=hit.source,
            content=hit.content,
            score=score,
            rank=rank,
        )


SMOKE_CASES = [
    ("本科生一次最多可以借几本书？", "10"),
    ("书没按时还回去会怎样？", "0.10"),
    ("一等奖学金对专业排名有什么要求？", "5%"),
    ("晚上回宿舍太晚需要登记吗？", "晚归"),
    ("体测不及格还能评优吗？", "评优"),
]


def smoke(qa: CampusQA) -> int:
    """看混合+重排的前三条里有没有关键事实。这不是论文指标，只用来确认链路没接错。"""
    print(f"稀疏检索：{qa.sparse_backend}")
    missed = 0
    for question, needle in SMOKE_CASES:
        hits = qa.search(question, "hybrid_rerank", top_n=3)
        blob = "\n".join(hit.content for hit in hits)
        ok = needle in blob
        title = hits[0].title if hits else "无结果"
        print(f"{'OK' if ok else 'MISS'}\t{question}\t-> {title}")
        if not ok:
            missed += 1
    print(f"冒烟通过 {len(SMOKE_CASES) - missed}/{len(SMOKE_CASES)}")
    return missed


def main() -> None:
    summary = ingest(preload_reranker=True)
    print(json.dumps(summary, ensure_ascii=False))
    smoke(CampusQA())


if __name__ == "__main__":
    main()
