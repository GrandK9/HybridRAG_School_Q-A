# 基于混合检索增强生成的校园智能问答系统

本科毕业设计。用 BM25 与向量检索两路召回校园制度，经融合和交叉编码器重排后，由大模型生成带来源的回答，并对比四种检索策略。

当前是可运行原型：检索、重排、生成和网页问答已经接通。语料是编写的「示范大学」制度，用来验证链路。评测集、对比实验数据和论文尚未开始。

## 题目要求

功能：

1. BM25 稀疏检索与向量稠密检索双路召回，并融合两路结果。
2. 交叉编码器重排序。
3. 校园问答评测集不少于 100 题。
4. 四种策略对比：纯向量、纯 BM25、混合、混合加重排。
5. Web 问答展示来源引用和置信度。

成果是可运行系统、对比实验数据和毕业论文，论文以对比实验为核心章节。

完成标准：混合加重排的回答准确率不低于 90%，并且显著高于任何单一策略。

技术上使用 Python 和向量库（FAISS 或 Chroma），检索侧使用 Elasticsearch 的 BM25、BGE 嵌入和重排模型。评测先固定指标口径和评测脚本，再比较检索策略；重排按批调用，避免一次评测把接口打满。

## 当前实现

一条切片有两个去处。正文写入 Elasticsearch 索引 `campus_chunks`，用 IK 做中文分词，供 BM25 使用。同一段正文经硅基流动的 `BAAI/bge-large-zh-v1.5` 变成 1024 维向量，写入本地 FAISS 文件。FAISS 只保存向量矩阵并返回行号，行号通过切片编号对应到 ES 里的正文。

提问时按页面所选策略检索：

| 策略 | 做法 |
| --- | --- |
| 纯向量 | 问题向量与 FAISS 中每一行做内积，取最相近的切片 |
| 纯 BM25 | Elasticsearch 关键词检索 |
| 混合 | 两路各召回 20 条，在应用层用 RRF（k=60）按名次融合 |
| 混合 + 重排 | 在融合结果上调用硅基流动 `BAAI/bge-reranker-v2-m3`，重排后取前 5 条 |

ES 自带的 RRF 需要付费许可，所以融合写在 `hybridrag/fusion.py`。前 5 条正文、来源和问题一起交给 DeepSeek（`deepseek-flash`）生成回答。页面用 Gradio，同一进程里完成检索和生成，没有再套一层 Web 后端框架。

LangChain 负责切片、嵌入客户端和对话客户端。FAISS 与 Elasticsearch 的查询在 `hybridrag/pipeline.py` 中自行实现，便于四种策略分开记录。

和题目之间还差三块：不少于 100 题的评测集、四种策略的实验数据，以及单独给出的置信度。页面上的分数是该策略自己的检索分或重排分，还不是校准后的置信度。

## 运行

Elasticsearch 使用本机已有容器，索引与其他项目的索引分开：

```powershell
docker start shumamall-es
cd D:\build-my-project\projects\HybridRAG_School_Q&A
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m hybridrag.ingest
.\.venv\Scripts\python app.py
```

浏览器打开 http://127.0.0.1:7860 。

密钥和模型名写在 `.env`，模板见 `.env.example`。`EMBEDDING_API_KEY` 用于硅基流动的嵌入和重排，`DEEPSEEK_API_KEY` 用于生成。生成密钥留空时，页面只返回检索到的原文。索引已经建好时，可以跳过 `ingest`。

## 目录

```
data/campus/          示例制度
hybridrag/config.py   读取 .env，并写明召回条数等固定参数
hybridrag/fusion.py   RRF
hybridrag/pipeline.py 切片、灌库、四种检索、生成
app.py                问答页
```

`data/index/` 中的向量文件和切片缓存由灌库生成，不提交。
