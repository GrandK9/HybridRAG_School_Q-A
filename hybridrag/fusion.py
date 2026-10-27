"""倒数排名融合。

两路检索的分数不可比：BM25 是无上界的词项权重，向量是余弦相似度。
直接相加或加权会让某一路主导排序，所以只使用名次。
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """把多路名次表融合成一份排序。

    score(d) = Σ 1 / (k + rank_i(d))，rank 从 1 开始。
    某一路没有召回到的文档，不给这一路加分。
    k 取 60 是 Cormack 等人原论文的常用取值，名次靠后的文档贡献会快速变小。
    """
    if k < 0:
        raise ValueError("k 必须大于等于 0")
    scores: dict[str, float] = {}
    for ranked_ids in ranked_lists:
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranked_ids, start=1):
            # 同一路里重复出现只计第一次，避免切片重复写入把分数抬高。
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
