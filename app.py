"""校园问答原型页面。"""

from __future__ import annotations

import gradio as gr

from hybridrag.config import STRATEGY_LABELS, STRATEGIES
from hybridrag.pipeline import Answer, CampusQA, Hit

LABEL_TO_KEY = {label: key for key, label in STRATEGY_LABELS.items()}
EXAMPLES = [
    "本科生一次最多可以借几本书？",
    "书没按时还回去会怎样？",
    "一等奖学金对专业排名有什么要求？",
    "晚上回宿舍太晚需要登记吗？",
    "体测不及格还能评优吗？",
]


def _rows(hits: list[Hit]) -> list[list[str]]:
    rows: list[list[str]] = []
    for hit in hits:
        snippet = hit.content.replace("\n", " ")
        if len(snippet) > 80:
            snippet = snippet[:80] + "…"
        rows.append([str(hit.rank), hit.title, f"{hit.score:.4f}", hit.source, snippet])
    return rows


def _compare_markdown(groups: dict[str, list[Hit]]) -> str:
    lines = ["| 策略 | 第 1 条 | 第 2 条 | 第 3 条 |", "| --- | --- | --- | --- |"]
    for key in STRATEGIES:
        titles = [hit.title for hit in groups.get(key, [])]
        while len(titles) < 3:
            titles.append("—")
        lines.append("| " + " | ".join([STRATEGY_LABELS[key], *titles[:3]]) + " |")
    return "\n".join(lines)


def build_demo(qa: CampusQA) -> gr.Blocks:
    def respond(question: str, strategy_label: str):
        strategy = LABEL_TO_KEY[strategy_label]
        try:
            result: Answer = qa.answer(question, strategy)
        except Exception as exc:
            return f"检索失败：{exc}", [], qa.sparse_backend
        note = "已调用大模型生成" if result.generated else "当前是检索摘录"
        header = f"**策略：** {strategy_label}  \n**稀疏检索：** {result.sparse_backend}  \n**回答来源：** {note}"
        return (
            f"{header}\n\n{result.text}",
            _rows(result.hits),
            f"稀疏检索后端：**{result.sparse_backend}**",
        )

    def compare(question: str):
        if not question.strip():
            return "先输入问题。"
        try:
            groups = qa.compare(question.strip(), top_n=3)
        except Exception as exc:
            return f"对比失败：{exc}"
        return _compare_markdown(groups)

    with gr.Blocks(title="校园 HybridRAG 原型") as demo:
        gr.Markdown(
            """# 校园 HybridRAG 原型

示例制度上的混合检索问答。稀疏检索用 Elasticsearch 的 BM25，稠密检索用 FAISS，两路用 RRF 融合，再用交叉编码器重排。
语料是为跑通链路编写的示范大学制度，不是学校正式文件。"""
        )
        status = gr.Markdown(f"稀疏检索后端：**{qa.sparse_backend}**")
        question = gr.Textbox(label="问题", placeholder="例如：本科生一次最多可以借几本书？", lines=2)
        strategy = gr.Radio(
            choices=[STRATEGY_LABELS[key] for key in STRATEGIES],
            value=STRATEGY_LABELS["hybrid_rerank"],
            label="检索策略",
        )
        with gr.Row():
            ask = gr.Button("回答", variant="primary")
            contrast = gr.Button("只对比四路召回")
        answer = gr.Markdown(label="回答")
        citations = gr.Dataframe(
            headers=["名次", "文档", "分数", "来源文件", "摘录"],
            label="引用",
            interactive=False,
            wrap=True,
        )
        contrast_view = gr.Markdown(label="四路对比")
        gr.Examples(examples=EXAMPLES, inputs=question)
        ask.click(respond, inputs=[question, strategy], outputs=[answer, citations, status])
        contrast.click(compare, inputs=question, outputs=contrast_view)
    return demo


def main() -> None:
    qa = CampusQA()
    print(qa.sparse_backend)
    build_demo(qa).launch(server_name="127.0.0.1", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
