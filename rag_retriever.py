"""
Step 3: RAG 检索器 — 知识库查询的核心模块
后续 Agent 和 FastAPI 都调用这个模块.

用法:
    from rag_retriever import RumorRetriever

    retriever = RumorRetriever("output/vector_store", "output/serving_rumor_KB.json")
    results = retriever.search("新冠病毒是人工合成的", top_k=3)
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError as e:
    raise ImportError(f"缺少依赖: {e}\n请运行: pip install sentence-transformers faiss-cpu")


@dataclass
class RetrievalResult:
    """单条检索结果"""
    kb_id: str
    score: float           # cosine similarity (0~1)
    rumor_text: str
    label: str
    evidence: str
    source: str
    date: str
    title: str
    tags: list
    original_label: str
    source_file: str
    rumormonger: str
    visit_times: int

    def to_dict(self) -> dict:
        return {
            "kb_id": self.kb_id,
            "score": round(self.score, 4),
            "rumor_text": self.rumor_text,
            "label": self.label,
            "evidence": self.evidence,
            "source": self.source,
            "date": self.date,
            "title": self.title,
            "tags": self.tags,
            "original_label": self.original_label,
            "source_file": self.source_file,
            "rumormonger": self.rumormonger,
            "visit_times": self.visit_times,
        }


class RumorRetriever:
    """
    谣言知识库检索器.

    职责单一: 接收一段文本, 返回最相似的知识库条目.
    不做分类判断, 不做置信度决策 — 那些是 Agent 层的事.
    """

    def __init__(
        self,
        store_dir: str = "output/vector_store",
        kb_path: str = "output/serving_rumor_KB.json",
        model_name: str | None = None,
        similarity_threshold: float = 0.75,
    ):
        store_path = Path(store_dir)

        # 加载元数据
        with open(store_path / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # 加载知识库记录（直接读源文件，不再维护副本）
        with open(kb_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

        # 加载 FAISS 索引
        self.index = faiss.read_index(str(store_path / "index.faiss"))

        # 加载 embedding 模型 (与构建时一致)
        effective_model = model_name or self.metadata["model_name"]
        self.model = SentenceTransformer(effective_model)

        self.similarity_threshold = similarity_threshold

        print(f"[RumorRetriever] 已加载:")
        print(f"  模型: {effective_model}")
        print(f"  知识库: {len(self.records)} 条")
        print(f"  相似度阈值: {self.similarity_threshold}")

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        检索最相似的知识库条目.

        Args:
            query: 待查询的谣言文本
            top_k: 返回前 k 条结果

        Returns:
            按相似度降序排列的 RetrievalResult 列表
        """
        # 编码查询文本
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).astype(np.float32)

        # FAISS 检索
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS 返回 -1 表示无效
                continue

            rec = self.records[idx]
            results.append(RetrievalResult(
                kb_id=rec["kb_id"],
                score=float(score),
                rumor_text=rec["rumor_text"],
                label=rec["label"],
                evidence=rec.get("evidence", ""),
                source=rec["source"],
                date=rec.get("date", ""),
                title=rec.get("title", ""),
                tags=rec.get("tags", []),
                original_label=rec.get("original_label", ""),
                source_file=rec.get("source_file", ""),
                rumormonger=rec.get("rumormonger", ""),
                visit_times=rec.get("visit_times", 0),
            ))

        return results

    def search_with_decision(
        self, query: str, top_k: int = 3
    ) -> dict:
        """
        检索 + 给出初步判断建议 (供 Agent 参考, 不是最终决策).

        返回:
        {
            "query": str,
            "match_level": "high" | "medium" | "low" | "none",
            "top_results": [...],
            "suggestion": str,  # 给 Agent 的建议
        }
        """
        results = self.search(query, top_k=top_k)

        if not results:
            return {
                "query": query,
                "match_level": "none",
                "top_results": [],
                "suggestion": "知识库中无匹配, 需要 LLM 独立分类",
            }

        best_score = results[0].score

        if best_score >= 0.90:
            match_level = "high"
            suggestion = (
                f"高度匹配 (score={best_score:.3f}), "
                f"建议直接采用知识库标签: [{results[0].label}]"
            )
        elif best_score >= self.similarity_threshold:
            match_level = "medium"
            suggestion = (
                f"中等匹配 (score={best_score:.3f}), "
                f"建议参考知识库标签但需 LLM 确认"
            )
        else:
            match_level = "low"
            suggestion = (
                f"低匹配 (score={best_score:.3f}), "
                f"知识库参考价值有限, 需 LLM 独立判断"
            )

        return {
            "query": query,
            "match_level": match_level,
            "top_results": [r.to_dict() for r in results],
            "suggestion": suggestion,
        }


# ──────────────────────────────────────────────
# CLI 测试入口
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="测试谣言检索器")
    parser.add_argument("--store", default="output/vector_store", help="向量存储目录")
    parser.add_argument("--kb", default="output/serving_rumor_KB.json", help="知识库路径")
    parser.add_argument("--query", type=str, required=True, help="查询文本")
    parser.add_argument("--top-k", type=int, default=3, help="返回条数")
    args = parser.parse_args()

    retriever = RumorRetriever(args.store, args.kb)
    result = retriever.search_with_decision(args.query, top_k=args.top_k)

    print(f"\n{'=' * 60}")
    print(f"查询: {result['query'][:80]}...")
    print(f"匹配级别: {result['match_level']}")
    print(f"建议: {result['suggestion']}")
    print(f"{'=' * 60}")

    for i, r in enumerate(result["top_results"], 1):
        print(f"\n--- Top {i} (score: {r['score']}) ---")
        print(f"  ID:    {r['kb_id']}")
        print(f"  标签:  {r['label']}")
        print(f"  谣言:  {r['rumor_text'][:100]}...")
        if r["evidence"]:
            print(f"  证据:  {r['evidence'][:100]}...")
