"""
处罚内容匹配检索器 — 基于 embedding 相似度匹配 train 处罚记录.

核心思路: "同话题同处罚" — 找到最相似的已判罚谣言，复用其处罚等级.
"""

import numpy as np
from collections import Counter

from config import (
    load_punishment_train,
    PUNISHMENT_TRAIN_PATH,
)


class PunishmentRetriever:
    """
    基于 embedding 内容匹配的处罚预测器.

    复用已有 SentenceTransformer 模型实例，避免重复加载。
    """

    def __init__(self, model, train_path: str = PUNISHMENT_TRAIN_PATH,
                 top_k: int = 3):
        """
        Args:
            model: SentenceTransformer 实例 (复用 RAG retriever 的模型)
            train_path: punishment train 数据路径
            top_k: 取 top_k 近邻做加权投票
        """
        self.model = model
        self.top_k = top_k

        # 加载并编码 train 数据
        self.records = load_punishment_train(train_path)
        texts = [r["rumorText"] for r in self.records]
        self.levels = np.array([r["level"] for r in self.records])

        print(f"[PunishmentRetriever] 编码 {len(texts)} 条处罚训练样本...")
        self.embeddings = model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
        ).astype(np.float32)
        print(f"[PunishmentRetriever] 就绪. 向量维度: {self.embeddings.shape}")

    def match(self, query_text: str) -> dict:
        """
        对单条文本查找最相似的处罚记录, 返回加权投票结果.

        返回:
        {
            "level": int,           # 预测处罚等级 (1-6)
            "score": float,         # top1 相似度
            "match_text": str,      # top1 匹配的谣言文本
            "match_result": str,    # top1 匹配的原始处罚结果
            "neighbors": [...]      # top_k 近邻详情
        }
        """
        query_vec = self.model.encode(
            [query_text],
            normalize_embeddings=True,
        ).astype(np.float32)

        # cosine similarity (已归一化 → dot product)
        sims = (query_vec @ self.embeddings.T)[0]
        top_indices = np.argsort(sims)[::-1][:self.top_k]

        neighbors = []
        level_weights = Counter()
        for idx in top_indices:
            score = float(sims[idx])
            level = int(self.levels[idx])
            neighbors.append({
                "index": int(idx),
                "score": round(score, 4),
                "level": level,
                "text": self.records[idx]["rumorText"][:80],
                "result": self.records[idx]["result"],
            })
            level_weights[level] += score

        voted_level = level_weights.most_common(1)[0][0]
        top1 = neighbors[0]

        return {
            "level": voted_level,
            "score": top1["score"],
            "match_text": top1["text"],
            "match_result": top1["result"],
            "neighbors": neighbors,
        }
