"""
Step 2: 将知识库向量化并构建 FAISS 索引
输入: output/serving_rumor_KB.json
输出: output/vector_store/index.faiss + metadata.json

运行: python build_vector_index.py
依赖: pip install sentence-transformers faiss-cpu
"""

import json
import argparse
import numpy as np
from pathlib import Path

# 延迟导入, 方便检查依赖
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请运行: pip install sentence-transformers faiss-cpu")
    exit(1)


# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
DEFAULT_MODEL = "BAAI/bge-base-zh-v1.5"
BATCH_SIZE = 64


def load_kb(filepath: str) -> list[dict]:
    """加载知识库"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="构建向量索引")
    parser.add_argument("--kb", default="output/serving_rumor_KB.json", help="知识库路径")
    parser.add_argument("--output-dir", default="output/vector_store", help="输出目录")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="embedding 模型名")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载知识库
    print(f"[1/3] 加载知识库: {args.kb}")
    records = load_kb(args.kb)
    print(f"  → {len(records)} 条记录")

    # 2. 编码向量（直接使用 KB 中预构建的 content 字段）
    documents = [rec["content"] for rec in records]
    print(f"[2/3] 加载模型并编码 (模型: {args.model})...")
    print("  (首次运行会下载模型, 约 500MB)")
    model = SentenceTransformer(args.model)
    embeddings = model.encode(
        documents,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # 归一化后 inner product = cosine similarity
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"  → 向量维度: {embeddings.shape}")

    # 3. 构建 FAISS 索引
    print("[3/3] 构建 FAISS 索引...")
    dimension = embeddings.shape[1]

    # 数据量 < 1000, 用 Flat 索引 (精确检索, 无损)
    index = faiss.IndexFlatIP(dimension)  # Inner Product (因为已归一化 = cosine)
    index.add(embeddings)

    # 保存索引
    faiss.write_index(index, str(output_dir / "index.faiss"))

    # 保存元数据（仅索引配置信息，检索时直接读 serving_rumor_KB.json）
    metadata = {
        "model_name": args.model,
        "dimension": dimension,
        "num_vectors": len(records),
        "index_type": "FlatIP",
        "kb_path": args.kb,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n输出:")
    print(f"  → 索引: {output_dir / 'index.faiss'}")
    print(f"  → 元数据: {output_dir / 'metadata.json'}")
    print(f"\n向量索引构建完成. 索引包含 {index.ntotal} 条向量.")


if __name__ == "__main__":
    main()
