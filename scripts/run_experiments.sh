#!/bin/bash
# 实验矩阵：3 种子 x 3 阈值组合 = 9 组实验
# 每个种子额外跑 1 组 no-rag 基线，共 12 组评估
set -e

SEEDS=(42 43 44)
# 阈值组合：(high, medium)
THRESHOLDS=("0.90,0.75" "0.85,0.70" "0.95,0.80")

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

EXP_ROOT="output/experiments"
mkdir -p "$EXP_ROOT"

for seed in "${SEEDS[@]}"; do
    echo "================================================"
    echo "  Seed = $seed"
    echo "================================================"

    SEED_DIR="$EXP_ROOT/seed_$seed"
    mkdir -p "$SEED_DIR"

    # ── Step 1: 数据划分 ──
    echo "[Step 1] 数据划分 (seed=$seed)..."
    python scripts/prepare_data.py \
        --seed "$seed" \
        --output-dir "$SEED_DIR"

    # ── Step 2: 构建知识库 ──
    echo "[Step 2] 构建知识库..."
    python build_kb.py \
        --exclude-dev "$SEED_DIR/split/dev_rumor_texts.json" \
        --output "$SEED_DIR/serving_rumor_KB.json"

    # ── Step 3: 构建向量索引 ──
    echo "[Step 3] 构建向量索引..."
    python build_vector_index.py \
        --kb "$SEED_DIR/serving_rumor_KB.json" \
        --output-dir "$SEED_DIR/vector_store"

    # ── Step 4: 各阈值组合评估 ──
    for threshold_pair in "${THRESHOLDS[@]}"; do
        IFS=',' read -r high med <<< "$threshold_pair"
        echo "[Step 4] 评估: seed=$seed, high=$high, medium=$med"

        # 分类评估
        python evaluate.py --task cls \
            --seed "$seed" \
            --data-dir "$SEED_DIR" \
            --high-threshold "$high" \
            --medium-threshold "$med"

        # 处罚评估
        python evaluate.py --task pun \
            --seed "$seed" \
            --data-dir "$SEED_DIR" \
            --high-threshold "$high" \
            --medium-threshold "$med"
    done

    # ── Step 5: no-rag 基线 ──
    echo "[Step 5] no-rag 基线: seed=$seed"
    python evaluate.py --task cls --no-rag \
        --seed "$seed" \
        --data-dir "$SEED_DIR"

    python evaluate.py --task pun --no-rag \
        --seed "$seed" \
        --data-dir "$SEED_DIR"
done

echo ""
echo "================================================"
echo "  所有实验完成，开始汇总..."
echo "================================================"
python scripts/summarize_experiments.py --exp-dir "$EXP_ROOT"
echo "Done."
