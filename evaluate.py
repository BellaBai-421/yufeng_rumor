"""
评估脚本 — 对比 RAG Agent vs 纯 LLM 基线.

用法:
    # 分类评估（RAG）
    python evaluate.py --task cls

    # 分类评估（无 RAG 基线）
    python evaluate.py --task cls --no-rag

    # 处罚评估
    python evaluate.py --task pun

    # 限制条数（调试用）
    python evaluate.py --task cls --limit 5
"""

import json
import sys
import argparse
from collections import defaultdict
from pathlib import Path

from rumor_agent import RumorAgent
from config import VALID_LABELS


def load_cls_data(path: str = "output/classification/dev.json") -> list[dict]:
    """加载分类验证集."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [
        {
            "rumor_text": item["rumorText"],
            "ground_truth": item["explain"],
            "forward_count": 0,
        }
        for item in raw
    ]


def load_pun_data(path: str = "output/punishment/dev.json") -> list[dict]:
    """加载处罚验证集."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [
        {
            "rumor_text": item["rumorText"],
            "ground_truth": item["explain"],
            "result": item.get("result", ""),
            "forward_count": item.get("forward", 0),
        }
        for item in raw
    ]


def compute_metrics(predictions: list[str], ground_truths: list[str]) -> dict:
    """
    计算分类指标: accuracy, per-class precision/recall/F1.

    Returns:
        {
            "accuracy": float,
            "total": int,
            "correct": int,
            "per_class": {label: {"precision", "recall", "f1", "support"}},
            "confusion_matrix": {true_label: {pred_label: count}},
        }
    """
    labels = VALID_LABELS
    correct = sum(p == g for p, g in zip(predictions, ground_truths))
    total = len(predictions)

    # 混淆矩阵
    cm = {tl: {pl: 0 for pl in labels} for tl in labels}
    for pred, gt in zip(predictions, ground_truths):
        if gt in cm and pred in cm[gt]:
            cm[gt][pred] += 1

    # Per-class 指标
    per_class = {}
    for label in labels:
        tp = cm[label][label]
        fp = sum(cm[other][label] for other in labels if other != label)
        fn = sum(cm[label][other] for other in labels if other != label)
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    return {
        "accuracy": round(correct / total, 4) if total > 0 else 0.0,
        "total": total,
        "correct": correct,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def print_report(metrics: dict, title: str = "分类评估结果"):
    """打印评估报告."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  准确率: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
    print(f"\n  {'标签':<8} {'精确率':>8} {'召回率':>8} {'F1':>8} {'支持数':>6}")
    print(f"  {'-' * 42}")
    for label, m in metrics["per_class"].items():
        print(f"  {label:<8} {m['precision']:>8.4f} {m['recall']:>8.4f} "
              f"{m['f1']:>8.4f} {m['support']:>6}")

    print(f"\n  混淆矩阵 (行=真实, 列=预测):")
    labels = VALID_LABELS
    header = f"  {'':>8}" + "".join(f"{l:>8}" for l in labels)
    print(header)
    for true_label in labels:
        row = f"  {true_label:>8}"
        for pred_label in labels:
            row += f"{metrics['confusion_matrix'][true_label][pred_label]:>8}"
        print(row)
    print()


def run_evaluation(task: str, no_rag: bool, limit: int | None, output_path: str):
    """执行评估."""
    # 加载数据
    if task == "cls":
        data = load_cls_data()
        title = "分类评估"
    else:
        data = load_pun_data()
        title = "处罚评估"

    if limit:
        data = data[:limit]

    mode = "no-rag" if no_rag else "rag"
    print(f"[评估] 模式={mode}, 任务={task}, 样本数={len(data)}", file=sys.stderr)

    # 初始化 Agent
    agent = RumorAgent(no_rag=no_rag)

    # 逐条分类
    predictions = []
    details = []
    for i, item in enumerate(data, 1):
        print(f"[{i}/{len(data)}] 分类中: {item['rumor_text'][:40]}...", file=sys.stderr)
        result = agent.classify(
            rumor_text=item["rumor_text"],
            forward_count=item.get("forward_count", 0),
        )
        pred_label = result.get("label", "未知")
        predictions.append(pred_label)
        details.append({
            "rumor_text": item["rumor_text"][:100],
            "ground_truth": item["ground_truth"],
            "predicted": pred_label,
            "correct": pred_label == item["ground_truth"],
            "confidence": result.get("confidence", 0),
            "reasoning": result.get("reasoning", ""),
            "kb_match_level": result.get("kb_match_level", ""),
            "punishment": result.get("punishment", {}),
        })

    ground_truths = [item["ground_truth"] for item in data]
    metrics = compute_metrics(predictions, ground_truths)

    # 打印报告
    full_title = f"{title} ({mode})"
    print_report(metrics, full_title)

    # 保存结果
    output = {
        "task": task,
        "mode": mode,
        "metrics": metrics,
        "details": details,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[评估] 结果已保存到 {output_path}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="谣言分类评估")
    parser.add_argument("--task", choices=["cls", "pun"], default="cls",
                        help="评估任务: cls=分类, pun=处罚")
    parser.add_argument("--no-rag", action="store_true", help="无 RAG 基线模式")
    parser.add_argument("--limit", type=int, default=None, help="限制评估条数")
    parser.add_argument("--output", type=str, default=None, help="结果输出路径")
    args = parser.parse_args()

    if args.output is None:
        mode_suffix = "no_rag" if args.no_rag else "rag"
        args.output = f"output/eval_{args.task}_{mode_suffix}.json"

    run_evaluation(args.task, args.no_rag, args.limit, args.output)
