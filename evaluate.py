"""
评估脚本 — 对比 RAG pipeline vs 纯 LLM 基线.

用法:
    python evaluate.py --task cls
    python evaluate.py --task cls --no-rag
    python evaluate.py --task pun
    python evaluate.py --task cls --limit 5
"""

import json
import sys
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
        }
        for item in raw
    ]


def compute_metrics(predictions: list[str], ground_truths: list[str]) -> dict:
    """计算 accuracy, per-class precision/recall/F1, 混淆矩阵."""
    labels = VALID_LABELS
    correct = sum(p == g for p, g in zip(predictions, ground_truths))
    total = len(predictions)

    cm = {tl: {pl: 0 for pl in labels} for tl in labels}
    for pred, gt in zip(predictions, ground_truths):
        if gt in cm and pred in cm[gt]:
            cm[gt][pred] += 1

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


def print_trace_summary(details: list[dict]):
    """打印 trace 统计摘要."""
    gate_counts = {"high_confidence": 0, "medium_confidence": 0,
                   "low_confidence": 0, "no_rag": 0}
    llm_called = 0
    for d in details:
        trace = d.get("trace", {})
        gate = trace.get("gate_decision", "no_rag")
        gate_counts[gate] = gate_counts.get(gate, 0) + 1
        if trace.get("llm_called"):
            llm_called += 1

    total = len(details)
    print(f"  Pipeline Trace 统计:")
    print(f"  {'决策门控':<20} {'数量':>6} {'占比':>8}")
    print(f"  {'-' * 36}")
    for gate, count in gate_counts.items():
        if count > 0:
            print(f"  {gate:<20} {count:>6} {count/total:>8.1%}")
    print(f"  LLM 调用次数: {llm_called}/{total} ({llm_called/total:.1%})")
    print()


def run_evaluation(task: str, no_rag: bool, limit: int | None, output_path: str):
    """执行评估."""
    if task == "cls":
        data = load_cls_data()
        title = "分类评估"
    else:
        data = load_pun_data()
        title = "处罚评估"

    if limit:
        data = data[:limit]

    mode = "no-rag" if no_rag else "rag"
    need_pun = task == "pun"
    print(f"[评估] 模式={mode}, 任务={task}, 样本数={len(data)}", file=sys.stderr)

    agent = RumorAgent(no_rag=no_rag)

    predictions = []
    details = []
    for i, item in enumerate(data, 1):
        print(f"[{i}/{len(data)}] {item['rumor_text'][:40]}...", file=sys.stderr)
        result = agent.classify(
            rumor_text=item["rumor_text"],
            need_punishment=need_pun,
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
            "trace": result.get("trace", {}),
            "punishment": result.get("punishment"),
        })

    ground_truths = [item["ground_truth"] for item in data]
    metrics = compute_metrics(predictions, ground_truths)

    print_report(metrics, f"{title} ({mode})")
    if not no_rag:
        print_trace_summary(details)

    # 处罚评估: 额外统计处罚等级准确率
    punishment_summary = None
    if need_pun:
        from config import normalize_punishment_result
        pun_correct = 0
        pun_total = 0
        for item, detail in zip(data, details):
            gt_result = item.get("result", "")
            gt_level = normalize_punishment_result(gt_result)
            pred_pun = detail.get("punishment")
            pred_level = pred_pun.get("level") if pred_pun else None
            if gt_level is not None:
                pun_total += 1
                if pred_level == gt_level:
                    pun_correct += 1
        pun_acc = pun_correct / pun_total if pun_total > 0 else 0
        punishment_summary = {
            "punishment_accuracy": round(pun_acc, 4),
            "punishment_correct": pun_correct,
            "punishment_total": pun_total,
        }
        print(f"\n  处罚等级准确率: {pun_acc:.2%} ({pun_correct}/{pun_total})")

    output = {
        "task": task,
        "mode": mode,
        "metrics": metrics,
        "punishment_summary": punishment_summary,
        "details": details,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[评估] 结果已保存到 {output_path}", file=sys.stderr)


if __name__ == "__main__":
    import argparse

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
