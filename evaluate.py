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
import time
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
    """计算 accuracy, per-class precision/recall/F1, macro/micro/weighted, 混淆矩阵."""
    labels = VALID_LABELS
    # 扩展标签列表，加入 INVALID 列处理非法预测
    all_labels = list(labels) + ["INVALID"]
    correct = sum(p == g for p, g in zip(predictions, ground_truths))
    total = len(predictions)

    # 统计非法预测
    invalid_count = sum(1 for p in predictions if p not in labels)

    # 构建含 INVALID 列的混淆矩阵
    cm = {tl: {pl: 0 for pl in all_labels} for tl in labels}
    for pred, gt in zip(predictions, ground_truths):
        if gt not in cm:
            continue
        pred_key = pred if pred in labels else "INVALID"
        cm[gt][pred_key] += 1

    # per-class 指标（INVALID 预测计入 FN）
    per_class = {}
    for label in labels:
        tp = cm[label][label]
        fp = sum(cm[other][label] for other in labels if other != label)
        fn = (sum(cm[label][other] for other in labels if other != label)
              + cm[label]["INVALID"])
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

    # Macro 指标：各类均值
    n_labels = len(labels)
    macro_precision = sum(pc["precision"] for pc in per_class.values()) / n_labels
    macro_recall = sum(pc["recall"] for pc in per_class.values()) / n_labels
    macro_f1 = sum(pc["f1"] for pc in per_class.values()) / n_labels

    # Micro 指标：全局 TP/FP/FN 聚合
    total_tp = sum(cm[l][l] for l in labels)
    total_fp = sum(
        sum(cm[o][l] for o in labels if o != l) for l in labels
    )
    total_fn = sum(
        sum(cm[l][o] for o in labels if o != l) + cm[l]["INVALID"]
        for l in labels
    )
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if (micro_precision + micro_recall) > 0 else 0.0)

    # Weighted 指标：按 support 加权
    total_support = sum(pc["support"] for pc in per_class.values())
    if total_support > 0:
        weighted_precision = sum(
            pc["precision"] * pc["support"] for pc in per_class.values()
        ) / total_support
        weighted_recall = sum(
            pc["recall"] * pc["support"] for pc in per_class.values()
        ) / total_support
        weighted_f1 = sum(
            pc["f1"] * pc["support"] for pc in per_class.values()
        ) / total_support
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    return {
        "accuracy": round(correct / total, 4) if total > 0 else 0.0,
        "total": total,
        "correct": correct,
        "invalid_prediction_count": invalid_count,
        "invalid_prediction_rate": round(invalid_count / total, 4) if total > 0 else 0.0,
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "micro_f1": round(micro_f1, 4),
        "weighted_precision": round(weighted_precision, 4),
        "weighted_recall": round(weighted_recall, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def print_report(metrics: dict, title: str = "分类评估结果"):
    """打印评估报告."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  准确率: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
    inv = metrics.get("invalid_prediction_count", 0)
    if inv > 0:
        print(f"  无效预测: {inv} ({metrics['invalid_prediction_rate']:.2%})")

    print(f"\n  {'标签':<8} {'精确率':>8} {'召回率':>8} {'F1':>8} {'支持数':>6}")
    print(f"  {'-' * 42}")
    for label, m in metrics["per_class"].items():
        print(f"  {label:<8} {m['precision']:>8.4f} {m['recall']:>8.4f} "
              f"{m['f1']:>8.4f} {m['support']:>6}")

    # 聚合指标
    print(f"\n  {'指标':<12} {'精确率':>8} {'召回率':>8} {'F1':>8}")
    print(f"  {'-' * 38}")
    for agg in ["macro", "micro", "weighted"]:
        p = metrics.get(f"{agg}_precision", 0)
        r = metrics.get(f"{agg}_recall", 0)
        f = metrics.get(f"{agg}_f1", 0)
        print(f"  {agg:<12} {p:>8.4f} {r:>8.4f} {f:>8.4f}")

    # 混淆矩阵（含 INVALID 列）
    print(f"\n  混淆矩阵 (行=真实, 列=预测):")
    all_labels = list(VALID_LABELS) + ["INVALID"]
    header = f"  {'':>8}" + "".join(f"{l:>8}" for l in all_labels)
    print(header)
    for true_label in VALID_LABELS:
        row = f"  {true_label:>8}"
        for pred_label in all_labels:
            row += f"{metrics['confusion_matrix'][true_label][pred_label]:>8}"
        print(row)
    print()


def compute_trace_summary(details: list[dict]) -> dict:
    """计算 trace 统计摘要，含 per-gate accuracy 和 avg_confidence."""
    gate_names = ["high_confidence", "medium_confidence",
                  "medium_confidence_downgraded", "low_confidence", "no_rag"]
    gate_data = {g: {"count": 0, "correct": 0, "confidences": [], "llm_calls": 0}
                 for g in gate_names}
    llm_called = 0

    for d in details:
        trace = d.get("trace", {})
        gate = trace.get("gate_decision", "no_rag")
        if gate not in gate_data:
            gate_data[gate] = {"count": 0, "correct": 0, "confidences": [], "llm_calls": 0}
        gate_data[gate]["count"] += 1
        if d.get("correct"):
            gate_data[gate]["correct"] += 1
        conf = d.get("confidence")
        if isinstance(conf, (int, float)):
            gate_data[gate]["confidences"].append(conf)
        if trace.get("llm_called"):
            llm_called += 1
            gate_data[gate]["llm_calls"] += 1

    total = len(details)
    per_gate = {}
    for gate, data in gate_data.items():
        if data["count"] == 0:
            continue
        confs = data["confidences"]
        per_gate[gate] = {
            "count": data["count"],
            "accuracy": round(data["correct"] / data["count"], 4),
            "avg_confidence": round(sum(confs) / len(confs), 4) if confs else 0.0,
            "llm_calls": data["llm_calls"],
        }

    return {
        "total": total,
        "llm_called": llm_called,
        "llm_call_rate": round(llm_called / total, 4) if total > 0 else 0.0,
        "per_gate": per_gate,
    }


def print_trace_summary(details: list[dict]) -> dict:
    """打印 trace 统计摘要并返回 summary dict."""
    summary = compute_trace_summary(details)
    total = summary["total"]

    print(f"  Pipeline Trace 统计:")
    print(f"  {'决策门控':<24} {'数量':>6} {'占比':>8} {'准确率':>8} {'平均置信':>8}")
    print(f"  {'-' * 58}")
    for gate, info in summary["per_gate"].items():
        print(f"  {gate:<24} {info['count']:>6} {info['count']/total:>8.1%} "
              f"{info['accuracy']:>8.2%} {info['avg_confidence']:>8.4f}")
    print(f"  LLM 调用次数: {summary['llm_called']}/{total} ({summary['llm_call_rate']:.1%})")
    print()
    return summary


# ── Step 4: RAG 检索质量 ─────────────────────────────────


def compute_rag_quality(details: list[dict]) -> dict:
    """计算 RAG 检索质量指标."""
    top1_total = 0
    top1_hits = 0
    topk_total = 0
    topk_hits = 0
    topk_conflict = 0
    high_conf_total = 0
    high_conf_correct = 0

    for d in details:
        trace = d.get("trace", {})
        gt = d.get("ground_truth")

        # top1 标签准确率
        top1_label = trace.get("rag_top1_label")
        if top1_label is not None:
            top1_total += 1
            if top1_label == gt:
                top1_hits += 1

        # topK 标签覆盖率
        topk_labels = trace.get("rag_topk_labels")
        if topk_labels:
            topk_total += 1
            if gt in topk_labels:
                topk_hits += 1
            if len(set(topk_labels)) > 1:
                topk_conflict += 1

        # 高置信准确率
        gate = trace.get("gate_decision")
        if gate == "high_confidence":
            high_conf_total += 1
            if d.get("correct"):
                high_conf_correct += 1

    return {
        "rag_top1_label_accuracy": round(top1_hits / top1_total, 4) if top1_total > 0 else None,
        "rag_top1_total": top1_total,
        "rag_topk_label_recall": round(topk_hits / topk_total, 4) if topk_total > 0 else None,
        "rag_topk_total": topk_total,
        "topk_conflict_rate": round(topk_conflict / topk_total, 4) if topk_total > 0 else None,
        "topk_conflict_count": topk_conflict,
        "high_confidence_accuracy": round(high_conf_correct / high_conf_total, 4) if high_conf_total > 0 else None,
        "high_confidence_total": high_conf_total,
    }


def print_rag_quality(rag_quality: dict):
    """打印 RAG 检索质量."""
    print(f"  RAG 检索质量:")
    if rag_quality["rag_top1_total"] > 0:
        print(f"  Top-1 标签准确率: {rag_quality['rag_top1_label_accuracy']:.2%} "
              f"({rag_quality['rag_top1_total']} 样本)")
    if rag_quality["rag_topk_total"] > 0:
        print(f"  TopK 标签覆盖率:  {rag_quality['rag_topk_label_recall']:.2%} "
              f"({rag_quality['rag_topk_total']} 样本)")
        print(f"  TopK 标签冲突率:  {rag_quality['topk_conflict_rate']:.2%} "
              f"({rag_quality['topk_conflict_count']} 条)")
    if rag_quality["high_confidence_total"] > 0:
        print(f"  高置信直采准确率: {rag_quality['high_confidence_accuracy']:.2%} "
              f"({rag_quality['high_confidence_total']} 条)")
    print()


# ── Step 5: 延迟 / 成本 ─────────────────────────────────


def compute_latency_stats(details: list[dict]) -> dict:
    """计算延迟和 LLM 调用统计."""
    latencies = [d["latency_seconds"] for d in details if "latency_seconds" in d]
    llm_calls = sum(1 for d in details if d.get("trace", {}).get("llm_called"))
    total = len(details)

    if not latencies:
        return {"avg": 0, "p50": 0, "p95": 0, "max": 0, "total_time": 0,
                "total_llm_calls": llm_calls, "llm_call_rate": 0, "total_samples": total}

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    return {
        "avg": round(sum(sorted_lat) / n, 3),
        "p50": round(sorted_lat[n // 2], 3),
        "p95": round(sorted_lat[int(n * 0.95)], 3),
        "max": round(sorted_lat[-1], 3),
        "total_time": round(sum(sorted_lat), 2),
        "total_llm_calls": llm_calls,
        "llm_call_rate": round(llm_calls / total, 4) if total > 0 else 0,
        "total_samples": total,
    }


def print_latency_stats(stats: dict):
    """打印延迟统计."""
    print(f"  延迟 / 成本统计:")
    print(f"  平均延迟: {stats['avg']:.3f}s | P50: {stats['p50']:.3f}s | "
          f"P95: {stats['p95']:.3f}s | Max: {stats['max']:.3f}s")
    print(f"  总耗时: {stats['total_time']:.2f}s | "
          f"LLM 调用: {stats['total_llm_calls']}/{stats['total_samples']} "
          f"({stats['llm_call_rate']:.1%})")
    print()


# ── Step 6: 错误案例分析 ─────────────────────────────────


def compute_error_analysis(details: list[dict]) -> dict:
    """按错误类型分组分析."""
    # 按 "真实→预测" 分组
    transition_groups = defaultdict(list)
    # 按 error_type 分组
    type_groups = defaultdict(list)

    for d in details:
        if d.get("correct"):
            continue
        gt = d.get("ground_truth", "")
        pred = d.get("predicted", "")
        trace = d.get("trace", {})
        gate = trace.get("gate_decision", "")
        text = d.get("rumor_text", "")

        transition_groups[f"{gt}→{pred}"].append(text)

        # 细分 error_type
        if pred == "未知" or pred not in VALID_LABELS:
            type_groups["unknown_prediction"].append(text)
        elif gate == "high_confidence":
            type_groups["rag_high_confidence_wrong"].append(text)
        elif trace.get("llm_called"):
            if gt == "不实信息" and pred != "不实信息":
                type_groups["false_negative_rumor"].append(text)
            elif gt != "不实信息" and pred == "不实信息":
                type_groups["false_positive_rumor"].append(text)
            else:
                type_groups["llm_wrong"].append(text)

    total_errors = sum(len(v) for v in transition_groups.values())

    # 排序并截取示例
    transitions = sorted(
        [{"type": k, "count": len(v), "examples": v[:3]}
         for k, v in transition_groups.items()],
        key=lambda x: x["count"], reverse=True,
    )
    error_types = sorted(
        [{"type": k, "count": len(v), "examples": v[:3]}
         for k, v in type_groups.items()],
        key=lambda x: x["count"], reverse=True,
    )

    return {
        "total_errors": total_errors,
        "transitions": transitions,
        "error_types": error_types,
    }


def print_error_analysis(analysis: dict):
    """打印错误案例分析."""
    print(f"  错误案例分析 (共 {analysis['total_errors']} 条错误):")
    if not analysis["transitions"]:
        print(f"  (无错误)")
        print()
        return

    print(f"\n  按转移方向:")
    for t in analysis["transitions"]:
        print(f"    {t['type']}: {t['count']} 条")
        for ex in t["examples"]:
            print(f"      - {ex[:60]}...")

    if analysis["error_types"]:
        print(f"\n  按错误类型:")
        for t in analysis["error_types"]:
            print(f"    {t['type']}: {t['count']} 条")
    print()


# ── Step 7: LLM 输出校对 ─────────────────────────────────


def compute_llm_validation_stats(details: list[dict]) -> dict:
    """聚合 LLM 输出校验统计."""
    total_llm_calls = 0
    json_parse_failed = 0
    invalid_label = 0
    invalid_confidence = 0
    llm_call_failed = 0
    missing_fields_breakdown = defaultdict(int)
    samples_with_missing = 0

    for d in details:
        validation = d.get("trace", {}).get("llm_validation")
        if validation is None:
            continue
        total_llm_calls += 1
        if validation.get("llm_call_failed"):
            llm_call_failed += 1
        if validation.get("json_parse_failed"):
            json_parse_failed += 1
        if validation.get("invalid_label"):
            invalid_label += 1
        if validation.get("invalid_confidence"):
            invalid_confidence += 1
        missing = validation.get("missing_fields", [])
        if missing:
            samples_with_missing += 1
            for f in missing:
                missing_fields_breakdown[f] += 1

    failure_count = json_parse_failed + invalid_label + llm_call_failed
    return {
        "total_llm_calls": total_llm_calls,
        "llm_call_failed": llm_call_failed,
        "json_parse_failed": json_parse_failed,
        "invalid_label": invalid_label,
        "invalid_confidence": invalid_confidence,
        "missing_fields_count": samples_with_missing,
        "missing_fields_breakdown": dict(missing_fields_breakdown),
        "failure_rate": round(failure_count / total_llm_calls, 4) if total_llm_calls > 0 else 0.0,
    }


def print_llm_validation_stats(stats: dict):
    """打印 LLM 输出校对统计."""
    print(f"  LLM 输出校对 (共 {stats['total_llm_calls']} 次调用):")
    if stats["total_llm_calls"] == 0:
        print(f"  (无 LLM 调用)")
        print()
        return
    print(f"  调用失败:     {stats['llm_call_failed']}")
    print(f"  JSON 解析失败: {stats['json_parse_failed']}")
    print(f"  无效标签:     {stats['invalid_label']}")
    print(f"  无效置信度:   {stats['invalid_confidence']}")
    print(f"  缺失字段:     {stats['missing_fields_count']}", end="")
    if stats["missing_fields_breakdown"]:
        print(f" ({stats['missing_fields_breakdown']})", end="")
    print()
    print(f"  总故障率:     {stats['failure_rate']:.2%}")
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
        t0 = time.perf_counter()
        result = agent.classify(
            rumor_text=item["rumor_text"],
            need_punishment=need_pun,
        )
        elapsed = time.perf_counter() - t0
        pred_label = result.get("label", "未知")
        predictions.append(pred_label)
        details.append({
            "rumor_text": item["rumor_text"][:100],
            "ground_truth": item["ground_truth"],
            "predicted": pred_label,
            "correct": pred_label == item["ground_truth"],
            "confidence": result.get("confidence", 0),
            "confidence_source": result.get("confidence_source", "unknown"),
            "reasoning": result.get("reasoning", ""),
            "trace": result.get("trace", {}),
            "punishment": result.get("punishment"),
            "latency_seconds": round(elapsed, 3),
        })

    ground_truths = [item["ground_truth"] for item in data]
    metrics = compute_metrics(predictions, ground_truths)

    # ── 控制台输出 ──
    print_report(metrics, f"{title} ({mode})")

    trace_summary = None
    rag_quality = None
    if not no_rag:
        trace_summary = print_trace_summary(details)
        rag_quality = compute_rag_quality(details)
        print_rag_quality(rag_quality)

    latency_stats = compute_latency_stats(details)
    print_latency_stats(latency_stats)

    error_analysis = compute_error_analysis(details)
    print_error_analysis(error_analysis)

    llm_validation_stats = compute_llm_validation_stats(details)
    print_llm_validation_stats(llm_validation_stats)

    # 处罚评估: 额外统计处罚等级准确率 + MAE + over/under
    punishment_summary = None
    if need_pun:
        from config import normalize_punishment_result
        pun_correct = 0
        pun_total = 0
        level_diffs = []
        over_pun = 0
        under_pun = 0
        for item, detail in zip(data, details):
            gt_result = item.get("result", "")
            gt_level = normalize_punishment_result(gt_result)
            pred_pun = detail.get("punishment")
            pred_level = pred_pun.get("level") if pred_pun else None
            if gt_level is not None:
                pun_total += 1
                if pred_level == gt_level:
                    pun_correct += 1
                if pred_level is not None:
                    diff = pred_level - gt_level
                    level_diffs.append(abs(diff))
                    if diff > 0:
                        over_pun += 1
                    elif diff < 0:
                        under_pun += 1

        pun_acc = pun_correct / pun_total if pun_total > 0 else 0
        pun_mae = sum(level_diffs) / len(level_diffs) if level_diffs else 0
        punishment_summary = {
            "punishment_accuracy": round(pun_acc, 4),
            "punishment_correct": pun_correct,
            "punishment_total": pun_total,
            "punishment_mae": round(pun_mae, 4),
            "over_punishment_count": over_pun,
            "over_punishment_rate": round(over_pun / pun_total, 4) if pun_total > 0 else 0,
            "under_punishment_count": under_pun,
            "under_punishment_rate": round(under_pun / pun_total, 4) if pun_total > 0 else 0,
        }
        print(f"\n  处罚评估:")
        print(f"  等级准确率: {pun_acc:.2%} ({pun_correct}/{pun_total})")
        print(f"  MAE: {pun_mae:.4f}")
        print(f"  过度处罚: {over_pun} ({over_pun/pun_total:.1%})" if pun_total > 0 else "")
        print(f"  处罚不足: {under_pun} ({under_pun/pun_total:.1%})" if pun_total > 0 else "")

    output = {
        "task": task,
        "mode": mode,
        "metrics": metrics,
        "trace_summary": trace_summary,
        "rag_quality": rag_quality,
        "latency_stats": latency_stats,
        "error_analysis": error_analysis,
        "llm_validation_stats": llm_validation_stats,
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
