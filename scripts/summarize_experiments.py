#!/usr/bin/env python3
"""
汇总实验结果 — 按 threshold 分组，计算跨 seed 的 mean±std.

用法:
    python scripts/summarize_experiments.py --exp-dir output/experiments
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def load_eval_results(exp_dir: str) -> list[dict]:
    """加载所有 eval_*.json 文件."""
    results = []
    for path in sorted(Path(exp_dir).rglob("eval_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_source_file"] = str(path)
        results.append(data)
    return results


def mean_std(values: list[float]) -> tuple[float, float]:
    """计算均值和标准差."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mu = sum(values) / n
    if n == 1:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in values) / (n - 1)
    return mu, math.sqrt(var)


def summarize(results: list[dict]) -> dict:
    """按 (task, threshold) 分组，跨 seed 聚合."""
    groups = defaultdict(list)

    for r in results:
        config = r.get("experiment_config", {})
        task = r.get("task", "unknown")
        mode = r.get("mode", "unknown")
        high = config.get("high_threshold", "?")
        med = config.get("medium_threshold", "?")
        seed = config.get("seed", "?")

        if mode == "no-rag":
            key = (task, "no-rag", "-", "-")
        else:
            key = (task, "rag", str(high), str(med))

        groups[key].append({
            "seed": seed,
            "accuracy": r.get("metrics", {}).get("accuracy", 0),
            "macro_f1": r.get("metrics", {}).get("macro_f1", 0),
            "micro_f1": r.get("metrics", {}).get("micro_f1", 0),
            "weighted_f1": r.get("metrics", {}).get("weighted_f1", 0),
            "punishment_accuracy": (
                r.get("punishment_summary", {}) or {}
            ).get("punishment_accuracy"),
            "source": r.get("_source_file", ""),
        })

    # 聚合
    summary_rows = []
    for (task, mode, high, med), entries in sorted(groups.items()):
        row = {
            "task": task,
            "mode": mode,
            "high_threshold": high,
            "medium_threshold": med,
            "n_seeds": len(entries),
            "seeds": [e["seed"] for e in entries],
        }

        for metric in ["accuracy", "macro_f1", "micro_f1", "weighted_f1"]:
            vals = [e[metric] for e in entries]
            mu, sd = mean_std(vals)
            row[f"{metric}_mean"] = round(mu, 4)
            row[f"{metric}_std"] = round(sd, 4)

        # 处罚准确率（可能为 None）
        pun_vals = [e["punishment_accuracy"] for e in entries
                    if e["punishment_accuracy"] is not None]
        if pun_vals:
            mu, sd = mean_std(pun_vals)
            row["punishment_accuracy_mean"] = round(mu, 4)
            row["punishment_accuracy_std"] = round(sd, 4)

        row["per_seed"] = entries
        summary_rows.append(row)

    return {"experiments": summary_rows}


def print_summary(summary: dict):
    """打印汇总表."""
    rows = summary["experiments"]

    print(f"\n{'=' * 90}")
    print(f"  实验结果汇总（跨 seed 聚合）")
    print(f"{'=' * 90}")

    # 分类结果
    cls_rows = [r for r in rows if r["task"] == "cls"]
    if cls_rows:
        print(f"\n  [分类任务]")
        print(f"  {'模式':<12} {'high':>6} {'med':>6} {'seeds':>6} "
              f"{'accuracy':>14} {'macro_f1':>14} {'weighted_f1':>14}")
        print(f"  {'-' * 78}")
        for r in sorted(cls_rows, key=lambda x: x.get("macro_f1_mean", 0), reverse=True):
            mode_str = r["mode"]
            if r["mode"] == "rag":
                mode_str = "rag"
            acc = f"{r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f}"
            mf1 = f"{r['macro_f1_mean']:.4f}±{r['macro_f1_std']:.4f}"
            wf1 = f"{r['weighted_f1_mean']:.4f}±{r['weighted_f1_std']:.4f}"
            print(f"  {mode_str:<12} {r['high_threshold']:>6} {r['medium_threshold']:>6} "
                  f"{r['n_seeds']:>6} {acc:>14} {mf1:>14} {wf1:>14}")

    # 处罚结果
    pun_rows = [r for r in rows if r["task"] == "pun"]
    if pun_rows:
        print(f"\n  [处罚任务]")
        print(f"  {'模式':<12} {'high':>6} {'med':>6} {'seeds':>6} "
              f"{'accuracy':>14} {'macro_f1':>14} {'pun_acc':>14}")
        print(f"  {'-' * 78}")
        for r in sorted(pun_rows, key=lambda x: x.get("macro_f1_mean", 0), reverse=True):
            acc = f"{r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f}"
            mf1 = f"{r['macro_f1_mean']:.4f}±{r['macro_f1_std']:.4f}"
            pun = (f"{r['punishment_accuracy_mean']:.4f}±{r['punishment_accuracy_std']:.4f}"
                   if "punishment_accuracy_mean" in r else "-")
            print(f"  {r['mode']:<12} {r['high_threshold']:>6} {r['medium_threshold']:>6} "
                  f"{r['n_seeds']:>6} {acc:>14} {mf1:>14} {pun:>14}")

    # 最优阈值推荐
    rag_cls = [r for r in cls_rows if r["mode"] == "rag"]
    if rag_cls:
        best = max(rag_cls, key=lambda x: (x["macro_f1_mean"], -x["macro_f1_std"]))
        print(f"\n  推荐阈值（分类 macro_f1 最高 + 方差最小）:")
        print(f"  high={best['high_threshold']}, medium={best['medium_threshold']} "
              f"→ macro_f1={best['macro_f1_mean']:.4f}±{best['macro_f1_std']:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="实验结果汇总")
    parser.add_argument("--exp-dir", default="output/experiments",
                        help="实验输出根目录")
    parser.add_argument("--output", default=None,
                        help="汇总 JSON 输出路径（默认 {exp-dir}/summary.json）")
    args = parser.parse_args()

    results = load_eval_results(args.exp_dir)
    if not results:
        print(f"[错误] 未找到评估结果文件: {args.exp_dir}/*/eval_*.json")
        return

    print(f"[汇总] 加载了 {len(results)} 个评估结果", flush=True)
    summary = summarize(results)
    print_summary(summary)

    out_path = args.output or f"{args.exp_dir}/summary.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[汇总] 已保存到 {out_path}")


if __name__ == "__main__":
    main()
