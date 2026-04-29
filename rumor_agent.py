"""
谣言分类 Agent — 确定性 pipeline + 4 模式入口.

用法:
    # 交互式菜单
    python rumor_agent.py

    # 指定模式
    python rumor_agent.py --mode classify_only --query "吃大蒜可以预防新冠病毒"
    python rumor_agent.py --mode classify_and_punish --query "..."
    python rumor_agent.py --mode batch_classify --input data.json
    python rumor_agent.py --mode batch_classify_punish --input data.json

    # 无 RAG 基线
    python rumor_agent.py --mode classify_only --query "..." --no-rag

    # 作为模块
    from rumor_agent import RumorAgent
    agent = RumorAgent()
    result = agent.classify("吃大蒜可以预防新冠病毒", need_punishment=True)
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import config as _cfg
from pipeline import case_pipeline


class RumorAgent:
    """
    谣言分类 Agent.

    确定性 pipeline：RAG → 门控 → 按需 LLM → 内容匹配判罚.
    高置信度直接采用 KB 标签，不调用 LLM。
    """

    def __init__(self, no_rag: bool = False):
        self.no_rag = no_rag
        self.retriever = None
        self._punishment_retriever = None

        if not no_rag:
            from rag_retriever import RumorRetriever
            self.retriever = RumorRetriever(
                store_dir=_cfg.RETRIEVER_STORE_DIR,
                kb_path=_cfg.RETRIEVER_KB_PATH,
            )

    @property
    def punishment_retriever(self):
        """懒加载判罚检索器，仅在首次需要时初始化."""
        if self._punishment_retriever is None:
            from punishment_retriever import PunishmentRetriever

            model = self.retriever.model if self.retriever else None
            if model is None:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer("BAAI/bge-base-zh-v1.5")

            self._punishment_retriever = PunishmentRetriever(model=model)

        return self._punishment_retriever

    def classify(self, rumor_text: str, need_punishment: bool = False) -> dict:
        """对单条谣言文本分类，按需判罚."""
        return case_pipeline(
            rumor_text=rumor_text,
            retriever=self.retriever,
            need_punishment=need_punishment,
            no_rag=self.no_rag,
            punishment_retriever=self.punishment_retriever if need_punishment else None,
        )

    def classify_batch(self, items: list[dict],
                       need_punishment: bool = False) -> list[dict]:
        """
        批量分类.

        Args:
            items: [{"rumorText": str, ...}, ...]
            need_punishment: 是否需要判罚
        """
        results = []
        total = len(items)
        for i, item in enumerate(items, 1):
            text = item.get("rumorText", item.get("rumor_text", ""))
            print(f"[{i}/{total}] {text[:40]}...", file=sys.stderr)
            t0 = time.perf_counter()
            result = self.classify(text, need_punishment=need_punishment)
            result["latency_seconds"] = round(time.perf_counter() - t0, 3)
            result["input_text"] = text
            results.append(result)
        return results

    @staticmethod
    def save_batch_report(results: list[dict], output_dir: str):
        """
        将批量结果输出到目录，包含：
        - results.json: 全量分类/判罚结果
        - review_queue.json: 需转人工复核的条目
        - stats.json: 分类/判罚统计
        - llm_log.json: LLM 调用统计
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1. 全量结果
        with open(out / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 2. 人工复核队列
        review_queue = [
            {
                "input_text": r["input_text"],
                "label": r["label"],
                "confidence": r["confidence"],
                "review_reason": r["review_reason"],
                "punishment": r.get("punishment"),
                "reasoning": r.get("reasoning", ""),
            }
            for r in results if r.get("review_required")
        ]
        with open(out / "review_queue.json", "w", encoding="utf-8") as f:
            json.dump(review_queue, f, ensure_ascii=False, indent=2)

        # 3. 分类/判罚统计
        label_counts = defaultdict(int)
        review_reasons = defaultdict(int)
        pun_level_counts = defaultdict(int)
        for r in results:
            label_counts[r["label"]] += 1
            if r.get("review_required"):
                review_reasons[r["review_reason"]] += 1
            pun = r.get("punishment")
            if pun and pun.get("level"):
                pun_level_counts[pun["level"]] += 1

        stats = {
            "total": len(results),
            "label_distribution": dict(label_counts),
            "review_required_total": len(review_queue),
            "review_by_reason": dict(review_reasons),
            "auto_pass": len(results) - len(review_queue),
            "punishment_level_distribution": dict(pun_level_counts),
        }
        with open(out / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        # 4. LLM 调用日志
        llm_calls = []
        for r in results:
            meta = r.get("trace", {}).get("llm_meta")
            if meta:
                llm_calls.append({
                    "input_text": r["input_text"][:60],
                    "gate_decision": r.get("trace", {}).get("gate_decision"),
                    "latency_ms": meta.get("latency_ms"),
                    "prompt_tokens": meta.get("prompt_tokens"),
                    "completion_tokens": meta.get("completion_tokens"),
                    "total_tokens": meta.get("total_tokens"),
                    "error": meta.get("error"),
                })

        total_tokens = sum(c["total_tokens"] or 0 for c in llm_calls)
        latencies = [c["latency_ms"] for c in llm_calls if c["latency_ms"]]
        llm_log = {
            "total_calls": len(llm_calls),
            "call_rate": round(len(llm_calls) / len(results), 4) if results else 0,
            "success_rate": round(
                sum(1 for c in llm_calls if not c.get("error")) / len(llm_calls), 4
            ) if llm_calls else 1.0,
            "total_tokens": total_tokens,
            "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "total_latency_seconds": round(sum(latencies) / 1000, 2) if latencies else 0,
            "calls": llm_calls,
        }
        with open(out / "llm_log.json", "w", encoding="utf-8") as f:
            json.dump(llm_log, f, ensure_ascii=False, indent=2)

        return stats, llm_log


# ── 交互式菜单 ───────────────────────────────────────────

MENU = """\
========================================
  谣言分类与处罚判断系统
========================================
请选择模式：
  [1] 单条分类
  [2] 单条分类 + 判罚
  [3] JSON 文件批量分类
  [4] JSON 文件批量分类 + 判罚
  [0] 退出
========================================"""


def _interactive():
    """交互式入口."""
    print(MENU)
    choice = input("请输入选项 (0-4): ").strip()

    if choice == "0":
        return

    if choice in ("1", "2"):
        text = input("请输入谣言文本: ").strip()
        if not text:
            print("文本不能为空", file=sys.stderr)
            return
        need_pun = choice == "2"

        agent = RumorAgent()
        result = agent.classify(text, need_punishment=need_pun)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif choice in ("3", "4"):
        path = input("请输入 JSON 文件路径: ").strip()
        if not path:
            print("路径不能为空", file=sys.stderr)
            return
        need_pun = choice == "4"

        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)

        agent = RumorAgent()
        results = agent.classify_batch(items, need_punishment=need_pun)

        out_dir = path.rsplit(".", 1)[0] + "_output"
        stats, llm_log = RumorAgent.save_batch_report(results, out_dir)
        print(f"\n结果已保存到 {out_dir}/")
        print(f"  总计: {stats['total']} 条, 转人工: {stats['review_required_total']} 条")
        print(f"  LLM 调用: {llm_log['total_calls']} 次, 平均延迟: {llm_log['avg_latency_ms']}ms")

    else:
        print("无效选项", file=sys.stderr)


# ── CLI 入口 ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="谣言分类 Agent")
    parser.add_argument(
        "--mode",
        choices=["classify_only", "classify_and_punish",
                 "batch_classify", "batch_classify_punish"],
        default=None,
        help="运行模式（不指定则进入交互式菜单）",
    )
    parser.add_argument("--query", type=str, help="单条模式的谣言文本")
    parser.add_argument("--input", type=str, help="批量模式的 JSON 文件路径")
    parser.add_argument("--no-rag", action="store_true", help="不使用 RAG，纯 LLM 分类")
    args = parser.parse_args()

    # 无参数 → 交互式菜单
    if args.mode is None and args.query is None and args.input is None:
        _interactive()
        sys.exit(0)

    # 有参数 → 命令行模式
    agent = RumorAgent(no_rag=args.no_rag)

    if args.mode in (None, "classify_only", "classify_and_punish"):
        if not args.query:
            parser.error("单条模式需要 --query 参数")
        need_pun = args.mode == "classify_and_punish"
        result = agent.classify(args.query, need_punishment=need_pun)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.mode in ("batch_classify", "batch_classify_punish"):
        if not args.input:
            parser.error("批量模式需要 --input 参数")
        need_pun = args.mode == "batch_classify_punish"
        with open(args.input, "r", encoding="utf-8") as f:
            items = json.load(f)
        results = agent.classify_batch(items, need_punishment=need_pun)

        out_dir = args.input.rsplit(".", 1)[0] + "_output"
        stats, llm_log = RumorAgent.save_batch_report(results, out_dir)
        print(f"结果已保存到 {out_dir}/")
        print(f"  总计: {stats['total']} 条, 转人工: {stats['review_required_total']} 条")
        print(f"  LLM 调用: {llm_log['total_calls']} 次, 平均延迟: {llm_log['avg_latency_ms']}ms")
