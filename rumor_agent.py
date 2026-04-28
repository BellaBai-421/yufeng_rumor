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

from config import RETRIEVER_STORE_DIR, RETRIEVER_KB_PATH
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
        self.punishment_retriever = None

        if not no_rag:
            from rag_retriever import RumorRetriever
            self.retriever = RumorRetriever(
                store_dir=RETRIEVER_STORE_DIR,
                kb_path=RETRIEVER_KB_PATH,
            )

        # 加载判罚检索器，复用 RAG retriever 的 embedding 模型
        from punishment_retriever import PunishmentRetriever
        model = self.retriever.model if self.retriever else None
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("BAAI/bge-base-zh-v1.5")
        self.punishment_retriever = PunishmentRetriever(model=model)

    def classify(self, rumor_text: str, need_punishment: bool = False) -> dict:
        """对单条谣言文本分类，按需判罚."""
        return case_pipeline(
            rumor_text=rumor_text,
            retriever=self.retriever,
            need_punishment=need_punishment,
            no_rag=self.no_rag,
            punishment_retriever=self.punishment_retriever,
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
            result = self.classify(text, need_punishment=need_punishment)
            result["input_text"] = text
            results.append(result)
        return results


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

        out_path = path.rsplit(".", 1)[0] + "_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到 {out_path}")

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

        out_path = args.input.rsplit(".", 1)[0] + "_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到 {out_path}")
