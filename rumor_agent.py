"""
谣言分类 Agent — 基于 Claude tool-use 的薄 Agent 层.

用法:
    # 单条测试
    python rumor_agent.py --query "吃大蒜可以预防新冠病毒"
    python rumor_agent.py --query "..." --forward-count 50

    # 无 RAG 基线
    python rumor_agent.py --query "..." --no-rag

    # 作为模块
    from rumor_agent import RumorAgent
    agent = RumorAgent()
    result = agent.classify("吃大蒜可以预防新冠病毒", forward_count=0)
"""

import json
import re
import sys

from config import (
    MODEL_NAME,
    RETRIEVER_STORE_DIR,
    RETRIEVER_KB_PATH,
    get_anthropic_client,
)
from tools import TOOL_DEFINITIONS, handle_tool_call
from prompts import SYSTEM_PROMPT, build_user_message


class RumorAgent:
    """
    谣言分类 Agent.

    封装 Claude tool-use 循环:
      用户消息 → Claude 调用 search_rumor_kb → 返回结果 →
      Claude 调用 lookup_punishment → 返回结果 → Claude 输出最终 JSON
    """

    def __init__(self, no_rag: bool = False):
        self.client = get_anthropic_client()
        self.no_rag = no_rag
        self.retriever = None

        if not no_rag:
            from rag_retriever import RumorRetriever
            self.retriever = RumorRetriever(
                store_dir=RETRIEVER_STORE_DIR,
                kb_path=RETRIEVER_KB_PATH,
            )

    def classify(self, rumor_text: str, forward_count: int = 0) -> dict:
        """
        对单条谣言文本进行分类 + 处罚判断.

        Returns:
            {
                "label": str,
                "confidence": float,
                "reasoning": str,
                "kb_match_level": str,
                "punishment": {"deduction": int, "action": str},
            }
        """
        user_msg = build_user_message(rumor_text, forward_count)
        messages = [{"role": "user", "content": user_msg}]

        # 无 RAG 模式不提供工具
        tools = None if self.no_rag else TOOL_DEFINITIONS

        # tool-use 循环
        max_turns = 6
        for _ in range(max_turns):
            kwargs = {
                "model": MODEL_NAME,
                "system": SYSTEM_PROMPT,
                "messages": messages,
                "max_tokens": 1024,
            }
            if tools:
                kwargs["tools"] = tools

            response = self.client.messages.create(**kwargs)

            # 检查是否有 tool_use
            if response.stop_reason == "tool_use":
                # 将 assistant 回复追加到消息
                messages.append({
                    "role": "assistant",
                    "content": response.content,
                })

                # 执行所有 tool call 并收集结果
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result_str = handle_tool_call(
                            block.name, block.input, self.retriever
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_str,
                        })

                messages.append({"role": "user", "content": tool_results})
            else:
                # 最终文本回复
                return self._parse_response(response)

        # 超过最大轮次
        return {
            "label": "未知",
            "confidence": 0.0,
            "reasoning": "超过最大工具调用轮次",
            "kb_match_level": "none",
            "punishment": {"deduction": 0, "action": "无法判断"},
        }

    def classify_batch(self, items: list[dict]) -> list[dict]:
        """
        批量分类.

        Args:
            items: [{"rumor_text": str, "forward_count": int}, ...]

        Returns:
            分类结果列表
        """
        results = []
        total = len(items)
        for i, item in enumerate(items, 1):
            print(f"[{i}/{total}] 分类中...", file=sys.stderr)
            result = self.classify(
                rumor_text=item["rumor_text"],
                forward_count=item.get("forward_count", 0),
            )
            result["input_text"] = item["rumor_text"]
            results.append(result)
        return results

    def _parse_response(self, response) -> dict:
        """从 Claude 的最终回复中提取 JSON."""
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        # 尝试提取 JSON 块
        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        # 尝试提取裸 JSON
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # 解析失败，返回原始文本
        return {
            "label": "未知",
            "confidence": 0.0,
            "reasoning": f"JSON 解析失败，原始回复: {text[:200]}",
            "kb_match_level": "none",
            "punishment": {"deduction": 0, "action": "无法判断"},
        }


# ── CLI 入口 ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="谣言分类 Agent")
    parser.add_argument("--query", type=str, required=True, help="待分类的谣言文本")
    parser.add_argument("--forward-count", type=int, default=0, help="直接转发数")
    parser.add_argument("--no-rag", action="store_true", help="不使用 RAG，纯 LLM 分类")
    args = parser.parse_args()

    agent = RumorAgent(no_rag=args.no_rag)
    result = agent.classify(args.query, forward_count=args.forward_count)

    print(json.dumps(result, ensure_ascii=False, indent=2))
