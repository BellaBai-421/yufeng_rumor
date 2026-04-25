"""
Claude tool-use 工具定义与执行函数.

两个工具:
  1. search_rumor_kb  — 包装 RumorRetriever.search_with_decision()
  2. lookup_punishment — 根据标签 + 转发数查扣分规则
"""

import json
from config import VALID_LABELS, load_credit_rules, get_deduction

# ── Tool Schema（Anthropic 格式）─────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "search_rumor_kb",
        "description": (
            "在谣言知识库中检索与输入文本最相似的已知谣言记录。"
            "返回匹配级别（high/medium/low/none）、最相似的知识库条目及建议。"
            "对每条待分类的谣言，应首先调用此工具获取知识库参考。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "待检索的谣言文本",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的最相似条目数量，默认 3",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "lookup_punishment",
        "description": (
            "根据谣言分类结果和转发数，查询微博信用扣分规则。"
            "仅在分类完成后调用。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "enum": VALID_LABELS,
                    "description": "谣言分类标签",
                },
                "forward_count": {
                    "type": "integer",
                    "description": "该谣言的直接转发数",
                },
            },
            "required": ["label", "forward_count"],
        },
    },
]


# ── Tool 执行函数 ────────────────────────────────────────

_credit_rules = None


def _get_rules():
    global _credit_rules
    if _credit_rules is None:
        _credit_rules = load_credit_rules()
    return _credit_rules


def execute_search(retriever, query: str, top_k: int = 3) -> dict:
    """调用 retriever.search_with_decision()."""
    return retriever.search_with_decision(query, top_k=top_k)


def execute_punishment_lookup(label: str, forward_count: int) -> dict:
    """根据标签和转发数查询处罚."""
    rules = _get_rules()

    if label == "确实如此":
        return {
            "label": label,
            "forward_count": forward_count,
            "deduction": 0,
            "action": rules["true_action"],
            "rule": "确实如此，不处罚",
        }

    if label == "尚无定论":
        return {
            "label": label,
            "forward_count": forward_count,
            "deduction": 0,
            "action": rules["uncertain_action"],
            "rule": "尚无定论，标记观察",
        }

    # 不实信息
    result = get_deduction(forward_count, rules)
    return {
        "label": label,
        "forward_count": forward_count,
        "deduction": result["deduction"],
        "action": f"扣除信用积分{result['deduction']}分",
        "rule": result["rule"],
    }


def handle_tool_call(tool_name: str, tool_input: dict, retriever) -> str:
    """
    分发工具调用，返回 JSON 字符串结果.

    Args:
        tool_name: 工具名称
        tool_input: 工具输入参数
        retriever: RumorRetriever 实例（仅 search_rumor_kb 需要）

    Returns:
        JSON 字符串
    """
    if tool_name == "search_rumor_kb":
        result = execute_search(
            retriever,
            query=tool_input["query"],
            top_k=tool_input.get("top_k", 3),
        )
    elif tool_name == "lookup_punishment":
        result = execute_punishment_lookup(
            label=tool_input["label"],
            forward_count=tool_input["forward_count"],
        )
    else:
        result = {"error": f"未知工具: {tool_name}"}

    return json.dumps(result, ensure_ascii=False)
