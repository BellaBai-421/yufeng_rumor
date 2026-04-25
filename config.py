"""
配置常量、扣分规则解析、LLM client 工厂.
"""

import os
import json

# ── DeepSeek 模型 ────────────────────────────────────────
MODEL_NAME = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# ── 分类标签 ─────────────────────────────────────────────
VALID_LABELS = ["不实信息", "尚无定论", "确实如此"]

# ── 置信度门控阈值 ───────────────────────────────────────
HIGH_CONFIDENCE_THRESHOLD = 0.90
MEDIUM_CONFIDENCE_THRESHOLD = 0.75

# ── 检索器默认路径 ───────────────────────────────────────
RETRIEVER_STORE_DIR = "output/vector_store"
RETRIEVER_KB_PATH = "output/serving_rumor_KB.json"

# ── 扣分规则路径 ─────────────────────────────────────────
CREDIT_RULES_PATH = "rules/weibo_credit_rules.json"


def get_llm_client():
    """返回 OpenAI-compatible 客户端（指向 DeepSeek）."""
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("请设置环境变量 DEEPSEEK_API_KEY")
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def load_credit_rules(path: str = CREDIT_RULES_PATH) -> dict:
    """
    解析 weibo_credit_rules.json，提取'发布不实信息'的扣分梯度.

    返回:
    {
        "tiers": [
            {"max_forward": 100,  "deduction": 10},
            {"max_forward": 1000, "deduction": 15},
            {"max_forward": None, "deduction": 20},
        ],
        "severe": "30+",
        "uncertain_action": "不扣分，标记观察并提醒",
        "true_action": "不扣分，不处罚",
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rumor_rules = None
    for item in raw["data"]:
        if item["user_behavior"] == "发布不实信息":
            rumor_rules = item["items"]
            break

    if rumor_rules is None:
        raise ValueError("weibo_credit_rules.json 中未找到'发布不实信息'规则")

    tiers = [
        {"max_forward": 100, "deduction": rumor_rules[0]["deduction_value"]},
        {"max_forward": 1000, "deduction": rumor_rules[1]["deduction_value"]},
        {"max_forward": None, "deduction": rumor_rules[2]["deduction_value"]},
    ]

    return {
        "tiers": tiers,
        "severe": str(rumor_rules[3]["deduction_value"]),
        "uncertain_action": "不扣分，标记观察并提醒",
        "true_action": "不扣分，不处罚",
    }


def get_deduction(forward_count: int, rules: dict | None = None) -> dict:
    """根据转发数计算扣分. 返回 {"deduction": int, "rule": str}."""
    if rules is None:
        rules = load_credit_rules()

    for tier in rules["tiers"]:
        if tier["max_forward"] is not None and forward_count <= tier["max_forward"]:
            return {
                "deduction": tier["deduction"],
                "rule": f"直接转发数不超过{tier['max_forward']}，扣{tier['deduction']}分",
            }

    last = rules["tiers"][-1]
    return {
        "deduction": last["deduction"],
        "rule": f"直接转发数超过1000，扣{last['deduction']}分",
    }
