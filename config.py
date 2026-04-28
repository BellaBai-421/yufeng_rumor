"""
配置常量、扣分规则解析、LLM client 工厂.
"""

import os
import json
import re

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
MINED_RULES_PATH = "rules/mined_rules.json"
PUNISHMENT_TRAIN_PATH = "output/punishment/train.json"


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
        # "情节恶劣"(30+) 无法自动判断，不纳入规则引擎
        "uncertain_action": "不扣分，标记观察并提醒",
        "true_action": "不扣分，不处罚",
    }


def get_deduction(forward_count: int, rules: dict | None = None) -> dict:
    """根据转发数计算扣分 (现行规则). 返回 {"deduction": int, "rule": str}."""
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


# ── 挖掘规则: 处罚等级体系 ──────────────────────────────

PUNISHMENT_LEVEL_DETAILS = {
    1: {"deduction": 2,  "ban_days": 0,    "action": "扣除信用积分2分"},
    2: {"deduction": 5,  "ban_days": 7,    "action": "扣除信用积分5分，禁言7天，禁被关注7天"},
    3: {"deduction": 10, "ban_days": 15,   "action": "扣除信用积分10分，禁言15天，禁被关注15天"},
    4: {"deduction": 10, "ban_days": 30,   "action": "扣除信用积分10分，禁言30天，禁被关注30天"},
    5: {"deduction": 20, "ban_days": 30,   "action": "扣除信用积分20分，禁言30天，禁被关注30天"},
    6: {"deduction": 0,  "ban_days": 99999, "action": "永久禁言"},
}


def normalize_punishment_result(result: str) -> int | None:
    """将原始 result 字符串归一化为处罚等级 (1-6), 无法归类返回 None."""
    if not result:
        return None
    if "永久禁言" in result:
        return 6
    ded_match = re.search(r"扣除信用积分(\d+)分", result)
    ban_match = re.search(r"禁言(\d+)天", result)
    deduction = int(ded_match.group(1)) if ded_match else 0
    ban_days = int(ban_match.group(1)) if ban_match else 0
    if deduction == 20 and ban_days >= 30:
        return 5
    if (deduction == 10 and ban_days >= 30) or (deduction == 0 and ban_days >= 30):
        return 4
    if deduction == 10 and ban_days >= 15:
        return 3
    if deduction == 5:
        return 2
    if deduction == 2:
        return 1
    if deduction == 0 and ban_days == 15:
        return 3
    return None


def load_punishment_train(path: str = PUNISHMENT_TRAIN_PATH) -> list[dict]:
    """加载判罚训练数据 (用于内容匹配)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = []
    for rec in raw:
        level = normalize_punishment_result(rec.get("result", ""))
        if level is None:
            continue
        records.append({
            "rumorText": rec["rumorText"],
            "level": level,
            "result": rec["result"],
            "forward": rec.get("forward", 0),
            "comment": rec.get("comment", 0),
            "visitTimes": rec.get("visitTimes", 0),
        })
    return records


def get_mined_deduction(level: int) -> dict:
    """根据挖掘的处罚等级返回扣分详情."""
    details = PUNISHMENT_LEVEL_DETAILS.get(level)
    if details is None:
        return {"deduction": 0, "action": "处罚等级无效", "level": level}
    return {
        "deduction": details["deduction"],
        "action": details["action"],
        "level": level,
        "ban_days": details["ban_days"],
    }
