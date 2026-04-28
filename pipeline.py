"""
确定性 Case Pipeline — 对单条谣言执行：RAG → 门控 → 按需 LLM → 规则判罚.

不使用 tool-use。LLM 仅在中/低置信度时作为纯 completion 调用。
"""

import json
import re
import sys
from pathlib import Path

# 复用 build_kb.py 使用的同一份 normalize_text，保证查询端与 KB 端文本处理一致
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.prepare_data import normalize_text

from config import (
    MODEL_NAME,
    VALID_LABELS,
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    get_llm_client,
    get_mined_deduction,
)
from prompts import (
    PROMPT_WITH_EVIDENCE,
    PROMPT_LOW_CONFIDENCE,
    format_evidence_block,
)


_llm_client = None


def _get_client():
    global _llm_client
    if _llm_client is None:
        _llm_client = get_llm_client()
    return _llm_client


def _call_llm(system_prompt: str, user_text: str) -> dict:
    """
    调用 DeepSeek LLM 进行分类（纯 completion，无 tool-use）.

    返回: {"label": str, "confidence": float, "reasoning": str}
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        max_tokens=512,
        temperature=0.1,
    )

    text = response.choices[0].message.content or ""
    return _parse_llm_json(text)


def _parse_llm_json(text: str) -> dict:
    """从 LLM 回复中提取 JSON."""
    # 尝试 ```json``` 块
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)

    # 尝试裸 JSON
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            # 校验 label
            if parsed.get("label") not in VALID_LABELS:
                parsed["label"] = "未知"
            return parsed
        except json.JSONDecodeError:
            pass

    return {"label": "未知", "confidence": 0.0, "reasoning": f"LLM 输出解析失败: {text[:100]}"}


def _apply_punishment(label: str, punishment_match: dict | None = None) -> dict:
    """
    规则引擎：根据标签和内容匹配结果计算处罚.

    基于训练数据中的实际处罚记录（"同话题同处罚"），通过 embedding
    相似度匹配最相似的已判罚案例，复用其处罚等级。
    """
    if label == "确实如此":
        return {"deduction": 0, "action": "不扣分，不处罚"}
    if label == "尚无定论":
        return {"deduction": 0, "action": "不扣分，标记观察并提醒"}
    if label != "不实信息":
        return {"deduction": 0, "action": "标签无效，无法判罚"}

    if punishment_match is None:
        return {"deduction": 0, "action": "无匹配判罚记录"}

    level = punishment_match["level"]
    detail = get_mined_deduction(level)
    detail["match_score"] = punishment_match.get("score", 0)
    detail["match_text"] = punishment_match.get("match_text", "")[:60]
    return detail


def case_pipeline(
    rumor_text: str,
    retriever,
    need_punishment: bool = False,
    no_rag: bool = False,
    punishment_retriever=None,
) -> dict:
    """
    对单条谣言执行确定性分类 pipeline.

    流程:
      1. RAG 检索 TopK（no_rag 时跳过）
      2. 置信度门控
         - 高置信：直接采用 KB 标签，不调用 LLM
         - 中置信：LLM 基于证据受限分类
         - 低置信/无匹配：LLM 独立判断
      3. 按需判罚（规则引擎，不调用 LLM）

    返回:
    {
        "label": str,
        "confidence": float,
        "reasoning": str,
        "trace": {
            "rag_top1_score": float | None,
            "rag_top1_kb_id": str | None,
            "rag_top1_label": str | None,
            "gate_decision": "high_confidence" | "medium_confidence" | "low_confidence" | "no_rag",
            "llm_called": bool,
        },
        "punishment": {"deduction": int, "action": str} | None,
    }
    """
    trace = {
        "rag_top1_score": None,
        "rag_top1_kb_id": None,
        "rag_top1_label": None,
        "gate_decision": "no_rag",
        "llm_called": False,
    }

    # ── Step 0: 文本清洗（与 KB 构建时对齐）─────────────
    cleaned_text = normalize_text(rumor_text)
    if not cleaned_text:
        cleaned_text = rumor_text  # fallback: 清洗后为空则用原文

    # ── Step 1: RAG 检索 ─────────────────────────────────
    if no_rag or retriever is None:
        rag_result = None
    else:
        rag_result = retriever.search_with_decision(cleaned_text, top_k=3)

    # ── Step 2: 置信度门控 ───────────────────────────────
    if rag_result and rag_result["top_results"]:
        top1 = rag_result["top_results"][0]
        top1_score = top1["score"]
        trace["rag_top1_score"] = round(top1_score, 4)
        trace["rag_top1_kb_id"] = top1["kb_id"]
        trace["rag_top1_label"] = top1["label"]

        if top1_score >= HIGH_CONFIDENCE_THRESHOLD:
            # 高置信：直接采用 KB 标签
            trace["gate_decision"] = "high_confidence"
            label = top1["label"]
            confidence = round(top1_score, 4)
            reasoning = (
                f"知识库高度匹配 (score={top1_score:.3f})，"
                f"直接采用 KB 标签。匹配谣言: {top1['rumor_text'][:60]}"
            )

        elif top1_score >= MEDIUM_CONFIDENCE_THRESHOLD:
            # 中置信：LLM 基于证据受限分类
            trace["gate_decision"] = "medium_confidence"
            trace["llm_called"] = True
            evidence_block = format_evidence_block(rag_result["top_results"])
            system = PROMPT_WITH_EVIDENCE.format(evidence_block=evidence_block)
            llm_out = _call_llm(system, cleaned_text)
            label = llm_out.get("label", "未知")
            confidence = llm_out.get("confidence", 0.5)
            reasoning = llm_out.get("reasoning", "")

        else:
            # 低置信：LLM 作为语义仲裁者（带弱匹配证据参考）
            trace["gate_decision"] = "low_confidence"
            trace["llm_called"] = True
            evidence_block = format_evidence_block(rag_result["top_results"])
            system = PROMPT_LOW_CONFIDENCE.format(evidence_block=evidence_block)
            llm_out = _call_llm(system, cleaned_text)
            label = llm_out.get("label", "未知")
            confidence = llm_out.get("confidence", 0.3)
            reasoning = llm_out.get("reasoning", "")
            trace["needs_human_review"] = True
            trace["suggested_verification"] = llm_out.get("suggested_verification", "")
            trace["uncertainty"] = llm_out.get("uncertainty", "")
    else:
        # 无 RAG 结果（no_rag 模式或无匹配）
        trace["gate_decision"] = "no_rag"
        trace["llm_called"] = True
        # no_rag 模式无证据可用，使用空证据块
        system = PROMPT_LOW_CONFIDENCE.format(evidence_block="（无匹配记录）")
        llm_out = _call_llm(system, cleaned_text)
        label = llm_out.get("label", "未知")
        confidence = llm_out.get("confidence", 0.3)
        reasoning = llm_out.get("reasoning", "")
        trace["needs_human_review"] = True
        trace["suggested_verification"] = llm_out.get("suggested_verification", "")
        trace["uncertainty"] = llm_out.get("uncertainty", "")

    # ── Step 3: 判罚（内容匹配，不调用 LLM）──────────────
    punishment = None
    if need_punishment:
        punishment_match = None
        if punishment_retriever is not None:
            punishment_match = punishment_retriever.match(cleaned_text)
        punishment = _apply_punishment(label, punishment_match)

    return {
        "label": label,
        "confidence": confidence,
        "reasoning": reasoning,
        "trace": trace,
        "punishment": punishment,
    }
