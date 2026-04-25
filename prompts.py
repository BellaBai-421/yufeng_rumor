"""
LLM 分类提示词.

仅在中/低置信度时调用 LLM，不涉及任何工具调用。
"""

# ── 中置信度 prompt：LLM 基于证据受限分类 ────────────────

PROMPT_WITH_EVIDENCE = """\
你是一个中国新冠疫情谣言分类专家。

## 分类标签（三选一）
1. **不实信息**：经核查证实为虚假、伪科学或伪常识的内容
2. **尚无定论**：当前科学证据不足以确认真伪，尚待进一步研究
3. **确实如此**：经核查证实内容属实

## 任务
知识库检索到了相关记录（相似度中等），请综合待分类文本和知识库证据进行判断。

知识库匹配记录：
{evidence_block}

## 输出格式
仅输出 JSON，不要包含其他内容：
```json
{{"label": "不实信息 或 尚无定论 或 确实如此", "confidence": 0.0-1.0, "reasoning": "简要理由"}}
```"""


# ── 低置信度 prompt：LLM 独立判断 ────────────────────────

PROMPT_WITHOUT_EVIDENCE = """\
你是一个中国新冠疫情谣言分类专家。

## 分类标签（三选一）
1. **不实信息**：经核查证实为虚假、伪科学或伪常识的内容
2. **尚无定论**：当前科学证据不足以确认真伪，尚待进一步研究
3. **确实如此**：经核查证实内容属实

## 任务
知识库中没有高度相关的匹配记录，请根据你的知识独立判断以下文本的分类。
注意：对于疫情期间广泛传播的典型谣言（如"吃大蒜预防新冠"），应结合常识判断。

## 输出格式
仅输出 JSON，不要包含其他内容：
```json
{{"label": "不实信息 或 尚无定论 或 确实如此", "confidence": 0.0-1.0, "reasoning": "简要理由"}}
```"""


def format_evidence_block(top_results: list[dict]) -> str:
    """将 RAG 检索结果格式化为给 LLM 的证据文本."""
    lines = []
    for i, r in enumerate(top_results, 1):
        lines.append(f"--- 匹配 {i} (相似度: {r['score']:.3f}) ---")
        lines.append(f"标签: {r['label']}")
        lines.append(f"谣言原文: {r['rumor_text'][:200]}")
        if r.get("evidence"):
            lines.append(f"辟谣证据: {r['evidence'][:200]}")
        lines.append("")
    return "\n".join(lines)
