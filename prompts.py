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


# ── 低置信度 prompt：LLM 作为语义仲裁者 ─────────────────

PROMPT_LOW_CONFIDENCE = """\
你是一个中国新冠疫情谣言**语义分析专家**。你的职责是提供分类意见和核查建议，而非做最终事实裁决。

## 分类标签（三选一）
1. **不实信息**：经核查证实为虚假、伪科学或伪常识的内容
2. **尚无定论**：当前科学证据不足以确认真伪，尚待进一步研究
3. **确实如此**：经核查证实内容属实

## 背景
知识库检索到了一些弱相关记录（相似度偏低），仅供语义参考。请综合以下信息给出分类倾向：

知识库弱匹配参考：
{evidence_block}

## 任务
1. 分析待分类文本的语义特征和传播意图
2. 结合弱匹配参考和你的知识，给出**分类倾向**（非最终裁决）
3. 说明不确定性来源
4. 建议人工核查方向（搜索关键词或可查证的事实点）

## 输出格式
仅输出 JSON，不要包含其他内容：
```json
{{
  "label": "不实信息 或 尚无定论 或 确实如此",
  "confidence": 0.0-1.0,
  "reasoning": "语义分析理由",
  "uncertainty": "不确定性来源说明",
  "suggested_verification": "建议核查方向/搜索关键词"
}}
```"""


# 向后兼容: 保留旧名称
PROMPT_WITHOUT_EVIDENCE = PROMPT_LOW_CONFIDENCE


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
