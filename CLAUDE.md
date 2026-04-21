# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

NLP 课程作业：基于中文新冠谣言数据集，构建一个**谣言核查智能体**。使用 RAG（检索增强生成）对辟谣知识库进行检索，结合 LLM 推理来识别和分类谣言。

## 数据集

- `fact.json` — 124 条人工核查记录（每行一个 JSON），字段包括 `rumor`（谣言原文）、`abstract`（辟谣说明）、`explain`（类别标签）、`tag`、`title`、`date`
- `rumor_weibo/` — 325 个 JSON 文件，微博平台的谣言投诉处理记录，以 `rumorCode` 为唯一标识
- `rumor_forward_comment/` — 266 个 JSON 文件，每条谣言的转发与评论互动数据（注意：目录中包含一个 `count.py` 统计脚本，非数据文件）
- 完整字段文档见 `docs/data-schema.md`

## 关键设计决策

- **无需 PyTorch / 无需 GPU** — 通过 API 调用 Claude + 本地轻量嵌入模型实现
- **技术栈**：Python、`anthropic` SDK（tool use）、`sentence-transformers`（文本嵌入）、`faiss-cpu`（向量检索）
- **架构**：RAG + Tool-use 智能体 — 对 `fact.json` 做向量检索，从 `rumor_forward_comment` 提取传播特征，Claude 作为推理引擎

## 语言

所有数据为**中文**。嵌入模型使用支持中文的多语言模型（如 `paraphrase-multilingual-MiniLM-L12-v2`）。代码变量名使用英文，用户交互输出使用中文。
