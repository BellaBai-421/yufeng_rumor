# 疫情谣言分类与处罚判断项目

## 项目目标

基于中文新冠谣言数据集，构建"谣言分类 + 处罚判断"系统。

**方案 A：RAG + 规则引擎 + LLM 辅助**

- RAG 和规则库为主：高置信度直接采用 KB 标签，规则引擎直接计算处罚
- LLM 仅辅助判断中/低置信度的模糊语义
- 不使用 langchain；LLM 后端切换为 DeepSeek API（OpenAI-compatible）

## 数据流水线

```
原始数据 (fact.json + rumor_weibo/)
  ↓
prepare_data.py — 在原始数据层面统一划分 train/dev
  ├─ 输出 train/dev 数据集 (classification/ + punishment/)
  └─ 输出 dev_rumor_texts.json（dev 样本文本列表）
  ↓
build_kb.py --exclude-dev — 仅用 train 数据构建知识库
  ↓
build_vector_index.py — 向量化 + FAISS 索引（仅含 train）
  ↓
evaluate.py — 在 dev 集上评估（KB 中无 dev 样本）
```

## Agent Pipeline

```
用户输入（单条文本 / JSON 文件）
  ↓
意图识别（规则式）→ 确定模式
  ├─ classify_only         单条分类
  ├─ classify_and_punish   单条分类 + 判罚
  ├─ batch_classify        批量分类
  └─ batch_classify_punish 批量分类 + 判罚
  ↓
对每条 rumorText 执行 Case Pipeline
  ↓
文本清洗
  ↓
RAG 检索 TopK
  ↓
置信度门控
  ├─ 高置信（score ≥ 0.90）：直接采用 KB 标签，跳过 LLM
  ├─ 中置信（0.75 ≤ score < 0.90）：LLM 基于证据受限分类
  └─ 低置信（score < 0.75）：LLM 基于弱匹配参考判断，标注低置信
  ↓
得到 label + confidence + evidence
  ↓
是否需要判罚？
  ├─ 否：输出分类结果
  └─ 是：内容匹配已判罚案例 → 6 档处罚等级
  ↓
结构化 JSON 输出（含 trace）
```

## 数据说明

| 数据源 | 路径 | 记录数 | 说明 |
|--------|------|--------|------|
| 辟谣知识库 | `data/fact.json` | 124 | 人工核查结果，含谣言原文、辟谣说明、类别标签 |
| 微博谣言投诉 | `data/rumor_weibo/` | 324 文件 | 平台投诉处理记录，273 条有 rumorText |
| 转发评论数据 | `data/rumor_forward_comment/` | 266 文件 | 每条谣言的转发与评论互动 |

完整字段文档见 `docs/data-schema.md`。

## 标签映射

fact.json 的 `explain` 字段映射为三分类标签（详见 `docs/label_policy.md`）：

| 原始值 | 映射标签 |
|--------|----------|
| 伪科学 | 不实信息 |
| 伪常识 | 不实信息 |
| 尚无定论 | 尚无定论 |
| 确实如此 | 确实如此 |

## 处罚判断任务

根据谣言分类结果，通过 embedding 相似度匹配训练数据中最相似的已判罚案例，复用其处罚等级（"同话题同处罚"）：

- **不实信息**：6 档处罚等级（L1-L6，含扣分+禁言天数）
- **尚无定论**：不扣分，标记观察并提醒
- **确实如此**：不扣分，不处罚

## 目录结构

```
rumor/
├── CLAUDE.md                 # 本文件
├── README_KB_GUIDE.md        # 知识库构建指南
│
├── # ── 数据准备与知识库构建 ──
├── build_kb.py               # Step 1: 构建统一知识库（支持 --exclude-dev）
├── build_vector_index.py     # Step 2: 向量化 + FAISS 索引
├── rag_retriever.py          # Step 3: RAG 检索器（只读）
│
├── # ── Agent 层 ──
├── config.py                 # 配置常量、扣分规则解析、DeepSeek client 工厂
├── prompts.py                # LLM 分类提示词（有/无证据两套）
├── pipeline.py               # 确定性 Case Pipeline（RAG → 门控 → LLM → 判罚）
├── rumor_agent.py            # 主 Agent（4模式入口 + 交互式菜单）
├── evaluate.py               # 评估脚本（含 --no-rag 基线对比 + trace 统计）
├── punishment_retriever.py   # 挖掘规则判罚检索器
├── abnormal_text_check.py    # 判罚训练数据异常文本检查
│
├── data/                     # 原始数据（gitignore）
│   ├── fact.json
│   ├── rumor_weibo/
│   └── rumor_forward_comment/
├── docs/
│   ├── data-schema.md        # 数据字段文档
│   ├── label_policy.md       # 标签映射规则
│   └── Todo.md               # 待办与设计决策
├── rules/
│   └── mined_rules.json         # 判罚规则（从训练数据挖掘）
├── scripts/
│   ├── prepare_data.py          # 数据划分 + 数据集生成（防泄露：先分后建）
│   ├── mine_punishment_rules.py # 判罚规则挖掘脚本
│   └── summarize_experiments.py # 实验结果汇总（跨 seed 聚合 mean±std）
└── output/                   # 脚本输出（gitignore）
    ├── serving_rumor_KB.json  # 统一知识库（仅含训练集数据）
    ├── kb_stats.json          # 知识库统计
    ├── split/                 # 数据划分信息
    │   └── dev_rumor_texts.json  # dev 样本文本（供 build_kb 排除）
    ├── classification/        # 谣言分类数据集
    │   ├── train.json
    │   ├── dev.json
    │   └── stats.json
    ├── punishment/            # 判罚数据集
    │   ├── train.json
    │   ├── dev.json
    │   └── stats.json
    ├── vector_store/          # 向量索引
    │   ├── index.faiss
    │   └── metadata.json
    └── experiments/           # 多 seed 实验结果
        ├── seed_*/            # 各 seed 的评估输出
        └── summary.json       # 跨 seed 汇总
```

## 技术栈

- **无需 PyTorch / 无需 GPU**
- Python、`openai` SDK（DeepSeek OpenAI-compatible API）、`sentence-transformers`（文本嵌入）、`faiss-cpu`（向量检索）
- LLM：DeepSeek API（`base_url="https://api.deepseek.com"`，模型 `deepseek-chat`）
- 嵌入模型：支持中文的多语言模型（如 `BAAI/bge-base-zh-v1.5`）
- 环境变量：`DEEPSEEK_API_KEY`（替代原 `ANTHROPIC_API_KEY`）

## 常用命令

```bash
# Step 0: 划分数据集（必须先于 build_kb，防止数据泄露）
python scripts/prepare_data.py
python scripts/prepare_data.py --task cls   # 仅分类
python scripts/prepare_data.py --task pun   # 仅判罚

# Step 1-2: 构建知识库 + 向量索引（排除 dev 样本）
python build_kb.py --exclude-dev output/split/dev_rumor_texts.json
python build_vector_index.py

# 测试检索
python rag_retriever.py --query "吃大蒜可以预防新冠病毒" --top-k 3

# Agent 交互式菜单
python rumor_agent.py

# Agent 命令行模式
python rumor_agent.py --mode classify_only --query "吃大蒜可以预防新冠病毒"
python rumor_agent.py --mode classify_and_punish --query "..."
python rumor_agent.py --mode batch_classify --input output/classification/dev.json
python rumor_agent.py --mode batch_classify_punish --input output/punishment/dev.json
python rumor_agent.py --mode classify_only --query "..." --no-rag  # 纯 LLM 基线

# 评估（RAG vs no-RAG 对比）
python evaluate.py --task cls
python evaluate.py --task cls --no-rag
python evaluate.py --task pun
python evaluate.py --task cls --limit 5    # 调试用
python evaluate.py --task cls --output custom_result.json  # 自定义输出路径

# 判罚规则挖掘
python scripts/mine_punishment_rules.py

# 实验结果汇总（跨 seed 聚合）
python scripts/summarize_experiments.py --exp-dir output/experiments
```

## 语言

所有数据为**中文**。代码变量名使用英文，用户交互输出使用中文。
