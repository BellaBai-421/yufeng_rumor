# 疫情谣言分类与处罚判断项目

## 项目目标

基于中文新冠谣言数据集，构建"谣言分类 + 处罚判断"系统，并对比两种方案：

| 方案 | 分类任务 | 处罚任务 |
|------|----------|----------|
| **方案 A：智能体+RAG** | RAG +  langchain 架构搭建智能体 | 同左 |

先完成谣言分类知识库RAG构建，再接智能体。跑通后再加入处罚判断系统。

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

根据谣言分类结果和传播特征（转发数），按 `rules/weibo_credit_rules.json` 中的规则判定信用扣分：

- **不实信息**：根据转发数梯度扣 10 / 15 / 20 / 30 分
- **尚无定论**：不扣分，标记观察并提醒
- **确实如此**：不扣分，不处罚

## 目录结构

```
rumor/
├── CLAUDE.md                 # 本文件
├── README_KB_GUIDE.md        # 知识库构建指南
├── build_kb.py               # Step 1: 构建统一知识库
├── build_vector_index.py     # Step 2: 向量化 + FAISS 索引
├── rag_retriever.py          # Step 3: RAG 检索器模块
├── data/                     # 原始数据（gitignore）
│   ├── fact.json
│   ├── rumor_weibo/
│   └── rumor_forward_comment/
├── docs/
│   ├── data-schema.md        # 数据字段文档
│   └── label_policy.md       # 标签映射规则
├── rules/
│   └── weibo_credit_rules.json  # 处罚扣分规则
├── scripts/
│   └── prepare_data.py       # 训练/验证数据集生成
└── output/                   # 脚本输出（gitignore）
    ├── serving_rumor_KB.json  # 统一知识库
    ├── kb_stats.json          # 知识库统计
    ├── classification/        # 谣言分类数据集
    │   ├── train.json
    │   ├── dev.json
    │   └── stats.json
    ├── punishment/            # 判罚数据集
    │   ├── train.json
    │   ├── dev.json
    │   └── stats.json
    └── vector_store/          # 向量索引
        ├── index.faiss
        ├── metadata.json
        └── kb_records.json
```

## 技术栈

- **无需 PyTorch / 无需 GPU**
- Python、`anthropic` SDK（tool use）、`sentence-transformers`（文本嵌入）、`faiss-cpu`（向量检索）
- 嵌入模型：支持中文的多语言模型（如 `paraphrase-multilingual-MiniLM-L12-v2`）

## 常用命令

```bash
# 生成训练/验证数据集
python scripts/prepare_data.py
python scripts/prepare_data.py --task cls   # 仅分类
python scripts/prepare_data.py --task pun   # 仅判罚

# 构建知识库 + 向量索引（方案 A）
python build_kb.py
python build_vector_index.py

# 测试检索
python rag_retriever.py --query "吃大蒜可以预防新冠病毒" --top-k 3
```

## 语言

所有数据为**中文**。代码变量名使用英文，用户交互输出使用中文。
