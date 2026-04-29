# 谣言风控 RAG 知识库 — 实施指南

## 目录结构 (知识库相关)

```
rumor/
├── build_kb.py                # Step 1: 清洗数据 → 统一知识库（支持 --exclude-dev）
├── build_vector_index.py      # Step 2: 向量化 → FAISS 索引
├── rag_retriever.py           # Step 3: 检索器模块 (Agent 调用此模块)
├── scripts/
│   └── prepare_data.py        # Step 0: 数据划分（必须先于 build_kb）
├── data/                      # 原始数据 (不修改)
│   ├── fact.json
│   ├── rumor_weibo/
│   └── rumor_forward_comment/
└── output/
    ├── serving_rumor_KB.json   # 统一知识库 (Step 1 输出，仅含训练集)
    ├── kb_stats.json           # 知识库统计 (Step 1 输出)
    ├── split/
    │   └── dev_rumor_texts.json  # dev 样本文本（供 build_kb 排除）
    └── vector_store/           # 向量存储 (Step 2 输出)
        ├── index.faiss
        └── metadata.json
```

## 执行步骤

### 前置依赖
```bash
pip install openai sentence-transformers faiss-cpu
# 可选：pip install jieba  （用于知识库去重的词级分词）
```

### Step 0: 划分数据集（防止数据泄露）
```bash
python scripts/prepare_data.py
```

预期输出:
- `output/split/dev_rumor_texts.json`: dev 样本文本列表
- `output/classification/`: 分类数据集 (train/dev)
- `output/punishment/`: 判罚数据集 (train/dev)

### Step 1: 构建知识库（排除 dev 样本）
```bash
python build_kb.py --exclude-dev output/split/dev_rumor_texts.json
# 可选参数: --fact <路径>  --weibo <路径>  --output <路径>  --dedup-threshold <值>
```

预期输出:
- `output/serving_rumor_KB.json`: 统一 schema 记录（去重 + 排除 dev 样本后）
- `output/kb_stats.json`: 统计信息

### Step 2: 构建向量索引
```bash
python build_vector_index.py
# 可选参数: --kb <路径>  --output-dir <路径>  --model <模型名>
```

首次运行会下载 embedding 模型 (~500MB). 预期输出:
- `output/vector_store/index.faiss`: FAISS 索引文件
- `output/vector_store/metadata.json`: 索引配置元数据

### Step 3: 测试检索
```bash
python rag_retriever.py --query "吃大蒜可以预防新冠病毒" --top-k 3
```

## 统一知识库 Schema

每条记录的字段说明（扁平化结构）:

| 字段 | 类型 | 说明 |
|------|------|------|
| kb_id | string | 唯一ID, "fact_{hash}" 或 "weibo_{rumorCode}" |
| source | string | 数据来源: "fact" 或 "weibo" |
| rumor_text | string | 谣言原文 (已清洗) |
| label | string | 分类标签: "不实信息" / "尚无定论" / "确实如此" |
| evidence | string | 辟谣证据或处理结果 |
| content | string | rumor_text + evidence 拼接, 供 embedding 使用 |
| date | string | 日期 YYYY-MM-DD |
| tags | list | 关键词标签 (仅 fact 数据有) |
| title | string | 标题摘要 |
| original_label | string | 原始标签 (fact.json explain 原值) |
| source_file | string | 来源文件名 |
| rumormonger | string | 发布谣言的用户名 (仅 weibo) |
| visit_times | int | 浏览次数 (仅 weibo) |

## 设计决策说明

### 为什么 embedding 使用 content 字段 (rumor_text + evidence)?
检索时 query 是一段谣言. 如果只编码 rumor_text, 向量只捕获"谣言说了什么";
加上 evidence 后, 向量同时编码"真相是什么", 提升语义区分度.
例如两条谣言文本相似但一个是伪科学一个确实如此, evidence 能帮助区分.

### 为什么 weibo 数据标签统一设为"不实信息"?
rumor_weibo 的所有记录均经平台审核判定为不实信息, 与 prepare_data.py 的处理保持一致.

### 为什么用 jieba 词级 Jaccard 去重?
Step 1 是纯数据工程, 不应该依赖模型. 词级 Jaccard 比字符级在中文场景下区分度更高,
能避免不同话题但用字相似的记录被误去重. 无 jieba 时自动回退到字符级.

### 为什么不在 vector_store 中保存知识库副本?
检索时直接读 `output/serving_rumor_KB.json`, 避免维护两份相同数据.
`vector_store/` 只存索引文件和配置元数据.
