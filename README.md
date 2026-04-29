# 疫情谣言分类与处罚判断系统

基于 RAG + 规则引擎 + LLM 辅助的中文新冠疫情谣言三分类系统，支持谣言真伪判定及微博信用积分处罚计算。

## 系统架构

```
用户输入（单条 / 批量 JSON）
  ↓
规则式意图识别 → 4 种模式
  ↓
对每条谣言执行确定性 Case Pipeline：
  ↓
文本清洗 → RAG 检索 TopK → 置信度门控
  ├─ 高置信（≥ 0.90）：直接采用知识库标签，跳过 LLM
  ├─ 中置信（0.75~0.90）：LLM 基于检索证据受限分类
  └─ 低置信（< 0.75）：LLM 基于弱匹配参考判断，标注低置信
  ↓
内容匹配判罚（按需，基于已判罚案例匹配） → 结构化 JSON 输出
```

核心设计：**RAG 和规则引擎为主，LLM 仅辅助模糊语义判断**。高置信度命中时完全不调用 LLM，降低延迟和成本。

## 三分类标签

| 标签 | 含义 | 处罚 |
|------|------|------|
| 不实信息 | 经核查证实为虚假/伪科学/伪常识 | 按内容匹配已判罚案例，6 档处罚等级（含扣分+禁言） |
| 尚无定论 | 当前科学证据不足，尚待进一步研究 | 不扣分，标记观察 |
| 确实如此 | 经核查证实内容属实 | 不扣分 |

## 数据来源

| 数据源 | 路径 | 记录数 | 说明 |
|--------|------|--------|------|
| 辟谣知识库 | `data/fact.json` | 124 | 人工核查结果，含 3 类标签 |
| 微博谣言投诉 | `data/rumor_weibo/` | 324 文件，273 条有 rumorText | 平台投诉处理记录，均为不实信息 |
| 转发评论数据 | `data/rumor_forward_comment/` | 266 文件 | 每条谣言的转发与评论互动 |

## 快速开始

### 1. 安装依赖

```bash
pip install openai sentence-transformers faiss-cpu
# 可选：pip install jieba  （用于知识库去重的词级分词）
```

### 2. 配置环境变量

```bash
export DEEPSEEK_API_KEY="your-api-key"
```

### 3. 构建数据流水线

按顺序执行（先划分数据集再建库，防止数据泄露）：

```bash
# Step 0: 划分 train/dev 数据集
python scripts/prepare_data.py

# Step 1: 构建知识库（排除 dev 样本）
python build_kb.py --exclude-dev output/split/dev_rumor_texts.json

# Step 2: 构建向量索引
python build_vector_index.py
```

### 4. 使用系统

**交互式菜单：**

```bash
python rumor_agent.py
```

**命令行模式：**

```bash
# 单条分类
python rumor_agent.py --mode classify_only --query "吃大蒜可以预防新冠病毒"

# 单条分类 + 判罚
python rumor_agent.py --mode classify_and_punish --query "吃大蒜可以预防新冠"

# 批量分类
python rumor_agent.py --mode batch_classify --input output/classification/dev.json

# 批量分类 + 判罚
python rumor_agent.py --mode batch_classify_punish --input output/punishment/dev.json

# 纯 LLM 基线（不使用 RAG）
python rumor_agent.py --mode classify_only --query "..." --no-rag
```

**作为模块调用：**

```python
from rumor_agent import RumorAgent

agent = RumorAgent()  # 会加载 embedding 模型 + 判罚训练数据
result = agent.classify("吃大蒜可以预防新冠病毒", need_punishment=True)
print(result["label"])       # "不实信息"
print(result["confidence"])  # 0.95
print(result["punishment"])  # 处罚详情（含等级、扣分、禁言天数）
```

### 5. 评估

```bash
# 分类评估（RAG 模式）
python evaluate.py --task cls

# 分类评估（纯 LLM 基线）
python evaluate.py --task cls --no-rag

# 处罚评估
python evaluate.py --task pun

# 调试（限制条数）
python evaluate.py --task cls --limit 5
```

## 输出格式

```json
{
  "label": "不实信息",
  "confidence": 0.9523,
  "reasoning": "知识库高度匹配 (score=0.952)，直接采用 KB 标签。",
  "trace": {
    "rag_top1_score": 0.9523,
    "rag_top1_kb_id": "fact_a1b2c3d4e5f6",
    "rag_top1_label": "不实信息",
    "gate_decision": "high_confidence",
    "llm_called": false
  },
  "punishment": {
    "deduction": 10,
    "action": "扣除信用积分10分，禁言15天，禁被关注15天",
    "level": 3,
    "ban_days": 15,
    "match_score": 0.9523,
    "match_text": "吃大蒜可以杀灭新冠病毒..."
  }
}
```

## 防数据泄露设计

```
原始数据 → prepare_data.py 统一划分 train/dev
                ├─ 输出 train/dev 数据集
                └─ 输出 dev_rumor_texts.json
           ↓
build_kb.py --exclude-dev → 仅用 train 构建知识库
           ↓
build_vector_index.py → 向量索引仅含 train 数据
           ↓
evaluate.py → dev 集评估，KB 中无 dev 样本
```

## 项目结构

```
rumor/
├── build_kb.py               # 统一知识库构建
├── build_vector_index.py     # FAISS 向量索引构建
├── rag_retriever.py          # RAG 检索器
├── config.py                 # 配置常量 + DeepSeek client
├── prompts.py                # LLM 分类提示词
├── pipeline.py               # 确定性 Case Pipeline
├── rumor_agent.py            # 主入口（4 模式 + 交互菜单）
├── evaluate.py               # 评估脚本
├── punishment_retriever.py   # 挖掘规则判罚检索器
├── abnormal_text_check.py   # 判罚训练数据异常文本检查
├── rules/
│   └── mined_rules.json         # 判罚规则（从训练数据挖掘）
├── scripts/
│   ├── prepare_data.py          # 数据划分 + 数据集生成
│   ├── mine_punishment_rules.py # 判罚规则挖掘脚本
│   └── summarize_experiments.py # 实验结果汇总（跨 seed 聚合）
└── docs/                     # 数据字段文档 + 标签映射规则
    ├── data-schema.md        # 数据说明
    └── label_policy.md       # 标签映射说明
```

## 技术栈

- **Python**（无需 PyTorch / 无需 GPU）
- **DeepSeek API**（OpenAI-compatible，模型 `deepseek-chat`）
- **sentence-transformers**（`BAAI/bge-base-zh-v1.5`，中文检索专项嵌入）
- **faiss-cpu**（FlatIP 精确余弦检索）
