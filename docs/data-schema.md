# 数据说明

## 目录结构

```
rumor/
├── fact.json                  # 辟谣知识库（每行一个JSON）
├── rumor_weibo/               # 微博谣言投诉记录（每条谣言一个JSON文件）
└── rumor_forward_comment/     # 谣言传播行为数据（每条谣言一个JSON文件）
```

---

## fact.json

辟谣机构的人工核查结果，共 124 条，每行一个 JSON 对象。

| 字段 | 类型 | 说明 |
|------|------|------|
| `date` | string | 核查日期，格式 `YYYY-MM-DD` |
| `explain` | string | 谣言类型标签，见下方枚举 |
| `tag` | list[string] | 关键词标签 |
| `rumor` | string | 谣言原文 |
| `abstract` | string | 事实核查说明（辟谣内容） |
| `title` | string | 谣言标题摘要 |

**`explain` 枚举值：**

| 值 | 含义 | 数量 |
|----|------|------|
| `伪科学` | 内容与科学事实相悖 | 61 |
| `尚无定论` | 目前科学界尚无定论 | 43 |
| `确实如此` | 内容属实，非谣言 | 19 |
| `伪常识` | 错误的常识性说法 | 1 |

---

## rumor_weibo/

微博平台的谣言投诉处理记录，共 324 个 JSON 文件，文件名格式为 `YYYY-MM-DD_<rumorCode>.json`。其中 `rumorText` 非空的有效记录共 **273 条**，另有 51 条 `rumorText` 为空。

| 字段 | 类型 | 说明 |
|------|------|------|
| `rumorCode` | string | 谣言唯一ID，与文件名后缀对应 |
| `title` | string | 投诉标题 |
| `rumorText` | string | 谣言正文内容 |
| `rumormongerName` | string | 发布谣言的用户名 |
| `rumormongerUrl` | string | 发布谣言的微博链接 |
| `informerName` | string | 举报人用户名 |
| `informerUrl` | string | 举报人主页链接 |
| `visitTimes` | int | 浏览次数 |
| `result` | string | 平台处理结果（含辟谣说明和处罚措施） |
| `publishTime` | string | 谣言发布日期，格式 `YYYY-MM-DD` |
| `related_url` | list[object] | 相关链接列表，每项含 `url` 和 `text` |

---

## rumor_forward_comment/

谣言微博的转发与评论记录，共 266 个 JSON 文件（另有 `count.py` 统计脚本一并存放于此），文件名与 `rumor_weibo/` 对应。每个文件是一个 JSON 数组。

| 字段 | 类型 | 说明 |
|------|------|------|
| `uid` | string | 用户ID |
| `text` | string | 转发或评论内容 |
| `date` | string | 时间，格式 `M月D日 HH:MM` |
| `comment_or_forward` | string | `comment`（评论）或 `forward`（转发） |

**规模统计：**
- 总互动记录：39,196 条
- 平均每条谣言：147.4 条互动
