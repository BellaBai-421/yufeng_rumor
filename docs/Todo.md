# 代码审查改进记录

## 已完成的修改（2026-04-28）

### P0 — Bug 修复 + 默认规则切换
- [x] 修复 `evaluate.py` legacy 处罚评估 bug（`pred_pun.get("level")` 在 legacy 模式永远为 None）
- [x] 将 `--rule-source` 默认值从 `legacy` 改为 `mined`（rumor_agent.py + evaluate.py）

### P1 — 消除重复代码
- [x] `rag_retriever.py` 阈值从 config 导入，不再硬编码
- [x] `mine_punishment_rules.py` 从 config 导入 `normalize_punishment_result` 和 `PUNISHMENT_LEVEL_DETAILS`
- [x] `build_kb.py` 从 `scripts.prepare_data` 导入 `LABEL_MAP`
- [x] 删除未使用的 `CLS_KEYS` 和 `PUN_KEYS`（prepare_data.py）
- [x] 删除未使用的 `normalize_punishment_result` 导入（punishment_retriever.py）

### P2 — 清理
- [x] 删除 config.py 中未使用的 `severe` 字段，加注释说明

### 不改动（设计决策）
- `PROMPT_WITHOUT_EVIDENCE` 别名保留（向后兼容）
- `RetrievalResult` 冗余字段保留（未来可能展示给用户）
- `match_level`/`suggestion` 保留（CLI 测试入口在用）
- 低置信/no_rag 分支重复逻辑不合并（保持可读性）

## 设计决策记录

### 为什么 mined 规则设为默认？
现行 legacy 规则仅按转发数分 3 档扣分，与实际处罚结果严重不符（forward=409 仅扣 2 分，forward=0 可扣 20 分）。
mined 规则基于"同话题同处罚"的内容匹配，6 档处罚等级，准确率显著优于 legacy。

### 为什么 mine_punishment_rules.py 从 config.py 导入？
config.py 是共享常量和纯函数层（不依赖网络/文件路径/pipeline），mine 脚本导入它不产生运行时耦合。
保持单一来源，避免两处维护同一逻辑。
