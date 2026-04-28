Step 1：pipeline.py — LLM 输出校验埋点
- _parse_llm_json() 仍返回 dict，不改 tuple。
- dict 内增加 "_validation" 字段。
- 保留 raw_label、raw_confidence、raw_output[:500]。
- confidence 支持字符串和 0-100 自动归一化。
- invalid_label 不静默丢弃，记录 raw_label。
- _call_llm() 增加异常捕获，返回 llm_call_failed。
- trace 初始化时加入 llm_validation=None。
- LLM 分支写入 trace["llm_validation"]。
- RAG 分支写入 trace["rag_topk"] 和 trace["rag_topk_labels"]，labels 保留顺序，不用 set。

Step 2：evaluate.py — 修复基础指标
- compute_metrics() 增加 INVALID 预测列。
- per-class FN/support 包含 INVALID。
- 增加 invalid_prediction_count/rate。
- 增加 macro/micro/weighted precision/recall/F1。

Step 3：evaluate.py — Trace 和分支质量
- print_trace_summary() 返回 summary dict，而不是只 print。
- 增加每个 gate 的 count、accuracy、avg_confidence、llm_call_count。

Step 4：evaluate.py — RAG 标签级质量
- compute_rag_quality(details)
- 只统计有 rag_top1_label 的样本。
- 输出 rag_top1_label_accuracy、rag_topk_label_recall、topk_conflict_rate、high_confidence_accuracy。
- 不要先叫 nDCG，除非有 gold_kb_id 或 relevance grade。

Step 5：evaluate.py — 延迟 / 成本
- run_evaluation() 外层用 perf_counter 记录 latency_seconds。
- compute_latency_stats(details)：avg、p50、p95、max、total。
- 成本第一版只统计 llm_call_count / llm_call_rate。
- 有 response.usage 后再加 token 和 estimated_cost。

Step 6：evaluate.py — 错误案例分析
- 按 "真实→预测" 分组。
- 按 error_type 分组。
- 区分 false_positive_rumor、false_negative_rumor、unknown_prediction、rag_high_confidence_wrong、llm_wrong。

Step 7：evaluate.py — LLM 输出校对
- compute_llm_validation_stats(details)
- 聚合 json_parse_failed、invalid_label、invalid_confidence、missing_fields、llm_call_failed。
- 输出调用级 failure rate。

Step 8：evaluate.py — 处罚评估增强
- 保留 punishment_accuracy。
- 增加 punishment_mae、over_punishment、under_punishment。
