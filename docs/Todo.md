建议
  build_kb.py

  1. weibo 标签设为"待分类"值得商榷（第127行）。rumor_weibo
  本身就是谣言投诉平台的数据，所有记录都经过平台审核判定为不实，prepare_data.py 中已统一标记为
  "不实信息"。建议与 prepare_data.py 保持一致，直接标为 "不实信息" 而非 "待分类"——否则知识库中 273
  条 weibo 记录的标签为"待分类"，RAG 检索命中后无法直接提供标签参考。
  2. clean_text() 与 prepare_data.py 的 normalize_text() 功能重复但实现不同。normalize_text 做了
  HTML unescape、特殊空白字符处理、标点整理等，clean_text 只做了基础的标签移除和空白合并。建议复用
  normalize_text 或提取为共享模块，保证知识库和训练集的文本清洗一致。
  3. 去重的 Jaccard 相似度是字符级（character-level）而非词级（第153行
  set(text)）。对中文来说，单字集合的 Jaccard
  区分度较低——两条讲不同话题但用词相似的谣言可能被误去重。建议改为基于 jieba
  分词后的词级集合，或降低阈值到 0.90。
  4. kb_id 使用行号 fact_{idx:03d}（第79行）。如果 fact.json 的行序发生变化，同一条记录的 kb_id
  就会改变，导致与向量索引不一致。建议用内容的 hash 或 title 字段生成更稳定的 ID。

  build_vector_index.py

  5. kb_records.json 是 serving_rumor_KB.json 
  的完整副本（第122行）。两份相同数据增加了维护负担，修改一处容易忘记另一处。建议 rag_retriever.py
  直接读取 output/serving_rumor_KB.json，不再在 vector_store/ 下保存副本。
  6. metadata.json 只存了 kb_id、label、source（第108-116行），检索时还是要加载完整 kb_records.json
  才能拿到 rumor_text 和 evidence。可以考虑去掉 metadata 中的 records
  列表，只保留索引配置信息（model_name、dimension 等），简化数据流。

  rag_retriever.py

  7. search_with_decision() 的阈值硬编码（0.90 /
  0.75）。这些阈值在实际运行后可能需要调整。建议提升为 __init__ 参数或配置项，方便实验调参。
  8. __init__ 中的 print 输出（第84-87行）在作为模块被 import 时会产生不期望的终端输出。建议改用
  logging 模块，或加一个 verbose 参数控制。

  README_KB_GUIDE.md

  9. 依赖中提到 langchain-openai 和 openai（第29行），但项目用的是 Claude/Anthropic SDK，不需要
  OpenAI 的包。应改为 anthropic 和 langchain-anthropic（如果用 langchain 的话）。
  10. 文件位置：这个文件放在项目根目录，但作为文档更适合放在 docs/ 下，与
  data-schema.md、label_policy.md 保持一致。