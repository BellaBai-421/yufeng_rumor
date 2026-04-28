"""
Step 1: 从 fact.json 和 rumor_weibo/ 构建统一的谣言知识库
输出: output/serving_rumor_KB.json — 每条记录包含统一 schema

运行: python build_kb.py --fact data/fact.json --weibo data/rumor_weibo/ --output output/serving_rumor_KB.json
"""

import json
import sys
import re
import hashlib
import argparse
from pathlib import Path
from datetime import datetime

# 复用 prepare_data.py 的文本清洗函数，保证 KB 与训练集文本处理一致
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.prepare_data import normalize_text, clean_tail, LABEL_MAP

# 延迟导入 jieba（仅去重时使用）
try:
    import jieba
except ImportError:
    jieba = None


# ──────────────────────────────────────────────
# 1. 统一知识库 Schema（扁平化）
# ──────────────────────────────────────────────
# {
#   "kb_id":           str,   # 唯一ID, 格式 "fact_{hash}" 或 "weibo_{rumorCode}"
#   "source":          str,   # "fact" | "weibo"
#   "rumor_text":      str,   # 谣言原文
#   "label":           str,   # "不实信息" | "尚无定论" | "确实如此"
#   "evidence":        str,   # 辟谣证据/处理结果
#   "content":         str,   # rumor_text + evidence 拼接，供 embedding 使用
#   "date":            str,   # YYYY-MM-DD
#   "tags":            list,  # 关键词标签(仅 fact 有)
#   "title":           str,   # 标题摘要
#   "original_label":  str,   # 原始标签(explain原值)
#   "source_file":     str,   # 来源文件名
#   "rumormonger":     str,   # 发布谣言的用户名(仅 weibo)
#   "visit_times":     int,   # 浏览次数(仅 weibo)
# }

def make_content(rumor_text: str, evidence: str) -> str:
    """拼接 rumor_text + evidence 作为 embedding 输入文本。"""
    parts = [rumor_text]
    if evidence:
        parts.append(evidence)
    return " ".join(parts)


def content_hash(text: str, length: int = 12) -> str:
    """基于文本内容生成短 hash，用于稳定的 kb_id。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]


def process_fact_json(filepath: str) -> list[dict]:
    """处理 fact.json: 每行一个 JSON 对象"""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [WARN] fact.json 第{idx}行 JSON 解析失败, 跳过")
                continue

            rumor_text = normalize_text(item.get("rumor", ""))
            if not rumor_text:
                print(f"  [WARN] fact.json 第{idx}行 rumor 为空, 跳过")
                continue

            original_label = item.get("explain", "")
            label = LABEL_MAP.get(original_label, "不实信息")
            evidence = normalize_text(item.get("abstract", ""))
            kb_id = f"fact_{content_hash(rumor_text)}"

            records.append({
                "kb_id": kb_id,
                "source": "fact",
                "rumor_text": rumor_text,
                "label": label,
                "evidence": evidence,
                "content": make_content(rumor_text, evidence),
                "date": item.get("date", ""),
                "tags": item.get("tag", []),
                "title": normalize_text(item.get("title", "")),
                "original_label": original_label,
                "source_file": "fact.json",
                "rumormonger": "",
                "visit_times": 0,
            })

    return records


def process_weibo_dir(dirpath: str) -> list[dict]:
    """处理 rumor_weibo/ 目录: 每个文件一条记录"""
    records = []
    json_files = sorted(Path(dirpath).glob("*.json"))

    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                item = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"  [WARN] {fpath.name} 解析失败: {e}")
            continue

        rumor_text = normalize_text(item.get("rumorText", ""))
        if not rumor_text:
            continue

        evidence = normalize_text(item.get("result", ""))

        # 从文件名提取日期: YYYY-MM-DD_rumorCode.json
        date_match = re.match(r"(\d{4}-\d{2}-\d{2})", fpath.name)
        date_str = date_match.group(1) if date_match else item.get("publishTime", "")

        rumor_code = item.get("rumorCode", fpath.stem)

        records.append({
            "kb_id": f"weibo_{rumor_code}",
            "source": "weibo",
            "rumor_text": rumor_text,
            "label": "不实信息",
            "evidence": evidence,
            "content": make_content(rumor_text, evidence),
            "date": date_str,
            "tags": [],
            "title": normalize_text(item.get("title", "")),
            "original_label": "",
            "source_file": fpath.name,
            "rumormonger": item.get("rumormongerName", ""),
            "visit_times": item.get("visitTimes", 0),
        })

    return records


def deduplicate(records: list[dict], threshold: float = 0.90) -> list[dict]:
    """
    基于词级 Jaccard 相似度去重。
    优先使用 jieba 分词；若 jieba 不可用，回退到字符级。
    """
    use_jieba = jieba is not None
    if not use_jieba:
        print("  [INFO] jieba 未安装，回退到字符级去重 (pip install jieba)")

    def tokenize(text: str) -> set:
        if use_jieba:
            return set(jieba.lcut(text))
        return set(text)

    seen_tokens = []
    unique_records = []

    for rec in records:
        text = rec["rumor_text"]
        tok_set = tokenize(text)
        is_dup = False

        for seen_set in seen_tokens:
            if not tok_set or not seen_set:
                continue
            jaccard = len(tok_set & seen_set) / len(tok_set | seen_set)
            if jaccard >= threshold:
                is_dup = True
                break

        if not is_dup:
            seen_tokens.append(tok_set)
            unique_records.append(rec)

    return unique_records


def build_stats(records: list[dict]) -> dict:
    """生成知识库统计信息"""
    stats = {
        "total": len(records),
        "by_source": {},
        "by_label": {},
        "with_evidence": 0,
        "avg_rumor_length": 0,
        "build_time": datetime.now().isoformat(),
    }

    total_len = 0
    for rec in records:
        src = rec["source"]
        stats["by_source"][src] = stats["by_source"].get(src, 0) + 1
        lbl = rec["label"]
        stats["by_label"][lbl] = stats["by_label"].get(lbl, 0) + 1
        if rec["evidence"]:
            stats["with_evidence"] += 1
        total_len += len(rec["rumor_text"])

    if records:
        stats["avg_rumor_length"] = round(total_len / len(records), 1)

    return stats


def main():
    parser = argparse.ArgumentParser(description="构建谣言知识库")
    parser.add_argument("--fact", default="data/fact.json", help="fact.json 路径")
    parser.add_argument("--weibo", default="data/rumor_weibo/", help="rumor_weibo 目录")
    parser.add_argument("--output", default="output/serving_rumor_KB.json", help="输出路径")
    parser.add_argument("--dedup-threshold", type=float, default=0.90, help="去重阈值")
    parser.add_argument("--exclude-dev", default=None,
                        help="排除 dev 样本的文本列表路径 (JSON)，"
                             "由 prepare_data.py 生成")
    args = parser.parse_args()

    print("=" * 50)
    print("谣言知识库构建")
    print("=" * 50)

    # Step 1: 处理 fact.json
    print(f"\n[1/4] 处理 fact.json: {args.fact}")
    fact_records = process_fact_json(args.fact)
    print(f"  → 有效记录: {len(fact_records)} 条")

    # Step 2: 处理 rumor_weibo/
    print(f"\n[2/4] 处理 rumor_weibo/: {args.weibo}")
    weibo_records = process_weibo_dir(args.weibo)
    print(f"  → 有效记录: {len(weibo_records)} 条")

    # Step 3: 合并 + 去重
    all_records = fact_records + weibo_records
    print(f"\n[3/4] 合并后共 {len(all_records)} 条, 去重中 (阈值={args.dedup_threshold})...")
    unique_records = deduplicate(all_records, threshold=args.dedup_threshold)
    removed = len(all_records) - len(unique_records)
    print(f"  → 去重后: {len(unique_records)} 条 (移除 {removed} 条)")

    # Step 3.5: 排除 dev 样本（防止数据泄露）
    if args.exclude_dev:
        print(f"\n[3.5/4] 排除 dev 样本: {args.exclude_dev}")
        with open(args.exclude_dev, "r", encoding="utf-8") as f:
            dev_texts = set(json.load(f))
        before_exclude = len(unique_records)
        unique_records = [r for r in unique_records
                          if r["rumor_text"] not in dev_texts]
        excluded = before_exclude - len(unique_records)
        print(f"  → 排除 {excluded} 条 dev 样本, 剩余 {len(unique_records)} 条")

    # Step 4: 输出
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(unique_records, f, ensure_ascii=False, indent=2)

    # 输出统计
    stats = build_stats(unique_records)
    stats_path = output_path.parent / "kb_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n[4/4] 输出完成:")
    print(f"  → 知识库: {output_path}")
    print(f"  → 统计:   {stats_path}")
    print(f"\n{'=' * 50}")
    print("统计摘要:")
    print(f"  总记录数:     {stats['total']}")
    for src, cnt in stats["by_source"].items():
        print(f"  来源 {src}:  {cnt}")
    for lbl, cnt in stats["by_label"].items():
        print(f"  标签 [{lbl}]: {cnt}")
    print(f"  有证据记录:   {stats['with_evidence']}")
    print(f"  平均谣言长度: {stats['avg_rumor_length']} 字")
    print("=" * 50)


if __name__ == "__main__":
    main()
