#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一数据准备脚本：生成谣言分类数据集 + 判罚数据集。

**防数据泄露**：先在原始数据层面统一划分 train/dev，再分别构建
分类和判罚数据集。同时输出 dev 谣言文本列表，供 build_kb.py
排除 dev 样本，确保知识库仅含训练集数据。

输出目录结构：
    output/
    ├── split/
    │   └── dev_rumor_texts.json # dev 样本谣言文本（供 build_kb 排除）
    ├── classification/          # 谣言分类数据集
    │   ├── train.json
    │   ├── dev.json
    │   └── stats.json
    └── punishment/              # 判罚数据集
        ├── train.json
        ├── dev.json
        └── stats.json

用法：
    python scripts/prepare_data.py                 # 生成全部
    python scripts/prepare_data.py --task cls       # 仅生成分类数据集
    python scripts/prepare_data.py --task pun       # 仅生成判罚数据集
"""

import argparse
import html
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# ── 路径 ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# ── 标签映射 ──────────────────────────────────────────────────────────────
LABEL_MAP = {
    "伪科学": "不实信息",
    "伪常识": "不实信息",
    "尚无定论": "尚无定论",
    "确实如此": "确实如此",
}

# ── 正则 ──────────────────────────────────────────────────────────────────
HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTI_SPACE_RE = re.compile(r"\s+")
REPEATED_PUNCT_RE = re.compile(r"([。！？；，、,.!?;:：])\1+")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([。！？；，、,.!?;:：])")
URL_RE = re.compile(r"https?://[^\s\u3000\"'<>]+", re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════════════════
#  通用工具
# ═══════════════════════════════════════════════════════════════════════════

def normalize_text(text) -> str:
    """HTML 反转义 → 去标签 → 去特殊空白 → 合并空格 → 整理标点。"""
    if text is None:
        return ""
    text = str(text)
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = HTML_TAG_RE.sub(" ", text)
    for ch in "\u00a0\u200b\u200c\u200d\ufeff":
        text = text.replace(ch, " " if ch == "\u00a0" else "")
    text = text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = REPEATED_PUNCT_RE.sub(r"\1", text)
    return text.strip()


def clean_tail(text: str) -> str:
    """去掉末尾残留的'详情：'和悬空标点。"""
    text = re.sub(r"详情[：:]\s*$", "", text).strip()
    text = re.sub(r"[，,；;、:：]\s*$", "", text).strip()
    return REPEATED_PUNCT_RE.sub(r"\1", text)


def safe_int(value, default=0) -> int:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).replace(",", "")
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else default


def read_jsonl(path: Path) -> List[Dict]:
    """读 JSONL（每行一个 JSON）。"""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_single_json(path: Path):
    """读单个 JSON 文件（对象或数组）。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── 数据集划分 ────────────────────────────────────────────────────────────

def stratified_split(records, label_key, dev_ratio=0.1, seed=42):
    """按标签分层 9:1 划分。"""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in records:
        groups[r[label_key]].append(r)

    train, dev = [], []
    for label, items in groups.items():
        items = items[:]
        rng.shuffle(items)
        dev_n = max(1, round(len(items) * dev_ratio))
        if dev_n >= len(items):
            dev_n = len(items) - 1
        dev.extend(items[:dev_n])
        train.extend(items[dev_n:])

    rng.shuffle(train)
    rng.shuffle(dev)
    return train, dev


def random_split(records, dev_ratio=0.1, seed=42):
    rng = random.Random(seed)
    items = records[:]
    rng.shuffle(items)
    if not items:
        return [], []
    dev_n = max(1, round(len(items) * dev_ratio))
    if dev_n >= len(items):
        dev_n = len(items) - 1
    return items[dev_n:], items[:dev_n]


# ── 写出 ──────────────────────────────────────────────────────────────────

def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════════════════════════

def load_fact(path: Path = DATA_DIR / "fact.json") -> List[Dict]:
    """读取 fact.json，返回原始字典列表。"""
    return read_jsonl(path)


def load_rumor_weibo(data_dir: Path = DATA_DIR / "rumor_weibo") -> List[Dict]:
    """读取 rumor_weibo/ 下所有 JSON，剔除 rumorText 为空的记录。
    从文件名提取日期（格式：YYYY-MM-DD_rumorCode.json）存入 _file_date。"""
    records = []
    for p in sorted(data_dir.glob("*.json")):
        obj = read_single_json(p)
        if obj.get("rumorText"):
            # 文件名格式: 2020-01-22_K1CaS7Qth660h.json
            obj["_file_date"] = p.stem.rsplit("_", 1)[0] if "_" in p.stem else ""
            records.append(obj)
    return records


def load_forward_comment_stats(
    data_dir: Path = DATA_DIR / "rumor_forward_comment",
) -> Dict[str, Dict[str, int]]:
    """统计每个 rumorCode 的转发/评论数，返回 {rumorCode: {forward, comment}}。"""
    stats = {}
    for p in sorted(data_dir.glob("*.json")):
        code = p.stem.split("_", 1)[1]
        items = read_single_json(p)
        fwd = sum(1 for i in items if i.get("comment_or_forward") == "forward")
        cmt = sum(1 for i in items if i.get("comment_or_forward") == "comment")
        stats[code] = {"forward": fwd, "comment": cmt}
    return stats


# ═══════════════════════════════════════════════════════════════════════════
#  谣言分类数据集
# ═══════════════════════════════════════════════════════════════════════════

def extract_abstract_from_result(result: str) -> str:
    """从 result 提取"被投诉人言论"之前的部分，去掉 URL 和'详情：'。"""
    text = normalize_text(result)
    prefix = text.split("被投诉人言论", 1)[0]
    # 有些用"被举报人言论"
    prefix = prefix.split("被举报人言论", 1)[0]
    abstract = URL_RE.sub("", prefix)
    abstract = re.sub(r"详情[：:]\s*", "", abstract)
    return clean_tail(normalize_text(abstract))


def build_cls_fact_records(raw_facts: List[Dict]) -> List[Dict]:
    records = []
    for obj in raw_facts:
        rumor = normalize_text(obj.get("rumor", ""))
        if not rumor:
            continue
        records.append({
            "rumorText": rumor,
            "explain": LABEL_MAP.get(obj.get("explain", ""), obj.get("explain", "")),
            "abstract": normalize_text(obj.get("abstract", "")),
            "date": obj.get("date", ""),
        })
    return records


def build_cls_weibo_records(raw_weibos: List[Dict]) -> List[Dict]:
    records = []
    for obj in raw_weibos:
        records.append({
            "rumorText": normalize_text(obj["rumorText"]),
            "explain": "不实信息",
            "abstract": extract_abstract_from_result(obj.get("result", "")),
            "date": obj.get("_file_date", ""),
        })
    return records


def build_classification_dataset(fact_train_raw, fact_dev_raw,
                                  weibo_train_raw, weibo_dev_raw, seed=42):
    """从预划分的原始数据构建分类数据集（不再内部划分，防止数据泄露）。"""
    fact_train = build_cls_fact_records(fact_train_raw)
    fact_dev = build_cls_fact_records(fact_dev_raw)
    weibo_train = build_cls_weibo_records(weibo_train_raw)
    weibo_dev = build_cls_weibo_records(weibo_dev_raw)

    rng = random.Random(seed)
    train = fact_train + weibo_train
    dev = fact_dev + weibo_dev
    rng.shuffle(train)
    rng.shuffle(dev)

    out_dir = OUTPUT_DIR / "classification"
    write_json(out_dir / "train.json", train)
    write_json(out_dir / "dev.json", dev)

    stats = {
        "fact_total": len(fact_train) + len(fact_dev),
        "weibo_total": len(weibo_train) + len(weibo_dev),
        "fact_train": len(fact_train),
        "fact_dev": len(fact_dev),
        "weibo_train": len(weibo_train),
        "weibo_dev": len(weibo_dev),
        "train_total": len(train),
        "dev_total": len(dev),
        "train_label_dist": dict(Counter(r["explain"] for r in train)),
        "dev_label_dist": dict(Counter(r["explain"] for r in dev)),
    }
    write_json(out_dir / "stats.json", stats)

    print("=== 谣言分类数据集 ===")
    print(f"fact.json:    {len(fact_train)+len(fact_dev)} 条 → 训练 {len(fact_train)} / 验证 {len(fact_dev)}")
    print(f"rumor_weibo:  {len(weibo_train)+len(weibo_dev)} 条 → 训练 {len(weibo_train)} / 验证 {len(weibo_dev)}")
    print(f"合计:         训练 {len(train)} / 验证 {len(dev)}")
    print(f"训练集标签: {dict(Counter(r['explain'] for r in train))}")
    print(f"验证集标签: {dict(Counter(r['explain'] for r in dev))}")
    print(f"输出到 {out_dir}/\n")


# ═══════════════════════════════════════════════════════════════════════════
#  判罚数据集
# ═══════════════════════════════════════════════════════════════════════════

def extract_penalty_result(result: str) -> str:
    """从 result 提取处罚措施（'处理如下：'之后的部分）。"""
    text = normalize_text(result)
    if not text:
        return ""
    m = re.search(r"处理如下[：:]\s*(.*?)\s*[。.]?上述处理", text)
    if m:
        return normalize_text(m.group(1)).rstrip("。；;，,")
    m = re.search(r"处理如下[：:]\s*(.*)$", text)
    if m:
        tail = re.sub(r"上述处理.*$", "", m.group(1)).strip()
        return normalize_text(tail).rstrip("。；;，,")
    return text


def build_punishment_records(raw_weibos, fc_stats):
    records = []
    for obj in raw_weibos:
        code = obj["rumorCode"]
        fc = fc_stats.get(code, {"forward": 0, "comment": 0})
        records.append({
            "rumorCode": code,
            "rumormongerName": normalize_text(obj.get("rumormongerName", "")),
            "rumorText": normalize_text(obj["rumorText"]),
            "visitTimes": safe_int(obj.get("visitTimes", 0)),
            "explain": "不实信息",
            "result": extract_penalty_result(obj.get("result", "")),
            "publishTime": normalize_text(obj.get("publishTime", "")),
            "forward": fc["forward"],
            "comment": fc["comment"],
        })
    return records


def build_punishment_dataset(weibo_train_raw, weibo_dev_raw, fc_stats, seed=42):
    """从预划分的原始数据构建判罚数据集（不再内部划分，防止数据泄露）。"""
    train = build_punishment_records(weibo_train_raw, fc_stats)
    dev = build_punishment_records(weibo_dev_raw, fc_stats)

    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(dev)

    out_dir = OUTPUT_DIR / "punishment"
    write_json(out_dir / "train.json", train)
    write_json(out_dir / "dev.json", dev)

    total = len(train) + len(dev)
    stats = {
        "total": total,
        "train": len(train),
        "dev": len(dev),
        "result_dist": dict(Counter(r["result"] for r in train + dev)),
    }
    write_json(out_dir / "stats.json", stats)

    print("=== 判罚数据集 ===")
    print(f"总计 {total} 条 → 训练 {len(train)} / 验证 {len(dev)}")
    print(f"处罚分布: {dict(Counter(r['result'] for r in train + dev))}")
    print(f"输出到 {out_dir}/\n")


# ═══════════════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="统一数据准备")
    parser.add_argument("--task", choices=["cls", "pun", "all"], default="all",
                        help="cls=分类, pun=判罚, all=全部（默认）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.1,
                        help="验证集比例（默认 0.1）")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出根目录（默认 output/）")
    args = parser.parse_args()

    # 覆盖全局 OUTPUT_DIR
    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载原始数据
    raw_facts = load_fact()
    raw_weibos = load_rumor_weibo()
    fc_stats = load_forward_comment_stats()

    print(f"已加载: fact.json {len(raw_facts)} 条, "
          f"rumor_weibo {len(raw_weibos)} 条（已剔除空 rumorText）, "
          f"forward_comment {len(fc_stats)} 条\n")

    # ── 在原始数据层面统一划分 train/dev（防止数据泄露）──────
    # 过滤有效 fact 记录（rumor 非空）
    valid_facts = [f for f in raw_facts if normalize_text(f.get("rumor", ""))]

    fact_train_raw, fact_dev_raw = stratified_split(
        valid_facts, "explain", dev_ratio=args.dev_ratio, seed=args.seed,
    )
    weibo_train_raw, weibo_dev_raw = random_split(
        raw_weibos, dev_ratio=args.dev_ratio, seed=args.seed,
    )

    print(f"原始数据划分: fact {len(fact_train_raw)}+{len(fact_dev_raw)}, "
          f"weibo {len(weibo_train_raw)}+{len(weibo_dev_raw)}")

    # ── 保存 dev 谣言文本，供 build_kb.py --exclude-dev 使用 ──
    dev_rumor_texts = set()
    for f in fact_dev_raw:
        t = normalize_text(f.get("rumor", ""))
        if t:
            dev_rumor_texts.add(t)
    for w in weibo_dev_raw:
        t = normalize_text(w.get("rumorText", ""))
        if t:
            dev_rumor_texts.add(t)

    split_dir = OUTPUT_DIR / "split"
    write_json(split_dir / "dev_rumor_texts.json", sorted(dev_rumor_texts))
    print(f"Dev 样本文本已保存: {split_dir / 'dev_rumor_texts.json'} "
          f"({len(dev_rumor_texts)} 条)\n")

    if args.task in ("cls", "all"):
        build_classification_dataset(
            fact_train_raw, fact_dev_raw,
            weibo_train_raw, weibo_dev_raw,
            seed=args.seed,
        )

    if args.task in ("pun", "all"):
        build_punishment_dataset(
            weibo_train_raw, weibo_dev_raw,
            fc_stats, seed=args.seed,
        )


if __name__ == "__main__":
    main()
