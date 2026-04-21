import json
import pandas as pd
from pathlib import Path


def load_rumor_weibo(data_dir: str = "rumor_weibo") -> pd.DataFrame:
    """
    读取 rumor_weibo 目录下所有 JSON 文件，汇总为一个 DataFrame。

    列说明：
        rumorCode       — 谣言唯一标识
        title           — 举报标题
        informerName    — 举报人昵称
        informerUrl     — 举报人主页链接
        rumormongerName — 被举报人昵称（可为 null）
        rumormongerUrl  — 被举报人主页链接（可为 null）
        rumorText       — 谣言原文（可为 null）
        visitTimes      — 浏览次数
        result          — 处理结果说明
        publishTime     — 发布日期
        related_url     — 相关链接列表（保留原始 list[dict] 结构）
    """
    records = []
    for path in sorted(Path(data_dir).glob("*.json")):
        with open(path, encoding="utf-8") as f:
            records.append(json.load(f))

    wb_df = pd.DataFrame(records, columns=[
        "rumorCode", "title", "informerName", "informerUrl",
        "rumormongerName", "rumormongerUrl", "rumorText",
        "visitTimes", "result", "publishTime", "related_url",
    ])
    wb_df["publishTime"] = pd.to_datetime(wb_df["publishTime"], errors="coerce")
    return wb_df


def load_forward_comment(data_dir: str = "rumor_forward_comment") -> pd.DataFrame:
    """
    统计 rumor_forward_comment 目录下每个 JSON 文件的转发和评论次数。

    列说明：
        rumorCode      — 从文件名中提取（YYYY-MM-DD_<rumorCode>.json 的后缀部分）
        forward_count  — 转发次数
        comment_count  — 评论次数
    """
    rows = []
    for path in sorted(Path(data_dir).glob("*.json")):
        rumorCode = path.stem.split("_", 1)[1]  # 去掉日期前缀
        with open(path, encoding="utf-8") as f:
            interactions = json.load(f)

        forward_count = sum(1 for item in interactions if item.get("comment_or_forward") == "forward")
        comment_count = sum(1 for item in interactions if item.get("comment_or_forward") == "comment")
        rows.append({"rumorCode": rumorCode, "forward_count": forward_count, "comment_count": comment_count})

    fc_df = pd.DataFrame(rows, columns=["rumorCode", "forward_count", "comment_count"])
    return fc_df


if __name__ == "__main__":
    wb_df = load_rumor_weibo()
    print(f"wb_df：共 {len(wb_df)} 条记录，列：{list(wb_df.columns)}")
    print(wb_df.head())

    fc_df = load_forward_comment()
    print(f"\nfc_df：共 {len(fc_df)} 条记录，列：{list(fc_df.columns)}")
    print(fc_df.head())
