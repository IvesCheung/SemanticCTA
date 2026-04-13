"""
统计 JSON 文件的总 token 数量（使用 Qwen tokenizer）

用法:
    python count_tokens.py <json_file> [--model /model/qwen3-0.6B-embedding] [--verbose]

示例:
    python count_tokens.py output/xxx.json
    python count_tokens.py output/xxx.json --verbose
    python count_tokens.py output/xxx.json --model /path/to/other/model
"""

import argparse
import json
import sys
from pathlib import Path


DEFAULT_MODEL = "./model/qwen3-0.6B-embedding"


def load_tokenizer(model_path: str):
    from transformers import AutoTokenizer
    print(f"加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)
    return tokenizer


def count_tokens_in_text(text: str, tokenizer) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return len(ids)


def json_to_text(obj) -> str:
    """将任意 JSON 对象序列化为字符串（紧凑格式，减少无效空白）"""
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))


def count_file_tokens(json_path: str, model_path: str, verbose: bool = False) -> dict:
    path = Path(json_path)
    if not path.exists():
        print(f"错误: 文件不存在 -> {json_path}", file=sys.stderr)
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokenizer = load_tokenizer(model_path)

    # 整体 token 数
    full_text = json_to_text(data)
    total_tokens = count_tokens_in_text(full_text, tokenizer)
    file_size_kb = path.stat().st_size / 1024

    result = {
        "file": str(path),
        "file_size_kb": round(file_size_kb, 2),
        "total_tokens": total_tokens,
    }

    # 如果顶层是列表，额外统计每条记录
    if verbose and isinstance(data, list):
        result["num_records"] = len(data)
        per_record = []
        for i, item in enumerate(data):
            t = count_tokens_in_text(json_to_text(item), tokenizer)
            per_record.append(t)
        result["avg_tokens_per_record"] = round(
            sum(per_record) / len(per_record), 1) if per_record else 0
        result["max_tokens_per_record"] = max(per_record) if per_record else 0
        result["min_tokens_per_record"] = min(per_record) if per_record else 0

    # 如果顶层是 dict，统计每个 key 的 token 数
    if verbose and isinstance(data, dict):
        key_tokens = {}
        for k, v in data.items():
            t = count_tokens_in_text(json_to_text(v), tokenizer)
            key_tokens[k] = t
        result["tokens_per_key"] = key_tokens

    return result


def main():
    parser = argparse.ArgumentParser(
        description="统计 JSON 文件的 token 数量（Qwen tokenizer）")
    parser.add_argument("json_file", help="JSON 文件路径")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"tokenizer 路径（默认: {DEFAULT_MODEL}）")
    parser.add_argument("--verbose", action="store_true",
                        help="显示详细统计（按记录/key 分析）")
    args = parser.parse_args()

    result = count_file_tokens(
        args.json_file, args.model, verbose=args.verbose)

    print("\n======== Token 统计结果 ========")
    print(f"文件      : {result['file']}")
    print(f"文件大小  : {result['file_size_kb']} KB")
    print(f"总 token 数: {result['total_tokens']:,}")

    if "num_records" in result:
        print(f"记录条数  : {result['num_records']:,}")
        print(f"平均 token: {result['avg_tokens_per_record']}")
        print(f"最大 token: {result['max_tokens_per_record']:,}")
        print(f"最小 token: {result['min_tokens_per_record']:,}")

    if "tokens_per_key" in result:
        print("各 key token 数:")
        for k, v in result["tokens_per_key"].items():
            print(f"  {k}: {v:,}")

    print("================================\n")


if __name__ == "__main__":
    main()
