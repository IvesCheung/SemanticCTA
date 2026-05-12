"""
从 SemanticCTA 模型中提取注意力权重，用于论文 Fig 2 的可视化。

该脚本提取最后一层注意力权重中，后缀（列描述触发点）对前缀（表格内容）的注意力分布，
并映射回源列（source columns），以分析列之间的依赖关系。
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None
    print("警告: 未安装 peft 库，将无法加载 LoRA 模型")


# ============== 常量定义 ==============

SYSTEM_PROMPT = """You are a table understanding assistant. Your task is to analyze table columns and determine their semantic types.

Consider the following when analyzing:
1. The column name and header
2. The data values in the column
3. Relationships with other columns
4. Context from the table name and description
5. Common patterns in semantic types

Provide your analysis in a clear and structured manner."""

SUFFIX_TEMPLATE = """Column '{column_name}' contains values: {sample_values}. What is the semantic type of this column? The semantic type is"""


# ============== 辅助函数 ==============

def get_basename(filepath: str) -> str:
    """获取文件基本名（去除扩展名）"""
    return Path(filepath).stem


def load_fold_table_ids(fold_dir: str, n_tables: int = 10) -> List[str]:
    """
    从 fold CSV 文件中加载前 n_tables 个唯一的 table_id

    Args:
        fold_dir: fold CSV 文件目录
        n_tables: 需要加载的表格数量

    Returns:
        table_id 列表
    """
    all_table_ids = set()

    # 遍历所有 fold 文件
    for fold_idx in range(5):
        fold_path = os.path.join(fold_dir, f"fold_{fold_idx}.csv")
        if not os.path.exists(fold_path):
            continue

        df = pd.read_csv(fold_path)
        if 'table_id' in df.columns:
            all_table_ids.update(df['table_id'].unique())

    # 转换为列表并限制数量
    table_ids = list(all_table_ids)[:n_tables]
    print(f"从 fold 目录加载了 {len(table_ids)} 个表格 ID")
    return table_ids


def find_table_file(table_id: str, table_dir: str) -> Optional[str]:
    """
    在 table_dir 中查找对应的表格文件

    Args:
        table_id: 表格 ID（不含扩展名）
        table_dir: 表格文件目录

    Returns:
        表格文件的完整路径，如果找不到则返回 None
    """
    # 尝试直接匹配
    direct_path = os.path.join(table_dir, f"{table_id}.csv")
    if os.path.exists(direct_path):
        return direct_path

    # 尝试在目录中搜索（处理可能的命名变体）
    table_dir_path = Path(table_dir)
    for file_path in table_dir_path.glob("*.csv"):
        if get_basename(file_path) == table_id:
            return str(file_path)

    return None


def serialize_table(df: pd.DataFrame, sample_rows: int = 5) -> str:
    """
    将 DataFrame 序列化为 Markdown 表格格式

    Args:
        df: 输入的 DataFrame
        sample_rows: 每列采样的行数

    Returns:
        Markdown 格式的表格字符串
    """
    # 限制行数
    df_sampled = df.head(min(sample_rows, len(df)))

    # 转换为 Markdown
    markdown = df_sampled.to_markdown(index=False)
    return markdown


def tokenize_prefix(
    tokenizer: AutoTokenizer,
    table_name: str,
    markdown_table: str,
    max_length: int = 2048
) -> torch.Tensor:
    """
    构建并 tokenize 前缀（系统提示 + 表格名称 + Markdown 表格）

    Args:
        tokenizer: 分词器
        table_name: 表格名称
        markdown_table: Markdown 格式的表格
        max_length: 最大 token 长度

    Returns:
        前缀 token IDs [1, seq_len]
    """
    prefix_text = f"{SYSTEM_PROMPT}\n\nTable: {table_name}\n{markdown_table}\n\n"

    # Tokenize
    inputs = tokenizer(
        prefix_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )

    return inputs["input_ids"]


def tokenize_suffixes_batch(
    tokenizer: AutoTokenizer,
    column_names: List[str],
    sample_values_list: List[str],
    max_suffix_length: int = 512
) -> torch.Tensor:
    """
    批量构建并 tokenize 后缀（每列的查询文本）

    Args:
        tokenizer: 分词器
        column_names: 列名列表
        sample_values_list: 每列的样本值列表
        max_suffix_length: 每个 suffix 的最大 token 长度

    Returns:
        后缀 token IDs [batch, seq_len]
    """
    suffix_texts = []
    for col_name, sample_values in zip(column_names, sample_values_list):
        suffix_text = SUFFIX_TEMPLATE.format(
            column_name=col_name,
            sample_values=sample_values
        )
        suffix_texts.append(suffix_text)

    # 批量 tokenize
    inputs = tokenizer(
        suffix_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_suffix_length
    )

    return inputs["input_ids"]


def map_tokens_to_columns(
    prefix_text: str,
    headers: List[str],
    tokenizer: AutoTokenizer
) -> Dict[int, int]:
    """
    将前缀 token 位置映射到对应的列

    通过在前缀 token 序列中搜索列名，将每个 token 位置映射到对应的列索引。

    Args:
        prefix_text: 前缀文本（系统提示 + 表格）
        headers: 列名列表
        tokenizer: 分词器

    Returns:
        token_to_column: {token_position: column_index}
        对于不属于任何列数据的 token（如系统提示、分隔符等），映射到 -1
    """
    # Tokenize 前缀文本
    tokens = tokenizer.encode(prefix_text, add_special_tokens=False)

    token_to_column = {}
    current_column = -1

    # 简化策略：查找列名在 token 序列中的位置
    # 注意：这是一个近似方法，因为列名可能会被拆分为多个 token
    for i, token_id in enumerate(tokens):
        # 解码 token
        token_text = tokenizer.decode([token_id]).strip()

        # 检查是否匹配某个列名（部分匹配）
        matched = False
        for col_idx, header in enumerate(headers):
            # 检查 token 是否是列名的一部分
            if header.lower() in token_text.lower() or token_text.lower() in header.lower():
                token_to_column[i] = col_idx
                current_column = col_idx
                matched = True
                break

        # 如果没有匹配到列名，保持当前的列（假设同一列的数据连续）
        if not matched and current_column >= 0:
            token_to_column[i] = current_column

    # 对于没有映射的 token（在第一个列名之前的），设为 -1
    for i in range(len(tokens)):
        if i not in token_to_column:
            token_to_column[i] = -1

    return token_to_column


def expand_kv_cache(past_key_values: Tuple, batch_size: int) -> Tuple:
    """
    将单个表格的 KV-cache 扩展到批次大小

    Args:
        past_key_values: 单个前向传播的 KV-cache
        batch_size: 目标批次大小

    Returns:
        扩展后的 KV-cache
    """
    expanded_cache = []
    for layer_cache in past_key_values:
        # layer_cache 是 (key, value) 的元组
        key, value = layer_cache

        # 复制 key: [batch, num_heads, seq_len, head_dim]
        key_expanded = key.repeat(batch_size, 1, 1, 1)

        # 复制 value
        value_expanded = value.repeat(batch_size, 1, 1, 1)

        expanded_cache.append((key_expanded, value_expanded))

    return tuple(expanded_cache)


# ============== 核心提取函数 ==============

def extract_attention_for_table(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    table_id: str,
    table_path: str,
    fold_df: pd.DataFrame,
    sample_rows: int = 5,
    max_prefix_length: int = 2048,
    max_suffix_length: int = 512
) -> Optional[Dict]:
    """
    为单个表格提取注意力权重

    Args:
        model: 语言模型
        tokenizer: 分词器
        table_id: 表格 ID
        table_path: 表格文件路径
        fold_df: fold DataFrame（包含列的标注信息）
        sample_rows: 每列采样的行数
        max_prefix_length: 前缀最大长度
        max_suffix_length: 后缀最大长度

    Returns:
        包含注意力信息的字典，如果失败则返回 None
    """
    try:
        # 加载表格
        df = pd.read_csv(table_path)
        table_name = get_basename(table_path)

        # 获取列信息（从 fold_df 中过滤出当前表格的列）
        table_cols = fold_df[fold_df['table_id'] == table_id]
        if len(table_cols) == 0:
            print(f"  警告: 在 fold 文件中找不到表格 {table_id} 的列信息")
            return None

        # 按 col_idx 排序并获取列名
        table_cols = table_cols.sort_values('col_idx')
        columns_to_process = table_cols['col_idx'].unique()

        # 构建前缀
        markdown_table = serialize_table(df, sample_rows)
        prefix_text = f"{SYSTEM_PROMPT}\n\nTable: {table_name}\n{markdown_table}\n\n"

        # Tokenize 前缀
        prefix_inputs = tokenizer(
            prefix_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_prefix_length
        )
        prefix_input_ids = prefix_inputs["input_ids"].to(model.device)

        # 前向传播前缀，获取 KV-cache 和注意力权重
        with torch.no_grad():
            prefix_outputs = model(
                input_ids=prefix_input_ids,
                use_cache=True,
                output_attentions=True,
                return_dict=True
            )

        past_key_values = prefix_outputs.past_key_values
        prefix_attentions = prefix_outputs.attentions  # Tuple of [batch, num_heads, seq_len, seq_len]
        prefix_len = prefix_input_ids.shape[1]

        # 映射 token 到列
        headers = df.columns.tolist()
        token_to_column = map_tokens_to_columns(prefix_text, headers, tokenizer)

        # 准备后缀批次
        column_names = []
        sample_values_list = []

        for col_idx in columns_to_process:
            col_name = df.columns[col_idx]
            column_names.append(col_name)

            # 采样该列的值
            col_values = df.iloc[:, col_idx].dropna().head(sample_rows)
            sample_values = ", ".join([str(v) for v in col_values])
            sample_values_list.append(sample_values)

        # Tokenize 后缀批次
        suffix_input_ids = tokenize_suffixes_batch(
            tokenizer,
            column_names,
            sample_values_list,
            max_suffix_length
        ).to(model.device)

        # 扩展 KV-cache
        batch_size = len(column_names)
        expanded_cache = expand_kv_cache(past_key_values, batch_size)

        # 前向传播后缀，获取注意力权重
        with torch.no_grad():
            suffix_outputs = model(
                input_ids=suffix_input_ids,
                past_key_values=expanded_cache,
                output_attentions=True,
                return_dict=True
            )

        suffix_attentions = suffix_outputs.attentions  # Tuple of [batch, num_heads, suffix_len, prefix_len+suffix_len]

        # 提取最后一层的注意力
        # suffix_attentions[-1]: [batch, num_heads, suffix_len, total_len]
        last_layer_attn = suffix_attentions[-1]  # [batch, num_heads, suffix_len, prefix_len + suffix_len]

        # 获取每列的后缀长度（非 padding 部分）
        suffix_lengths = []
        for i in range(batch_size):
            # 找到第一个 padding token 的位置
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            suffix_len = suffix_input_ids.shape[1]
            for j in range(suffix_input_ids.shape[1]):
                if suffix_input_ids[i, j].item() == pad_token_id:
                    suffix_len = j
                    break
            suffix_lengths.append(suffix_len)

        # 对每列，提取最后一个非 padding token 的注意力权重
        # 该 token 对应于生成列描述的时刻
        column_attention = {}

        for col_idx, (actual_col_idx, suffix_len) in enumerate(zip(columns_to_process, suffix_lengths)):
            if suffix_len == 0:
                continue

            # 获取该列最后一个 token 的注意力权重
            # last_layer_attn[col_idx, :, suffix_len-1, :]: [num_heads, total_len]
            # 我们只关心对前缀的注意力，即 [:prefix_len]
            attn_weights = last_layer_attn[col_idx, :, suffix_len - 1, :prefix_len]  # [num_heads, prefix_len]

            # 平均所有注意力头
            attn_weights = attn_weights.mean(dim=0)  # [prefix_len]

            # 将注意力权重映射到源列
            column_attn = defaultdict(float)

            for token_pos in range(min(prefix_len, len(token_to_column))):
                src_col = token_to_column.get(token_pos, -1)
                if src_col >= 0:  # 忽略系统提示等非列内容
                    column_attn[str(src_col)] += attn_weights[token_pos].item()

            # 归一化
            total = sum(column_attn.values())
            if total > 0:
                column_attn = {k: v / total for k, v in column_attn.items()}

            column_attention[str(actual_col_idx)] = dict(column_attn)

        # 构建结果
        result = {
            "table_name": table_name,
            "headers": headers,
            "column_attention": column_attention
        }

        return result

    except Exception as e:
        print(f"  错误: 处理表格 {table_id} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============== 主函数 ==============

def main():
    parser = argparse.ArgumentParser(
        description="从 SemanticCTA 模型中提取注意力权重，用于论文 Fig 2 的可视化"
    )

    # 模型参数
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="LLM 模型路径（可以是 LoRA checkpoint 的基础模型）"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="LoRA 适配器路径（可选，如果使用 LoRA checkpoint）"
    )

    # 数据参数
    parser.add_argument(
        "--fold_dir",
        type=str,
        required=True,
        help="fold CSV 文件目录"
    )
    parser.add_argument(
        "--table_dir",
        type=str,
        required=True,
        help="表格 CSV 文件目录"
    )

    # 输出参数
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录"
    )

    # 提取参数
    parser.add_argument(
        "--n_tables",
        type=int,
        default=10,
        help="提取的表格数量（默认: 10）"
    )
    parser.add_argument(
        "--sample_rows",
        type=int,
        default=5,
        help="每列采样的行数（默认: 5）"
    )
    parser.add_argument(
        "--max_prefix_length",
        type=int,
        default=2048,
        help="前缀最大 token 长度（默认: 2048）"
    )
    parser.add_argument(
        "--max_suffix_length",
        type=int,
        default=512,
        help="后缀最大 token 长度（默认: 512）"
    )

    # 硬件参数
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="GPU ID（可选，不指定则使用 CPU）"
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置设备
    if args.gpu_id is not None:
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"使用 GPU: {args.gpu_id}")
    else:
        device = torch.device("cpu")
        print("使用 CPU")

    # 加载模型和分词器
    print(f"加载模型: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # 设置 padding token（如果不存在）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto" if args.gpu_id is not None else None,
        trust_remote_code=True
    )

    if args.gpu_id is not None:
        model = model.to(device)

    model.eval()

    # 加载 LoRA 适配器（如果提供）
    if args.lora_path is not None:
        if PeftModel is None:
            raise ValueError("未安装 peft 库，无法加载 LoRA 模型。请安装: pip install peft")

        print(f"加载 LoRA 适配器: {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()  # 合并 LoRA 权重到基础模型

    # 加载 fold 数据
    print("加载 fold 数据...")
    fold_data = []
    for fold_idx in range(5):
        fold_path = os.path.join(args.fold_dir, f"fold_{fold_idx}.csv")
        if os.path.exists(fold_path):
            fold_data.append(pd.read_csv(fold_path))

    if not fold_data:
        raise ValueError(f"在 {args.fold_dir} 中找不到任何 fold CSV 文件")

    # 合并所有 fold 数据
    fold_df = pd.concat(fold_data, ignore_index=True)

    # 加载表格 ID
    table_ids = load_fold_table_ids(args.fold_dir, args.n_tables)

    if not table_ids:
        raise ValueError("未能加载任何表格 ID")

    # 提取注意力权重
    print(f"\n开始提取 {len(table_ids)} 个表格的注意力权重...")
    results = {}

    for table_id in tqdm(table_ids, desc="处理表格"):
        # 查找表格文件
        table_path = find_table_file(table_id, args.table_dir)

        if table_path is None:
            print(f"  跳过: 找不到表格文件 {table_id}")
            continue

        # 提取注意力
        result = extract_attention_for_table(
            model=model,
            tokenizer=tokenizer,
            table_id=table_id,
            table_path=table_path,
            fold_df=fold_df,
            sample_rows=args.sample_rows,
            max_prefix_length=args.max_prefix_length,
            max_suffix_length=args.max_suffix_length
        )

        if result is not None:
            results[table_id] = result

    # 保存结果
    output_path = os.path.join(args.output_dir, "attention_weights.json")
    print(f"\n保存结果到: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n提取完成！成功处理 {len(results)}/{len(table_ids)} 个表格")
    print(f"输出文件: {output_path}")


if __name__ == "__main__":
    main()
