"""
CTA 效率基准测试脚本 — SemanticCTA 论文 Q5 效率实验（Table 9 + Fig 5）

对比三种 CTA 流水线的效率指标：
1. semantic_cta_cached: 默认 SemanticCTA（KV-cache 复用）
2. semantic_cta_no_cache: SemanticCTA 不使用 KV-cache（每列独立前向传播）
3. description_then_encode: 两阶段流水线（生成描述 → BERT 编码）

测量指标：
- 总耗时、平均延迟、吞吐量
- GPU 峰值显存、token 消耗

输出：benchmark_results.json
"""

import os
import sys
import json
import argparse
import time
import pandas as pd
import torch
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# ---------------------------------------------------------------------------
#  表格读取 & 查找工具（内联，自包含）
# ---------------------------------------------------------------------------


def get_basename(file_path: str) -> str:
    """获取文件基本名（不含扩展名）"""
    base = os.path.basename(file_path)
    return os.path.splitext(base)[0]


def load_fold_table_ids(fold_dir: str) -> List[str]:
    """从 fold_0~4.csv 中加载所有唯一的 table_id"""
    all_table_ids = set()
    for i in range(5):
        fold_path = os.path.join(fold_dir, f"fold_{i}.csv")
        if os.path.exists(fold_path):
            df = pd.read_csv(fold_path)
            all_table_ids.update(df["table_id"].unique())
    return sorted(list(all_table_ids))


def find_table_file(table_id: str, table_dir: str, file_type: str = ".csv") -> Optional[str]:
    """查找表文件路径"""
    direct = os.path.join(table_dir, f"{table_id}{file_type}")
    if os.path.exists(direct):
        return direct
    for root, dirs, files in os.walk(table_dir):
        for f in files:
            if f == f"{table_id}{file_type}" or f.startswith(table_id):
                return os.path.join(root, f)
    return None


def load_table(table_path: str, sample_rows: int) -> Tuple[List[str], List[List[str]]]:
    """加载表数据并返回 headers 和 data"""
    df = pd.read_csv(table_path, nrows=sample_rows, lineterminator="\n")
    df = df.dropna(how="all")
    headers = df.columns.tolist()
    data = df.values.tolist()
    return headers, data


# ---------------------------------------------------------------------------
#  Profiling 数据加载（与 index_cta_llm_hidden.py 相同）
# ---------------------------------------------------------------------------


from functools import lru_cache


@lru_cache(maxsize=128)
def _load_profilling_json(profilling_path: str):
    with open(profilling_path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=128)
def _get_profilling_index(profilling_path: str):
    descriptions_data = _load_profilling_json(profilling_path)
    norm_map = {}
    base_map = {}
    keys = list(descriptions_data.keys())
    for json_path, value in descriptions_data.items():
        try:
            norm = os.path.normpath(json_path)
        except Exception:
            norm = json_path
        norm_map[norm] = value
        base = get_basename(json_path)
        base_map.setdefault(base, []).append(value)
    return descriptions_data, norm_map, base_map, keys


def get_table_profilling(csv_file_path: str, profilling_path: str) -> Optional[dict]:
    """查找给定表对应的 profiling 数据"""
    descriptions_data, norm_map, base_map, keys = _get_profilling_index(profilling_path)
    normalized = os.path.normpath(csv_file_path)
    if normalized in norm_map:
        return norm_map[normalized]
    base = get_basename(normalized)
    candidates = base_map.get(base)
    if candidates:
        if len(candidates) == 1:
            return candidates[0]
        for json_path in keys:
            if get_basename(json_path) == base and os.path.normpath(json_path) in normalized:
                return descriptions_data.get(json_path)
        return candidates[0]
    for json_path in keys:
        if (os.path.normpath(json_path) == normalized
                or get_basename(normalized) == get_basename(json_path)
                or os.path.normpath(json_path) in normalized):
            return descriptions_data.get(json_path)
    return None


# ---------------------------------------------------------------------------
#  Pipeline 1 & 2: SemanticCTA（从 index_cta_llm_hidden.py 导入 LLMHiddenStateEncoder）
# ---------------------------------------------------------------------------


def run_semantic_cta_pipeline(
    encoder,
    table_ids: List[str],
    table_dir: str,
    sample_rows: int,
    profilling_path: Optional[str] = None,
    use_cache: bool = True,
) -> Dict:
    """
    运行 SemanticCTA 流水线并测量效率指标

    Args:
        encoder: LLMHiddenStateEncoder 实例
        table_ids: 要处理的表 ID 列表
        table_dir: 表文件目录
        sample_rows: 每表采样行数
        profilling_path: profiling 数据路径（可选）
        use_cache: 是否使用 KV-cache 复用

    Returns:
        效率指标字典
    """
    # 清理 GPU 缓存并重置显存统计
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    total_cols = 0
    total_tokens = 0
    start_time = time.time()

    for table_id in tqdm(table_ids, desc=f"SemanticCTA {'cached' if use_cache else 'no-cache'}"):
        table_path = find_table_file(table_id, table_dir)
        if table_path is None:
            continue

        try:
            headers, data = load_table(table_path, sample_rows)
            table_name = get_basename(table_path)

            # 获取 profiling（可选）
            profile = None
            if profilling_path:
                profile = get_table_profilling(table_path, profilling_path)

            # 编码表（在 encode_table 内部记录 token 数）
            col_embeddings, token_count = encoder.encode_table_with_token_count(
                table_name=table_name,
                headers=headers,
                data=data,
                n_rows=sample_rows,
                profile=profile,
                use_cache=use_cache,
            )

            total_cols += col_embeddings.shape[0]
            total_tokens += token_count

        except Exception as e:
            print(f"  Error processing {table_id}: {e}")
            continue

    total_time = time.time() - start_time
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return {
        "total_time_sec": round(total_time, 3),
        "avg_latency_per_table_ms": round(total_time / len(table_ids) * 1000, 2),
        "throughput_cols_per_sec": round(total_cols / total_time, 2),
        "peak_gpu_memory_gb": round(peak_memory_gb, 2),
        "total_tokens": total_tokens,
        "num_tables_processed": len(table_ids),
        "num_columns_processed": total_cols,
    }


# ---------------------------------------------------------------------------
#  Pipeline 3: Description-then-Encode（两阶段流水线）
# ---------------------------------------------------------------------------


def build_column_description_text(
    col_name: str,
    col_values: List[str],
    table_name: str,
    profile: Optional[dict] = None,
) -> str:
    """为单列构建描述文本（Stage 1）"""
    values_str = ", ".join(str(v) for v in col_values if v) if col_values else "(empty)"

    parts = [f"Column '{col_name}' from table '{table_name}'"]

    if profile and col_name in profile:
        col_profile = profile[col_name]
        if isinstance(col_profile, dict):
            desc_parts = []
            for k, v in col_profile.items():
                if k == "__type__":
                    desc_parts.insert(0, f"Type: {v}")
                elif k == "__table__":
                    continue
                else:
                    desc_parts.append(str(v))
            if desc_parts:
                parts.append("; ".join(desc_parts))
        elif isinstance(col_profile, str):
            parts.append(col_profile)

    parts.append(f"Example values: {values_str}")
    return ". ".join(parts) + "."


def run_description_then_encode_pipeline(
    table_ids: List[str],
    table_dir: str,
    sample_rows: int,
    bert_model_name: str,
    profilling_path: Optional[str] = None,
    gpu_id: Optional[int] = None,
) -> Dict:
    """
    运行两阶段流水线：生成描述 → BERT 编码

    Args:
        table_ids: 要处理的表 ID 列表
        table_dir: 表文件目录
        sample_rows: 每表采样行数
        bert_model_name: BERT 模型名称
        profilling_path: profiling 数据路径（可选）
        gpu_id: GPU ID

    Returns:
        效率指标字典
    """
    from transformers import AutoModel, AutoTokenizer

    device = f"cuda:{gpu_id}" if gpu_id is not None else "cuda"

    # 清理 GPU 缓存并重置显存统计
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 加载 BERT 模型
    print(f"  Loading BERT model: {bert_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    model = AutoModel.from_pretrained(bert_model_name).to(device)
    model.eval()

    total_cols = 0
    total_tokens = 0
    start_time = time.time()

    for table_id in tqdm(table_ids, desc="Description-then-Encode"):
        table_path = find_table_file(table_id, table_dir)
        if table_path is None:
            continue

        try:
            headers, data = load_table(table_path, sample_rows)
            table_name = get_basename(table_path)

            # 获取 profiling（可选）
            profile = None
            if profilling_path:
                profile = get_table_profilling(table_path, profilling_path)

            # Stage 1: 生成描述文本
            col_descriptions = []
            for col_idx, col_name in enumerate(headers):
                col_values = [str(row[col_idx]) for row in data if col_idx < len(row)]
                desc_text = build_column_description_text(
                    col_name=col_name,
                    col_values=col_values,
                    table_name=table_name,
                    profile=profile,
                )
                col_descriptions.append(desc_text)

            # Stage 2: BERT 编码
            with torch.no_grad():
                encoded = tokenizer(
                    col_descriptions,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}

                # 统计 token 数
                total_tokens += encoded["input_ids"].shape[0] * encoded["input_ids"].shape[1]

                outputs = model(**encoded)
                # 提取 [CLS] token 表示
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (num_cols, hidden_dim)

            total_cols += cls_embeddings.shape[0]

        except Exception as e:
            print(f"  Error processing {table_id}: {e}")
            continue

    total_time = time.time() - start_time
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # 释放模型
    del model
    torch.cuda.empty_cache()

    return {
        "total_time_sec": round(total_time, 3),
        "avg_latency_per_table_ms": round(total_time / len(table_ids) * 1000, 2),
        "throughput_cols_per_sec": round(total_cols / total_time, 2),
        "peak_gpu_memory_gb": round(peak_memory_gb, 2),
        "total_tokens": total_tokens,
        "num_tables_processed": len(table_ids),
        "num_columns_processed": total_cols,
    }


# ---------------------------------------------------------------------------
#  扩展 LLMHiddenStateEncoder 以支持 token 统计
# ---------------------------------------------------------------------------


def add_token_count_method_to_encoder(encoder_class):
    """动态添加 encode_table_with_token_count 方法到 LLMHiddenStateEncoder"""

    def encode_table_with_token_count(self, table_name, headers, data, n_rows=5,
                                      profile=None, use_cache=True, save_attention=False):
        """encode_table 的扩展版本，返回 (embeddings, token_count)"""
        device = self.device
        num_cols = len(headers)

        # 提取列值
        col_values_list = []
        for col_idx in range(num_cols):
            vals = [str(row[col_idx]) for row in data[:n_rows] if col_idx < len(row)]
            col_values_list.append(vals)

        if not use_cache:
            # No-cache 模式：每列独立前向传播
            prefix_text = self._build_prefix_text(table_name, headers, data, n_rows, profile)
            result_parts = []
            total_tokens = 0

            for col_idx in range(num_cols):
                name = headers[col_idx]
                vals = col_values_list[col_idx]
                values_str = ", ".join(str(v) for v in vals) if vals else "(empty)"
                suffix_text = self.SUFFIX_TEMPLATE.format(col_name=name, col_values=values_str)
                full_user_text = prefix_text + "\n" + suffix_text

                messages = [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": full_user_text},
                ]
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(device)

                total_tokens += input_ids.shape[-1]

                out = self.model(input_ids=input_ids, output_hidden_states=True)
                last_pos = input_ids.shape[-1] - 1

                col_reprs = []
                for abs_layer_idx in self._abs_layers:
                    h = out.hidden_states[abs_layer_idx]
                    col_reprs.append(h[0, last_pos])

                if len(col_reprs) == 1:
                    col_repr = col_reprs[0]
                else:
                    col_repr = torch.stack(col_reprs, dim=0).mean(dim=0)
                result_parts.append(col_repr)
                del out

            result = torch.stack(result_parts, dim=0)
            result = torch.nn.functional.normalize(result, p=2, dim=-1)
            return result.float().cpu(), total_tokens

        else:
            # Cached 模式：KV-cache 复用
            prefix_text = self._build_prefix_text(table_name, headers, data, n_rows, profile)
            prefix_ids = self._tokenize_prefix(prefix_text).to(device)

            total_tokens = prefix_ids.shape[-1]

            prefix_out = self.model(input_ids=prefix_ids, use_cache=True, output_hidden_states=False)
            kv_cache = prefix_out.past_key_values
            del prefix_out

            suffix_ids, suffix_mask = self._tokenize_suffixes_batch(headers, col_values_list)
            suffix_ids = suffix_ids.to(device)
            suffix_mask = suffix_mask.to(device)

            total_tokens += suffix_ids.shape[-1] * num_cols

            from transformers import DynamicCache
            batch_kv_cache = DynamicCache()
            for layer_idx in range(len(kv_cache)):
                k, v = kv_cache[layer_idx]
                batch_kv_cache.update(
                    k.expand(num_cols, -1, -1, -1).contiguous(),
                    v.expand(num_cols, -1, -1, -1).contiguous(),
                    layer_idx,
                )
            del kv_cache

            out = self.model(
                input_ids=suffix_ids,
                attention_mask=suffix_mask,
                past_key_values=batch_kv_cache,
                output_hidden_states=True,
                output_attentions=save_attention,
            )

            last_token_positions = suffix_mask.sum(dim=1) - 1
            all_hidden = out.hidden_states
            col_reprs = []
            for abs_layer_idx in self._abs_layers:
                h = all_hidden[abs_layer_idx]
                gathered = h[torch.arange(num_cols, device=device), last_token_positions]
                col_reprs.append(gathered)

            if len(col_reprs) == 1:
                result = col_reprs[0]
            else:
                result = torch.stack(col_reprs, dim=0).mean(dim=0)

            result = torch.nn.functional.normalize(result, p=2, dim=-1)

            del batch_kv_cache, out, col_reprs
            for gpu_idx in range(torch.cuda.device_count()):
                with torch.cuda.device(gpu_idx):
                    torch.cuda.empty_cache()

            return result.float().cpu(), total_tokens

    # 添加必需的常量
    import sys
    import importlib
    idx_module = importlib.import_module("column_type_annotation.index_cta_llm_hidden")
    encoder_class.SYSTEM_PROMPT = idx_module.SYSTEM_PROMPT
    encoder_class.SUFFIX_TEMPLATE = idx_module.SUFFIX_TEMPLATE

    # 绑定方法
    encoder_class.encode_table_with_token_count = encode_table_with_token_count


# ---------------------------------------------------------------------------
#  主流程
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="CTA 效率基准测试 — SemanticCTA 论文 Q5 效率实验"
    )

    # 基本参数
    parser.add_argument("--model_path", type=str, required=True,
                        help="LLM 模型路径（如 Qwen2.5-7B-Instruct）")
    parser.add_argument("--fold_dir", type=str, required=True,
                        help="fold CSV 文件目录")
    parser.add_argument("--table_dir", type=str, required=True,
                        help="表文件目录")
    parser.add_argument("--result_dir", type=str, required=True,
                        help="结果输出目录")

    # 数据处理
    parser.add_argument("--n_tables", type=int, default=50,
                        help="基准测试的表数量（默认 50）")
    parser.add_argument("--sample_rows", type=int, default=5,
                        help="每表采样行数（默认 5）")

    # BERT 配置（两阶段流水线）
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased",
                        help="BERT 模型名称（默认 bert-base-uncased）")

    # 可选参数
    parser.add_argument("--profilling_path", type=str, default=None,
                        help="Profiling JSON 路径（可选，用于两阶段流水线）")
    parser.add_argument("--gpu_id", type=int, default=None,
                        help="GPU ID（默认 auto）")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("CTA 效率基准测试 — SemanticCTA 论文 Q5 效率实验")
    print("=" * 70)
    print(f"LLM 模型路径:        {args.model_path}")
    print(f"Fold 目录:           {args.fold_dir}")
    print(f"表目录:              {args.table_dir}")
    print(f"结果目录:            {args.result_dir}")
    print(f"基准测试表数量:      {args.n_tables}")
    print(f"每表采样行数:        {args.sample_rows}")
    print(f"BERT 模型:           {args.bert_model}")
    print(f"Profiling 路径:      {args.profilling_path or '(未使用)'}")
    print(f"GPU ID:              {args.gpu_id if args.gpu_id is not None else 'auto'}")
    print("=" * 70)

    # 创建结果目录
    os.makedirs(args.result_dir, exist_ok=True)

    # 1. 加载 table_id
    print("\n[Step 1] 加载表 ID...")
    all_table_ids = load_fold_table_ids(args.fold_dir)
    print(f"  总表数: {len(all_table_ids)}")

    # 采样指定数量的表
    import random
    random.seed(42)
    table_ids = random.sample(all_table_ids, min(args.n_tables, len(all_table_ids)))
    print(f"  采样表数: {len(table_ids)}")

    # 2. 导入 LLMHiddenStateEncoder
    print("\n[Step 2] 导入 LLMHiddenStateEncoder...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from index_cta_llm_hidden import LLMHiddenStateEncoder

    # 添加 token 统计方法
    add_token_count_method_to_encoder(LLMHiddenStateEncoder)

    # 3. 初始化 encoder
    print(f"\n[Step 3] 初始化 LLM encoder: {args.model_path}")
    device = f"cuda:{args.gpu_id}" if args.gpu_id is not None else "auto"
    encoder = LLMHiddenStateEncoder(
        model_path=args.model_path,
        max_prefix_length=2048,
        layers=[-1],
        device=device,
    )

    # 4. 运行基准测试
    results = {}

    # Pipeline 1: SemanticCTA with KV-cache
    print("\n" + "=" * 70)
    print("[Pipeline 1] SemanticCTA (KV-cache 复用)")
    print("=" * 70)
    results["semantic_cta_cached"] = run_semantic_cta_pipeline(
        encoder=encoder,
        table_ids=table_ids,
        table_dir=args.table_dir,
        sample_rows=args.sample_rows,
        profilling_path=args.profilling_path,
        use_cache=True,
    )

    # Pipeline 2: SemanticCTA without KV-cache
    print("\n" + "=" * 70)
    print("[Pipeline 2] SemanticCTA (无 KV-cache)")
    print("=" * 70)
    results["semantic_cta_no_cache"] = run_semantic_cta_pipeline(
        encoder=encoder,
        table_ids=table_ids,
        table_dir=args.table_dir,
        sample_rows=args.sample_rows,
        profilling_path=args.profilling_path,
        use_cache=False,
    )

    # Pipeline 3: Description-then-Encode
    print("\n" + "=" * 70)
    print("[Pipeline 3] Description-then-Encode (两阶段流水线)")
    print("=" * 70)
    results["description_then_encode"] = run_description_then_encode_pipeline(
        table_ids=table_ids,
        table_dir=args.table_dir,
        sample_rows=args.sample_rows,
        bert_model_name=args.bert_model,
        profilling_path=args.profilling_path,
        gpu_id=args.gpu_id,
    )

    # 5. 保存结果
    print("\n" + "=" * 70)
    print("[Step 5] 保存基准测试结果")
    print("=" * 70)

    output_path = os.path.join(args.result_dir, "benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存至: {output_path}")

    # 打印结果摘要
    print("\n" + "=" * 70)
    print("基准测试结果摘要")
    print("=" * 70)
    print(f"{'指标':<30} {'SemanticCTA (cached)':<25} {'SemanticCTA (no-cache)':<25} {'Desc-then-Encode':<25}")
    print("-" * 100)

    metrics = [
        ("总耗时 (秒)", "total_time_sec"),
        ("平均延迟/表 (毫秒)", "avg_latency_per_table_ms"),
        ("吞吐量 (列/秒)", "throughput_cols_per_sec"),
        ("峰值显存 (GB)", "peak_gpu_memory_gb"),
        ("总 Token 数", "total_tokens"),
    ]

    for metric_name, metric_key in metrics:
        row = [
            metric_name,
            results["semantic_cta_cached"][metric_key],
            results["semantic_cta_no_cache"][metric_key],
            results["description_then_encode"][metric_key],
        ]
        print(f"{row[0]:<30} {row[1]:<25} {row[2]:<25} {row[3]:<25}")

    print("\n" + "=" * 70)
    print("基准测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
