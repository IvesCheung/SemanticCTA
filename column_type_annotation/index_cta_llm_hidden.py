"""
CTA Embedding 生成脚本 — 基于 LLM 隐藏层状态（Prefix-cache + Per-column Suffix 方案）

核心思路：
    不走 "LLM → 生成 description 文本 → embedding 模型编码" 的老路，
    而是直接提取 LLM 前向传播时的隐藏层状态作为列的语义表示。

    1. 将整张表序列化为 prefix prompt，前向传播一次，拿到 KV-cache
    2. 对每个列拼接列级 suffix，利用 KV-cache 做轻量前向传播
    3. 提取 suffix 最后一个 token 位置的隐藏状态 → 该列的 embedding

输出格式与 index_cta.py 完全一致：
    pickle 字典 { table_id: tensor(num_cols, hidden_dim) }
    可直接对接 train_cta.py（自动检测 embedding 维度）。

用法：
    python column_type_annotation/index_cta_llm_hidden.py \
        --model_path ./model/Qwen3-32B \
        --fold_dir datasets/gittables-semtab22/ \
        --table_dir datasets/gittables-semtab22/ \
        --output_path results/cta/gittables/embeddings_llm_hidden.pkl \
        --sample_rows 5

    # 可选：加入 profiling 数据增强 prefix
    python column_type_annotation/index_cta_llm_hidden.py \
        ... \
        --profilling_path results/profiling/gittables.json
"""

import os
import sys
import pickle
import argparse
import json
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

# ---------------------------------------------------------------------------
#  表格读取 & 查找工具（自包含，不依赖 index_cta.py 以避免改动现有代码）
# ---------------------------------------------------------------------------


def get_basename(file_path: str) -> str:
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
            print(f"  Loaded fold_{i}.csv: {len(df['table_id'].unique())} unique tables")
        else:
            print(f"  Warning: fold_{i}.csv not found")
    print(f"  Total unique tables: {len(all_table_ids)}")
    return list(all_table_ids)


def find_table_file(table_id: str, table_dir: str, file_type: str = ".csv") -> Optional[str]:
    direct = os.path.join(table_dir, f"{table_id}{file_type}")
    if os.path.exists(direct):
        return direct
    for root, dirs, files in os.walk(table_dir):
        for f in files:
            if f == f"{table_id}{file_type}" or f.startswith(table_id):
                return os.path.join(root, f)
    return None


# ---------------------------------------------------------------------------
#  Profiling 数据加载
# ---------------------------------------------------------------------------


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
#  表格序列化
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a semantic type annotator for tabular data. "
    "Analyze each column's semantic meaning, data type, value patterns, "
    "cross-column relationships, and domain-specific interpretation."
)

SUFFIX_TEMPLATE = (
    'Column "{col_name}" contains values: {col_values}. '
    "Based on the table above, the semantic type and meaning of this column is"
)


def serialize_table(headers: List[str], data: List[List[str]], n_rows: int = 5) -> str:
    """将表格序列化为 markdown 格式文本"""
    lines = []
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in data[:n_rows]:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def build_profilling_text(headers: List[str], profile: dict) -> str:
    """将 profiling 数据格式化为可追加到 prefix 的文本"""
    lines = ["\nColumn descriptions:"]
    for col_name in headers:
        col_data = profile.get(col_name, {})
        if isinstance(col_data, dict):
            desc_parts = []
            for k, v in col_data.items():
                if k == "__type__":
                    desc_parts.insert(0, f"Type: {v}")
                elif k == "__table__":
                    continue
                else:
                    desc_parts.append(str(v))
            if desc_parts:
                lines.append(f'  - "{col_name}": {"; ".join(desc_parts)}')
        elif isinstance(col_data, str):
            lines.append(f'  - "{col_name}": {col_data}')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  核心类：LLM Hidden State Encoder
# ---------------------------------------------------------------------------


class LLMHiddenStateEncoder:
    """使用 LLM 隐藏层状态为表格的每一列生成 embedding"""

    def __init__(
        self,
        model_path: str,
        max_prefix_length: int = 2048,
        layers: List[int] = None,
        device: str = "auto",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_prefix_length = max_prefix_length
        self.layers = layers or [-1]  # 默认只取最后一层

        print(f"[LLMHiddenStateEncoder] Loading model from {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        model_kwargs = {
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
            "output_hidden_states": True,
        }
        if device != "auto":
            model_kwargs["device_map"] = device
        else:
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self.model.eval()

        # 获取 hidden_size
        self.hidden_dim = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers + 1  # +1 for embedding layer

        print(f"  hidden_size = {self.hidden_dim}")
        print(f"  num_layers  = {self.num_layers}")
        print(f"  extracting layers = {self._resolve_layer_indices()}")
        print(f"  device_map = {getattr(self.model, 'hf_device_map', 'N/A')}")

        # 将 layers 转为绝对索引
        self._abs_layers = self._resolve_layer_indices()
        self.device = next(self.model.parameters()).device

    def _resolve_layer_indices(self) -> List[int]:
        """将负索引转为绝对索引"""
        total = self.num_layers  # num_hidden_layers + 1
        return [idx if idx >= 0 else total + idx for idx in self.layers]

    def _build_prefix_text(
        self,
        table_name: str,
        headers: List[str],
        data: List[List[str]],
        n_rows: int = 5,
        profile: Optional[dict] = None,
    ) -> str:
        """构建 prefix prompt 文本（系统提示 + 表格内容 + 可选 profiling）"""
        table_md = serialize_table(headers, data, n_rows)

        user_parts = [f"Table name: {table_name}\n"]
        user_parts.append(table_md)

        if profile:
            prof_text = build_profilling_text(headers, profile)
            user_parts.append(prof_text)

        return "\n".join(user_parts)

    def _tokenize_prefix(self, text: str) -> torch.Tensor:
        """Tokenize prefix 文本并截断到 max_prefix_length"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        # apply_chat_template 返回 token ids（不含 generation prompt）
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # 截断
        if input_ids.shape[-1] > self.max_prefix_length:
            input_ids = input_ids[:, -self.max_prefix_length:]
        return input_ids

    def _tokenize_suffixes_batch(
        self, col_names: List[str], col_values_list: List[List[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize 多个列的 suffix 并 padding 到等长

        Args:
            col_names: 列名列表
            col_values_list: 每列的采样值列表，如 [["25", "30"], ["Alice", "Bob"]]

        Returns:
            input_ids: (num_cols, max_suffix_len)
            attention_mask: (num_cols, max_suffix_len)
        """
        texts = []
        for name, vals in zip(col_names, col_values_list):
            values_str = ", ".join(str(v) for v in vals) if vals else "(empty)"
            texts.append(SUFFIX_TEMPLATE.format(col_name=name, col_values=values_str))
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        return encoded.input_ids, encoded.attention_mask

    @torch.no_grad()
    def encode_table(
        self,
        table_name: str,
        headers: List[str],
        data: List[List[str]],
        n_rows: int = 5,
        profile: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        为一张表的所有列生成 embedding。

        Args:
            table_name: 表名
            headers: 列名列表
            data: 表格数据 (2D list)
            n_rows: 序列化行数
            profile: 可选 profiling 数据

        Returns:
            tensor(num_cols, hidden_dim) — float32
        """
        device = self.device
        num_cols = len(headers)

        # Step 1: 构建 & tokenize prefix
        prefix_text = self._build_prefix_text(table_name, headers, data, n_rows, profile)
        prefix_ids = self._tokenize_prefix(prefix_text).to(device)

        # Step 2: 前向传播 prefix，获取 KV-cache
        prefix_out = self.model(input_ids=prefix_ids, use_cache=True, output_hidden_states=True)
        kv_cache = prefix_out.past_key_values

        # Step 3: Batch suffix 前向传播
        # 3a. 提取每列的采样值
        col_values_list = []
        for col_idx in range(num_cols):
            vals = [str(row[col_idx]) for row in data[:n_rows] if col_idx < len(row)]
            col_values_list.append(vals)

        # 3b. Tokenize 所有列的 suffix 并 padding
        suffix_ids, suffix_mask = self._tokenize_suffixes_batch(headers, col_values_list)
        suffix_ids = suffix_ids.to(device)    # (num_cols, max_suffix_len)
        suffix_mask = suffix_mask.to(device)  # (num_cols, max_suffix_len)

        # 3c. 将 KV-cache 沿 batch 维度复制 num_cols 份
        # 用 DynamicCache.update() 写入，兼容不同 transformers 版本的内部存储格式
        from transformers import DynamicCache
        batch_kv_cache = DynamicCache()
        for layer_idx in range(len(kv_cache)):
            k, v = kv_cache[layer_idx]  # (1, num_heads, prefix_len, head_dim)
            batch_kv_cache.update(
                k.expand(num_cols, -1, -1, -1).contiguous(),
                v.expand(num_cols, -1, -1, -1).contiguous(),
                layer_idx,
            )

        # 3d. 单次 batch 前向传播
        out = self.model(
            input_ids=suffix_ids,
            attention_mask=suffix_mask,
            past_key_values=batch_kv_cache,
            output_hidden_states=True,
        )

        # 3e. 提取每个序列最后一个有效 token 的隐藏状态
        # last_token_positions: (num_cols,) — 每行最后一个非 pad 位置
        last_token_positions = suffix_mask.sum(dim=1) - 1  # (num_cols,)

        all_hidden = out.hidden_states  # tuple: (num_layers,), each (num_cols, max_suffix_len, dim)
        col_reprs = []
        for abs_layer_idx in self._abs_layers:
            h = all_hidden[abs_layer_idx]  # (num_cols, max_suffix_len, dim)
            # 取每个序列最后一个有效 token
            gathered = h[torch.arange(num_cols, device=device), last_token_positions]
            col_reprs.append(gathered)

        if len(col_reprs) == 1:
            result = col_reprs[0]
        else:
            # 多层 mean pooling
            result = torch.stack(col_reprs, dim=0).mean(dim=0)

        # L2 归一化：统一向量尺度，避免数值范围差异干扰分类器
        result = torch.nn.functional.normalize(result, p=2, dim=-1)

        # Step 4: 清理 KV-cache 释放显存
        del prefix_out, kv_cache, batch_kv_cache, out
        torch.cuda.empty_cache()

        return result.float().cpu()  # (num_cols, hidden_dim)


# ---------------------------------------------------------------------------
#  主流程
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="CTA Embedding 生成 — 基于 LLM 隐藏层状态（Prefix-cache + Per-column Suffix）"
    )

    # 数据路径
    parser.add_argument("--model_path", type=str, required=True,
                        help="Qwen3-32B 本地模型路径 (HuggingFace 格式)")
    parser.add_argument("--fold_dir", type=str, required=True,
                        help="包含 fold_*.csv 文件的目录")
    parser.add_argument("--table_dir", type=str, required=True,
                        help="原始表文件所在目录")
    parser.add_argument("--output_path", type=str, required=True,
                        help="输出 embedding pickle 文件路径")

    # 数据处理
    parser.add_argument("--sample_rows", type=int, default=5,
                        help="每张表序列化的行数 (默认 5)")
    parser.add_argument("--file_type", type=str, default=".csv",
                        help="表文件类型 (默认 .csv)")

    # Profiling
    parser.add_argument("--profilling_path", type=str, default=None,
                        help="Profilling JSON 路径（可选，注入到 prefix）")
    parser.add_argument("--require_profile", action="store_true", default=True,
                        help="是否要求必须有 profilling 数据（默认 True）")
    parser.add_argument("--no_require_profile", action="store_false", dest="require_profile",
                        help="关闭 profilling 必须要求")

    # 模型配置
    parser.add_argument("--max_prefix_length", type=int, default=2048,
                        help="Prefix 最大 token 数 (默认 2048)")
    parser.add_argument("--layers", type=str, default="-1",
                        help="提取哪些层的隐藏状态，逗号分隔 (默认 -1 即最后一层; "
                             "例: -1,-4,-8,-12 取4层做 mean pooling)")
    parser.add_argument("--gpu_id", type=int, default=None,
                        help="指定 GPU ID (默认 auto)")

    return parser.parse_args()


def parse_layers(layers_str: str) -> List[int]:
    """解析 --layers 参数"""
    parts = [s.strip() for s in layers_str.split(",")]
    return [int(p) for p in parts if p]


def _diagnose_embeddings(embeddings_map: Dict[str, torch.Tensor]) -> dict:
    """快速诊断 embedding 质量：余弦相似度分布、方差等"""
    all_embs = torch.cat(list(embeddings_map.values()), dim=0)  # (total_cols, dim)
    total_cols, dim = all_embs.shape

    # L2 norm
    norms = torch.norm(all_embs, p=2, dim=1)

    # Per-dim std（越高说明该维度有区分度）
    per_dim_std = all_embs.std(dim=0)

    # Pairwise cosine similarity（采样计算，避免 O(N^2)）
    n_sample = min(1000, total_cols)
    indices = torch.randperm(total_cols)[:n_sample]
    sample = all_embs[indices]
    # L2 normalized already, cosine sim = dot product
    sim_matrix = sample @ sample.T
    # 取上三角（排除对角线）
    mask = torch.triu(torch.ones_like(sim_matrix, dtype=torch.bool), diagonal=1)
    sim_values = sim_matrix[mask]

    return {
        'total_cols': total_cols,
        'dim': dim,
        'norm_mean': norms.mean().item(),
        'norm_std': norms.std().item(),
        'cosine_sim_mean': sim_values.mean().item(),
        'cosine_sim_std': sim_values.std().item(),
        'per_dim_std_mean': per_dim_std.mean().item(),
        'per_dim_std_min': per_dim_std.min().item(),
    }


def _diagnose_class_separation(embeddings_map: Dict[str, torch.Tensor], fold_dir: str) -> dict:
    """诊断 embedding 是否按类别聚簇：同类内 vs 类间余弦相似度"""
    # 加载 fold 标签
    all_rows = []
    for i in range(5):
        fp = os.path.join(fold_dir, f"fold_{i}.csv")
        if os.path.exists(fp):
            all_rows.append(pd.read_csv(fp))
    if not all_rows:
        return {}
    labels_df = pd.concat(all_rows, ignore_index=True)
    labels_df = labels_df[labels_df['class_id'] != -1]

    # 构建 (col_embedding, class_id) 对
    embs, class_ids = [], []
    for _, row in labels_df.iterrows():
        tid = row['table_id']
        cidx = int(row['col_idx'])
        if tid in embeddings_map:
            t_emb = embeddings_map[tid]
            if cidx < len(t_emb):
                embs.append(t_emb[cidx])
                class_ids.append(int(row['class_id']))

    if len(embs) < 100:
        return {}

    embs = torch.stack(embs)
    class_ids = torch.tensor(class_ids)
    unique_classes = class_ids.unique()

    # 只取样本数 >= 5 的类别
    class_counts = torch.bincount(class_ids)
    valid_classes = (class_counts >= 5).nonzero(as_tuple=True)[0]

    within_sims, between_sims = [], []
    for cls_id in valid_classes[:50]:  # 最多50个类
        cls_mask = class_ids == cls_id.item()
        cls_embs = embs[cls_mask]
        other_embs = embs[~cls_mask]

        # Within-class: 采样 200 对
        n_cls = len(cls_embs)
        if n_cls < 2:
            continue
        n_within = min(200, n_cls * (n_cls - 1) // 2)
        for _ in range(n_within):
            i, j = torch.randperm(n_cls)[:2]
            within_sims.append(torch.dot(cls_embs[i], cls_embs[j]).item())

        # Between-class: 采样 200 对
        n_other = len(other_embs)
        for _ in range(200):
            i = torch.randint(n_cls, (1,)).item()
            j = torch.randint(n_other, (1,)).item()
            between_sims.append(torch.dot(cls_embs[i], other_embs[j]).item())

    if not within_sims or not between_sims:
        return {}

    within_mean = sum(within_sims) / len(within_sims)
    between_mean = sum(between_sims) / len(between_sims)
    gap = within_mean - between_mean

    return {
        'within_class_sim': within_mean,
        'between_class_sim': between_mean,
        'class_separation_gap': gap,
        'num_classes_sampled': min(len(valid_classes), 50),
    }


def generate_embeddings(
    table_ids: List[str],
    table_dir: str,
    encoder: LLMHiddenStateEncoder,
    file_type: str = ".csv",
    sample_rows: int = 5,
    profilling_path: Optional[str] = None,
    require_profile: bool = True,
    fold_dir: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """为所有表生成 embedding"""
    embeddings_map = {}
    missing_tables = []
    error_tables = []

    for table_id in tqdm(table_ids, desc="Generating LLM hidden-state embeddings"):
        table_path = find_table_file(table_id, table_dir, file_type)
        if table_path is None:
            missing_tables.append(table_id)
            continue

        try:
            # 读取表
            df = pd.read_csv(table_path, nrows=sample_rows, lineterminator="\n")
            df = df.dropna(how="all")
            headers = df.columns.tolist()
            data = df.values.tolist()

            # 获取 profiling（可选）
            profile = None
            if profilling_path:
                profile = get_table_profilling(table_path, profilling_path)
                if profile is None and require_profile:
                    print(f"  Skipping {table_id}: no profiling data")
                    continue
                if profile is None:
                    profile = None  # explicit None

            # 生成列 embedding
            table_name = get_basename(table_path)
            col_embeddings = encoder.encode_table(
                table_name=table_name,
                headers=headers,
                data=data,
                n_rows=sample_rows,
                profile=profile,
            )

            embeddings_map[table_id] = col_embeddings

        except Exception as e:
            error_tables.append((table_id, str(e)))
            import traceback
            traceback.print_exc()
            continue

    # 统计
    print(f"\n{'='*50}")
    print(f"Embedding generation completed:")
    print(f"  - Success: {len(embeddings_map)}")
    print(f"  - Missing tables: {len(missing_tables)}")
    print(f"  - Errors: {len(error_tables)}")

    # ---- Embedding 质量诊断 ----
    if embeddings_map:
        diag = _diagnose_embeddings(embeddings_map)
        print(f"\n[Embedding Diagnostics]")
        print(f"  - Total columns:          {diag['total_cols']}")
        print(f"  - Embedding dim:          {diag['dim']}")
        print(f"  - Norm mean:              {diag['norm_mean']:.4f} (L2 normalized should be ~1.0)")
        print(f"  - Norm std:               {diag['norm_std']:.6f}")
        print(f"  - Pairwise cosine sim:    {diag['cosine_sim_mean']:.4f} (lower = more diverse)")
        print(f"  - Pairwise cosine sim std:{diag['cosine_sim_std']:.4f}")
        print(f"  - Per-dim std (mean):     {diag['per_dim_std_mean']:.6f} (higher = more signal)")
        print(f"  - Per-dim std (min):      {diag['per_dim_std_min']:.6f}")
        if diag['cosine_sim_mean'] > 0.95:
            print(f"  ⚠ WARNING: Cosine similarity > 0.95 — embeddings are nearly identical!")
            print(f"    Raw LLM hidden states may lack discriminative power for this task.")
            print(f"    Consider: LoRA fine-tuning (approach 3) or a contrastive projection head.")
        elif diag['cosine_sim_mean'] > 0.8:
            print(f"  ⚠ NOTE: Cosine similarity > 0.8 — embeddings have limited diversity.")
        else:
            print(f"  ✓ Embeddings show reasonable diversity.")

    # ---- 类别级诊断 ----
    if embeddings_map and fold_dir:
        cls_diag = _diagnose_class_separation(embeddings_map, fold_dir)
        if cls_diag:
            print(f"\n[Class Separation Diagnostics]")
            print(f"  - Within-class cosine sim:  {cls_diag['within_class_sim']:.4f}")
            print(f"  - Between-class cosine sim: {cls_diag['between_class_sim']:.4f}")
            print(f"  - Gap (within - between):   {cls_diag['class_separation_gap']:.4f}")
            print(f"  - Classes sampled:          {cls_diag['num_classes_sampled']}")
            if cls_diag['class_separation_gap'] < 0.02:
                print(f"  ⚠ WARNING: Gap < 0.02 — embeddings do NOT separate by class!")
                print(f"    Within-class and between-class similarity are nearly equal.")
                print(f"    The embedding diversity exists but does NOT align with column types.")
                print(f"    Raw LLM hidden states lack class-discriminative signal.")
            elif cls_diag['class_separation_gap'] < 0.1:
                print(f"  ⚠ NOTE: Gap = {cls_diag['class_separation_gap']:.4f} — weak class separation.")
                print(f"    Some signal exists but may not be sufficient for good classification.")
            else:
                print(f"  ✓ Good class separation — embeddings carry type-discriminative signal.")

    if missing_tables and len(missing_tables) <= 10:
        print(f"\nMissing tables: {missing_tables}")
    elif missing_tables:
        print(f"\nFirst 10 missing tables: {missing_tables[:10]}")

    if error_tables and len(error_tables) <= 5:
        print(f"\nError tables: {error_tables}")
    elif error_tables:
        print(f"\nFirst 5 error tables: {error_tables[:5]}")

    return embeddings_map


def main():
    args = parse_args()

    layers = parse_layers(args.layers)

    print("=" * 60)
    print("CTA Embedding Generation — LLM Hidden States")
    print("=" * 60)
    print(f"Model path:       {args.model_path}")
    print(f"Fold directory:   {args.fold_dir}")
    print(f"Table directory:  {args.table_dir}")
    print(f"Output path:      {args.output_path}")
    print(f"Sample rows:      {args.sample_rows}")
    print(f"Max prefix len:   {args.max_prefix_length}")
    print(f"Layers:           {layers}")
    print(f"Profiling:        {args.profilling_path or '(none)'}")
    print(f"Require profile:  {args.require_profile}")
    if args.gpu_id is not None:
        print(f"GPU ID:           {args.gpu_id}")
    print("=" * 60)

    # 1. 加载 table_id
    print("\n[Step 1] Loading table IDs from fold files...")
    table_ids = load_fold_table_ids(args.fold_dir)
    if not table_ids:
        print("Error: No table IDs found!")
        sys.exit(1)

    # 2. 初始化 encoder
    print("\n[Step 2] Initializing LLM encoder...")
    device = f"cuda:{args.gpu_id}" if args.gpu_id is not None else "auto"
    encoder = LLMHiddenStateEncoder(
        model_path=args.model_path,
        max_prefix_length=args.max_prefix_length,
        layers=layers,
        device=device,
    )

    # 3. 生成 embedding
    print("\n[Step 3] Generating embeddings...")
    embeddings_map = generate_embeddings(
        table_ids=table_ids,
        table_dir=args.table_dir,
        encoder=encoder,
        file_type=args.file_type,
        sample_rows=args.sample_rows,
        profilling_path=args.profilling_path,
        require_profile=args.require_profile,
        fold_dir=args.fold_dir,
    )

    # 4. 保存
    print("\n[Step 4] Saving embeddings...")
    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, "wb") as f:
        pickle.dump(embeddings_map, f)

    print(f"Embeddings saved to: {args.output_path}")
    print(f"Total tables: {len(embeddings_map)}")

    if embeddings_map:
        sample_id = list(embeddings_map.keys())[0]
        sample_emb = embeddings_map[sample_id]
        print(f"\nSample embedding info:")
        print(f"  - Table ID: {sample_id}")
        print(f"  - Shape: {sample_emb.shape}")
        print(f"  - Dtype: {sample_emb.dtype}")

    print(f"\n{'='*60}")
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
