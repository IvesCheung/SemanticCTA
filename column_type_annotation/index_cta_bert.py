"""
CTA Embedding 生成脚本 — 基于 BERT / Sentence-BERT 模型

核心思路：
    使用预训练的 BERT 或 Sentence-BERT 模型为表格的每一列生成语义嵌入向量，
    用于 SemanticCTA 论文中的 Q2A 跨系统对比实验。

    支持两种模式：
    1. Normal 模式（默认）：将每列序列化为 "Column '{name}' contains values: {val1}, {val2}, ..."
       然后通过 BERT 编码，取 [CLS] token 的隐藏状态作为列 embedding。
    2. Description 模式（--use_description）：使用 profiling 数据（如果存在），
       通过 add_profilling.encode_column() 生成更丰富的列描述文本，再编码。

输出格式：
    pickle 字典 { table_id: tensor(num_cols, 768) }
    与 index_cta_llm_hidden.py 和 index_cta.py 完全兼容，可直接对接 train_cta.py。

用法示例：
    # Normal 模式 - 使用 BERT-base
    python column_type_annotation/index_cta_bert.py \
        --encoder_model bert-base-uncased \
        --fold_dir datasets/gittables-semtab22-db-all_wrangled/ \
        --table_dir datasets/gittables-semtab22-db-all_wrangled/ \
        --output_path results/cta_bert/gittables/embeddings.pkl \
        --sample_rows 5

    # Description 模式 - 使用 Sentence-BERT + profiling 数据
    python column_type_annotation/index_cta_bert.py \
        --encoder_model sentence-transformers/all-mpnet-base-v2 \
        --fold_dir datasets/gittables-semtab22-db-all_wrangled/ \
        --table_dir datasets/gittables-semtab22-db-all_wrangled/ \
        --output_path results/cta_bert/gittables/embeddings_desc.pkl \
        --sample_rows 5 \
        --profilling_path results/profiling/gittables.json \
        --use_description

    # 使用 GPU 加速
    python column_type_annotation/index_cta_bert.py \
        ... \
        --gpu_id 0
"""

import os
import sys
import pickle
import argparse
import json
import pandas as pd
import torch
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

# ---------------------------------------------------------------------------
#  表格读取 & 查找工具（复制自 index_cta_llm_hidden.py，保持自包含）
# ---------------------------------------------------------------------------


def get_basename(file_path: str) -> str:
    """获取文件名（不含路径和扩展名）"""
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
    """在 table_dir 中查找表文件"""
    direct = os.path.join(table_dir, f"{table_id}{file_type}")
    if os.path.exists(direct):
        return direct
    for root, dirs, files in os.walk(table_dir):
        for f in files:
            if f == f"{table_id}{file_type}" or f.startswith(table_id):
                return os.path.join(root, f)
    return None


# ---------------------------------------------------------------------------
#  Profiling 数据加载（复制自 index_cta_llm_hidden.py）
# ---------------------------------------------------------------------------


@lru_cache(maxsize=128)
def _load_profilling_json(profilling_path: str):
    """加载并缓存 profiling JSON 文件"""
    with open(profilling_path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=128)
def _get_profilling_index(profilling_path: str):
    """构建 profiling 数据的快速查找索引"""
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
#  列序列化工具（复制自 add_profilling.py，保持一致性）
# ---------------------------------------------------------------------------


def encode_column(
    table_name: str,
    header_name: str,
    col_type: Optional[str],
    profile: Optional[dict],
    column_samples: Optional[List[str]] = None,
    split: str = ",",
    separator: str = "\n"
) -> str:
    """
    使用 profiling 信息编码列描述文本（复制自 add_profilling.py）

    Args:
        table_name: 表名
        header_name: 列名
        col_type: 列类型（从 profile 的 __type__ 获取）
        profile: 该列的 profiling 数据字典
        column_samples: 列的采样值列表
        split: 值分隔符
        separator: 字段分隔符

    Returns:
        编码后的文本描述
    """
    if not isinstance(profile, dict):
        profile = {}

    # 提取描述信息（排除 __type__ 和 __table__）
    description_values = [v for k, v in profile.items() if k not in ["__type__", "__table__"]]
    description = "\n+ ".join([f"{v}" for v in description_values if v]) if description_values else ""

    samples_text = split.join([str(sample) for sample in column_samples]) if column_samples else None

    _encoded_column = [
        f"Column name: {header_name}",
        f"Column type: {col_type}" if col_type else "",
        f"Description of this column: {description}",
        f"From table: {table_name}, {profile.get('__table__', '')}",
        f"Example data: {samples_text}" if samples_text else "",
    ]
    encoded_column = [_ for _ in _encoded_column if _ != ""]
    return separator.join(encoded_column)


def serialize_column_normal(header_name: str, column_samples: List[str]) -> str:
    """
    Normal 模式：简单的列序列化格式

    Args:
        header_name: 列名
        column_samples: 列的采样值列表

    Returns:
        "Column '{name}' contains values: {val1}, {val2}, ..."
    """
    values_str = ", ".join(str(v) for v in column_samples if pd.notna(v))
    return f'Column "{header_name}" contains values: {values_str}'


# ---------------------------------------------------------------------------
#  核心类：BERT Encoder
# ---------------------------------------------------------------------------


class BERTEncoder:
    """使用 BERT / Sentence-BERT 为列文本生成 embedding"""

    def __init__(
        self,
        encoder_model: str,
        device: str = "auto",
        max_length: int = 512,
    ):
        """
        Args:
            encoder_model: HuggingFace 模型名称或路径
            device: 设备（"auto", "cuda:0", "cpu" 等）
            max_length: 最大序列长度
        """
        from transformers import AutoTokenizer, AutoModel

        self.encoder_model = encoder_model
        self.max_length = max_length

        print(f"[BERTEncoder] Loading model: {encoder_model}")

        # 加载 tokenizer 和 model
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
        self.model = AutoModel.from_pretrained(encoder_model)

        # 设置设备
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                print(f"  Using CPU")
        else:
            self.device = torch.device(device)
            print(f"  Using device: {device}")

        self.model = self.model.to(self.device)
        self.model.eval()

        # 获取 hidden_size
        self.hidden_dim = self.model.config.hidden_size
        print(f"  hidden_size = {self.hidden_dim}")

    @torch.no_grad()
    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        批量编码文本，返回 [CLS] token 的隐藏状态

        Args:
            texts: 文本列表
            batch_size: 批大小
            show_progress: 是否显示进度条

        Returns:
            tensor: (num_texts, hidden_dim)，L2 归一化
        """
        all_embeddings = []

        # 批处理
        range_iter = range(0, len(texts), batch_size)
        if show_progress:
            range_iter = tqdm(range_iter, desc="Encoding texts")

        for i in range_iter:
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            # 移到设备
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            # 前向传播
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            # 提取 [CLS] token (第一个 token) 的隐藏状态
            # outputs.last_hidden_state: (batch, seq_len, hidden_dim)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_dim)

            all_embeddings.append(cls_embeddings.cpu())

        # 拼接所有批次
        embeddings = torch.cat(all_embeddings, dim=0)  # (num_texts, hidden_dim)

        # L2 归一化
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        return embeddings


# ---------------------------------------------------------------------------
#  主流程
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="CTA Embedding 生成 — 基于 BERT / Sentence-BERT"
    )

    # 模型参数
    parser.add_argument("--encoder_model", type=str, required=True,
                        help="HuggingFace 模型名称或路径 (例如: bert-base-uncased, sentence-transformers/all-mpnet-base-v2)")

    # 数据路径
    parser.add_argument("--fold_dir", type=str, required=True,
                        help="包含 fold_*.csv 文件的目录")
    parser.add_argument("--table_dir", type=str, required=True,
                        help="原始表文件所在目录")
    parser.add_argument("--output_path", type=str, required=True,
                        help="输出 embedding pickle 文件路径")

    # 数据处理
    parser.add_argument("--sample_rows", type=int, default=5,
                        help="每张表采样的行数 (默认 5)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="编码批大小 (默认 64)")
    parser.add_argument("--file_type", type=str, default=".csv",
                        help="表文件类型 (默认 .csv)")

    # Profiling
    parser.add_argument("--profilling_path", type=str, default=None,
                        help="Profiling JSON 路径（可选，用于 --use_description 模式）")
    parser.add_argument("--use_description", action="store_true", default=False,
                        help="使用 profiling 描述文本而非原始值（需要 --profilling_path）")

    # 设备
    parser.add_argument("--gpu_id", type=int, default=None,
                        help="GPU ID（默认 auto，即使用第一个可用的 GPU）")

    return parser.parse_args()


def generate_embeddings(
    table_ids: List[str],
    table_dir: str,
    encoder: BERTEncoder,
    file_type: str = ".csv",
    sample_rows: int = 5,
    batch_size: int = 64,
    profilling_path: Optional[str] = None,
    use_description: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    为所有表生成 BERT embedding

    Args:
        table_ids: 表 ID 列表
        table_dir: 表文件目录
        encoder: BERTEncoder 实例
        file_type: 文件扩展名
        sample_rows: 每表采样行数
        batch_size: 编码批大小
        profilling_path: profiling JSON 路径
        use_description: 是否使用描述模式

    Returns:
        {table_id: tensor(num_cols, hidden_dim)}
    """
    embeddings_map = {}
    missing_tables = []
    error_tables = []

    for table_id in tqdm(table_ids, desc="Processing tables"):
        table_path = find_table_file(table_id, table_dir, file_type)
        if table_path is None:
            missing_tables.append(table_id)
            continue

        try:
            # 读取表
            df = pd.read_csv(table_path, nrows=sample_rows, lineterminator="\n")
            df = df.dropna(how="all")
            headers = df.columns.tolist()

            # 提取每列的采样值
            column_samples_list = []
            for col_name in headers:
                col_data = df[col_name].dropna().astype(str).tolist()
                column_samples_list.append(col_data)

            # 根据 mode 生成文本描述
            column_texts = []
            table_name = get_basename(table_path)

            if use_description and profilling_path:
                # Description 模式：使用 profiling 数据
                profile = get_table_profilling(table_path, profilling_path)

                for col_name, col_samples in zip(headers, column_samples_list):
                    if profile and col_name in profile:
                        col_data = profile[col_name]
                        if isinstance(col_data, dict):
                            col_type = col_data.get("__type__", "unknown")
                            # 过滤掉 __type__ 和 __table__
                            col_profile = {k: v for k, v in col_data.items()
                                         if k not in ["__type__", "__table__"]}
                        else:
                            col_type = "unknown"
                            col_profile = None

                        text = encode_column(
                            table_name=table_name,
                            header_name=col_name,
                            col_type=col_type,
                            profile=col_profile or {},
                            column_samples=col_samples[:sample_rows],
                        )
                    else:
                        # 如果没有 profiling 数据，回退到 normal 模式
                        text = serialize_column_normal(col_name, col_samples[:sample_rows])

                    column_texts.append(text)
            else:
                # Normal 模式：简单序列化
                for col_name, col_samples in zip(headers, column_samples_list):
                    text = serialize_column_normal(col_name, col_samples[:sample_rows])
                    column_texts.append(text)

            # 批量编码所有列
            if column_texts:
                col_embeddings = encoder.encode_texts(
                    column_texts,
                    batch_size=batch_size,
                    show_progress=False,
                )
                embeddings_map[table_id] = col_embeddings
            else:
                # 空表：创建空 tensor
                embeddings_map[table_id] = torch.empty(0, encoder.hidden_dim)

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

    # 打印前几个错误（如果有）
    if error_tables and len(error_tables) <= 5:
        print(f"\nError tables: {error_tables}")
    elif error_tables:
        print(f"\nFirst 5 error tables: {error_tables[:5]}")

    if missing_tables and len(missing_tables) <= 10:
        print(f"\nMissing tables: {missing_tables}")
    elif missing_tables:
        print(f"\nFirst 10 missing tables: {missing_tables[:10]}")

    return embeddings_map


def main():
    args = parse_args()

    print("=" * 60)
    print("CTA Embedding Generation — BERT / Sentence-BERT")
    print("=" * 60)
    print(f"Encoder model:    {args.encoder_model}")
    print(f"Fold directory:   {args.fold_dir}")
    print(f"Table directory:  {args.table_dir}")
    print(f"Output path:      {args.output_path}")
    print(f"Sample rows:      {args.sample_rows}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Profiling:        {args.profilling_path or '(none)'}")
    print(f"Use description:  {args.use_description}")
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
    print("\n[Step 2] Initializing BERT encoder...")
    device = f"cuda:{args.gpu_id}" if args.gpu_id is not None else "auto"
    encoder = BERTEncoder(
        encoder_model=args.encoder_model,
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
        batch_size=args.batch_size,
        profilling_path=args.profilling_path,
        use_description=args.use_description,
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

        # 统计总列数
        total_cols = sum(emb.shape[0] for emb in embeddings_map.values())
        print(f"\nTotal columns across all tables: {total_cols}")

    print(f"\n{'='*60}")
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
