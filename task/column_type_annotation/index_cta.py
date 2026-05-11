"""
CTA (Column Type Annotation) 任务的 Embedding 生成脚本

功能：
1. 读取 fold CSV 文件，获取所有需要编码的表
2. 使用 Qwen Embedding 模型（或本地训练的模型）生成列的 embedding
3. 输出格式为 pickle 字典：{ table_id: tensor(num_cols, embedding_dim) }

输出格式兼容 train_cta.py
"""

import os
import sys
import pickle
import argparse
import uuid
import random
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Optional

# 设置项目根目录
root_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(
    root_dir, 'task', 'dataset_discovery', 'Tabert'))

# 导入所需模块
try:
    from EmbeddingModel import EmbeddingModelFactory
    from add_profilling import read_table, get_table_profilling
except ImportError as e:
    print(f"Import error: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CTA Embedding 生成脚本')

    # 数据路径
    parser.add_argument('--fold_dir', type=str, required=True,
                        help='包含 fold_*.csv 文件的目录')
    parser.add_argument('--table_dir', type=str, required=True,
                        help='原始表文件所在目录')
    parser.add_argument('--output_path', type=str, required=True,
                        help='输出 embedding pickle 文件路径')

    # 模型配置
    parser.add_argument('--encoder', type=str, default='qwen',
                        choices=['qwen', 'tabert', 'bge'],
                        help='Embedding 模型类型')
    parser.add_argument('--qwen_model', type=str, default='qwen3-embedding-0.6b',
                        help='Qwen 模型名称')
    parser.add_argument('--tabert_model_path', type=str,
                        default='./model/tabert_base_k3/model.bin',
                        help='TabERT 模型路径')
    parser.add_argument('--local_model_path', type=str, default=None,
                        help='本地训练的模型权重路径 (.pth)')
    parser.add_argument('--base_model_path', type=str, default=None,
                        help='基础模型路径')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Embedding 批次大小')

    # 数据处理配置
    parser.add_argument('--sample_rows', type=int, default=3,
                        help='每列采样行数')
    parser.add_argument('--file_type', type=str, default='.csv',
                        help='表文件类型')
    parser.add_argument('--profilling_path', type=str, default=None,
                        help='Profilling 数据路径（可选）')
    parser.add_argument('--require_profile', action='store_true', default=True,
                        help='是否要求必须有 profilling 数据（默认 True，没有则跳过表；设为 False 则即使没有也继续生成 embedding）')
    parser.add_argument('--no_require_profile', action='store_false', dest='require_profile',
                        help='关闭必须要求 profilling 数据（没有也继续生成 embedding）')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='噪声比例 (0-1)，将对应比例的单元格替换为随机字符串')
    parser.add_argument('--gpu_id', type=int, default=None,
                        help='指定使用的 GPU ID（如 0, 1, 2），默认自动选择')

    return parser.parse_args()


def load_fold_table_ids(fold_dir: str) -> List[str]:
    """从 fold 文件中加载所有唯一的 table_id

    Args:
        fold_dir: fold CSV 文件所在目录

    Returns:
        唯一 table_id 列表
    """
    all_table_ids = set()

    for i in range(5):
        fold_path = os.path.join(fold_dir, f'fold_{i}.csv')
        if os.path.exists(fold_path):
            df = pd.read_csv(fold_path)
            all_table_ids.update(df['table_id'].unique())
            print(
                f"Loaded fold_{i}.csv: {len(df['table_id'].unique())} unique tables")
        else:
            print(f"Warning: fold_{i}.csv not found")

    print(f"Total unique tables: {len(all_table_ids)}")
    return list(all_table_ids)


def find_table_file(table_id: str, table_dir: str, file_type: str = '.csv') -> Optional[str]:
    """查找表文件路径

    Args:
        table_id: 表 ID
        table_dir: 表文件目录
        file_type: 文件类型

    Returns:
        表文件完整路径，找不到返回 None
    """
    # 尝试直接匹配
    direct_path = os.path.join(table_dir, f"{table_id}{file_type}")
    if os.path.exists(direct_path):
        return direct_path

    # 尝试在子目录中查找
    for root, dirs, files in os.walk(table_dir):
        for f in files:
            if f == f"{table_id}{file_type}" or f.startswith(table_id):
                return os.path.join(root, f)
    print(f"Warning: Table file for {table_id} not found in {table_dir}")
    return None


def apply_noise(data: List[List[str]], noise_ratio: float) -> List[List[str]]:
    """对表格数据施加噪声，以 noise_ratio 概率将单元格替换为随机字符串

    Args:
        data: 表格数据 (2D list)
        noise_ratio: 噪声比例 (0-1)

    Returns:
        修改后的 data（原地修改并返回）
    """
    if noise_ratio <= 0:
        return data
    for row in data:
        for j in range(len(row)):
            if random.random() < noise_ratio:
                row[j] = uuid.uuid4().hex[:8]
    return data


def generate_embeddings(
    table_ids: List[str],
    table_dir: str,
    embedding_model,
    file_type: str = '.csv',
    sample_rows: int = 3,
    profilling_path: str = None,
    require_profile: bool = True,
    noise: float = 0.0
) -> Dict[str, torch.Tensor]:
    """为所有表生成 embedding

    Args:
        table_ids: 表 ID 列表
        table_dir: 表文件目录
        embedding_model: Embedding 模型实例
        file_type: 文件类型
        sample_rows: 采样行数
        profilling_path: Profilling 数据路径

    Returns:
        embedding 字典：{ table_id: tensor(num_cols, embedding_dim) }
    """
    embeddings_map = {}
    missing_tables = []
    error_tables = []

    for table_id in tqdm(table_ids, desc="Generating embeddings"):
        # 查找表文件
        table_path = find_table_file(table_id, table_dir, file_type)

        if table_path is None:
            missing_tables.append(table_id)
            continue

        try:
            # 读取表数据
            df = read_table(table_path, sample_rows=sample_rows)
            headers = df.columns.tolist()
            data = df.values.tolist()

            # 获取 profilling 数据（如果有）
            profile = {}
            if profilling_path:
                profile = get_table_profilling(table_path, profilling_path) or {}
                if not profile and require_profile:
                    print(f"  Skipping {table_id}: no profilling data found")
                    continue
                if not profile:
                    print(f"  Warning: {table_id} has no profilling data, continuing with empty profile")

            # 施加噪声
            if noise > 0:
                data = apply_noise(data, noise)

            # 生成列 embedding
            column_embeddings = embedding_model.encode_columns(
                table_id=table_id,
                headers=headers,
                data=data,
                sample_rows=sample_rows,
                profilling_data=profile
            )

            # 转换为 tensor
            embeddings_map[table_id] = torch.tensor(
                column_embeddings, dtype=torch.float32)

        except Exception as e:
            error_tables.append((table_id, str(e)))
            continue

    # 打印统计信息
    print(f"\n{'='*50}")
    print(f"Embedding generation completed:")
    print(f"  - Success: {len(embeddings_map)}")
    print(f"  - Missing tables: {len(missing_tables)}")
    print(f"  - Errors: {len(error_tables)}")

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

    print("="*60)
    print("CTA Embedding Generation")
    print("="*60)
    print(f"Fold directory: {args.fold_dir}")
    print(f"Table directory: {args.table_dir}")
    print(f"Output path: {args.output_path}")
    print(f"Encoder: {args.encoder}")
    print(f"Noise: {args.noise}")
    if args.encoder == 'tabert':
        print(f"TabERT model path: {args.tabert_model_path}")
    else:
        print(f"Model: {args.qwen_model}")
    if args.local_model_path:
        print(f"Local model: {args.local_model_path}")
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        print(f"Using GPU: {args.gpu_id}")
    print("="*60)

    # 1. 加载所有 table_id
    print("\n[Step 1] Loading table IDs from fold files...")
    table_ids = load_fold_table_ids(args.fold_dir)

    if not table_ids:
        print("Error: No table IDs found!")
        sys.exit(1)

    # 2. 初始化 embedding 模型
    print(f"\n[Step 2] Initializing embedding model (encoder={args.encoder})...")
    if args.encoder == 'tabert':
        embedding_model = EmbeddingModelFactory.create(
            'tabert',
            model_path=args.tabert_model_path,
        )
    else:
        # qwen / bge
        embedding_model = EmbeddingModelFactory.create(
            args.encoder,
            model_name=args.qwen_model,
            batch_size=args.batch_size,
            local_model_path=args.local_model_path,
            base_model_path=args.base_model_path,
        )

    # 3. 生成 embedding
    print("\n[Step 3] Generating embeddings...")
    embeddings_map = generate_embeddings(
        table_ids=table_ids,
        table_dir=args.table_dir,
        embedding_model=embedding_model,
        file_type=args.file_type,
        sample_rows=args.sample_rows,
        profilling_path=args.profilling_path,
        require_profile=args.require_profile,
        noise=args.noise
    )

    # 4. 保存结果
    print("\n[Step 4] Saving embeddings...")
    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, 'wb') as f:
        pickle.dump(embeddings_map, f)

    print(f"Embeddings saved to: {args.output_path}")
    print(f"Total tables: {len(embeddings_map)}")

    # 打印样本信息
    if embeddings_map:
        sample_id = list(embeddings_map.keys())[0]
        sample_emb = embeddings_map[sample_id]
        print(f"\nSample embedding info:")
        print(f"  - Table ID: {sample_id}")
        print(f"  - Shape: {sample_emb.shape}")
        print(f"  - Dtype: {sample_emb.dtype}")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    main()
