"""
CTA Embedding 质量评估脚本

为 SemanticCTA 论文 Q2C 实验 (Table 6) 计算 4 个 embedding 质量指标：
  1. kNN 标签纯度 (k=5, k=10)
  2. 线性探针 Macro-F1
  3. 类内/类间余弦相似度比值
  4. 描述-标签一致性 (可选，需要 profiling JSON)

用法：
    python column_type_annotation/eval_embedding_quality.py \
        --embedding_path results/cta/embeddings.pkl \
        --fold_dir datasets/gittables-semtab22-db-all_wrangled/ \
        --output_path results/cta/quality_metrics.json \
        --profilling_path profile/gittables-semtab22-db-all_wrangled_single_column_profilling_row0.json
"""

import os
import sys
import json
import pickle
import argparse
from typing import Dict, Optional, Tuple
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


# ==================== 工具函数 ====================

def load_embeddings(path: str) -> Dict[str, torch.Tensor]:
    """加载 embedding pickle 文件"""
    print(f"Loading embeddings from: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"  Loaded {len(data)} tables")
    if data:
        sample_key = list(data.keys())[0]
        print(f"  Sample shape: {data[sample_key].shape}")
    return data


def load_fold_data(fold_dir: str) -> pd.DataFrame:
    """加载所有 fold CSV 文件"""
    all_rows = []
    for i in range(5):
        fp = os.path.join(fold_dir, f"fold_{i}.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            df['fold'] = i
            all_rows.append(df)
    if not all_rows:
        raise FileNotFoundError(f"No fold files found in {fold_dir}")
    labels_df = pd.concat(all_rows, ignore_index=True)
    # 过滤掉 class_id = -1 的样本
    labels_df = labels_df[labels_df['class_id'] != -1]
    print(f"Loaded {len(labels_df)} labeled samples from {len(all_rows)} folds")
    print(f"  Unique classes: {labels_df['class_id'].nunique()}")
    return labels_df


def match_embeddings_and_labels(
    embeddings_map: Dict[str, torch.Tensor],
    labels_df: pd.DataFrame
) -> Tuple[torch.Tensor, torch.Tensor]:
    """将 embedding 与标签对齐，返回 (embeddings, labels)"""
    embs, labels = [], []
    skipped = 0

    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Matching embeddings"):
        tid = row['table_id']
        cidx = int(row['col_idx'])
        label = int(row['class_id'])

        if tid in embeddings_map:
            table_emb = embeddings_map[tid]
            if cidx < len(table_emb):
                embs.append(table_emb[cidx])
                labels.append(label)
            else:
                skipped += 1
        else:
            skipped += 1

    if skipped > 0:
        print(f"  Skipped {skipped} samples (no embedding or col_idx out of range)")

    if not embs:
        raise ValueError("No valid embeddings found after matching")

    embs = torch.stack(embs)
    labels = torch.tensor(labels)
    print(f"  Matched {len(embs)} samples")
    return embs, labels


def normalize_embeddings(embs: torch.Tensor) -> torch.Tensor:
    """L2 归一化"""
    return torch.nn.functional.normalize(embs, p=2, dim=-1)


# ==================== 指标 1: kNN 标签纯度 ====================

def compute_knn_purity(embs: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """计算 kNN 标签纯度"""
    # 使用 sklearn 的 NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    nbrs.fit(embs.numpy())
    distances, indices = nbrs.kneighbors(embs.numpy())

    # 跳过第一个（自己）
    neighbor_labels = labels[indices[:, 1:]]
    target_labels = labels.unsqueeze(1).expand_as(neighbor_labels)

    # 计算每个样本的邻居中同类标签的比例
    purity = (neighbor_labels == target_labels).float().mean(dim=1).mean().item()
    return purity


# ==================== 指标 2: 线性探针 Macro-F1 ====================

def compute_linear_probe_macro_f1(
    embs: torch.Tensor,
    labels: torch.Tensor,
    labels_df: pd.DataFrame
) -> float:
    """使用 5-fold CV 的线性探针评估 Macro-F1"""
    fold_f1_scores = []

    for fold_id in range(5):
        # 划分训练集和测试集
        train_mask = labels_df['fold'] != fold_id
        test_mask = labels_df['fold'] == fold_id

        # 由于 labels_df 已经和 embs/labels 对齐过，需要重新对齐索引
        # 这里简化处理：使用整个数据集的 fold 划分
        train_df = labels_df[train_mask].reset_index(drop=True)
        test_df = labels_df[test_mask].reset_index(drop=True)

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        # 提取训练和测试样本
        train_embs, train_labels = [], []
        for _, row in train_df.iterrows():
            tid = row['table_id']
            cidx = int(row['col_idx'])
            if tid in labels_df['table_id'].values:
                # 通过索引查找
                idx = labels_df[(labels_df['table_id'] == tid) &
                               (labels_df['col_idx'] == cidx)].index
                if len(idx) > 0:
                    train_embs.append(embs[idx[0]])
                    train_labels.append(labels[idx[0]])

        test_embs, test_labels = [], []
        for _, row in test_df.iterrows():
            tid = row['table_id']
            cidx = int(row['col_idx'])
            if tid in labels_df['table_id'].values:
                idx = labels_df[(labels_df['table_id'] == tid) &
                               (labels_df['col_idx'] == cidx)].index
                if len(idx) > 0:
                    test_embs.append(embs[idx[0]])
                    test_labels.append(labels[idx[0]])

        if len(train_embs) == 0 or len(test_embs) == 0:
            continue

        train_X = torch.stack(train_embs).numpy()
        train_y = torch.tensor(train_labels).numpy()
        test_X = torch.stack(test_embs).numpy()
        test_y = torch.tensor(test_labels).numpy()

        # 训练逻辑斯谛回归
        clf = LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            n_jobs=-1,
            random_state=42
        )
        clf.fit(train_X, train_y)

        # 预测并计算 Macro-F1
        y_pred = clf.predict(test_X)
        f1 = f1_score(test_y, y_pred, average='macro')
        fold_f1_scores.append(f1)

    if not fold_f1_scores:
        return 0.0

    mean_f1 = np.mean(fold_f1_scores)
    print(f"  Fold-wise F1 scores: {[f'{f:.4f}' for f in fold_f1_scores]}")
    return mean_f1


# ==================== 指标 3: 类内/类间余弦相似度 ====================

def compute_class_separation(embs: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float, float]:
    """计算类内和类间余弦相似度，以及它们的差距"""
    # L2 归一化
    embs_norm = normalize_embeddings(embs)

    unique_labels = torch.unique(labels)
    within_sims = []
    between_sims = []

    # 采样计算以提高效率
    max_samples_per_class = 500
    max_between_samples = 5000

    for label in tqdm(unique_labels, desc="Computing class separation"):
        mask = labels == label
        class_embs = embs_norm[mask]

        if len(class_embs) < 2:
            continue

        # 类内相似度：采样同类对
        n_class = len(class_embs)
        n_within_samples = min(max_samples_per_class, n_class * (n_class - 1) // 2)

        if n_within_samples > 0:
            indices = torch.randperm(n_class)[:min(10, n_class)]
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    sim = torch.dot(class_embs[indices[i]], class_embs[indices[j]]).item()
                    within_sims.append(sim)

        # 类间相似度：采样该类与其它类的样本对
        other_mask = labels != label
        other_embs = embs_norm[other_mask]

        n_between = min(max_between_samples // len(unique_labels), len(class_embs) * 10)
        for _ in range(min(n_between, 100)):
            i = torch.randint(len(class_embs), (1,)).item()
            j = torch.randint(len(other_embs), (1,)).item()
            sim = torch.dot(class_embs[i], other_embs[j]).item()
            between_sims.append(sim)

    if not within_sims or not between_sims:
        return 0.0, 0.0, 0.0

    within_mean = np.mean(within_sims)
    between_mean = np.mean(between_sims)
    gap = within_mean - between_mean

    return within_mean, between_mean, gap


# ==================== 指标 4: 描述-标签一致性 ====================

def normalize_text(text: str) -> str:
    """标准化文本用于匹配"""
    return text.strip().lower().replace("_", " ").replace("-", " ")


def match_description_to_class(
    description: str,
    class_names: Dict[int, str]
) -> Tuple[int, float]:
    """将描述文本匹配到类别标签"""
    desc_norm = normalize_text(description)

    best_id = -1
    best_score = 0.0

    for cid, cname in class_names.items():
        cname_norm = normalize_text(cname)

        # 子串匹配
        if cname_norm in desc_norm or desc_norm in cname_norm:
            return cid, 1.0

        # SequenceMatcher 模糊匹配
        ratio = SequenceMatcher(None, desc_norm, cname_norm).ratio()
        if ratio > best_score:
            best_score = ratio
            best_id = cid

    # 设置阈值，低于阈值认为不匹配
    if best_score < 0.3:
        return -1, 0.0
    return best_id, best_score


def compute_description_label_agreement(
    labels_df: pd.DataFrame,
    profilling_path: str,
    class_names_path: Optional[str] = None
) -> Optional[float]:
    """计算描述与标签的一致性"""
    if not os.path.exists(profilling_path):
        print(f"  Profiling file not found: {profilling_path}")
        return None

    print(f"Loading profiling data from: {profilling_path}")
    with open(profilling_path, 'r', encoding='utf-8') as f:
        profilling_data = json.load(f)

    # 加载类别名称
    if class_names_path and os.path.exists(class_names_path):
        with open(class_names_path, 'r', encoding='utf-8') as f:
            class_names_data = json.load(f)
        if isinstance(class_names_data, dict):
            class_names = {int(k): v for k, v in class_names_data.items()}
        else:
            class_names = {i: v for i, v in enumerate(class_names_data)}
    else:
        # 从标签中推断类别名称
        print("  No class_names.json provided, using numeric labels")
        class_names = {cid: str(cid) for cid in labels_df['class_id'].unique()}

    agreements = []
    total = 0

    # 注意：这里简化处理，实际需要 col_idx → column_name 的映射
    # 由于 profiling JSON 使用列名而非索引，这里只做演示性实现
    # 实际使用时需要根据 table 文件获取列名映射

    print("  Warning: Description-label agreement requires col_idx → column_name mapping")
    print("  This is a simplified implementation")

    # 如果 profiling 数据中有列名信息，可以尝试匹配
    # 这里返回 None 表示该指标在此配置下不可用
    return None


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(
        description="CTA Embedding 质量评估 - 计算 4 个质量指标"
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        required=True,
        help="Embedding pickle 文件路径"
    )
    parser.add_argument(
        "--fold_dir",
        type=str,
        required=True,
        help="Fold CSV 目录"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="输出 JSON 路径（默认与 embedding 同目录）"
    )
    parser.add_argument(
        "--profilling_path",
        type=str,
        default=None,
        help="Profiling JSON 路径（可选，用于描述-标签一致性）"
    )
    parser.add_argument(
        "--class_names_path",
        type=str,
        default=None,
        help="类别名称 JSON 路径（可选）"
    )

    args = parser.parse_args()

    # 设置默认输出路径
    if args.output_path is None:
        embedding_dir = os.path.dirname(args.embedding_path)
        args.output_path = os.path.join(embedding_dir, "quality_metrics.json")

    print("="*60)
    print("CTA Embedding 质量评估")
    print("="*60)

    # 加载数据
    embeddings_map = load_embeddings(args.embedding_path)
    labels_df = load_fold_data(args.fold_dir)

    # 对齐 embedding 和标签
    embs, labels = match_embeddings_and_labels(embeddings_map, labels_df)
    print(f"\nEmbedding shape: {embs.shape}")
    print(f"Labels shape: {labels.shape}")

    results = {}

    # 指标 1: kNN 标签纯度
    print("\n" + "="*60)
    print("指标 1: kNN 标签纯度")
    print("="*60)
    embs_norm = normalize_embeddings(embs)
    purity_k5 = compute_knn_purity(embs_norm, labels, k=5)
    purity_k10 = compute_knn_purity(embs_norm, labels, k=10)
    results["knn_purity_k5"] = float(purity_k5)
    results["knn_purity_k10"] = float(purity_k10)
    print(f"  k=5:  {purity_k5:.4f}")
    print(f"  k=10: {purity_k10:.4f}")

    # 指标 2: 线性探针 Macro-F1
    print("\n" + "="*60)
    print("指标 2: 线性探针 Macro-F1 (5-fold CV)")
    print("="*60)
    macro_f1 = compute_linear_probe_macro_f1(embs, labels, labels_df)
    results["linear_probe_macro_f1"] = float(macro_f1)
    print(f"  Macro-F1: {macro_f1:.4f}")

    # 指标 3: 类内/类间余弦相似度
    print("\n" + "="*60)
    print("指标 3: 类内/类间余弦相似度")
    print("="*60)
    within_sim, between_sim, gap = compute_class_separation(embs, labels)
    results["within_class_cosine_sim"] = float(within_sim)
    results["between_class_cosine_sim"] = float(between_sim)
    results["class_separation_gap"] = float(gap)
    print(f"  Within-class:  {within_sim:.4f}")
    print(f"  Between-class: {between_sim:.4f}")
    print(f"  Gap (W - B):   {gap:.4f}")

    # 指标 4: 描述-标签一致性
    print("\n" + "="*60)
    print("指标 4: 描述-标签一致性")
    print("="*60)
    if args.profilling_path:
        agreement = compute_description_label_agreement(
            labels_df,
            args.profilling_path,
            args.class_names_path
        )
        results["description_label_agreement"] = (
            float(agreement) if agreement is not None else None
        )
        if agreement is not None:
            print(f"  Agreement rate: {agreement:.4f}")
        else:
            print("  Skipped (requires col_idx → column_name mapping)")
    else:
        results["description_label_agreement"] = None
        print("  Skipped (no profiling JSON provided)")

    # 保存结果
    print("\n" + "="*60)
    print("保存结果")
    print("="*60)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to: {args.output_path}")

    # 打印摘要
    print("\n" + "="*60)
    print("结果摘要")
    print("="*60)
    for key, value in results.items():
        if value is not None:
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: null")


if __name__ == "__main__":
    main()
