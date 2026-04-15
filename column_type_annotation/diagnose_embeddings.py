"""
CTA Embedding 质量诊断脚本

直接读取已生成的 embedding pkl 文件 + fold CSV，输出：
  1. 整体统计：余弦相似度分布、维度方差
  2. 类别级分析：within-class vs between-class 相似度，判断 embedding 是否按类别聚簇

用法：
    python column_type_annotation/diagnose_embeddings.py \
        --embedding_path results/cta/embeddings.pkl \
        --fold_dir datasets/gittables-semtab22/
"""

import os
import sys
import pickle
import argparse
import torch
import pandas as pd
import numpy as np
from typing import Dict, Optional


def load_embeddings(path: str) -> Dict[str, torch.Tensor]:
    print(f"Loading embeddings from: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"  Loaded {len(data)} tables")
    if data:
        sample_key = list(data.keys())[0]
        print(f"  Sample shape: {data[sample_key].shape}")
    return data


def diagnose_overall(embeddings_map: Dict[str, torch.Tensor]):
    """整体 embedding 统计"""
    all_embs = torch.cat(list(embeddings_map.values()), dim=0)
    total_cols, dim = all_embs.shape

    norms = torch.norm(all_embs, p=2, dim=1)
    per_dim_std = all_embs.std(dim=0)

    # Pairwise cosine sim（采样）
    n_sample = min(2000, total_cols)
    indices = torch.randperm(total_cols)[:n_sample]
    sample = all_embs[indices]
    # 检查是否 L2 normalized
    sample_norms = torch.norm(sample, p=2, dim=1)
    if sample_norms.mean() < 0.9:
        sample = torch.nn.functional.normalize(sample, p=2, dim=-1)

    sim_matrix = sample @ sample.T
    mask = torch.triu(torch.ones_like(sim_matrix, dtype=torch.bool), diagonal=1)
    sim_values = sim_matrix[mask]

    # 分布百分位
    percentiles = [10, 25, 50, 75, 90]
    pct_values = torch.quantile(sim_values, torch.tensor([p/100 for p in percentiles]))

    print(f"\n{'='*60}")
    print(f"[Overall Statistics]")
    print(f"  Total columns:     {total_cols}")
    print(f"  Embedding dim:     {dim}")
    print(f"  Norm mean:         {norms.mean():.4f}")
    print(f"  Norm std:          {norms.std():.6f}")
    print(f"  Per-dim std (mean):{per_dim_std.mean():.6f}")
    print(f"  Per-dim std (min): {per_dim_std.min():.6f}")
    print(f"\n  Pairwise cosine similarity:")
    print(f"    Mean:   {sim_values.mean():.4f}")
    print(f"    Std:    {sim_values.std():.4f}")
    for p, v in zip(percentiles, pct_values):
        print(f"    P{p:02d}:    {v:.4f}")


def diagnose_class_separation(embeddings_map: Dict[str, torch.Tensor], fold_dir: str):
    """类别级诊断：同类 vs 异类相似度"""
    # 加载标签
    all_rows = []
    for i in range(5):
        fp = os.path.join(fold_dir, f"fold_{i}.csv")
        if os.path.exists(fp):
            all_rows.append(pd.read_csv(fp))
    if not all_rows:
        print("No fold files found, skipping class diagnostics.")
        return

    labels_df = pd.concat(all_rows, ignore_index=True)
    labels_df = labels_df[labels_df['class_id'] != -1]
    print(f"\n  Labels loaded: {len(labels_df)} rows, {labels_df['class_id'].nunique()} classes")

    # 构建对
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
        print("  Too few matched embeddings, skipping.")
        return

    embs = torch.stack(embs)
    class_ids = torch.tensor(class_ids)

    # L2 normalize
    embs = torch.nn.functional.normalize(embs, p=2, dim=-1)

    class_counts = torch.bincount(class_ids)
    valid_classes = (class_counts >= 5).nonzero(as_tuple=True)[0]
    print(f"  Classes with >= 5 samples: {len(valid_classes)}")

    # 计算每个类的中心
    class_centroids = {}
    for cls_id in valid_classes:
        cls_mask = class_ids == cls_id.item()
        class_centroids[cls_id.item()] = embs[cls_mask].mean(dim=0)

    # Within-class: 随机采样同类对
    within_sims = []
    for cls_id in valid_classes[:80]:
        cls_mask = class_ids == cls_id.item()
        cls_embs = embs[cls_mask]
        n_cls = len(cls_embs)
        if n_cls < 2:
            continue
        n_pairs = min(300, n_cls * (n_cls - 1) // 2)
        for _ in range(n_pairs):
            i, j = torch.randperm(n_cls)[:2]
            within_sims.append(torch.dot(cls_embs[i], cls_embs[j]).item())

    # Between-class: 随机采样异类对
    between_sims = []
    n_total = len(embs)
    for _ in range(len(within_sims)):
        i = torch.randint(n_total, (1,)).item()
        j = torch.randint(n_total, (1,)).item()
        if class_ids[i] != class_ids[j]:
            between_sims.append(torch.dot(embs[i], embs[j]).item())

    if not within_sims or not between_sims:
        print("  Not enough pairs sampled.")
        return

    within_mean = np.mean(within_sims)
    within_std = np.std(within_sims)
    between_mean = np.mean(between_sims)
    between_std = np.std(between_sims)
    gap = within_mean - between_mean

    print(f"\n{'='*60}")
    print(f"[Class Separation]")
    print(f"  Within-class sim:  {within_mean:.4f} ± {within_std:.4f}")
    print(f"  Between-class sim: {between_mean:.4f} ± {between_std:.4f}")
    print(f"  Gap (W - B):       {gap:.4f}")
    print(f"  Pairs sampled:     {len(within_sims)} within, {len(between_sims)} between")
    print(f"  Classes sampled:   {min(len(valid_classes), 80)}")

    # Top-5 / Bottom-5 类内聚度
    print(f"\n  Per-class cohesion (top 5 most separable):")
    cohesion = {}
    for cls_id in valid_classes:
        cls_mask = class_ids == cls_id.item()
        cls_embs = embs[cls_mask]
        centroid = class_centroids[cls_id.item()]
        sim_to_centroid = torch.nn.functional.cosine_similarity(cls_embs, centroid.unsqueeze(0), dim=-1)
        cohesion[cls_id.item()] = sim_to_centroid.mean().item()

    sorted_cls = sorted(cohesion.items(), key=lambda x: x[1], reverse=True)
    for cls_id, score in sorted_cls[:5]:
        cnt = (class_ids == cls_id).sum().item()
        print(f"    class {cls_id:>3d} (n={cnt:>4d}): cohesion={score:.4f}")

    print(f"\n  Per-class cohesion (bottom 5 least separable):")
    for cls_id, score in sorted_cls[-5:]:
        cnt = (class_ids == cls_id).sum().item()
        print(f"    class {cls_id:>3d} (n={cnt:>4d}): cohesion={score:.4f}")

    # 判定
    print(f"\n  Verdict:")
    if gap < 0.02:
        print(f"  ✗ Gap = {gap:.4f} < 0.02 — embeddings do NOT separate by class.")
        print(f"    Within-class ≈ between-class: embeddings carry no type signal.")
        print(f"    Suggestion: need LoRA fine-tuning or contrastive projection head.")
    elif gap < 0.1:
        print(f"  △ Gap = {gap:.4f} — weak class separation.")
        print(f"    Some signal exists but may need better classifier or more data.")
    else:
        print(f"  ✓ Gap = {gap:.4f} — good class separation.")


def main():
    parser = argparse.ArgumentParser(description="CTA Embedding 质量诊断")
    parser.add_argument("--embedding_path", type=str, required=True,
                        help="Embedding pickle 文件路径")
    parser.add_argument("--fold_dir", type=str, default=None,
                        help="Fold CSV 目录（用于类别级诊断）")
    args = parser.parse_args()

    embeddings_map = load_embeddings(args.embedding_path)
    diagnose_overall(embeddings_map)

    if args.fold_dir:
        diagnose_class_separation(embeddings_map, args.fold_dir)
    else:
        print(f"\n  (Pass --fold_dir for class-level diagnostics)")


if __name__ == "__main__":
    main()
