from utils import list_csv_files, get_basename, split_dataset_and_create_loaders
import torch.nn.functional as F
from enum import Enum
from sentence_transformers.util import pairwise_cos_sim, pairwise_euclidean_sim, pairwise_manhattan_sim
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from add_profilling import read_table, get_table_profilling, encode_column
import argparse
import random
import glob
from contextlib import nullcontext
import sys
import os

MAX_SEQ_LENGTH = 512  # 最大序列长度，2048 对 embedding 训练显存压力过大
DEFAULT_LOSS_TYPE = "infonce"  # 可选: "infonce", "triplet"
# 添加项目根目录到sys.path，以便导入根目录下的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .augment import augment, random_column_augment
except Exception as e:
    try:
        from augment import augment, random_column_augment
    except Exception as e2:
        print(f"Warning: Failed to import augment functions: {e}, {e2}")
        augment = None
        random_column_augment = None


class DistanceMetric(Enum):
    """The metric for the triplet loss"""
    def COSINE(x, y):
        return 1 - pairwise_cos_sim(x, y)

    def EUCLIDEAN(x, y):
        return pairwise_euclidean_sim(x, y)

    def MANHATTAN(x, y):
        return pairwise_manhattan_sim(x, y)


def contrastive_sample(data, anchor_indices=None):
    """
    对比学习数据采样函数
    目前还没有实现,我的设想是这么做:
    1.对所有数据取出erased_embedding,使用这个信息聚类,得到分类信息
    2.遍历anchor_indices,对于每一个anchor,随机取出同一个类里的数据索引当作正样本,随机取出其他类里的数据索引当作负样本
    3.需要注意的是,每个(anchor_indice, positive_indice/negative_indice)只能出现一次
    4.返回一个n*3的矩阵,每一行是一个(样本索引,正样本索引,负样本索引)
    """
    # 1.对所有数据取出erased_embedding,使用这个信息聚类,得到分类信息
    erased_embeddings = np.array(data['erased_embedding'].tolist())
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=8, random_state=42).fit(erased_embeddings)
    cluster_labels = kmeans.labels_
    # 如果没有提供anchor_indices，使用所有索引
    if anchor_indices is None:
        anchor_indices = np.arange(len(data))

    sample_indices = []
    used_pairs = set()  # 用于记录已经使用过的组合

    # AI生成,性能不太行,但是效果经过我的修改应该是符合要去的
    for anchor_idx in anchor_indices:
        anchor_cluster = cluster_labels[anchor_idx]

        # 获取同类的所有索引（排除anchor自身）
        positive_candidates = np.where(cluster_labels == anchor_cluster)[0]
        positive_candidates = positive_candidates[positive_candidates != anchor_idx]

        # 获取其他类的所有索引
        negative_candidates = np.where(cluster_labels != anchor_cluster)[0]

        # 随机选择正样本和负样本
        if len(positive_candidates) > 0 and len(negative_candidates) > 0:
            # 尝试找到未使用过的组合
            max_attempts = 100
            for _ in range(max_attempts):
                pos_idx = np.random.choice(positive_candidates)
                neg_idx = np.random.choice(negative_candidates)

                # 检查组合是否已经使用过
                def judge_set_pair(anchor_idx, pos_idx, neg_idx, used_pairs):
                    if (anchor_idx, pos_idx) in used_pairs or (anchor_idx, neg_idx) in used_pairs:
                        return False
                    # 翻转
                    elif (pos_idx, anchor_idx) in used_pairs or (neg_idx, anchor_idx) in used_pairs:
                        return False
                    else:
                        used_pairs.add((anchor_idx, pos_idx))
                        used_pairs.add((anchor_idx, neg_idx))
                        return True

                if judge_set_pair(anchor_idx, pos_idx, neg_idx, used_pairs):
                    sample_indices.append([anchor_idx, pos_idx, neg_idx])
                    break
            else:
                # 如果无法找到未使用的组合，跳过这个anchor
                continue

    # 返回一个n*3的矩阵,每一行是一个(样本索引,正样本索引,负样本索引)
    return np.array(sample_indices)


class ColumnContrastiveDataset(Dataset):
    """
    表格列对比学习数据集
    正例：同一列的原始数据 + 多个profilling文件
    负例：batch中其他列

    采用懒加载策略：__init__ 只扫描并记录元数据索引，不预先加载表格内容，
    __getitem__ 按需读取，避免 6000+ 张表同时驻留内存导致 OOM。
    Profilling JSON 文件体积较小且被多列复用，统一缓存在内存中。
    """

    def __init__(self, csv_files, anchor_profilling, positive_profillings, sample_rows=5, use_aug=False):
        """
        csv_files: CSV文件路径列表
        anchor_profilling: anchor的profilling文件路径
        positive_profillings: 用于正例的profilling文件路径列表
        sample_rows: 每个表格采样的行数
        use_aug: 是否对表格进行列级别的增强操作来构建正例
        """
        self.anchor_profilling = anchor_profilling
        self.positive_profillings = positive_profillings if isinstance(
            positive_profillings, list) else [positive_profillings]
        self.sample_rows = sample_rows
        self.use_aug = use_aug

        # 缓存 profilling JSON（体积小，多列复用，一次性加载）
        self._prof_cache = {}  # key: (csv_file, prof_path) -> dict or None

        # 只记录轻量元数据：(csv_file, col_name, col_idx, pos_prof_path_idx)
        # 不读取表格内容，不构建任何文本
        self.column_data = []
        for csv_file in tqdm(csv_files, desc="Scanning tables"):
            # 只读表头，不把整张表加载进来，也不写入 _table_cache
            try:
                header_table = pd.read_csv(
                    csv_file, nrows=0, lineterminator='\n', low_memory=False)
                col_names = list(header_table.columns)
            except Exception as e:
                print(f"Warning: Failed to read header of {csv_file}: {e}")
                continue

            # 预加载并缓存该文件的 profilling
            anchor_prof = self._load_prof(csv_file, anchor_profilling)
            if anchor_prof is None:
                continue

            valid_pos_indices = []
            for i, prof_path in enumerate(self.positive_profillings):
                prof = self._load_prof(csv_file, prof_path)
                if prof is not None:
                    valid_pos_indices.append(i)

            if len(valid_pos_indices) == 0:
                continue

            for col_idx, col_name in enumerate(col_names):
                for pos_idx in valid_pos_indices:
                    self.column_data.append(
                        (csv_file, col_name, col_idx, pos_idx))

        print(
            f"Total training samples indexed: {len(self.column_data)} (columns × positive profillings)")

    def _load_prof(self, csv_file, prof_path):
        """带缓存地读取 profilling JSON，避免重复 IO。"""
        key = (csv_file, prof_path)
        if key not in self._prof_cache:
            self._prof_cache[key] = get_table_profilling(csv_file, prof_path)
        return self._prof_cache[key]

    def __len__(self):
        return len(self.column_data)

    def __getitem__(self, idx):
        csv_file, col_name, col_idx, pos_prof_path_idx = self.column_data[idx]

        # 按需读取表格，直接用 pd.read_csv 绕开全局 _table_cache，
        # 避免 6000+ 张表逐渐积累在内存里。
        try:
            table = pd.read_csv(
                csv_file, lineterminator='\n', low_memory=False)
        except Exception as e:
            return {'anchor': '', 'positive': '', 'column_name': col_name, 'csv_file': csv_file}

        col_values = table[col_name].tolist(
        ) if col_name in table.columns else []

        # Anchor profilling
        anchor_prof = self._load_prof(csv_file, self.anchor_profilling) or {}
        anchor_prof_info = anchor_prof.get(col_name, {})
        if not isinstance(anchor_prof_info, dict):
            anchor_prof_info = {}
        anchor_prof_type = anchor_prof_info.get("__type__", "")

        anchor_text = encode_column(
            get_basename(csv_file),
            col_name,
            anchor_prof_type,
            anchor_prof_info,
            np.random.choice(col_values, np.random.randint(
                0, min(self.sample_rows, len(col_values))), replace=False).tolist() if len(col_values) > 0 else []
        )

        # Positive profilling
        pos_prof_path = self.positive_profillings[pos_prof_path_idx]
        pos_prof = self._load_prof(csv_file, pos_prof_path) or {}
        pos_prof_info = pos_prof.get(col_name, {})
        if not isinstance(pos_prof_info, dict):
            pos_prof_info = {}
        pos_prof_type = pos_prof_info.get("__type__", "")

        # 如果启用增强，对该列数据做增强后作为正例输入
        if self.use_aug and random_column_augment is not None:
            try:
                augmented_table = random_column_augment(table.copy())
                pos_col_values = augmented_table[col_name].tolist(
                ) if col_name in augmented_table.columns else col_values
            except Exception:
                pos_col_values = col_values
        else:
            pos_col_values = col_values

        positive_text = encode_column(
            get_basename(csv_file),
            col_name,
            pos_prof_type,
            pos_prof_info,
            np.random.choice(pos_col_values, np.random.randint(
                0, min(self.sample_rows, len(pos_col_values))), replace=False).tolist() if len(pos_col_values) > 0 else []
        )

        return {
            'anchor': anchor_text,
            'positive': positive_text,
            'column_name': col_name,
            'csv_file': csv_file
        }


def column_collate_fn(batch):
    """
    自定义collate函数
    """
    anchors = [item['anchor'] for item in batch]
    positives = [item['positive'] for item in batch]

    return {
        'anchor': anchors,
        'positive': positives
    }


def InfoNCE_loss(anchor, positive, negative, margin=0.2):
    """
    对比损失函数：让anchor和positive更接近，和negative更远离
    """
    pos_distance = torch.nn.functional.cosine_similarity(anchor, positive)
    neg_distance = torch.nn.functional.cosine_similarity(anchor, negative)

    # InfoNCE风格的对比损失
    loss = -torch.log(torch.exp(pos_distance) /
                      (torch.exp(pos_distance) + torch.exp(neg_distance)))

    return loss.mean()


def triplet_loss(anchor, positive, negative, positive_weight=1, negative_weight=1, margin=0.1, metric=DistanceMetric.COSINE):
    """
    triplet loss实现
    计算triplet loss
    anchor: 锚点样本的embedding
    positive: 正样本的embedding
    negative: 负样本的embedding
    margin: triplet loss的margin参数
    metric: 距离度量指标,默认是cosine相似度
    """
    pos_dist = 1 - F.cosine_similarity(anchor, positive, dim=-1)
    neg_dist = 1 - F.cosine_similarity(anchor, negative, dim=-1)
    loss = F.relu(positive_weight * pos_dist -
                  negative_weight * neg_dist + margin)
    return loss.mean()


class WeightedTripletLoss(nn.Module):
    def __init__(self, margin=0.5, init_alpha=1.0, init_beta=1.0):
        super(WeightedTripletLoss, self).__init__()
        self.margin = margin
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.beta = nn.Parameter(torch.tensor(init_beta))

    def forward(self, anchor, positive, negative):
        pos_dist = 1 - F.cosine_similarity(anchor, positive, dim=-1)
        neg_dist = 1 - F.cosine_similarity(anchor, negative, dim=-1)

        # 使用可学习的权重
        loss = F.relu(self.beta * pos_dist -
                      self.alpha * neg_dist + self.margin)
        return loss.mean()


def batch_hard_triplet_loss(embeddings, labels, margin=0.2, squared=False):
    """
    实现批量硬三元组挖掘的Triplet Loss

    embeddings: 特征向量 [batch_size, embed_dim]
    labels: 标签 [batch_size]
    """
    # 计算欧氏距离矩阵
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

    # 对每个锚点，找到最难的正例(最远的同类样本)
    mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask_pos = mask_pos.float() - \
        torch.eye(mask_pos.shape[0]).to(mask_pos.device)
    mask_pos[mask_pos < 0] = 0
    hardest_positive_dist = (pairwise_dist * mask_pos).max(dim=1)[0]

    # 对每个锚点，找到最难的负例(最近的不同类样本)
    mask_neg = labels.unsqueeze(0) != labels.unsqueeze(1)
    hardest_negative_dist = (pairwise_dist * mask_neg.float()).min(dim=1)[0]

    # 计算三元组损失
    triplet_loss = F.relu(hardest_positive_dist -
                          hardest_negative_dist + margin)
    return triplet_loss.mean()


class ContrastiveEmbeddingModel(nn.Module):
    def __init__(self, base_model='', device=None, max_seq_length=MAX_SEQ_LENGTH, gradient_checkpointing=True):
        super(ContrastiveEmbeddingModel, self).__init__()
        self.model = SentenceTransformer(base_model)
        # 显式设置 tokenizer 截断长度，避免极长样本把显存直接打满。
        self.model.max_seq_length = max_seq_length
        if device is None:
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.gradient_checkpointing_enabled = False
        if gradient_checkpointing:
            self.gradient_checkpointing_enabled = self._enable_gradient_checkpointing()

        self.model.to(self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        # self.adapter = nn.Linear(self.embedding_dim, target_embedding_dim)

    def _enable_gradient_checkpointing(self):
        try:
            first_module = self.model._first_module() if hasattr(
                self.model, '_first_module') else None
            auto_model = getattr(first_module, 'auto_model', None)
            if auto_model is None or not hasattr(auto_model, 'gradient_checkpointing_enable'):
                print(
                    '[Memory] Gradient checkpointing is not available for this model.')
                return False

            auto_model.gradient_checkpointing_enable()
            if hasattr(auto_model, 'config') and hasattr(auto_model.config, 'use_cache'):
                auto_model.config.use_cache = False
            print('[Memory] Gradient checkpointing enabled.')
            return True
        except Exception as e:
            print(f'[Memory] Failed to enable gradient checkpointing: {e}')
            return False

    def forward(self, texts):
        tokenized = self.model.tokenize(texts)
        max_seq_length = getattr(self.model, 'max_seq_length', None)
        if max_seq_length is not None:
            for key in ('input_ids', 'attention_mask', 'token_type_ids'):
                value = tokenized.get(key)
                if value is not None and value.ndim == 2 and value.shape[1] > max_seq_length:
                    tokenized[key] = value[:, :max_seq_length]
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        out = self.model(tokenized)              # returns dict
        embeddings = out['sentence_embedding']      # (B, D)
        return F.normalize(embeddings, dim=-1)


def _print_length_stats(title: str, lengths):
    if lengths is None or len(lengths) == 0:
        print(f"{title}: (empty)")
        return
    arr = np.asarray(lengths, dtype=np.int64)
    p50, p90, p95, p99 = np.percentile(arr, [50, 90, 95, 99]).tolist()
    print(
        f"{title}: min={int(arr.min())}, mean={arr.mean():.2f}, p50={p50:.0f}, p90={p90:.0f}, p95={p95:.0f}, p99={p99:.0f}, max={int(arr.max())}"
    )


def sample_csv_files(csv_files, sampling_ratio=1.0, sampling_seed=42):
    """Sample a subset of csv files for training when the dataset is too large."""
    if not csv_files:
        return csv_files

    ratio = float(sampling_ratio)
    if ratio <= 0 or ratio > 1:
        raise ValueError("sampling_ratio must be within (0, 1].")

    if ratio >= 1.0:
        return csv_files

    sample_size = max(1, int(len(csv_files) * ratio))
    rng = random.Random(sampling_seed)
    sampled_csv_files = rng.sample(list(csv_files), sample_size)
    sampled_csv_files.sort()
    print(
        f"[Sampling] Selected {len(sampled_csv_files)}/{len(csv_files)} CSV files with sampling_ratio={ratio} and sampling_seed={sampling_seed}"
    )
    return sampled_csv_files


def print_dataset_text_length_stats(dataset, model_path: str, max_samples: int = 0, batch_size: int = 256, max_seq_length: int = MAX_SEQ_LENGTH):
    """Print char length + tokenizer token length stats for dataset['anchor'/'positive'].

    This is meant to catch rare extremely-long samples that can cause occasional OOM.
    """
    n_total = len(dataset)
    if n_total == 0:
        print("[LengthStats] Dataset is empty.")
        return

    n = n_total if (max_samples is None or int(max_samples) <= 0) else min(
        n_total, int(max_samples))
    print(
        f"[LengthStats] Scanning {n}/{n_total} samples using tokenizer from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    def _short_text(text: str, max_chars: int = 200) -> str:
        if text is None:
            return ''
        s = str(text).replace('\n', ' ').replace('\r', ' ').strip()
        if len(s) <= max_chars:
            return s
        return s[:max_chars] + ' ...'

    def _meta_str(meta: dict) -> str:
        if not meta:
            return ''
        csv_file = meta.get('csv_file', '')
        column_name = meta.get('column_name', '')
        idx = meta.get('index', None)
        parts = []
        if idx is not None:
            parts.append(f"index={idx}")
        if csv_file:
            parts.append(f"csv_file={csv_file}")
        if column_name:
            parts.append(f"column_name={column_name}")
        return ', '.join(parts)

    best_anchor_chars = {'len': -1, 'meta': None, 'text': ''}
    best_positive_chars = {'len': -1, 'meta': None, 'text': ''}
    best_anchor_tokens = {'len': -1, 'meta': None, 'text': ''}
    best_positive_tokens = {'len': -1, 'meta': None, 'text': ''}

    anchor_char_lens = []
    positive_char_lens = []
    anchor_token_lens = []
    positive_token_lens = []

    # Iterate in batches to keep memory stable
    for start in tqdm(range(0, n, batch_size), desc="[LengthStats] Tokenizing", leave=False):
        end = min(start + batch_size, n)
        anchors = []
        positives = []
        metas = []
        for i in range(start, end):
            item = dataset[i]
            a = item.get('anchor', '')
            p = item.get('positive', '')
            anchors.append(a)
            positives.append(p)
            meta = {
                'index': i,
                'csv_file': item.get('csv_file', ''),
                'column_name': item.get('column_name', ''),
            }
            metas.append(meta)

            a_len = len(a)
            if a_len > best_anchor_chars['len']:
                best_anchor_chars = {'len': a_len, 'meta': meta, 'text': a}
            p_len = len(p)
            if p_len > best_positive_chars['len']:
                best_positive_chars = {'len': p_len, 'meta': meta, 'text': p}

            anchor_char_lens.append(len(a))
            positive_char_lens.append(len(p))

        # Measure raw token lengths (添加truncation防止超长序列)
        a_tok = tokenizer(
            anchors,
            add_special_tokens=True,
            padding=False,
            truncation=True,  # ⚠️ 改为True，防止超长文本
            max_length=max_seq_length,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        p_tok = tokenizer(
            positives,
            add_special_tokens=True,
            padding=False,
            truncation=True,  # ⚠️ 改为True，防止超长文本
            max_length=max_seq_length,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        anchor_token_lens.extend([len(ids) for ids in a_tok['input_ids']])
        positive_token_lens.extend([len(ids) for ids in p_tok['input_ids']])

        # Track longest tokenized samples with metadata
        for j, ids in enumerate(a_tok['input_ids']):
            tok_len = len(ids)
            if tok_len > best_anchor_tokens['len']:
                best_anchor_tokens = {
                    'len': tok_len,
                    'meta': metas[j] if j < len(metas) else None,
                    'text': anchors[j] if j < len(anchors) else '',
                }
        for j, ids in enumerate(p_tok['input_ids']):
            tok_len = len(ids)
            if tok_len > best_positive_tokens['len']:
                best_positive_tokens = {
                    'len': tok_len,
                    'meta': metas[j] if j < len(metas) else None,
                    'text': positives[j] if j < len(positives) else '',
                }

    print("[LengthStats] Character lengths")
    _print_length_stats("  anchor chars", anchor_char_lens)
    _print_length_stats("  positive chars", positive_char_lens)
    print("[LengthStats] Token lengths")
    _print_length_stats("  anchor tokens", anchor_token_lens)
    _print_length_stats("  positive tokens", positive_token_lens)

    print("[LengthStats] Longest samples (with table info)")
    if best_anchor_chars['len'] >= 0:
        print(
            f"  anchor chars: len={best_anchor_chars['len']} | {_meta_str(best_anchor_chars['meta'])}\n"
            f"    preview: {_short_text(best_anchor_chars['text'])}"
        )
    if best_positive_chars['len'] >= 0:
        print(
            f"  positive chars: len={best_positive_chars['len']} | {_meta_str(best_positive_chars['meta'])}\n"
            f"    preview: {_short_text(best_positive_chars['text'])}"
        )
    if best_anchor_tokens['len'] >= 0:
        print(
            f"  anchor tokens: len={best_anchor_tokens['len']} | {_meta_str(best_anchor_tokens['meta'])}\n"
            f"    preview: {_short_text(best_anchor_tokens['text'])}"
        )
    if best_positive_tokens['len'] >= 0:
        print(
            f"  positive tokens: len={best_positive_tokens['len']} | {_meta_str(best_positive_tokens['meta'])}\n"
            f"    preview: {_short_text(best_positive_tokens['text'])}"
        )


def train_column_contrastive_model(
    model_path,
    csv_dir,
    anchor_profilling,
    positive_profillings,
    batch_size=32,
    save_path=None,
    epoch_num=10,
    sample_rows=10,
    temperature=0.07,
    device=None,
    use_aug=False,
    loss_type=DEFAULT_LOSS_TYPE,
    max_seq_length=MAX_SEQ_LENGTH,
    sampling_ratio=1.0,
    sampling_seed=42,
    use_amp=True,
    gradient_checkpointing=True,
    print_length_stats=False,
    length_stats_max_samples=0,
    length_stats_batch_size=256
):
    """
    训练列级别对比学习模型

    Args:
        model_path: 基础模型路径
        csv_dir: CSV文件目录或文件列表
        anchor_profilling: anchor的profilling文件路径
        positive_profillings: 用于正例的profilling文件路径列表
        batch_size: 批次大小
        save_path: 模型保存路径
        epoch_num: 训练轮数
        sample_rows: 每个表格采样的行数
        temperature: InfoNCE loss的温度参数
        device: 训练设备
    """
    # 获取CSV文件列表
    if isinstance(csv_dir, str):
        if os.path.isdir(csv_dir):
            csv_files = list_csv_files(csv_dir)
        else:
            raise ValueError(
                "csv_dir should be a directory path or a list of file paths")
    else:
        raise ValueError(
            "csv_dir should be a directory path or a list of file paths")
    print(f"Found {len(csv_files)} CSV files")
    csv_files = sample_csv_files(
        csv_files,
        sampling_ratio=sampling_ratio,
        sampling_seed=sampling_seed,
    )

    # 创建数据集
    dataset = ColumnContrastiveDataset(
        csv_files=csv_files,
        anchor_profilling=anchor_profilling,
        positive_profillings=positive_profillings,
        sample_rows=sample_rows,
        use_aug=use_aug
    )

    if len(dataset) == 0:
        raise ValueError("No valid columns found in the dataset")

    if print_length_stats:
        try:
            print_dataset_text_length_stats(
                dataset,
                model_path=model_path,
                max_samples=length_stats_max_samples,
                batch_size=length_stats_batch_size,
                max_seq_length=max_seq_length,
            )
        except Exception as e:
            print(f"[LengthStats] Failed to compute length stats: {e}")

    # 创建数据加载器
    train_loader, val_loader, test_loader = split_dataset_and_create_loaders(
        dataset,
        train_ratio=0.85,
        val_ratio=0.05,
        test_ratio=0.10,
        batch_size=batch_size,
        collate_fn=column_collate_fn
    )

    # 初始化模型
    model = ContrastiveEmbeddingModel(
        base_model=model_path,
        device=device,
        max_seq_length=max_seq_length,
        gradient_checkpointing=gradient_checkpointing,
    )
    print(f'[Memory] Using max_seq_length={model.model.max_seq_length}')

    amp_enabled = bool(use_amp and model.device.type == 'cuda')
    if amp_enabled:
        print('[Memory] AMP enabled (float16 autocast on CUDA).')
    elif use_amp:
        print('[Memory] AMP requested but CUDA is unavailable, running without AMP.')

    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    def _compute_inbatch_infonce_loss(anchor_embeddings, positive_embeddings, temperature, criterion):
        """In-batch InfoNCE via cross-entropy on anchor-positive similarity matrix."""
        logits = torch.matmul(anchor_embeddings, positive_embeddings.T)
        if temperature is not None:
            logits = logits / temperature

        labels = torch.arange(logits.shape[0], device=anchor_embeddings.device)
        return criterion(logits, labels)

    def _compute_inbatch_triplet_loss(anchor_embeddings, positive_embeddings):
        """Use hardest in-batch positives-as-negatives for triplet loss."""
        if anchor_embeddings.shape[0] < 2:
            raise ValueError("triplet loss requires batch_size >= 2")

        sim_matrix = torch.matmul(anchor_embeddings, positive_embeddings.T)
        diagonal_mask = torch.eye(
            sim_matrix.shape[0], device=sim_matrix.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(diagonal_mask, float('-inf'))
        negative_indices = sim_matrix.argmax(dim=1)
        negative_embeddings = positive_embeddings[negative_indices]

        return triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)

    criterion = nn.CrossEntropyLoss()

    # 训练历史记录
    train_history = []
    val_history = []
    best_val_loss = float('inf')

    for epoch in range(epoch_num):
        model.train()
        total_loss = 0

        pbar = tqdm(
            train_loader, desc=f'Epoch {epoch+1}/{epoch_num}', leave=False)

        for batch_idx, batch in enumerate(pbar):
            # ⚠️ 确保梯度清零，设置set_to_none=True更彻底
            optimizer.zero_grad(set_to_none=True)

            autocast_context = torch.autocast(
                device_type='cuda', dtype=torch.float16) if amp_enabled else nullcontext()

            with autocast_context:
                # 获取embeddings
                anchor_embeddings = model(batch['anchor'])
                positive_embeddings = model(batch['positive'])

                if loss_type == 'infonce':
                    loss = _compute_inbatch_infonce_loss(
                        anchor_embeddings,
                        positive_embeddings,
                        temperature,
                        criterion,
                    )
                elif loss_type == 'triplet':
                    loss = _compute_inbatch_triplet_loss(
                        anchor_embeddings,
                        positive_embeddings,
                    )
                else:
                    raise ValueError(f"Unsupported loss_type: {loss_type}")

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            # 更新进度条
            avg_loss_so_far = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                'batch_loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss_so_far:.4f}'
            })
            # ⚠️ 删除所有中间变量释放显存
            del loss, anchor_embeddings, positive_embeddings

            # ⚠️ 每个batch都清理显存（改为每次都清理更安全）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 计算epoch平均loss
        epoch_avg_loss = total_loss / len(train_loader)
        train_history.append(epoch_avg_loss)

        print(f'Epoch {epoch+1}/{epoch_num} completed. '
              f'Average Loss: {epoch_avg_loss:.4f}')

        # 验证
        if val_loader:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    autocast_context = torch.autocast(
                        device_type='cuda', dtype=torch.float16) if amp_enabled else nullcontext()
                    with autocast_context:
                        anchor_embeddings = model(batch['anchor'])
                        positive_embeddings = model(batch['positive'])

                        if loss_type == 'infonce':
                            loss = _compute_inbatch_infonce_loss(
                                anchor_embeddings,
                                positive_embeddings,
                                temperature,
                                criterion,
                            )
                        elif loss_type == 'triplet':
                            loss = _compute_inbatch_triplet_loss(
                                anchor_embeddings,
                                positive_embeddings,
                            )
                        else:
                            raise ValueError(
                                f"Unsupported loss_type: {loss_type}")

                    total_val_loss += loss.item()

                    # 清理显存
                    del anchor_embeddings, positive_embeddings, loss

            # 验证后清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            val_loss = total_val_loss / len(val_loader)
            val_history.append(val_loss)
            print(f'Validation Loss: {val_loss:.4f}')

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f'Best model saved with validation loss: {val_loss:.4f}')

    # 如果没有保存过最佳模型（例如没有验证集），保存最终模型
    if best_val_loss == float('inf') and save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f'Final model saved to: {save_path}')

    # 测试
    if test_loader:
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                autocast_context = torch.autocast(
                    device_type='cuda', dtype=torch.float16) if amp_enabled else nullcontext()
                with autocast_context:
                    anchor_embeddings = model(batch['anchor'])
                    positive_embeddings = model(batch['positive'])

                    if loss_type == 'infonce':
                        loss = _compute_inbatch_infonce_loss(
                            anchor_embeddings,
                            positive_embeddings,
                            temperature,
                            criterion,
                        )
                    elif loss_type == 'triplet':
                        loss = _compute_inbatch_triplet_loss(
                            anchor_embeddings,
                            positive_embeddings,
                        )
                    else:
                        raise ValueError(f"Unsupported loss_type: {loss_type}")

                total_test_loss += loss.item()
                del anchor_embeddings, positive_embeddings, loss
        # 测试后清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        test_loss = total_test_loss / len(test_loader)
        print(f'Test Loss: {test_loss:.4f}')

    return train_history, val_history


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train a contrastive learning model.")
    parser.add_argument('--model_path', type=str,
                        default='Qwen/Qwen3-Embedding-0.6B', help='Base model path or name.')
    parser.add_argument('--sample_method', type=str, default='default',
                        choices=['default', 'more_negative', 'nerfarneg'], help='Sampling method for contrastive learning.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training.')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save the trained model.')
    parser.add_argument('--epoch_num', type=int, default=10,
                        help='Number of training epochs.')
    parser.add_argument('--positive_weight', type=float, default=1.0,
                        help='Weight for positive pairs in the loss function.')
    parser.add_argument('--negative_weight', type=float, default=1.0,
                        help='Weight for negative pairs in the loss function.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index to use (default 0). Ignored if no CUDA.')
    parser.add_argument('--csv_dir', type=str, default='./datasets/opendata_SG',
                        help='Directory containing CSV files for column-level training.')
    parser.add_argument('--anchor_profilling', type=str,
                        default='./output/profilling_result.json',
                        help='Path to the anchor profilling JSON file.')
    parser.add_argument('--positive_profillings', type=str, nargs='+',
                        default=['./output/profilling_result.json'],
                        help='Paths to the positive profilling JSON files (can specify multiple).')
    parser.add_argument('--sample_rows', type=int, default=10,
                        help='Number of rows to sample from each table.')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature parameter for InfoNCE loss.')
    parser.add_argument('--loss_type', type=str, default=DEFAULT_LOSS_TYPE,
                        choices=['triplet', 'infonce'], help='Loss type for training.')
    parser.add_argument('--aug', action='store_true',
                        help='Enable column-level augmentation for positive examples.')
    parser.add_argument('--max_seq_length', type=int, default=MAX_SEQ_LENGTH,
                        help='Maximum token length after tokenizer truncation.')
    parser.add_argument('--sampling_ratio', type=float, default=1.0,
                        help='Sample only a ratio of CSV files for training, within (0, 1].')
    parser.add_argument('--sampling_seed', type=int, default=42,
                        help='Random seed used when sampling CSV files.')
    parser.add_argument('--disable_amp', action='store_true',
                        help='Disable automatic mixed precision on CUDA.')
    parser.add_argument('--disable_gradient_checkpointing', action='store_true',
                        help='Disable gradient checkpointing on the encoder when supported.')

    parser.add_argument('--print_length_stats', action='store_true',
                        help='Print dataset text/token length statistics before training.')
    parser.add_argument('--length_stats_max_samples', type=int, default=0,
                        help='Max samples to scan for length stats (0 means all).')
    parser.add_argument('--length_stats_batch_size', type=int, default=256,
                        help='Batch size for tokenizer when computing length stats.')

    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    # 定义使用设备
    if torch.cuda.is_available():
        if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
            print(f'Invalid GPU index {args.gpu}. Using GPU 0.')
            args.gpu = 0
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    # 列级别对比学习
    print("Training column-level contrastive model...")

    modelname2path = {
        "Qwen/Qwen3-Embedding-0.6B": "./model/qwen3-0.6B-embedding"
    }
    if args.model_path in modelname2path:
        args.model_path = modelname2path[args.model_path]

    if args.save_path is None:
        monthday = datetime.now().strftime("%m%d")
        modeldir = args.model_path.split('/')[-1]
        os.makedirs(f'./model/cl/{modeldir}', exist_ok=True)
        args.save_path = f'./model/cl/{modeldir}/column_contrastive_{args.loss_type}_{monthday}_temp{args.temperature}_ep{args.epoch_num}.pth'

    print(f"Model will be saved to: {args.save_path}")

    train_history, val_history = train_column_contrastive_model(
        model_path=args.model_path,
        csv_dir=args.csv_dir,
        anchor_profilling=args.anchor_profilling,
        positive_profillings=args.positive_profillings,
        batch_size=args.batch_size,
        save_path=args.save_path,
        epoch_num=args.epoch_num,
        sample_rows=args.sample_rows,
        temperature=args.temperature,
        device=device,
        use_aug=args.aug,
        loss_type=args.loss_type,
        max_seq_length=args.max_seq_length,
        sampling_ratio=args.sampling_ratio,
        sampling_seed=args.sampling_seed,
        use_amp=not args.disable_amp,
        gradient_checkpointing=not args.disable_gradient_checkpointing,
        print_length_stats=args.print_length_stats,
        length_stats_max_samples=args.length_stats_max_samples,
        length_stats_batch_size=args.length_stats_batch_size
    )

    print("\nTraining completed!")
    print(f"Final training loss: {train_history[-1]:.4f}")
    if val_history:
        print(f"Final validation loss: {val_history[-1]:.4f}")
