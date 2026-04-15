"""
CTA LoRA Fine-tuning — 端到端训练 LLM 生成类型区分性隐状态

方案3：冻结 LLM backbone → 添加 LoRA adapters + 解冻最后几层 + 分类头
输入格式：prefix-cache (表内容) + per-column suffix (列焦点)
训练：5-fold CV，Focal Loss，梯度累积，Early Stopping，Ensemble 测试

用法：
    python column_type_annotation/train_cta_lora.py \
        --model_path ./model/Qwen2.5-7B-Instruct \
        --fold_dir datasets/gittables-semtab22-db-all_wrangled/ \
        --table_dir datasets/gittables-semtab22-db-all_wrangled/ \
        --result_dir results/cta_lora/gittables/ \
        --gpu_id 0
"""

import os
import sys
import json
import math
import pickle
import random
import argparse
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
#  Constants
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

# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------


def get_basename(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def serialize_table(headers: List[str], data: List[List[str]], n_rows: int = 5) -> str:
    lines = []
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in data[:n_rows]:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


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
#  Focal Loss
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0,
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.weight = weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pt = torch.exp(-F.cross_entropy(input, target, reduction='none'))
        ce = F.cross_entropy(input, target, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction='none')
        return ((1.0 - pt) ** self.gamma * ce).mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss — 同类拉近，异类推远"""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: (N, D) — raw hidden states
        labels:   (N,)
        """
        features = F.normalize(features, p=2, dim=-1)
        device = features.device
        N = features.shape[0]

        if N < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        sim = features @ features.T / self.temperature

        label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        diag_mask = ~torch.eye(N, dtype=torch.bool, device=device)
        pos_mask = label_mask & diag_mask

        if pos_mask.sum() == 0:
            # 无同类配对 → diversity loss: 最小化平均余弦相似度 (推开异类)
            avg_sim = (sim * diag_mask.float()).sum() / (N * (N - 1))
            return avg_sim

        # 标准 SupCon
        logits_max = sim.max(dim=1, keepdim=True).values.detach()
        logits = sim - logits_max
        exp_logits = torch.exp(logits) * diag_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        num_pos = pos_mask.sum(dim=1)
        mean_log_prob = (pos_mask.float() * log_prob).sum(dim=1) / (num_pos + 1e-12)

        has_pos = num_pos > 0
        return -mean_log_prob[has_pos].mean()


# ---------------------------------------------------------------------------
#  Early Stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float('inf')
        self.counter = 0

    def __call__(self, score: float) -> bool:
        if score > self.best + self.min_delta:
            self.best = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
#  CTA Dataset (grouped by table)
# ---------------------------------------------------------------------------


def load_folds(fold_dir: str):
    """加载 5 个 fold CSV，返回 {fold_idx: DataFrame} 和类别数"""
    folds = {}
    for i in range(5):
        fp = os.path.join(fold_dir, f"fold_{i}.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            folds[i] = df
    if not folds:
        raise FileNotFoundError(f"No fold_*.csv found in {fold_dir}")
    max_class = max(df['class_id'].max() for df in folds.values())
    return folds, int(max_class) + 1


class CTATableDataset:
    """按表组织数据：返回 (table_id, [(col_idx, class_id), ...])"""

    def __init__(self, fold_dfs: List[pd.DataFrame], table_dir: str, file_type: str = ".csv",
                 sample_rows: int = 5):
        df = pd.concat(fold_dfs, ignore_index=True)
        df = df[df['class_id'] != -1].reset_index(drop=True)
        self.table_dir = table_dir
        self.file_type = file_type
        self.sample_rows = sample_rows

        # 按 table_id 分组
        self.groups = list(df.groupby('table_id'))
        self.table_cache = {}

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        table_id, group_df = self.groups[idx]
        col_labels = [(int(row['col_idx']), int(row['class_id']))
                      for _, row in group_df.iterrows()]

        # 读取并缓存表
        if table_id not in self.table_cache:
            table_path = find_table_file(table_id, self.table_dir, self.file_type)
            if table_path is None:
                return None
            df = pd.read_csv(table_path, nrows=self.sample_rows, lineterminator="\n")
            df = df.dropna(how="all")
            self.table_cache[table_id] = {
                'headers': df.columns.tolist(),
                'data': df.values.tolist(),
                'name': get_basename(table_path),
            }
            # 限制缓存大小
            if len(self.table_cache) > 500:
                oldest = list(self.table_cache.keys())[0]
                del self.table_cache[oldest]

        return table_id, col_labels, self.table_cache[table_id]


def collate_skip_none(batch):
    """跳过 None (找不到文件的表)"""
    return [item for item in batch if item is not None]


# ---------------------------------------------------------------------------
#  Model Setup: LoRA + Classification Head
# ---------------------------------------------------------------------------


class CTAClassifierHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def setup_model(model_path: str, num_classes: int, lora_r: int, lora_alpha: int,
                lora_dropout: float, num_unfrozen_layers: int,
                device: str = "auto"):
    """加载模型，添加 LoRA，解冻最后几层，加分类头"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print(f"[Model] Loading {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
        "device_map": device,
    }
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    hidden_dim = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    print(f"  hidden_size={hidden_dim}, num_layers={num_layers}")

    # 冻结全部参数
    for param in model.parameters():
        param.requires_grad = False

    # 解冻最后 N 层
    if num_unfrozen_layers > 0:
        layers_to_unfreeze = model.model.layers[-(min(num_unfrozen_layers, num_layers)):]
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"  Unfrozen last {len(layers_to_unfreeze)} layers")

    # 添加 LoRA — 同时覆盖 attention 和 MLP 层
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout,
        task_type="FEATURE_EXTRACTION",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 分类头
    classifier = CTAClassifierHead(hidden_dim, num_classes).to(model.device)

    trainable = sum(p.numel() for p in classifier.parameters())
    print(f"  Classifier params: {trainable:,}")

    return model, tokenizer, classifier, hidden_dim


# ---------------------------------------------------------------------------
#  Prefix / Suffix Tokenization
# ---------------------------------------------------------------------------


def tokenize_prefix(tokenizer, table_name: str, headers: List[str],
                    data: List[List[str]], n_rows: int, max_length: int) -> torch.Tensor:
    table_md = serialize_table(headers, data, n_rows)
    user_text = f"Table name: {table_name}\n{table_md}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    if input_ids.shape[-1] > max_length:
        input_ids = input_ids[:, -max_length:]
    return input_ids


def tokenize_suffixes_batch(tokenizer, col_names: List[str],
                            col_values_list: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
    texts = []
    for name, vals in zip(col_names, col_values_list):
        values_str = ", ".join(str(v) for v in vals) if vals else "(empty)"
        texts.append(SUFFIX_TEMPLATE.format(col_name=name, col_values=values_str))
    encoded = tokenizer(texts, padding=True, truncation=True,
                        return_tensors="pt", add_special_tokens=False)
    return encoded.input_ids, encoded.attention_mask


# ---------------------------------------------------------------------------
#  Training / Eval per fold
# ---------------------------------------------------------------------------


def expand_kv_cache(kv_cache, num_cols: int):
    """将 KV-cache 从 batch=1 扩展到 batch=num_cols"""
    from transformers import DynamicCache
    batch_cache = DynamicCache()
    for layer_idx in range(len(kv_cache)):
        k, v = kv_cache[layer_idx]
        batch_cache.update(
            k.expand(num_cols, -1, -1, -1).contiguous(),
            v.expand(num_cols, -1, -1, -1).contiguous(),
            layer_idx,
        )
    return batch_cache


def process_table(model, tokenizer, classifier, device,
                  table_info, col_labels, sample_rows, max_prefix_length):
    """处理一张表：prefix no_grad → suffix with grad → logits + hidden states"""
    table_name = table_info['name']
    headers = table_info['headers']
    data = table_info['data']

    # 过滤有效列
    valid_cols = [(cidx, cid) for cidx, cid in col_labels if cidx < len(headers)]
    if not valid_cols:
        return None, None, None

    col_indices, labels = zip(*valid_cols)
    col_names = [headers[i] for i in col_indices]
    col_values = [[str(row[i]) for row in data[:sample_rows] if i < len(row)]
                  for i in col_indices]
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    num_cols = len(col_names)

    # Step 1: Prefix no_grad
    prefix_ids = tokenize_prefix(tokenizer, table_name, headers, data,
                                 sample_rows, max_prefix_length).to(device)
    with torch.no_grad():
        prefix_out = model(input_ids=prefix_ids, use_cache=True)
        kv_cache = prefix_out.past_key_values
        del prefix_out

    # Step 2: Batch suffix
    suffix_ids, suffix_mask = tokenize_suffixes_batch(tokenizer, col_names, col_values)
    suffix_ids = suffix_ids.to(device)
    suffix_mask = suffix_mask.to(device)

    batch_kv_cache = expand_kv_cache(kv_cache, num_cols)
    del kv_cache

    # Step 3: Forward with gradients
    out = model(input_ids=suffix_ids, attention_mask=suffix_mask,
                past_key_values=batch_kv_cache, output_hidden_states=True)

    # 取最后一层最后有效 token
    last_pos = suffix_mask.sum(dim=1) - 1
    hidden = out.hidden_states[-1][torch.arange(num_cols, device=device), last_pos]
    # hidden: (num_cols, hidden_dim) — float16

    logits = classifier(hidden.float())

    del out, batch_kv_cache
    torch.cuda.empty_cache()

    return logits, labels, hidden


def train_one_fold(model, tokenizer, classifier, optimizer, scheduler,
                   criterion, con_criterion, contrastive_weight,
                   train_dataset, val_dataset,
                   num_epochs, grad_accum_steps, patience,
                   device, sample_rows, max_prefix_length,
                   fold_idx, result_dir, save_model):
    """训练单个 fold"""
    best_f1 = -1.0
    best_state = None
    early_stopper = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        classifier.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_con_loss = 0.0
        train_steps = 0
        random.shuffle(train_dataset.groups)

        pbar = tqdm(CTATableDataset.__len__(train_dataset) and range(len(train_dataset)),
                     desc=f"Fold {fold_idx} Epoch {epoch+1}/{num_epochs}")
        for i in pbar:
            item = train_dataset[i]
            if item is None:
                continue

            table_id, col_labels, table_info = item
            logits, labels, hidden = process_table(
                model, tokenizer, classifier, device,
                table_info, col_labels, sample_rows, max_prefix_length
            )
            if logits is None:
                continue

            # 分类 loss
            cls_loss = criterion(logits, labels)
            # 对比 loss — 直接在 hidden states 上优化类间分离
            con_loss = con_criterion(hidden.float(), labels)
            total_loss = (cls_loss + contrastive_weight * con_loss) / grad_accum_steps

            total_loss.backward()
            train_loss += total_loss.item() * grad_accum_steps
            train_cls_loss += cls_loss.item()
            train_con_loss += con_loss.item()
            train_steps += 1

            if train_steps % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(classifier.parameters()), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=f"{train_loss/max(train_steps,1):.4f}",
                             cls=f"{train_cls_loss/max(train_steps,1):.4f}",
                             con=f"{train_con_loss/max(train_steps,1):.4f}")

        avg_train_loss = train_loss / max(train_steps, 1)

        # --- Validate ---
        val_metrics = evaluate(model, tokenizer, classifier, val_dataset,
                               criterion, device, sample_rows, max_prefix_length)
        val_acc = val_metrics['accuracy']
        val_macro_f1 = val_metrics['macro_f1']
        val_micro_f1 = val_metrics['micro_f1']
        val_loss = val_metrics['loss']

        print(f"  Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Macro F1: {val_macro_f1:.4f} | "
              f"Micro F1: {val_micro_f1:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save best
        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            best_state = {
                'model': deepcopy(model.state_dict()),
                'classifier': deepcopy(classifier.state_dict()),
            }
            if save_model:
                path = os.path.join(result_dir, f"fold_{fold_idx}_best.pt")
                torch.save(best_state, path)
            print(f"  → New best Macro F1: {best_f1:.4f}")

        if early_stopper(val_macro_f1):
            print(f"  Early stopping at epoch {epoch+1}")
            break

    return best_state, best_f1


@torch.no_grad()
def evaluate(model, tokenizer, classifier, dataset, criterion,
             device, sample_rows, max_prefix_length):
    """评估"""
    model.eval()
    classifier.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_steps = 0

    for i in tqdm(range(len(dataset)), desc="Evaluating", leave=False):
        item = dataset[i]
        if item is None:
            continue

        table_id, col_labels, table_info = item
        logits, labels, _ = process_table(
            model, tokenizer, classifier, device,
            table_info, col_labels, sample_rows, max_prefix_length
        )
        if logits is None:
            continue

        loss = criterion(logits, labels)
        total_loss += loss.item()
        total_steps += 1

        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    if not all_preds:
        return {'accuracy': 0, 'macro_f1': 0, 'micro_f1': 0, 'loss': 0}

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'loss': total_loss / max(total_steps, 1),
    }


@torch.no_grad()
def ensemble_test(models_states, model_template, tokenizer, classifier_template,
                  test_dataset, device, sample_rows, max_prefix_length):
    """Ensemble 多个 fold 模型的 softmax 概率"""
    all_probs = []
    all_labels = []

    for i in tqdm(range(len(test_dataset)), desc="Ensemble test"):
        item = test_dataset[i]
        if item is None:
            continue

        table_id, col_labels, table_info = item

        # 收集每列的值
        headers = table_info['headers']
        data = table_info['data']
        valid_cols = [(cidx, cid) for cidx, cid in col_labels if cidx < len(headers)]
        if not valid_cols:
            continue

        col_indices, labels = zip(*valid_cols)
        labels = torch.tensor(labels, dtype=torch.long)
        col_names = [headers[i] for i in col_indices]
        col_values = [[str(row[i]) for row in data[:sample_rows] if i < len(row)]
                      for i in col_indices]

        col_probs = torch.zeros(len(valid_cols), classifier_template.head[-1].out_features)

        for state in models_states:
            model_template.load_state_dict(state['model'])
            classifier_template.load_state_dict(state['classifier'])
            model_template.eval()
            classifier_template.eval()

            logits, _, _ = process_table(
                model_template, tokenizer, classifier_template, device,
                table_info, list(zip(col_indices, labels.tolist())),
                sample_rows, max_prefix_length
            )
            if logits is not None:
                col_probs += torch.softmax(logits.float().cpu(), dim=-1)

        col_probs /= len(models_states)
        all_probs.append(col_probs)
        all_labels.append(labels)

    if not all_probs:
        return {'accuracy': 0, 'macro_f1': 0, 'micro_f1': 0}

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    preds = all_probs.argmax(axis=-1)

    from sklearn.metrics import accuracy_score, f1_score
    return {
        'accuracy': accuracy_score(all_labels, preds),
        'macro_f1': f1_score(all_labels, preds, average='macro', zero_division=0),
        'micro_f1': f1_score(all_labels, preds, average='micro', zero_division=0),
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="CTA LoRA Fine-tuning")

    # 数据
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--fold_dir", type=str, required=True)
    parser.add_argument("--table_dir", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--sample_rows", type=int, default=5)
    parser.add_argument("--file_type", type=str, default=".csv")
    parser.add_argument("--max_prefix_length", type=int, default=1024)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.3)
    parser.add_argument("--num_unfrozen_layers", type=int, default=0)

    # 训练
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.15)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--contrastive_weight", type=float, default=0.1,
                        help="对比学习 loss 权重 (0 表示关闭, 默认 0.1)")
    parser.add_argument("--contrastive_temperature", type=float, default=0.07,
                        help="SupCon 温度参数 (默认 0.07)")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # GPU
    parser.add_argument("--gpu_id", type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = f"cuda:{args.gpu_id}" if args.gpu_id is not None else "auto"
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    print("=" * 60)
    print("CTA LoRA Fine-tuning")
    print("=" * 60)
    print(f"  Model:        {args.model_path}")
    print(f"  Fold dir:     {args.fold_dir}")
    print(f"  Table dir:    {args.table_dir}")
    print(f"  Result dir:   {args.result_dir}")
    print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"  Target modules: q/k/v/o_proj + gate/up/down_proj")
    print(f"  Unfrozen layers: {args.num_unfrozen_layers}")
    print(f"  LR:           {args.learning_rate}")
    print(f"  Epochs:       {args.num_epochs}")
    print(f"  Grad accum:   {args.grad_accum_steps}")
    print(f"  Contrastive:  weight={args.contrastive_weight}, T={args.contrastive_temperature}")
    print("=" * 60)

    os.makedirs(args.result_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(args.result_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)

    # 加载 fold 数据
    folds_data, num_classes = load_folds(args.fold_dir)
    print(f"\n  {len(folds_data)} folds, {num_classes} classes")

    # 加载模型
    model, tokenizer, classifier, hidden_dim = setup_model(
        args.model_path, num_classes,
        args.lora_r, args.lora_alpha, args.lora_dropout,
        args.num_unfrozen_layers, device,
    )

    # 类别权重
    all_labels = pd.concat(folds_data.values())
    class_counts = Counter(all_labels['class_id'].values)
    total = sum(class_counts.values())
    class_weights = torch.zeros(num_classes)
    for cid, cnt in class_counts.items():
        if cid >= 0:
            class_weights[cid] = total / (num_classes * cnt)
    class_weights = class_weights.to(model.device)

    criterion = FocalLoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing,
                          weight=class_weights)

    # 5-fold CV
    fold_results = []
    all_best_states = []

    for fold_idx in range(len(folds_data)):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx}/{len(folds_data)-1}")
        print(f"{'='*60}")

        # Split
        val_df = folds_data[fold_idx]
        train_dfs = [folds_data[i] for i in range(len(folds_data)) if i != fold_idx]

        train_dataset = CTATableDataset(train_dfs, args.table_dir, args.file_type, args.sample_rows)
        val_dataset = CTATableDataset([val_df], args.table_dir, args.file_type, args.sample_rows)
        print(f"  Train: {len(train_dataset)} tables, Val: {len(val_dataset)} tables")

        # 重新初始化 LoRA + classifier
        model, tokenizer, classifier, hidden_dim = setup_model(
            args.model_path, num_classes,
            args.lora_r, args.lora_alpha, args.lora_dropout,
            args.num_unfrozen_layers, device,
        )
        class_weights = class_weights.to(model.device)
        criterion = FocalLoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing,
                              weight=class_weights)
        con_criterion = SupConLoss(temperature=args.contrastive_temperature)

        # Optimizer
        trainable_params = list(model.parameters()) + list(classifier.parameters())
        optimizer = torch.optim.AdamW(
            trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay
        )

        # Scheduler
        total_steps = (len(train_dataset) // args.grad_accum_steps + 1) * args.num_epochs
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.learning_rate, total_steps=total_steps,
            pct_start=args.warmup_ratio, anneal_strategy='cos',
        )

        # Train
        best_state, best_f1 = train_one_fold(
            model, tokenizer, classifier, optimizer, scheduler,
            criterion, con_criterion, args.contrastive_weight,
            train_dataset, val_dataset,
            args.num_epochs, args.grad_accum_steps, args.patience,
            model.device, args.sample_rows, args.max_prefix_length,
            fold_idx, args.result_dir, args.save_model,
        )

        all_best_states.append(best_state)
        fold_results.append({'fold': fold_idx, 'best_val_macro_f1': best_f1})
        print(f"  Fold {fold_idx} best Macro F1: {best_f1:.4f}")

        # 清理显存
        del model, classifier, optimizer, scheduler
        torch.cuda.empty_cache()

    # Ensemble test
    print(f"\n{'='*60}")
    print("Ensemble Test")
    print(f"{'='*60}")

    model_template, _, classifier_template, _ = setup_model(
        args.model_path, num_classes,
        args.lora_r, args.lora_alpha, args.lora_dropout,
        args.num_unfrozen_layers, device,
    )

    ensemble_results = {}
    for fold_idx in range(len(folds_data)):
        test_df = folds_data[fold_idx]
        test_dataset = CTATableDataset([test_df], args.table_dir, args.file_type, args.sample_rows)
        other_states = [all_best_states[i] for i in range(len(folds_data)) if i != fold_idx]

        metrics = ensemble_test(
            other_states, model_template, tokenizer, classifier_template,
            test_dataset, model_template.device, args.sample_rows, args.max_prefix_length,
        )
        ensemble_results[f"fold_{fold_idx}"] = metrics
        print(f"  Fold {fold_idx} test — Acc: {metrics['accuracy']:.4f} | "
              f"Macro F1: {metrics['macro_f1']:.4f} | Micro F1: {metrics['micro_f1']:.4f}")

    # 汇总
    avg_acc = np.mean([m['accuracy'] for m in ensemble_results.values()])
    avg_macro_f1 = np.mean([m['macro_f1'] for m in ensemble_results.values()])
    avg_micro_f1 = np.mean([m['micro_f1'] for m in ensemble_results.values()])

    final_results = {
        'per_fold_cv': fold_results,
        'ensemble_test': ensemble_results,
        'ensemble_avg': {
            'accuracy': avg_acc,
            'macro_f1': avg_macro_f1,
            'micro_f1': avg_micro_f1,
        },
    }

    results_path = os.path.join(args.result_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Results saved to: {results_path}")
    print(f"Ensemble Avg — Acc: {avg_acc:.4f} | Macro F1: {avg_macro_f1:.4f} | Micro F1: {avg_micro_f1:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
