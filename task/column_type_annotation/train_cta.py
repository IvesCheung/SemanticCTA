import os
import sys
import argparse
import copy
from collections import Counter
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import pickle
import json

# ================= 命令行参数 =================


def parse_args():
    parser = argparse.ArgumentParser(description='CTA 训练和评估脚本')

    parser.add_argument('--fold_dir', type=str,
                        default='datasets/gittables-semtab22-db-all_wrangled/',
                        help='fold csv 所在目录')
    parser.add_argument('--embedding_path', type=str,
                        default='path/to/your/embeddings.pkl',
                        help='预计算好的 embedding 文件路径')
    parser.add_argument('--result_dir', type=str, default=None,
                        help='结果输出目录（保存模型和结果）')
    parser.add_argument('--input_dim', type=int, default=768,
                        help='Embedding 的维度')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='MLP 隐藏层维度')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='每个 fold 训练的 epoch 数')
    parser.add_argument('--test_path', type=str, default=None,
                        help='独立测试集路径（可选）')
    parser.add_argument('--save_model', action='store_true',
                        help='是否保存每折的最佳模型')
    parser.add_argument('--gpu_id', type=int, default=None,
                        help='指定使用的 GPU ID（如 0, 1, 2），默认自动选择')
    parser.add_argument('--num_residual_blocks', type=int, default=2,
                        help='残差块数量（默认 2）')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='Dropout 比率（默认 0.3）')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='AdamW 权重衰减（默认 1e-2）')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='标签平滑系数（默认 0.1，0 表示关闭）')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='是否对 loss 使用类别频率倒数权重（缓解类别不平衡）')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                        help='线性 warmup 的 epoch 数（默认 2）')
    parser.add_argument('--save_metric', type=str, default='f1',
                        choices=['acc', 'f1'],
                        help='按哪个指标保存最佳模型（默认 f1）')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss 的 gamma 参数（0 表示退化为普通 CE，默认 2.0）')
    parser.add_argument('--use_sampler', action='store_true',
                        help='使用 WeightedRandomSampler 平衡每个 batch 的类别分布')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='MixUp 增强的 alpha（0 表示关闭，默认 0.2）')
    parser.add_argument('--patience', type=int, default=8,
                        help='Early Stopping 的容忍 epoch 数（0 表示关闭，默认 8）')
    parser.add_argument('--embedding_model', type=str, default=None,
                        help='生成 embedding 所用的模型名称（记录到 results.json）')
    return parser.parse_args()


# ================= 配置参数 =================
def get_config(args=None):
    if args is None:
        args = parse_args()

    config = {
        'fold_dir': args.fold_dir,
        'embedding_path': args.embedding_path,
        'result_dir': args.result_dir or os.path.dirname(args.embedding_path),
        'input_dim': args.input_dim,
        'hidden_dim': args.hidden_dim,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'device': (f'cuda:{args.gpu_id}' if args.gpu_id is not None and torch.cuda.is_available()
                   else 'cuda' if torch.cuda.is_available() else 'cpu'),
        'num_classes': None,  # 将自动从数据中计算
        'test_path': args.test_path,
        'save_model': args.save_model,
        'num_residual_blocks': args.num_residual_blocks,
        'dropout_rate': args.dropout_rate,
        'weight_decay': args.weight_decay,
        'label_smoothing': args.label_smoothing,
        'use_class_weights': args.use_class_weights,
        'warmup_epochs': args.warmup_epochs,
        'save_metric': args.save_metric,
        'focal_gamma': args.focal_gamma,
        'use_sampler': args.use_sampler,
        'mixup_alpha': args.mixup_alpha,
        'patience': args.patience,
        'embedding_model': args.embedding_model,
    }
    return config


CONFIG = None  # 将在 main() 中初始化

# ================= 1. 数据集定义 =================


class CTADataset(Dataset):
    def __init__(self, dataframe, embeddings_map, input_dim=None):
        """
        Args:
            dataframe (pd.DataFrame): 包含 table_id, col_idx, class_id 的数据框
            embeddings_map (dict): 预加载的embedding字典 {table_id: tensor_of_shape(num_cols, input_dim)}
            input_dim (int): embedding 维度，用于生成零向量（可选）
        """
        # 过滤掉 class_id 为 -1 的无效列，以及没有 embedding 的表（如 API 限流导致缺失）
        has_emb = dataframe['table_id'].isin(embeddings_map.keys())
        valid_class = dataframe['class_id'] != -1
        valid_mask = valid_class & has_emb
        self.data = dataframe[valid_mask].reset_index(drop=True)
        missing_tables = dataframe.loc[valid_class &
                                       ~has_emb, 'table_id'].nunique()
        if missing_tables > 0:
            dropped_rows = (valid_class & ~has_emb).sum()
            print(f"[CTADataset] Skipped {missing_tables} tables ({dropped_rows} rows) "
                  f"with missing embeddings — likely due to API rate limiting.")
        self.embeddings_map = embeddings_map

        # 自动检测 embedding 维度
        if input_dim is not None:
            self.input_dim = input_dim
        elif CONFIG is not None and CONFIG.get('input_dim'):
            self.input_dim = CONFIG['input_dim']
        elif embeddings_map:
            # 从 embedding 中自动检测
            sample_key = list(embeddings_map.keys())[0]
            sample_emb = embeddings_map[sample_key]
            self.input_dim = sample_emb.shape[-1] if hasattr(
                sample_emb, 'shape') else 768
        else:
            self.input_dim = 768  # 默认值

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        table_id = row['table_id']
        col_idx = int(row['col_idx'])
        label = int(row['class_id'])

        # 获取 Embedding
        # 假设 embeddings_map[table_id] 返回该表所有列的embedding矩阵
        # 格式通常为 shape (num_cols, input_dim)
        if table_id in self.embeddings_map:
            # 注意：需确保 col_idx 不越界，且 embedding 数据已转为 Tensor
            table_emb = self.embeddings_map[table_id]

            # 容错处理：如果 col_idx 超过了 embedding 的长度
            if col_idx >= len(table_emb):
                # 这种情况通常不应发生，除非数据不一致，这里给一个零向量作为fallback
                embedding = torch.zeros(self.input_dim, dtype=torch.float32)
            else:
                embedding = table_emb[col_idx]
                # 确保是 tensor 类型
                if not isinstance(embedding, torch.Tensor):
                    embedding = torch.tensor(embedding, dtype=torch.float32)
        else:
            # 如果找不到对应的 embedding，返回零向量
            embedding = torch.zeros(self.input_dim, dtype=torch.float32)

        return embedding, label


class ResidualBlock(nn.Module):
    """带 LayerNorm + GELU + Dropout 的残差块"""

    def __init__(self, dim: int, dropout_rate: float = 0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff1 = nn.Linear(dim, dim * 2)
        self.ff2 = nn.Linear(dim * 2, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Pre-norm 残差结构：更稳定的梯度流
        residual = x
        x = self.norm1(x)
        x = self.act(self.ff1(x))
        x = self.drop(x)
        x = self.ff2(x)
        x = self.drop(x)
        x = residual + x
        return self.norm2(x)


class ImprovedCTAClassifier(nn.Module):
    """残差 MLP 分类器：输入投影 -> N 个残差块 -> 分类头"""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int,
                 dropout_rate: float = 0.3, num_residual_blocks: int = 2):
        super().__init__()

        # 输入投影层
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # 残差块堆叠
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate)
            for _ in range(num_residual_blocks)
        ])

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.residual_blocks:
            x = block(x)
        return self.classifier(x)


# 保留旧名以兼容已有代码
SimpleMLP = ImprovedCTAClassifier


class FocalLoss(nn.Module):
    """Focal Loss，可选 label smoothing 和 class weights"""

    def __init__(self, gamma: float = 2.0, weight=None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.gamma > 0:
            with torch.no_grad():
                pt = torch.exp(-F.cross_entropy(input,
                               target, reduction='none'))
            ce = F.cross_entropy(input, target, weight=self.weight,
                                 label_smoothing=self.label_smoothing, reduction='none')
            return ((1.0 - pt) ** self.gamma * ce).mean()
        return F.cross_entropy(input, target, weight=self.weight,
                               label_smoothing=self.label_smoothing)


class EarlyStopping:
    """基于验证指标的 Early Stopping"""

    def __init__(self, patience: int = 8, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = -float('inf')

    def __call__(self, score: float) -> bool:
        """返回 True 表示应该停止训练"""
        if score > self.best + self.min_delta:
            self.best = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def make_weighted_sampler(dataset: 'CTADataset') -> WeightedRandomSampler:
    """为训练集构建 WeightedRandomSampler，使每个 batch 中各类别出现概率相同"""
    labels = dataset.data['class_id'].astype(int).tolist()
    class_counts = Counter(labels)
    weights = [1.0 / class_counts[lbl] for lbl in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """在嵌入空间进行 MixUp 数据增强"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def load_all_folds(fold_dir):
    """读取所有 fold csv 文件"""
    folds = {}
    max_class_id = 0

    for i in range(5):
        file_path = os.path.join(fold_dir, f'fold_{i}.csv')
        df = pd.read_csv(file_path)
        folds[i] = df

        # 计算最大类别ID以确定输出层维度
        current_max = df[df['class_id'] != -1]['class_id'].max()
        if current_max > max_class_id:
            max_class_id = current_max

    return folds, int(max_class_id) + 1


def load_embeddings(path):
    """
    加载预计算的 Embedding。
    假设文件是一个 pickle 字典: { 'GitTables_100277': numpy_array_or_tensor, ... }
    如果你的存储格式不同（例如每个表一个文件），请在此处修改。
    """
    print(f"Loading embeddings from {path} ...")
    if not os.path.exists(path):
        # 这是一个用于测试的 Mock 数据生成器，如果你没有实际文件，代码也可以跑通
        print("Embeddings file not found. Generating RANDOM embeddings for test...")
        return "MOCK_MODE"

    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def train_one_epoch(model, dataloader, criterion, optimizer, device, mixup_alpha=0.0):
    model.train()
    running_loss = 0.0
    n_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if mixup_alpha > 0.0:
            mixed_inputs, labels_a, labels_b, lam = mixup_data(
                inputs, labels, mixup_alpha)
            outputs = model(mixed_inputs)
            loss = lam * criterion(outputs, labels_a) + \
                (1.0 - lam) * criterion(outputs, labels_b)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        n_samples += inputs.size(0)

    return running_loss / n_samples


def evaluate(model, dataloader, device, return_details=False):
    """评估模型，返回 (accuracy, macro_f1, micro_f1) 或加上 (preds, targets)"""
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.numpy())

    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
    micro_f1 = f1_score(targets, preds, average='micro', zero_division=0)

    if return_details:
        return acc, macro_f1, micro_f1, preds, targets
    return acc, macro_f1, micro_f1


def test_ensemble(models, test_loader, device, num_classes):
    """使用所有 fold 模型进行集成测试（投票）

    Args:
        models: 所有 fold 的模型列表
        test_loader: 测试数据加载器
        device: 设备
        num_classes: 类别数量

    Returns:
        (accuracy, macro_f1, predictions, targets)
    """
    all_probs = []
    targets = []

    for model in models:
        model.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            batch_probs = torch.zeros(inputs.size(0), num_classes).to(device)

            # 每个模型的预测概率求和
            for model in models:
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                batch_probs += probs

            # 平均概率
            batch_probs /= len(models)
            all_probs.append(batch_probs)
            targets.extend(labels.numpy())

    # 合并所有批次
    all_probs = torch.cat(all_probs, dim=0)
    preds = torch.argmax(all_probs, dim=1).cpu().numpy()

    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
    micro_f1 = f1_score(targets, preds, average='micro', zero_division=0)

    return acc, macro_f1, micro_f1, preds, targets


def main():
    global CONFIG
    args = parse_args()
    CONFIG = get_config(args)

    print("="*60)
    print("CTA Training and Evaluation")
    print("="*60)
    print(f"Fold directory: {CONFIG['fold_dir']}")
    print(f"Embedding path: {CONFIG['embedding_path']}")
    print(f"Device: {CONFIG['device']}")
    print("="*60)

    # 1. 读取数据元信息
    folds_data, num_classes = load_all_folds(CONFIG['fold_dir'])
    CONFIG['num_classes'] = num_classes
    print(f"Total classes detected: {num_classes}")

    # 2. 读取 Embedding
    embeddings_map = load_embeddings(CONFIG['embedding_path'])

    # 如果是测试模式，构建一个假的 map 生成器
    is_mock = False
    if embeddings_map == "MOCK_MODE":
        is_mock = True
        print("Creating mock embeddings dictionary for all tables in folds...")
        embeddings_map = {}
        all_dfs = pd.concat(folds_data.values())
        unique_tables = all_dfs['table_id'].unique()
        for tid in unique_tables:
            embeddings_map[tid] = torch.randn(20, CONFIG['input_dim'])
    else:
        # 自动检测 embedding 维度
        sample_key = list(embeddings_map.keys())[0]
        sample_emb = embeddings_map[sample_key]
        if hasattr(sample_emb, 'shape'):
            detected_dim = sample_emb.shape[-1]
            if detected_dim != CONFIG['input_dim']:
                print(
                    f"Warning: Detected embedding dim {detected_dim} differs from config {CONFIG['input_dim']}")
                print(f"Using detected dimension: {detected_dim}")
                CONFIG['input_dim'] = detected_dim

    # 创建结果目录
    os.makedirs(CONFIG['result_dir'], exist_ok=True)

    # 3. 五折交叉验证循环
    results = {
        'val_acc': [],
        'val_macro_f1': [],
        'val_micro_f1': [],
        'best_models': []
    }

    all_fold_models = []

    for fold_idx in range(5):
        print(f"\n{'='*20} Fold {fold_idx}/4 {'='*20}")

        # --- 数据划分 ---
        val_df = folds_data[fold_idx]
        train_dfs = [folds_data[i] for i in range(5) if i != fold_idx]
        train_df = pd.concat(train_dfs, ignore_index=True)

        train_samples = len(train_df[train_df['class_id'] != -1])
        val_samples = len(val_df[val_df['class_id'] != -1])
        print(f"Train samples: {train_samples}, Val samples: {val_samples}")

        # --- 构建 DataLoader ---
        train_dataset = CTADataset(train_df, embeddings_map)
        val_dataset = CTADataset(val_df, embeddings_map)

        if CONFIG['use_sampler']:
            sampler = make_weighted_sampler(train_dataset)
            train_loader = DataLoader(
                train_dataset, batch_size=CONFIG['batch_size'], sampler=sampler)
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

        # --- 初始化模型 ---
        model = ImprovedCTAClassifier(
            input_dim=CONFIG['input_dim'],
            hidden_dim=CONFIG['hidden_dim'],
            num_classes=CONFIG['num_classes'],
            dropout_rate=CONFIG['dropout_rate'],
            num_residual_blocks=CONFIG['num_residual_blocks'],
        )
        model = model.to(CONFIG['device'])

        # 可选：类别权重（对抗类别不平衡)
        cls_weights = None
        if CONFIG['use_class_weights']:
            class_counts = train_dataset.data['class_id'].value_counts()
            w = torch.zeros(CONFIG['num_classes'])
            for cls_id, cnt in class_counts.items():
                w[int(cls_id)] = 1.0 / cnt
            cls_weights = (
                w / w.sum() * CONFIG['num_classes']).to(CONFIG['device'])

        # Focal Loss 或普通 CrossEntropy
        gamma = CONFIG['focal_gamma']
        if gamma > 0:
            criterion = FocalLoss(gamma=gamma, weight=cls_weights,
                                  label_smoothing=CONFIG['label_smoothing'])
            print(f"Using Focal Loss (gamma={gamma})")
        else:
            criterion = FocalLoss(gamma=0, weight=cls_weights,
                                  label_smoothing=CONFIG['label_smoothing'])

        # AdamW + 余弦退火调度器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CONFIG['learning_rate'],
            weight_decay=CONFIG['weight_decay'],
        )
        warmup_epochs = CONFIG['warmup_epochs']
        cosine_epochs = CONFIG['num_epochs'] - warmup_epochs

        def warmup_lambda(ep):
            if ep < warmup_epochs:
                return (ep + 1) / max(warmup_epochs, 1)
            return 1.0

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=warmup_lambda)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(cosine_epochs, 1), eta_min=CONFIG['learning_rate'] * 1e-2
        )

        # --- 训练循环 ---
        best_val_acc = 0.0
        best_val_f1 = 0.0
        best_val_micro_f1 = 0.0
        best_model_state = None
        save_by_f1 = CONFIG['save_metric'] == 'f1'
        early_stopper = (EarlyStopping(patience=CONFIG['patience'])
                         if CONFIG['patience'] > 0 else None)

        for epoch in range(CONFIG['num_epochs']):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, CONFIG['device'],
                mixup_alpha=CONFIG['mixup_alpha'])
            val_acc, val_f1, val_micro_f1 = evaluate(
                model, val_loader, CONFIG['device'])

            # 学习率调度
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch [{epoch+1}/{CONFIG['num_epochs']}] Loss: {train_loss:.4f} "
                f"| Val Acc: {val_acc:.4f} | Macro F1: {val_f1:.4f} | Micro F1: {val_micro_f1:.4f} | LR: {current_lr:.2e}")

            improved = (val_f1 > best_val_f1) if save_by_f1 else (
                val_acc > best_val_acc)
            if improved:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_val_micro_f1 = val_micro_f1
                best_model_state = copy.deepcopy(model.state_dict())

            # Early Stopping（基于监控指标）
            monitor = val_f1 if save_by_f1 else val_acc
            if early_stopper is not None and early_stopper(monitor):
                print(f"Early stopping triggered at epoch {epoch+1} "
                      f"(best Macro F1={best_val_f1:.4f})")
                break

        print(
            f"Fold {fold_idx} Best: Acc={best_val_acc:.4f}, Macro F1={best_val_f1:.4f}, Micro F1={best_val_micro_f1:.4f}")

        results['val_acc'].append(best_val_acc)
        results['val_macro_f1'].append(best_val_f1)
        results['val_micro_f1'].append(best_val_micro_f1)

        # 保存最佳模型
        if CONFIG['save_model'] and best_model_state is not None:
            model_path = os.path.join(
                CONFIG['result_dir'], f'model_fold_{fold_idx}.pth')
            torch.save(best_model_state, model_path)
            print(f"Model saved to {model_path}")

        # 加载最佳模型用于测试
        if best_model_state is not None:
            best_model = ImprovedCTAClassifier(
                input_dim=CONFIG['input_dim'],
                hidden_dim=CONFIG['hidden_dim'],
                num_classes=CONFIG['num_classes'],
                dropout_rate=CONFIG['dropout_rate'],
                num_residual_blocks=CONFIG['num_residual_blocks'],
            )
            best_model.load_state_dict(best_model_state)
            best_model.to(CONFIG['device'])
            all_fold_models.append(best_model)

    # 4. 打印验证集结果汇总
    print(f"\n{'='*60}")
    print("5-Fold Cross-Validation Results (Validation Set)")
    print("="*60)
    print(
        f"Average Accuracy:  {np.mean(results['val_acc']):.4f} (+/- {np.std(results['val_acc']):.4f})")
    print(
        f"Average Macro F1:  {np.mean(results['val_macro_f1']):.4f} (+/- {np.std(results['val_macro_f1']):.4f})")
    print(
        f"Average Micro F1:  {np.mean(results['val_micro_f1']):.4f} (+/- {np.std(results['val_micro_f1']):.4f})")

    for i, (acc, macro_f1, micro_f1) in enumerate(zip(results['val_acc'], results['val_macro_f1'], results['val_micro_f1'])):
        print(
            f"  Fold {i}: Acc={acc:.4f}, Macro F1={macro_f1:.4f}, Micro F1={micro_f1:.4f}")

    # 5. 独立测试集评估（如果有）
    if CONFIG['test_path'] and os.path.exists(CONFIG['test_path']):
        print(f"\n{'='*60}")
        print("Test Set Evaluation (Ensemble)")
        print("="*60)

        test_df = pd.read_csv(CONFIG['test_path'])
        test_dataset = CTADataset(test_df, embeddings_map)
        test_loader = DataLoader(
            test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

        test_acc, test_f1, test_micro_f1, test_preds, test_targets = test_ensemble(
            all_fold_models, test_loader, CONFIG['device'], CONFIG['num_classes'])

        print(f"Test Accuracy:  {test_acc:.4f}")
        print(f"Test Macro F1:  {test_f1:.4f}")
        print(f"Test Micro F1:  {test_micro_f1:.4f}")

        results['test_acc'] = test_acc
        results['test_macro_f1'] = test_f1
        results['test_micro_f1'] = test_micro_f1

        # 保存详细的分类报告
        report = classification_report(
            test_targets, test_preds, output_dict=True)
        report_path = os.path.join(
            CONFIG['result_dir'], 'classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Classification report saved to {report_path}")

    # 6. 保存结果汇总
    results_summary = {
        'config': {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v
                   for k, v in CONFIG.items()},
        'val_accuracy_mean': float(np.mean(results['val_acc'])),
        'val_accuracy_std': float(np.std(results['val_acc'])),
        'val_macro_f1_mean': float(np.mean(results['val_macro_f1'])),
        'val_macro_f1_std': float(np.std(results['val_macro_f1'])),
        'val_micro_f1_mean': float(np.mean(results['val_micro_f1'])),
        'val_micro_f1_std': float(np.std(results['val_micro_f1'])),
        'fold_results': [
            {'fold': i, 'val_acc': float(acc), 'val_macro_f1': float(
                macro_f1), 'val_micro_f1': float(micro_f1)}
            for i, (acc, macro_f1, micro_f1) in enumerate(zip(results['val_acc'], results['val_macro_f1'], results['val_micro_f1']))
        ]
    }

    if 'test_acc' in results:
        results_summary['test_accuracy'] = results['test_acc']
        results_summary['test_macro_f1'] = results['test_macro_f1']
        results_summary['test_micro_f1'] = results['test_micro_f1']

    results_path = os.path.join(CONFIG['result_dir'], 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print(f"\n{'='*60}")
    print("Training completed!")
    print("="*60)


if __name__ == '__main__':
    main()
