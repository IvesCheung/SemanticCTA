#!/usr/bin/env python
"""
CTA (Column Type Annotation) 任务统一运行脚本

功能：
1. 运行 index_cta.py 生成 embedding
2. 运行 train_cta.py 进行训练和评估

数据集目录结构（fold 文件和表文件在同一目录下）：
    gittables-semtab22-db-all_wrangled/
        fold_0.csv
        fold_1.csv
        ...
        GitTables_6.csv
        GitTables_12.csv
        ...

使用示例：
    # 完整流程（生成 embedding + 训练）
    # --table_dir 默认与 --fold_dir 相同，可省略
    python run_cta.py --fold_dir datasets/gittables-semtab22-db-all_wrangled/ \
                      --result_dir results/cta/gittables-semtab22-db-all_wrangled/
    
    # 仅训练（已有 embedding）
    python run_cta.py --fold_dir datasets/gittables-semtab22-db-all_wrangled/ \
                      --result_dir results/cta/gittables-semtab22-db-all_wrangled/ \
                      --skip_index
    
    # 仅生成 embedding
    python run_cta.py --fold_dir datasets/gittables-semtab22-db-all_wrangled/ \
                      --result_dir results/cta/gittables-semtab22-db-all_wrangled/ \
                      --skip_train
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='CTA 任务统一运行脚本')

    # 必选参数
    parser.add_argument('--fold_dir', type=str, required=True,
                        help='包含 fold_*.csv 文件的目录')
    parser.add_argument('--result_dir', type=str, default=None,
                        help='结果根目录（默认 results/cta/<fold_dir 的 basename>）')

    # 可选参数 - 索引阶段
    parser.add_argument('--table_dir', type=str, default=None,
                        help='原始表文件目录（默认与 --fold_dir 相同）')
    parser.add_argument('--skip_index', action='store_true',
                        help='跳过 embedding 生成（使用已有的 embedding）')

    # 可选参数 - 训练阶段
    parser.add_argument('--skip_train', action='store_true',
                        help='跳过训练阶段')

    # 模型配置
    parser.add_argument('--encoder', type=str, default='qwen',
                        choices=['qwen', 'tabert', 'bge'],
                        help='Embedding 模型类型 (qwen/tabert/bge)')
    parser.add_argument('--qwen_model', type=str, default='qwen3-embedding-0.6b',
                        help='Qwen 模型名称')
    parser.add_argument('--tabert_model_path', type=str,
                        default='./model/tabert_base_k3/model.bin',
                        help='TabERT 模型路径')
    parser.add_argument('--local_model_path', type=str, default=None,
                        help='本地训练的模型权重路径')
    parser.add_argument('--base_model_path', type=str, default=None,
                        help='基础模型路径')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Embedding 批次大小')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='噪声比例 (0-1)，将对应比例的单元格替换为随机字符串')
    parser.add_argument('--sample_rows', type=int, default=3,
                        help='每列采样行数')
    parser.add_argument('--profilling_path', type=str, default=None,
                        help='Profilling 数据路径')
    parser.add_argument('--require_profile', action='store_true', default=True,
                        help='是否要求必须有 profilling 数据（默认 True，没有则跳过表；设为 False 则即使没有也继续生成 embedding）')
    parser.add_argument('--no_require_profile', action='store_false', dest='require_profile',
                        help='关闭必须要求 profilling 数据（没有也继续生成 embedding）')

    # GPU 配置
    parser.add_argument('--gpu_id', type=int, default=None,
                        help='指定使用的 GPU ID（如 0, 1, 2），默认自动选择')

    # 训练配置
    parser.add_argument('--input_dim', type=int, default=768,
                        help='Embedding 维度')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='MLP 隐藏层维度')
    parser.add_argument('--train_batch_size', type=int, default=64,
                        help='训练批次大小')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='学习率（AdamW 推荐 3e-4）')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--num_residual_blocks', type=int, default=2,
                        help='残差块数量')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='Dropout 比率')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='AdamW 权重衰减')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='标签平滑系数')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='对 loss 使用类别频率倒数权重')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help='线性 warmup 的 epoch 数')
    parser.add_argument('--save_metric', type=str, default='f1',
                        choices=['acc', 'f1'],
                        help='按哪个指标保存最佳模型')
    parser.add_argument('--save_model', action='store_true',
                        help='是否保存每折的最佳模型')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss 的 gamma（0 表示普通 CE，默认 2.0）')
    parser.add_argument('--use_sampler', action='store_true',
                        help='使用 WeightedRandomSampler 平衡每个 batch')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='MixUp alpha（0 表示关闭，默认 0.2）')
    parser.add_argument('--patience', type=int, default=8,
                        help='Early Stopping 容忍 epoch 数（0 表示关闭，默认 8）')
    return parser.parse_args()


def run_index(args):
    """运行 embedding 生成"""
    print("\n" + "="*60)
    print("Stage 1: Generating Embeddings")
    print("="*60)

    if args.table_dir is None:
        args.table_dir = args.fold_dir
        print(f"--table_dir not specified, using --fold_dir: {args.fold_dir}")

    # 构建命令
    script_path = os.path.join(os.path.dirname(__file__), 'index_cta.py')
    embedding_path = os.path.join(args.result_dir, 'embeddings.pkl')

    cmd = [
        sys.executable, script_path,
        '--fold_dir', args.fold_dir,
        '--table_dir', args.table_dir,
        '--output_path', embedding_path,
        '--encoder', args.encoder,
        '--batch_size', str(args.batch_size),
        '--sample_rows', str(args.sample_rows),
        '--noise', str(args.noise),
    ]

    if args.encoder == 'tabert':
        cmd.extend(['--tabert_model_path', args.tabert_model_path])
    else:
        cmd.extend(['--qwen_model', args.qwen_model])

    if args.local_model_path:
        cmd.extend(['--local_model_path', args.local_model_path])
    if args.base_model_path:
        cmd.extend(['--base_model_path', args.base_model_path])
    if args.profilling_path:
        cmd.extend(['--profilling_path', args.profilling_path])
    if not args.require_profile:
        cmd.append('--no_require_profile')
    if args.gpu_id is not None:
        cmd.extend(['--gpu_id', str(args.gpu_id)])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("Error: Embedding generation failed!")
        sys.exit(1)

    return embedding_path


def run_train(args, embedding_path: str, run_dir: str):
    """运行训练，结果写入时间戳子目录 run_dir"""
    print("\n" + "="*60)
    print("Stage 2: Training and Evaluation")
    print("="*60)

    # 构建命令
    script_path = os.path.join(os.path.dirname(__file__), 'train_cta.py')

    cmd = [
        sys.executable, script_path,
        '--fold_dir', args.fold_dir,
        '--embedding_path', embedding_path,
        '--result_dir', run_dir,
        '--input_dim', str(args.input_dim),
        '--hidden_dim', str(args.hidden_dim),
        '--batch_size', str(args.train_batch_size),
        '--learning_rate', str(args.learning_rate),
        '--num_epochs', str(args.num_epochs),
        '--num_residual_blocks', str(args.num_residual_blocks),
        '--dropout_rate', str(args.dropout_rate),
        '--weight_decay', str(args.weight_decay),
        '--label_smoothing', str(args.label_smoothing),
        '--warmup_epochs', str(args.warmup_epochs),
        '--save_metric', args.save_metric,
        '--focal_gamma', str(args.focal_gamma),
        '--mixup_alpha', str(args.mixup_alpha),
        '--patience', str(args.patience),
        '--embedding_model', args.qwen_model if args.encoder == 'qwen' else args.encoder,
    ]

    if args.use_class_weights:
        cmd.append('--use_class_weights')
    if args.use_sampler:
        cmd.append('--use_sampler')
    if args.save_model:
        cmd.append('--save_model')
    if args.gpu_id is not None:
        cmd.extend(['--gpu_id', str(args.gpu_id)])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("Error: Training failed!")
        sys.exit(1)


def main():
    args = parse_args()

    # 推导默认 result_dir
    if args.result_dir is None:
        dataset_name = os.path.basename(args.fold_dir.rstrip('/\\'))
        args.result_dir = os.path.join('results', 'cta', dataset_name)

    # 创建时间戳子目录（仅用于训练产物）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.result_dir, timestamp)

    print("="*60)
    print("CTA Task Runner")
    print("="*60)
    print(f"Fold directory:  {args.fold_dir}")
    print(f"Result root:     {args.result_dir}")
    print(f"Run directory:   {run_dir}")
    print(f"Encoder:         {args.encoder}")
    print(f"Skip index:      {args.skip_index}")
    print(f"Skip train:      {args.skip_train}")
    if args.gpu_id is not None:
        print(f"GPU ID:          {args.gpu_id}")
    print("="*60)

    # 创建目录
    os.makedirs(args.result_dir, exist_ok=True)
    if not args.skip_train:
        os.makedirs(run_dir, exist_ok=True)
        # 保存本次实验配置
        config_path = os.path.join(run_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
        print(f"Config saved to: {config_path}")

    # embedding 文件始终放在顶层 result_dir（跨实验共用）
    embedding_path = os.path.join(args.result_dir, 'embeddings.pkl')

    # Stage 1: 生成 embedding
    if not args.skip_index:
        embedding_path = run_index(args)
    else:
        if not os.path.exists(embedding_path):
            print(f"Error: Embedding file not found: {embedding_path}")
            print("Please run without --skip_index first")
            sys.exit(1)
        print(f"Using existing embeddings: {embedding_path}")

    # Stage 2: 训练（结果输出到时间戳子目录）
    if not args.skip_train:
        run_train(args, embedding_path, run_dir)

    print("\n" + "="*60)
    print("All stages completed!")
    if not args.skip_train:
        print(f"Results saved to: {run_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
