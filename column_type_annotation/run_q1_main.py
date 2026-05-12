"""
Q1 主实验 Runner — 在所有数据集上运行 SemanticCTA (LoRA)，收集主结果

用法:
    python column_type_annotation/run_q1_main.py \
        --model_path ./model/Qwen2.5-7B-Instruct \
        --datasets datasets/gittables-semtab22-db-all_wrangled \
                    datasets/gittables-semtab22-sc-all_wrangled \
        --result_dir results/q1_main \
        --gpu_id 0
"""

import os
import sys
import json
import argparse
import subprocess


def run_experiment(model_path, fold_dir, table_dir, result_dir, gpu_id, extra_args=None):
    """运行单个数据集的 LoRA 训练"""
    cmd = [
        sys.executable, "column_type_annotation/train_cta_lora.py",
        "--model_path", model_path,
        "--fold_dir", fold_dir,
        "--table_dir", table_dir,
        "--result_dir", result_dir,
        "--gpu_id", str(gpu_id),
    ]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Q1 主实验 Runner")
    parser.add_argument("--model_path", type=str, required=True, help="LLM 模型路径")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="数据集 fold 目录列表（每个即 fold_dir 也作 table_dir）")
    parser.add_argument("--result_dir", type=str, default="results/q1_main", help="总结果目录")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--contrastive_weight", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--save_model", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    summary = {}
    for fold_dir in args.datasets:
        dataset_name = os.path.basename(fold_dir.rstrip("/\\"))
        run_dir = os.path.join(args.result_dir, dataset_name)
        os.makedirs(run_dir, exist_ok=True)

        extra = [
            "--lora_r", str(args.lora_r),
            "--lora_alpha", str(args.lora_alpha),
            "--contrastive_weight", str(args.contrastive_weight),
            "--num_epochs", str(args.num_epochs),
        ]
        if args.save_model:
            extra.append("--save_model")

        # table_dir 默认与 fold_dir 相同
        table_dir = fold_dir
        run_experiment(args.model_path, fold_dir, table_dir, run_dir, args.gpu_id, extra)

        # 读取结果
        results_path = os.path.join(run_dir, "results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                summary[dataset_name] = json.load(f)
            print(f"  ✓ {dataset_name}: loaded results.json")
        else:
            print(f"  ✗ {dataset_name}: results.json not found")

    # 保存汇总
    summary_path = os.path.join(args.result_dir, "main_results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")
    print(f"Datasets completed: {len(summary)}/{len(args.datasets)}")


if __name__ == "__main__":
    main()
