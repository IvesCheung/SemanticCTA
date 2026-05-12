"""
Q2B Backbone 敏感性 Runner — 在不同 decoder backbone 上运行 SemanticCTA (LoRA)

测试 SemanticCTA 方法对不同 LLM backbone 的泛化能力：
    - Qwen2.5-7B-Instruct（默认 backbone）
    - Qwen2.5-3B-Instruct
    - Qwen2.5-14B-Instruct
    - Llama-3.1-8B-Instruct

在 2 个数据集上各跑一次，汇总 backbone × dataset 的结果矩阵。

用法:
    python column_type_annotation/run_q2b_backbone.py \
        --backbones ./model/Qwen2.5-7B-Instruct ./model/Qwen2.5-3B-Instruct \
        --datasets datasets/gittables-semtab22-db-all_wrangled \
                    datasets/gittables-semtab22-sc-all_wrangled \
        --result_dir results/q2b_backbone \
        --gpu_id 0
"""

import os
import sys
import json
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Q2B Backbone 敏感性 Runner")
    parser.add_argument("--backbones", nargs="+", required=True,
                        help="LLM backbone 路径列表")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="数据集 fold 目录列表")
    parser.add_argument("--result_dir", type=str, default="results/q2b_backbone")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--contrastive_weight", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    summary = {}

    for model_path in args.backbones:
        backbone_name = os.path.basename(model_path.rstrip("/\\"))
        for fold_dir in args.datasets:
            dataset_name = os.path.basename(fold_dir.rstrip("/\\"))
            run_name = f"{backbone_name}_{dataset_name}"
            run_dir = os.path.join(args.result_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"Backbone: {backbone_name} | Dataset: {dataset_name}")
            print(f"{'='*60}")

            cmd = [
                sys.executable, "column_type_annotation/train_cta_lora.py",
                "--model_path", model_path,
                "--fold_dir", fold_dir,
                "--table_dir", fold_dir,
                "--result_dir", run_dir,
                "--lora_r", str(args.lora_r),
                "--lora_alpha", str(args.lora_alpha),
                "--contrastive_weight", str(args.contrastive_weight),
                "--num_epochs", str(args.num_epochs),
                "--gpu_id", str(args.gpu_id),
            ]

            try:
                print(f"  [Train] {backbone_name} on {dataset_name}")
                subprocess.run(cmd, check=True)

                results_path = os.path.join(run_dir, "results.json")
                if os.path.exists(results_path):
                    with open(results_path) as f:
                        summary[run_name] = json.load(f)
                    print(f"  ✓ {run_name}: loaded results.json")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ {run_name} failed: {e}")
                summary[run_name] = {"error": str(e)}

    # 保存汇总
    summary_path = os.path.join(args.result_dir, "backbone_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
