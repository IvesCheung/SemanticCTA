"""
Q3 列关系消融 Runner — 运行 5 种 column relation 消融变体

消融变体（对应论文 Table 5）：
    1. full:      完整 SemanticCTA（复用 Q1 结果，不重新跑）
    2. no_prefix:     prefix 仅保留 system prompt，无表格内容
    3. single_column: prefix 中仅保留目标列，移除其他列
    4. shuffle_columns: prefix 中打乱列顺序
    5. no_profile:    不注入 profiling 文本

用法:
    python column_type_annotation/run_q3_relation.py \
        --model_path ./model/Qwen2.5-7B-Instruct \
        --fold_dir datasets/gittables-semtab22-sc-all_wrangled \
        --result_dir results/q3_relation \
        --gpu_id 0
"""

import os
import sys
import json
import argparse
import subprocess


ABLATION_VARIANTS = [
    "no_prefix",
    "single_column",
    "shuffle_columns",
    "no_profile",
]


def main():
    parser = argparse.ArgumentParser(description="Q3 列关系消融 Runner")
    parser.add_argument("--model_path", type=str, required=True,
                        help="LLM 模型路径")
    parser.add_argument("--fold_dir", type=str, required=True,
                        help="数据集 fold 目录")
    parser.add_argument("--table_dir", type=str, default=None,
                        help="表格目录（默认与 fold_dir 相同）")
    parser.add_argument("--result_dir", type=str, default="results/q3_relation")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--contrastive_weight", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--profilling_path", type=str, default=None,
                        help="Profiling JSON 路径（no_profile 变体以外的变体可用）")
    parser.add_argument("--variants", nargs="+", default=None,
                        choices=ABLATION_VARIANTS,
                        help="指定要运行的消融变体（默认全部）")
    args = parser.parse_args()

    table_dir = args.table_dir or args.fold_dir
    variants = args.variants or ABLATION_VARIANTS

    os.makedirs(args.result_dir, exist_ok=True)
    summary = {}

    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Ablation variant: {variant}")
        print(f"{'='*60}")

        run_dir = os.path.join(args.result_dir, variant)
        os.makedirs(run_dir, exist_ok=True)

        cmd = [
            sys.executable, "column_type_annotation/train_cta_lora.py",
            "--model_path", args.model_path,
            "--fold_dir", args.fold_dir,
            "--table_dir", table_dir,
            "--result_dir", run_dir,
            "--lora_r", str(args.lora_r),
            "--lora_alpha", str(args.lora_alpha),
            "--contrastive_weight", str(args.contrastive_weight),
            "--num_epochs", str(args.num_epochs),
            "--ablation", variant,
            "--gpu_id", str(args.gpu_id),
        ]
        if args.profilling_path:
            cmd += ["--profilling_path", args.profilling_path]

        try:
            print(f"  [Train] ablation={variant}")
            subprocess.run(cmd, check=True)

            results_path = os.path.join(run_dir, "results.json")
            if os.path.exists(results_path):
                with open(results_path) as f:
                    summary[variant] = json.load(f)
                print(f"  ✓ {variant}: loaded results.json")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ {variant} failed: {e}")
            summary[variant] = {"error": str(e)}

    # 保存汇总
    summary_path = os.path.join(args.result_dir, "relation_ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
