"""
附录 Layer Sweep Runner — 遍历不同中间层提取 hidden state 的性能

测试 {-1, -2, -4, -8, -12, -16} 层在 2 个数据集上的表现，
用于论文附录 layer sensitivity 分析（Table 10）。

用法:
    python column_type_annotation/run_appendix_layer_sweep.py \
        --model_path ./model/Qwen2.5-7B-Instruct \
        --datasets datasets/gittables-semtab22-db-all_wrangled \
                    datasets/gittables-semtab22-sc-all_wrangled \
        --result_dir results/appendix_layer_sweep \
        --gpu_id 0
"""

import os
import sys
import json
import argparse
import subprocess


# 层 sweep 配置
LAYER_CONFIGS = ["-1", "-2", "-4", "-8", "-12", "-16"]


def main():
    parser = argparse.ArgumentParser(description="附录 Layer Sweep Runner")
    parser.add_argument("--model_path", type=str, required=True,
                        help="LLM 模型路径")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="数据集 fold 目录列表")
    parser.add_argument("--table_dir", type=str, default=None,
                        help="表格目录（默认与各 fold_dir 相同）")
    parser.add_argument("--result_dir", type=str,
                        default="results/appendix_layer_sweep")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--sample_rows", type=int, default=5)
    parser.add_argument("--layers", nargs="+", default=None,
                        help="指定层列表（默认全部）")
    args = parser.parse_args()

    layers = args.layers or LAYER_CONFIGS

    os.makedirs(args.result_dir, exist_ok=True)
    summary = {}

    for fold_dir in args.datasets:
        dataset_name = os.path.basename(fold_dir.rstrip("/\\"))
        table_dir = args.table_dir or fold_dir

        for layer_str in layers:
            run_name = f"{dataset_name}_layer{layer_str}"
            run_dir = os.path.join(args.result_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name} | Layer: {layer_str}")
            print(f"{'='*60}")

            emb_path = os.path.join(run_dir, "embeddings.pkl")

            # Step 1: 生成 embedding
            cmd_index = [
                sys.executable, "column_type_annotation/index_cta_llm_hidden.py",
                "--model_path", args.model_path,
                "--fold_dir", fold_dir,
                "--table_dir", table_dir,
                "--output_path", emb_path,
                "--layers", layer_str,
                "--sample_rows", str(args.sample_rows),
                "--no_require_profile",
                "--gpu_id", str(args.gpu_id),
            ]

            try:
                print(f"  [Index] layer={layer_str}")
                subprocess.run(cmd_index, check=True)

                # Step 2: 训练 MLP
                cmd_train = [
                    sys.executable, "column_type_annotation/train_cta.py",
                    "--fold_dir", fold_dir,
                    "--embedding_path", emb_path,
                    "--result_dir", run_dir,
                    "--gpu_id", str(args.gpu_id),
                ]
                print(f"  [Train] MLP")
                subprocess.run(cmd_train, check=True)

                results_path = os.path.join(run_dir, "results.json")
                if os.path.exists(results_path):
                    with open(results_path) as f:
                        summary[run_name] = json.load(f)
                    print(f"  ✓ {run_name}: loaded results.json")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ {run_name} failed: {e}")
                summary[run_name] = {"error": str(e)}

    # 保存汇总
    summary_path = os.path.join(args.result_dir, "layer_sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
