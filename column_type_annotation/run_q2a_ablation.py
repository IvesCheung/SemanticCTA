"""
Q2A Embedding 消融 Runner — 运行 D1-D5 五种 embedding 提取策略 + MLP 训练

五种策略：
    D1: Final-layer hidden state (--layers -1)
    D2: Mean pool of layers L-1,L-4,L-8,L-12 (--layers -1,-4,-8,-12)
    D3: Intermediate layer L-4 (--layers -4) — SemanticCTA 默认
    D4: L-4 without table prefix (--layers -4 --no_prefix)
    D5: L-4 without profiling text (--layers -4 --no_require_profile)

用法:
    python column_type_annotation/run_q2a_ablation.py \
        --model_path ./model/Qwen2.5-7B-Instruct \
        --fold_dir datasets/gittables-semtab22-sc-all_wrangled \
        --result_dir results/q2a_ablation \
        --gpu_id 0
"""

import os
import sys
import json
import argparse
import subprocess


# 定义 5 种消融变体
ABLATION_VARIANTS = {
    "D1_final_layer": {"layers": "-1"},
    "D2_multi_layer": {"layers": "-1,-4,-8,-12"},
    "D3_intermediate": {"layers": "-4"},
    "D4_no_prefix": {"layers": "-4", "no_prefix": True},
    "D5_no_profile": {"layers": "-4", "no_require_profile": True},
}


def run_index_and_train(model_path, fold_dir, table_dir, result_dir, gpu_id,
                        layers, no_prefix=False, no_require_profile=False,
                        sample_rows=5):
    """运行 index + train 两步"""
    os.makedirs(result_dir, exist_ok=True)
    emb_path = os.path.join(result_dir, "embeddings.pkl")

    # Step 1: 提取 embedding
    cmd_index = [
        sys.executable, "column_type_annotation/index_cta_llm_hidden.py",
        "--model_path", model_path,
        "--fold_dir", fold_dir,
        "--table_dir", table_dir,
        "--output_path", emb_path,
        "--layers", layers,
        "--sample_rows", str(sample_rows),
        "--no_require_profile",
        "--gpu_id", str(gpu_id),
    ]
    if no_prefix:
        cmd_index.append("--no_prefix")

    print(f"\n  [Index] layers={layers}, no_prefix={no_prefix}")
    subprocess.run(cmd_index, check=True)

    # Step 2: 训练 MLP
    cmd_train = [
        sys.executable, "column_type_annotation/train_cta.py",
        "--fold_dir", fold_dir,
        "--embedding_path", emb_path,
        "--result_dir", result_dir,
        "--gpu_id", str(gpu_id),
    ]
    print(f"  [Train] MLP on {emb_path}")
    subprocess.run(cmd_train, check=True)


def main():
    parser = argparse.ArgumentParser(description="Q2A Embedding 消融 Runner")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--fold_dir", type=str, required=True)
    parser.add_argument("--table_dir", type=str, default=None,
                        help="表格目录（默认与 fold_dir 相同）")
    parser.add_argument("--result_dir", type=str, default="results/q2a_ablation")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--sample_rows", type=int, default=5)
    parser.add_argument("--variants", nargs="+", default=None,
                        choices=list(ABLATION_VARIANTS.keys()),
                        help="指定要运行的变体（默认全部）")
    args = parser.parse_args()

    table_dir = args.table_dir or args.fold_dir
    variants = args.variants or list(ABLATION_VARIANTS.keys())

    summary = {}
    for name, config in ABLATION_VARIANTS.items():
        if name not in variants:
            continue
        print(f"\n{'='*60}")
        print(f"Variant: {name} — {config}")
        print(f"{'='*60}")

        run_dir = os.path.join(args.result_dir, name)
        try:
            run_index_and_train(
                args.model_path, args.fold_dir, table_dir, run_dir, args.gpu_id,
                layers=config["layers"],
                no_prefix=config.get("no_prefix", False),
                no_require_profile=config.get("no_require_profile", False),
                sample_rows=args.sample_rows,
            )
            results_path = os.path.join(run_dir, "results.json")
            if os.path.exists(results_path):
                with open(results_path) as f:
                    summary[name] = json.load(f)
        except subprocess.CalledProcessError as e:
            print(f"  ✗ {name} failed: {e}")
            summary[name] = {"error": str(e)}

    # 汇总
    summary_path = os.path.join(args.result_dir, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
