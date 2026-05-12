"""
Q4 分类头消融 Runner — 运行 5 种分类头 / loss 变体

变体（对应论文 Table 7）：
    1. full_semantic_cta: 完整 SemanticCTA（LoRA + Residual MLP + SupCon loss），复用 Q1 结果
    2. linear_head:      Linear 分类头（frozen embedding + nn.Linear）
    3. mlp_1layer:       单层 MLP 分类头
    4. mlp_2layer:       两层 MLP 分类头
    5. no_contrastive:   LoRA + Residual MLP 但无 SupCon loss

其中变体 2-4 先提取一次 frozen embedding，再用不同 head 训练 MLP。
变体 5 直接跑 LoRA 训练但关闭 contrastive loss。

用法:
    python column_type_annotation/run_q4_head.py \
        --model_path ./model/Qwen2.5-7B-Instruct \
        --fold_dir datasets/gittables-semtab22-sc-all_wrangled \
        --result_dir results/q4_head \
        --gpu_id 0
"""

import os
import sys
import json
import argparse
import subprocess


HEAD_VARIANTS = ["linear", "mlp_1layer", "mlp_2layer"]


def main():
    parser = argparse.ArgumentParser(description="Q4 分类头消融 Runner")
    parser.add_argument("--model_path", type=str, required=True,
                        help="LLM 模型路径")
    parser.add_argument("--fold_dir", type=str, required=True,
                        help="数据集 fold 目录")
    parser.add_argument("--table_dir", type=str, default=None,
                        help="表格目录（默认与 fold_dir 相同）")
    parser.add_argument("--result_dir", type=str, default="results/q4_head")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--contrastive_weight", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--profilling_path", type=str, default=None)
    parser.add_argument("--variants", nargs="+", default=None,
                        choices=HEAD_VARIANTS + ["no_contrastive", "all"],
                        help="指定要运行的变体（默认全部）")
    args = parser.parse_args()

    table_dir = args.table_dir or args.fold_dir
    variants = args.variants if args.variants and "all" not in (args.variants or []) else (
        HEAD_VARIANTS + ["no_contrastive"]
    )

    os.makedirs(args.result_dir, exist_ok=True)
    summary = {}

    # Step 1: 生成一次 frozen embedding（供 head 变体使用）
    shared_emb_dir = os.path.join(args.result_dir, "shared_embedding")
    os.makedirs(shared_emb_dir, exist_ok=True)
    emb_path = os.path.join(shared_emb_dir, "embeddings.pkl")

    need_embedding = any(v in HEAD_VARIANTS for v in variants)
    if need_embedding and not os.path.exists(emb_path):
        print(f"\n{'='*60}")
        print("Step 1: 生成 frozen embedding（供分类头变体共用）")
        print(f"{'='*60}")

        cmd_index = [
            sys.executable, "column_type_annotation/index_cta_llm_hidden.py",
            "--model_path", args.model_path,
            "--fold_dir", args.fold_dir,
            "--table_dir", table_dir,
            "--output_path", emb_path,
            "--layers", "-4",
            "--no_require_profile",
            "--gpu_id", str(args.gpu_id),
        ]
        if args.profilling_path:
            cmd_index += ["--profilling_path", args.profilling_path]

        try:
            print("  [Index] Generating frozen embeddings...")
            subprocess.run(cmd_index, check=True)
            print(f"  ✓ Embeddings saved to {emb_path}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Embedding generation failed: {e}")
            # 无法继续 head 变体
            for v in HEAD_VARIANTS:
                if v in variants:
                    summary[v] = {"error": f"Embedding generation failed: {e}"}
            # 但仍可运行 no_contrastive
            if "no_contrastive" in variants:
                need_embedding = False
            else:
                # 全部失败，保存并退出
                summary_path = os.path.join(args.result_dir, "head_ablation_summary.json")
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2, default=str)
                return

    # Step 2: 分类头变体（frozen embedding + 不同 head）
    for head_type in HEAD_VARIANTS:
        if head_type not in variants:
            continue

        print(f"\n{'='*60}")
        print(f"Variant: {head_type} head")
        print(f"{'='*60}")

        run_dir = os.path.join(args.result_dir, head_type)
        os.makedirs(run_dir, exist_ok=True)

        cmd_train = [
            sys.executable, "column_type_annotation/train_cta.py",
            "--fold_dir", args.fold_dir,
            "--embedding_path", emb_path,
            "--result_dir", run_dir,
            "--head_type", head_type,
            "--gpu_id", str(args.gpu_id),
        ]

        try:
            print(f"  [Train] head_type={head_type}")
            subprocess.run(cmd_train, check=True)

            results_path = os.path.join(run_dir, "results.json")
            if os.path.exists(results_path):
                with open(results_path) as f:
                    summary[head_type] = json.load(f)
                print(f"  ✓ {head_type}: loaded results.json")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ {head_type} failed: {e}")
            summary[head_type] = {"error": str(e)}

    # Step 3: no_contrastive 变体（LoRA 训练，关闭 SupCon loss）
    if "no_contrastive" in variants:
        print(f"\n{'='*60}")
        print("Variant: no_contrastive (LoRA without SupCon loss)")
        print(f"{'='*60}")

        run_dir = os.path.join(args.result_dir, "no_contrastive")
        os.makedirs(run_dir, exist_ok=True)

        cmd_lora = [
            sys.executable, "column_type_annotation/train_cta_lora.py",
            "--model_path", args.model_path,
            "--fold_dir", args.fold_dir,
            "--table_dir", table_dir,
            "--result_dir", run_dir,
            "--lora_r", str(args.lora_r),
            "--lora_alpha", str(args.lora_alpha),
            "--contrastive_weight", str(args.contrastive_weight),
            "--num_epochs", str(args.num_epochs),
            "--no_contrastive",
            "--gpu_id", str(args.gpu_id),
        ]
        if args.profilling_path:
            cmd_lora += ["--profilling_path", args.profilling_path]

        try:
            print("  [Train] LoRA without contrastive loss")
            subprocess.run(cmd_lora, check=True)

            results_path = os.path.join(run_dir, "results.json")
            if os.path.exists(results_path):
                with open(results_path) as f:
                    summary["no_contrastive"] = json.load(f)
                print("  ✓ no_contrastive: loaded results.json")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ no_contrastive failed: {e}")
            summary["no_contrastive"] = {"error": str(e)}

    # 保存汇总
    summary_path = os.path.join(args.result_dir, "head_ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
