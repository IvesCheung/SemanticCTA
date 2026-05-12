"""
Q2A 跨系统对比 Runner — 5 种 frozen embedding 源 + 同一 MLP 训练

5 种 embedding 源：
    1. BERT frozen [CLS]
    2. Sentence-BERT frozen
    3. Qwen3-Embedding frozen
    4. Description → BERT (用 profiling 描述文本代替原始值)
    5. SemanticCTA frozen (LLM hidden state L-4)

用法:
    python column_type_annotation/run_q2a_cross_system.py \
        --fold_dir datasets/gittables-semtab22-sc-all_wrangled \
        --result_dir results/q2a_cross_system \
        --gpu_id 0
"""

import os
import sys
import json
import argparse
import subprocess


EMBEDDING_SOURCES = {
    "bert_frozen": {
        "script": "index_cta_bert.py",
        "extra_args": ["--encoder_model", "bert-base-uncased"],
    },
    "sentence_bert": {
        "script": "index_cta_bert.py",
        "extra_args": ["--encoder_model", "sentence-transformers/all-mpnet-base-v2"],
    },
    "qwen3_embedding": {
        "script": "index_cta.py",
        "extra_args": ["--encoder", "qwen", "--qwen_model", "qwen3-embedding-0.6b"],
    },
    "desc_then_bert": {
        "script": "index_cta_bert.py",
        "extra_args": ["--encoder_model", "bert-base-uncased", "--use_description"],
    },
    "semantic_cta_frozen": {
        "script": "index_cta_llm_hidden.py",
        "extra_args": ["--layers", "-4", "--no_require_profile"],
        "needs_model": True,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Q2A 跨系统 Embedding 对比 Runner")
    parser.add_argument("--model_path", type=str, default="./model/Qwen2.5-7B-Instruct",
                        help="LLM 模型路径（SemanticCTA frozen 需要）")
    parser.add_argument("--fold_dir", type=str, required=True)
    parser.add_argument("--table_dir", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default="results/q2a_cross_system")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--profilling_path", type=str, default=None,
                        help="Profiling JSON（description 模式需要）")
    parser.add_argument("--sources", nargs="+", default=None,
                        choices=list(EMBEDDING_SOURCES.keys()))
    args = parser.parse_args()

    table_dir = args.table_dir or args.fold_dir
    sources = args.sources or list(EMBEDDING_SOURCES.keys())

    summary = {}
    for name, config in EMBEDDING_SOURCES.items():
        if name not in sources:
            continue
        print(f"\n{'='*60}")
        print(f"Embedding source: {name}")
        print(f"{'='*60}")

        run_dir = os.path.join(args.result_dir, name)
        os.makedirs(run_dir, exist_ok=True)
        emb_path = os.path.join(run_dir, "embeddings.pkl")

        # Step 1: 生成 embedding
        cmd = [sys.executable, f"column_type_annotation/{config['script']}"]
        if config["script"] == "index_cta_llm_hidden.py":
            cmd += ["--model_path", args.model_path]
        cmd += ["--fold_dir", args.fold_dir, "--table_dir", table_dir,
                "--output_path", emb_path, "--gpu_id", str(args.gpu_id)]
        cmd += config["extra_args"]
        if args.profilling_path and name in ("desc_then_bert", "semantic_cta_frozen"):
            cmd += ["--profilling_path", args.profilling_path]

        try:
            print(f"  [Index] {config['script']}")
            subprocess.run(cmd, check=True)

            # Step 2: 训练 MLP
            cmd_train = [
                sys.executable, "column_type_annotation/train_cta.py",
                "--fold_dir", args.fold_dir,
                "--embedding_path", emb_path,
                "--result_dir", run_dir,
                "--gpu_id", str(args.gpu_id),
            ]
            print(f"  [Train] MLP")
            subprocess.run(cmd_train, check=True)

            results_path = os.path.join(run_dir, "results.json")
            if os.path.exists(results_path):
                with open(results_path) as f:
                    summary[name] = json.load(f)
        except subprocess.CalledProcessError as e:
            print(f"  ✗ {name} failed: {e}")
            summary[name] = {"error": str(e)}

    summary_path = os.path.join(args.result_dir, "cross_system_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
