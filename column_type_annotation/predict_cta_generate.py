"""
CTA via LLM Generative Zero-shot Classification

核心思路：
    不提取隐状态，而是让 LLM 直接生成列的语义类型名称。
    利用 LLM 的文本生成能力（而非隐状态的表示能力），
    通过精心设计的 prompt 引导模型输出类型标签。

    1. 表序列化 → prefix prompt (含候选类型列表) → KV-cache
    2. 对每个列 → suffix prompt → 生成类型名称 (≤15 tokens)
    3. 文本匹配 → 映射到 class_id
    4. 5-fold 评估

优势：
    - 无需训练，直接利用 LLM 的语义理解能力
    - 输出的是类型名称（人类可读），而非隐向量（不可解释）
    - KV-cache 复用：表级 prefix 只算一次

用法：
    # 准备 class_names.json:  {"0": "personName", "1": "age", ...}
    python column_type_annotation/predict_cta_generate.py \
        --model_path ./model/Qwen2.5-7B-Instruct \
        --fold_dir datasets/gittables-semtab22-db-all_wrangled/ \
        --table_dir datasets/gittables-semtab22-db-all_wrangled/ \
        --class_names_path datasets/class_names.json \
        --result_dir results/cta_generate/gittables/ \
        --gpu_id 0
"""

import os
import sys
import json
import argparse
import random
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
from collections import Counter

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a semantic type classifier for tabular data columns. "
    "Given a table and a column, classify the column's semantic type. "
    "Output ONLY the exact type name from the provided list, nothing else."
)

SUFFIX_TEMPLATE = (
    'Column "{col_name}" contains values: {col_values}.\n'
    "The semantic type of this column is:"
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
#  Class Name Loading & Matching
# ---------------------------------------------------------------------------


def load_class_names(path: str) -> Dict[int, str]:
    """加载 class_id → class_name 映射 (JSON/CSV/TXT)"""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {int(k): str(v) for k, v in data.items()}
        elif isinstance(data, list):
            return {i: str(v) for i, v in enumerate(data)}
    elif ext == '.csv':
        df = pd.read_csv(path)
        cols = df.columns.tolist()
        return {int(row[cols[0]]): str(row[cols[1]]) for _, row in df.iterrows()}
    elif ext == '.txt':
        with open(path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        return {i: name for i, name in enumerate(lines)}
    raise ValueError(f"Unsupported class names format: {ext}")


def build_type_list_text(class_names: Dict[int, str]) -> str:
    """构建逗号分隔的类型名称列表"""
    return ", ".join(str(v) for v in class_names.values())


def _normalize(text: str) -> str:
    """统一格式用于匹配"""
    return text.strip().lower().replace("_", " ").replace("-", " ").strip(".")


def match_type_name(generated_text: str, class_names: Dict[int, str]) -> Tuple[int, float]:
    """
    将 LLM 生成的文本匹配到 class_id。
    返回 (class_id, confidence)。未匹配返回 (-1, 0.0)。
    """
    text = _normalize(generated_text)

    # 提取第一个有效行（去掉换行后多余内容）
    text = text.split("\n")[0].strip()
    # 去掉常见前缀
    for prefix in ["the type is", "type:", "this column is", "this is"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    best_id = -1
    best_score = 0.0

    for cid, cname in class_names.items():
        cname_clean = _normalize(cname)
        # 短名: 取最后一个 / 或 . 之后的部分
        short_name = cname_clean.split("/")[-1].split(".")[-1]

        # 1. 完全匹配
        if text == short_name or text == cname_clean:
            return cid, 1.0

        # 2. 包含匹配
        if short_name and (short_name in text or text in short_name):
            score = min(len(text), len(short_name)) / max(len(text), len(short_name))
            if score > best_score:
                best_score = score
                best_id = cid

        # 3. 模糊匹配
        if short_name:
            ratio = SequenceMatcher(None, text, short_name).ratio()
            if ratio > best_score:
                best_score = ratio
                best_id = cid

    return (best_id, best_score) if best_score > 0.45 else (-1, 0.0)


# ---------------------------------------------------------------------------
#  Fold Loading
# ---------------------------------------------------------------------------


def load_folds(fold_dir: str):
    folds = {}
    for i in range(5):
        fp = os.path.join(fold_dir, f"fold_{i}.csv")
        if os.path.exists(fp):
            folds[i] = pd.read_csv(fp)
    if not folds:
        raise FileNotFoundError(f"No fold_*.csv found in {fold_dir}")
    max_class = max(df['class_id'].max() for df in folds.values())
    return folds, int(max_class) + 1


# ---------------------------------------------------------------------------
#  LLM Generator
# ---------------------------------------------------------------------------


class CTAGenerator:
    """使用 LLM 直接生成列类型名称"""

    def __init__(self, model_path: str, max_prefix_length: int = 2048,
                 max_new_tokens: int = 15, device: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[CTAGenerator] Loading {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
            "device_map": device,
        }
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self.model.eval()

        self.max_prefix_length = max_prefix_length
        self.max_new_tokens = max_new_tokens
        self.device = next(self.model.parameters()).device
        print(f"  device = {self.device}")

    def _build_user_text(self, table_name, headers, data, n_rows, type_list_text):
        table_md = serialize_table(headers, data, n_rows)
        return (
            f"Table name: {table_name}\n{table_md}\n\n"
            f"Classify each column into one of these types:\n{type_list_text}\n\n"
            f"Output ONLY the exact type name from the list above."
        )

    def _build_suffix(self, col_name, col_values):
        values_str = ", ".join(str(v) for v in col_values) if col_values else "(empty)"
        return SUFFIX_TEMPLATE.format(col_name=col_name, col_values=values_str)

    @torch.no_grad()
    def classify_table(self, table_name, headers, data,
                       col_names, col_values_list,
                       class_names, type_list_text,
                       n_rows=5, verbose=False):
        """为一张表的所有列生成类型名称"""
        device = self.device

        # Step 1: 构建 & tokenize prefix
        user_text = self._build_user_text(
            table_name, headers, data, n_rows, type_list_text
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        prefix_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        if prefix_ids.shape[-1] > self.max_prefix_length:
            prefix_ids = prefix_ids[:, -self.max_prefix_length:]
        prefix_ids = prefix_ids.to(device)

        # Step 2: 前向传播 → KV-cache
        prefix_out = self.model(input_ids=prefix_ids, use_cache=True)
        kv_cache = prefix_out.past_key_values
        prefix_len = prefix_ids.shape[-1]
        del prefix_out

        # Step 3: 逐列生成
        predictions = []
        raw_outputs = []
        for col_idx in range(len(col_names)):
            suffix_text = self._build_suffix(
                col_names[col_idx], col_values_list[col_idx]
            )
            suffix_ids = self.tokenizer.encode(
                suffix_text, add_special_tokens=False, return_tensors="pt"
            ).to(device)

            # attention mask = prefix_ones + suffix
            full_mask = torch.ones(
                1, prefix_len + suffix_ids.shape[-1],
                dtype=torch.long, device=device
            )

            output_ids = self.model.generate(
                input_ids=suffix_ids,
                attention_mask=full_mask,
                past_key_values=kv_cache,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            generated_ids = output_ids[0, suffix_ids.shape[-1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            pred_class, confidence = match_type_name(generated_text, class_names)
            predictions.append(pred_class)
            raw_outputs.append({
                "col_name": col_names[col_idx],
                "generated": generated_text.strip(),
                "matched_id": pred_class,
                "confidence": confidence,
            })

            if verbose:
                print(f"    Col '{col_names[col_idx]}': "
                      f"generated='{generated_text.strip()}' → "
                      f"class {pred_class} (conf={confidence:.2f})")

        del kv_cache
        torch.cuda.empty_cache()

        return predictions, raw_outputs


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------


def evaluate_fold(generator, fold_df, table_dir, class_names, type_list_text,
                  file_type, sample_rows, max_prefix_length, verbose_first_n=5):
    """评估一个 fold"""
    all_preds = []
    all_labels = []
    raw_examples = []
    table_cache = {}
    verbose_count = 0

    groups = list(fold_df.groupby('table_id'))

    for table_id, group_df in tqdm(groups, desc="Evaluating tables"):
        # 读取表
        if table_id not in table_cache:
            table_path = find_table_file(table_id, table_dir, file_type)
            if table_path is None:
                continue
            try:
                df = pd.read_csv(table_path, nrows=sample_rows, lineterminator="\n")
                df = df.dropna(how="all")
                table_cache[table_id] = {
                    'headers': df.columns.tolist(),
                    'data': df.values.tolist(),
                    'name': get_basename(table_path),
                }
                if len(table_cache) > 500:
                    oldest = list(table_cache.keys())[0]
                    del table_cache[oldest]
            except Exception:
                continue

        table_info = table_cache[table_id]
        headers = table_info['headers']
        data = table_info['data']

        valid_cols = [(int(row['col_idx']), int(row['class_id']))
                      for _, row in group_df.iterrows()
                      if int(row['col_idx']) < len(headers) and int(row['class_id']) >= 0]
        if not valid_cols:
            continue

        col_indices, labels = zip(*valid_cols)
        col_names = [headers[i] for i in col_indices]
        col_values = [[str(row[i]) for row in data[:sample_rows] if i < len(row)]
                      for i in col_indices]

        verbose = verbose_count < verbose_first_n
        preds, raw_out = generator.classify_table(
            table_info['name'], headers, data,
            col_names, col_values,
            class_names, type_list_text,
            sample_rows, verbose=verbose
        )
        if verbose:
            verbose_count += 1

        all_preds.extend(preds)
        all_labels.extend(labels)
        raw_examples.extend(raw_out)

    if not all_preds:
        return {'accuracy': 0, 'macro_f1': 0, 'micro_f1': 0, 'unknown_rate': 1.0}

    from sklearn.metrics import accuracy_score, f1_score

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    unknown_rate = (all_preds == -1).mean()

    # 匹配统计
    match_conf = [r['confidence'] for r in raw_examples]
    exact_match_rate = sum(1 for c in match_conf if c == 1.0) / len(match_conf)

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'unknown_rate': unknown_rate,
        'exact_match_rate': exact_match_rate,
        'avg_confidence': np.mean(match_conf),
        'num_samples': len(all_preds),
        'raw_examples': raw_examples[:20],  # 只保存前20个
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="CTA via LLM Generative Zero-shot Classification"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--fold_dir", type=str, required=True)
    parser.add_argument("--table_dir", type=str, required=True)
    parser.add_argument("--class_names_path", type=str, required=True,
                        help="class_id → type name 映射文件 (JSON: {\"0\":\"name\",...})")
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--sample_rows", type=int, default=5)
    parser.add_argument("--file_type", type=str, default=".csv")
    parser.add_argument("--max_prefix_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=15)
    parser.add_argument("--verbose_first_n", type=int, default=3,
                        help="每个 fold 打印前 N 张表的详细输出")
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = f"cuda:{args.gpu_id}" if args.gpu_id is not None else "auto"
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    print("=" * 60)
    print("CTA via LLM Generative Zero-shot Classification")
    print("=" * 60)
    print(f"  Model:          {args.model_path}")
    print(f"  Fold dir:       {args.fold_dir}")
    print(f"  Table dir:      {args.table_dir}")
    print(f"  Class names:    {args.class_names_path}")
    print(f"  Result dir:     {args.result_dir}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Max prefix len: {args.max_prefix_length}")
    print("=" * 60)

    os.makedirs(args.result_dir, exist_ok=True)

    # 加载类型名称
    class_names = load_class_names(args.class_names_path)
    type_list_text = build_type_list_text(class_names)
    print(f"\n  Loaded {len(class_names)} class names")
    print(f"  Type list: {len(type_list_text)} chars, ~{len(type_list_text)//4} tokens")

    # 打印前10个类型名称
    sample_names = list(class_names.values())[:10]
    print(f"  Sample types: {', '.join(sample_names)}, ...")

    # 保存配置
    config = vars(args).copy()
    config['class_names'] = class_names
    with open(os.path.join(args.result_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, default=str)

    # 加载 fold
    folds_data, num_classes = load_folds(args.fold_dir)
    print(f"  {len(folds_data)} folds, {num_classes} classes in fold data")
    if num_classes != len(class_names):
        print(f"  WARNING: num_classes={num_classes} != len(class_names)={len(class_names)}")

    # 初始化 generator
    generator = CTAGenerator(
        args.model_path, args.max_prefix_length, args.max_new_tokens, device
    )

    # 5-fold 评估
    fold_results = {}
    for fold_idx in range(len(folds_data)):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx}/{len(folds_data)-1} "
              f"({len(folds_data[fold_idx])} labeled columns)")
        print(f"{'='*60}")

        metrics = evaluate_fold(
            generator, folds_data[fold_idx],
            args.table_dir, class_names, type_list_text,
            args.file_type, args.sample_rows, args.max_prefix_length,
            verbose_first_n=args.verbose_first_n,
        )
        fold_results[f"fold_{fold_idx}"] = metrics
        print(f"\n  Acc: {metrics['accuracy']:.4f} | "
              f"Macro F1: {metrics['macro_f1']:.4f} | "
              f"Micro F1: {metrics['micro_f1']:.4f}")
        print(f"  Unknown rate: {metrics['unknown_rate']:.4f} | "
              f"Exact match: {metrics['exact_match_rate']:.4f} | "
              f"Avg confidence: {metrics['avg_confidence']:.4f}")

    # 汇总
    avg_acc = np.mean([m['accuracy'] for m in fold_results.values()])
    avg_macro_f1 = np.mean([m['macro_f1'] for m in fold_results.values()])
    avg_micro_f1 = np.mean([m['micro_f1'] for m in fold_results.values()])
    avg_unknown = np.mean([m['unknown_rate'] for m in fold_results.values()])

    final_results = {
        'per_fold': {k: {kk: vv for kk, vv in v.items() if kk != 'raw_examples'}
                     for k, v in fold_results.items()},
        'average': {
            'accuracy': avg_acc,
            'macro_f1': avg_macro_f1,
            'micro_f1': avg_micro_f1,
            'unknown_rate': avg_unknown,
        },
    }

    results_path = os.path.join(args.result_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    # 保存原始输出样本
    examples_path = os.path.join(args.result_dir, 'raw_examples.json')
    examples = {}
    for fold_key, metrics in fold_results.items():
        examples[fold_key] = metrics.get('raw_examples', [])
    with open(examples_path, 'w') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*60}")
    print(f"Results saved to: {results_path}")
    print(f"Raw examples:     {examples_path}")
    print(f"Average — Acc: {avg_acc:.4f} | Macro F1: {avg_macro_f1:.4f} | "
          f"Micro F1: {avg_micro_f1:.4f} | Unknown: {avg_unknown:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
