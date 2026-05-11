import os.path as osp
import os
import json
import pandas as pd
import time
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from itertools import product
from sklearn.metrics.pairwise import cosine_similarity
from task.BaseTask import BaseTask, TaskResult
from task.dataset_discovery.Tabert.EmbeddingModel import TabertEmbedding
from llm_tool.util import BinaryMetric


CONFIG = {
    "model_path": "./model/tabert_base_k3/model.bin",
    "sample_rows": 3,
    "threshold": 0.7,
    "mask_header": False,
    "use_profilling": True,
    "description": "使用 TaBERT 进行 Schema Matching 任务",
}

SUPPORTED_DATASETS = {"GDC", "MIMIC", "Wikidata", "HDXSM"}


class TabertSchemaMatchingTask(BaseTask):
    """
    使用 TaBERT 进行 Schema Matching 任务。
    通过计算列嵌入之间的余弦相似度来判断列是否匹配。
    """

    def prepare(self):
        super().prepare()
        assert "dataset_root" in self.args, "dataset_root is required."
        assert "dataset" in self.args, "dataset is required."
        assert "model_path" in self.args, "model_path is required."

    @staticmethod
    def _normalize_profile_key(key: str) -> str:
        return str(key).replace("\\", "/").strip("/")

    @classmethod
    def _build_profile_lookup(cls, profilling_data: dict) -> dict:
        normalized_profile = {}
        suffix_index = defaultdict(list)

        for key, value in profilling_data.items():
            normalized_key = cls._normalize_profile_key(key)
            normalized_profile[normalized_key] = value

            parts = normalized_key.split("/")
            if len(parts) >= 3:
                suffix_index["/".join(parts[-3:])].append(value)
            if len(parts) >= 4:
                suffix_index["/".join(parts[-4:])].append(value)

        return {
            "exact": normalized_profile,
            "suffix": dict(suffix_index),
        }

    @classmethod
    def _lookup_table_profile(cls, profile_lookup: dict, dataset_name: str, table_pair_dir: str, file_name: str) -> dict:
        normalized_profile = {
            "exact": profile_lookup["exact"],
            "suffix": profile_lookup["suffix"],
        }
        canonical_key = cls._normalize_profile_key(
            "/".join([dataset_name, table_pair_dir, file_name])
        )
        direct_candidates = [
            canonical_key,
            f"datasets/{canonical_key}",
            f"task/schema_matching/{canonical_key}",
        ]
        for candidate in direct_candidates:
            if candidate in normalized_profile["exact"]:
                return normalized_profile["exact"][candidate]

        suffix_candidates = [
            canonical_key,
            f"datasets/{canonical_key}",
        ]
        matched_values = []
        for candidate in suffix_candidates:
            matched_values.extend(
                normalized_profile["suffix"].get(candidate, []))

        unique_values = []
        seen_ids = set()
        for value in matched_values:
            value_id = id(value)
            if value_id in seen_ids:
                continue
            seen_ids.add(value_id)
            unique_values.append(value)

        if len(unique_values) == 1:
            return unique_values[0]
        if len(unique_values) > 1:
            print(
                f"Warning: Multiple profiling entries matched {canonical_key}, using the first one.")
            return unique_values[0]
        return {}

    @staticmethod
    def _enrich_headers(cols, profilling_dict):
        """将 profilling 描述拼接进列名，与 index.py 中 csv_to_list 的处理方式一致。"""
        enriched = list(cols)
        for i, col in enumerate(enriched):
            col_profile = profilling_dict.get(col)
            if col_profile:
                description_parts = [
                    f"{k}: {v}" for k, v in col_profile.items() if k != "__type__"
                ]
                if description_parts:
                    enriched[i] = f"{col} ({' | '.join(description_parts)})"
        return enriched

    def execute(self):
        dataset_root = self.get_arg("dataset_root")
        dataset_name = self.get_arg("dataset")
        model_path = self.get_arg("model_path", CONFIG["model_path"])
        sample_rows = self.get_arg("sample_rows", CONFIG["sample_rows"])
        threshold = self.get_arg("threshold", CONFIG["threshold"])
        mask_header = self.get_arg("mask_header", CONFIG["mask_header"])
        use_profilling = self.get_arg(
            "use_profilling", CONFIG["use_profilling"])
        profile_path = self.get_arg("profile_path", None)

        if dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        results = {"dataset": dataset_name, "threshold": threshold}

        # 初始化 TaBERT 模型
        print(f"Loading TaBERT model from {model_path}...")
        embedding_model = TabertEmbedding(model_path=model_path)

        # 加载 profilling 数据（如果提供）
        profilling_data = {}
        profile_lookup = self._build_profile_lookup(profilling_data)
        if use_profilling and profile_path and osp.exists(profile_path):
            with open(profile_path, 'r', encoding='utf-8') as f:
                profilling_data = json.load(f)
            profile_lookup = self._build_profile_lookup(profilling_data)
            print(f"Loaded profilling data from {profile_path}")

        if dataset_name in ["GDC", "MIMIC"]:
            for table_pair_dir in os.listdir(dataset_root):
                if not osp.isdir(osp.join(dataset_root, table_pair_dir)):
                    continue
                results[table_pair_dir] = []
                table_pair_path = osp.join(dataset_root, table_pair_dir)
                gt_file_path = osp.join(table_pair_path, "groundtruth.csv")
                src_file_path = osp.join(table_pair_path, "source.csv")
                tgt_file_path = osp.join(table_pair_path, "target.csv")

                # 读取 ground truth
                gt_frame = pd.read_csv(gt_file_path)
                labels = []
                for gt_case in gt_frame.iterrows():
                    gt_case = gt_case[1].to_dict()
                    src, tgt = gt_case.get(
                        "source", None), gt_case.get("target", None)
                    labels.append((src, tgt))
                print(f"{dataset_name}-{table_pair_dir}下共扫描到{len(labels)}条gt")

                # 读取表数据
                src_frame = pd.read_csv(src_file_path)
                tgt_frame = pd.read_csv(tgt_file_path)
                src_cols = list(src_frame.columns.values)
                tgt_cols = list(tgt_frame.columns.values)

                # 准备表数据（采样）
                src_data = src_frame.head(sample_rows).values.tolist()
                tgt_data = tgt_frame.head(sample_rows).values.tolist()

                # 获取 profilling 数据
                src_table_identifier = "/".join(
                    [dataset_name, table_pair_dir, "source.csv"])
                tgt_table_identifier = "/".join(
                    [dataset_name, table_pair_dir, "target.csv"])
                src_profilling = self._lookup_table_profile(
                    profile_lookup, dataset_name, table_pair_dir, "source.csv")
                tgt_profilling = self._lookup_table_profile(
                    profile_lookup, dataset_name, table_pair_dir, "target.csv")

                # 将 profilling 描述拼入 header（与 index.py 保持一致）；原始列名保留用于 label 匹配
                src_headers = self._enrich_headers(
                    src_cols, src_profilling) if use_profilling else src_cols
                tgt_headers = self._enrich_headers(
                    tgt_cols, tgt_profilling) if use_profilling else tgt_cols

                # 编码源表列
                print(f"Encoding source table columns for {table_pair_dir}...")
                src_embeddings = embedding_model.encode_columns(
                    table_id=src_table_identifier,
                    headers=src_headers,
                    data=src_data,
                    mask_header=mask_header
                )

                # 编码目标表列
                print(f"Encoding target table columns for {table_pair_dir}...")
                tgt_embeddings = embedding_model.encode_columns(
                    table_id=tgt_table_identifier,
                    headers=tgt_headers,
                    data=tgt_data,
                    mask_header=mask_header
                )

                # 计算所有列对的相似度
                all_col_pairs = product(src_cols, tgt_cols)
                for col_pair in tqdm(all_col_pairs, "分类所有列对"):
                    src_col, tgt_col = col_pair
                    src_idx = src_cols.index(src_col)
                    tgt_idx = tgt_cols.index(tgt_col)

                    # 计算余弦相似度
                    src_emb = src_embeddings[src_idx:src_idx+1]
                    tgt_emb = tgt_embeddings[tgt_idx:tgt_idx+1]
                    similarity = cosine_similarity(src_emb, tgt_emb)[0][0]

                    # 基于阈值预测
                    pred = similarity >= threshold
                    is_positive = (src_col, tgt_col) in labels

                    case = {
                        "src_col": src_col,
                        "tgt_col": tgt_col,
                        "similarity": float(similarity),
                        "pred": pred,
                        "label": is_positive
                    }
                    results[table_pair_dir].append(case)

        elif dataset_name in ["Wikidata"]:
            for table_pair_dir in os.listdir(dataset_root):
                if not osp.isdir(osp.join(dataset_root, table_pair_dir)):
                    continue
                results[table_pair_dir] = []
                table_pair_path = osp.join(dataset_root, table_pair_dir)
                gt_file_path = osp.join(
                    table_pair_path, f"{table_pair_dir.lower()}_mapping.json")
                src_file_path = osp.join(
                    table_pair_path, f"{table_pair_dir.lower()}_source.csv")
                tgt_file_path = osp.join(
                    table_pair_path, f"{table_pair_dir.lower()}_target.csv")

                # 读取 ground truth
                gt_json = json.load(open(gt_file_path, 'r', encoding='utf-8'))
                labels = []
                for gt_case in gt_json["matches"]:
                    src, tgt = gt_case.get("source_column", None), gt_case.get(
                        "target_column", None)
                    labels.append((src, tgt))
                print(f"{dataset_name}-{table_pair_dir}下共扫描到{len(labels)}条gt")

                # 读取表数据
                src_frame = pd.read_csv(src_file_path)
                tgt_frame = pd.read_csv(tgt_file_path)
                src_cols = list(src_frame.columns.values)
                tgt_cols = list(tgt_frame.columns.values)

                src_data = src_frame.head(sample_rows).values.tolist()
                tgt_data = tgt_frame.head(sample_rows).values.tolist()

                # 获取 profilling 数据
                src_table_identifier = "/".join(
                    [dataset_name, table_pair_dir, f"{table_pair_dir.lower()}_source.csv"])
                tgt_table_identifier = "/".join(
                    [dataset_name, table_pair_dir, f"{table_pair_dir.lower()}_target.csv"])
                src_profilling = self._lookup_table_profile(
                    profile_lookup, dataset_name, table_pair_dir, f"{table_pair_dir.lower()}_source.csv")
                tgt_profilling = self._lookup_table_profile(
                    profile_lookup, dataset_name, table_pair_dir, f"{table_pair_dir.lower()}_target.csv")

                # 将 profilling 描述拼入 header（与 index.py 保持一致）；原始列名保留用于 label 匹配
                src_headers = self._enrich_headers(
                    src_cols, src_profilling) if use_profilling else src_cols
                tgt_headers = self._enrich_headers(
                    tgt_cols, tgt_profilling) if use_profilling else tgt_cols

                # 编码
                print(f"Encoding tables for {table_pair_dir}...")
                src_embeddings = embedding_model.encode_columns(
                    table_id=src_table_identifier,
                    headers=src_headers,
                    data=src_data,
                    mask_header=mask_header
                )
                tgt_embeddings = embedding_model.encode_columns(
                    table_id=tgt_table_identifier,
                    headers=tgt_headers,
                    data=tgt_data,
                    mask_header=mask_header
                )

                # 计算相似度
                all_col_pairs = product(src_cols, tgt_cols)
                for col_pair in tqdm(all_col_pairs, "分类所有列对"):
                    src_col, tgt_col = col_pair
                    src_idx = src_cols.index(src_col)
                    tgt_idx = tgt_cols.index(tgt_col)

                    src_emb = src_embeddings[src_idx:src_idx+1]
                    tgt_emb = tgt_embeddings[tgt_idx:tgt_idx+1]
                    similarity = cosine_similarity(src_emb, tgt_emb)[0][0]

                    pred = similarity >= threshold
                    is_positive = (src_col, tgt_col) in labels

                    case = {
                        "src_col": src_col,
                        "tgt_col": tgt_col,
                        "similarity": float(similarity),
                        "pred": pred,
                        "label": is_positive
                    }
                    results[table_pair_dir].append(case)

        elif dataset_name in ["HDXSM"]:
            for table_pair_dir in os.listdir(dataset_root):
                if not osp.isdir(osp.join(dataset_root, table_pair_dir)):
                    continue
                results[table_pair_dir] = []
                table_pair_path = osp.join(dataset_root, table_pair_dir)
                gt_file_path = osp.join(
                    table_pair_path, f"{table_pair_dir.lower()}_mapping.json")
                src_file_path = osp.join(table_pair_path, f"Table1.csv")
                tgt_file_path = osp.join(table_pair_path, f"Table2.csv")

                # 读取 ground truth
                gt_json = json.load(open(gt_file_path, 'r', encoding='utf-8'))
                labels = []
                for gt_case in gt_json["matches"]:
                    src, tgt = gt_case.get("source_column", None), gt_case.get(
                        "target_column", None)
                    labels.append((src, tgt))
                print(f"{dataset_name}-{table_pair_dir}下共扫描到{len(labels)}条gt")

                # 读取表数据
                src_frame = pd.read_csv(src_file_path)
                tgt_frame = pd.read_csv(tgt_file_path)
                src_cols = list(src_frame.columns.values)
                tgt_cols = list(tgt_frame.columns.values)

                src_data = src_frame.head(sample_rows).values.tolist()
                tgt_data = tgt_frame.head(sample_rows).values.tolist()

                # 获取 profilling 数据
                src_table_identifier = "/".join(
                    [dataset_name, table_pair_dir, f"Table1.csv"])
                tgt_table_identifier = "/".join(
                    [dataset_name, table_pair_dir, f"Table2.csv"])
                src_profilling = self._lookup_table_profile(
                    profile_lookup, dataset_name, table_pair_dir, "Table1.csv")
                tgt_profilling = self._lookup_table_profile(
                    profile_lookup, dataset_name, table_pair_dir, "Table2.csv")

                # 将 profilling 描述拼入 header（与 index.py 保持一致）；原始列名保留用于 label 匹配
                src_headers = self._enrich_headers(
                    src_cols, src_profilling) if use_profilling else src_cols
                tgt_headers = self._enrich_headers(
                    tgt_cols, tgt_profilling) if use_profilling else tgt_cols

                # 编码
                print(f"Encoding tables for {table_pair_dir}...")
                src_embeddings = embedding_model.encode_columns(
                    table_id=src_table_identifier,
                    headers=src_headers,
                    data=src_data,
                    mask_header=mask_header
                )
                tgt_embeddings = embedding_model.encode_columns(
                    table_id=tgt_table_identifier,
                    headers=tgt_headers,
                    data=tgt_data,
                    mask_header=mask_header
                )

                # 计算相似度
                all_col_pairs = product(src_cols, tgt_cols)
                for col_pair in tqdm(all_col_pairs, "分类所有列对"):
                    src_col, tgt_col = col_pair
                    src_idx = src_cols.index(src_col)
                    tgt_idx = tgt_cols.index(tgt_col)

                    src_emb = src_embeddings[src_idx:src_idx+1]
                    tgt_emb = tgt_embeddings[tgt_idx:tgt_idx+1]
                    similarity = cosine_similarity(src_emb, tgt_emb)[0][0]

                    pred = similarity >= threshold
                    is_positive = (src_col, tgt_col) in labels

                    case = {
                        "src_col": src_col,
                        "tgt_col": tgt_col,
                        "similarity": float(similarity),
                        "pred": pred,
                        "label": is_positive
                    }
                    results[table_pair_dir].append(case)

        return results

    def validate(self, result: TaskResult):
        data = result.data
        overall_metric = BinaryMetric()
        threshold = data.get("threshold", 0.7)

        for split in data.keys():
            if split in ["dataset", "threshold"]:
                continue
            split_metric = BinaryMetric()
            print(f"统计{split}上的metrics (threshold={threshold})")
            for case in data[split]:
                split_metric.update(x=case["pred"], y=case["label"])
                overall_metric.update(x=case["pred"], y=case["label"])
            print(split_metric.stat_str())

        # 保存结果
        log_dir = osp.join("output", "results", "schema_matching", "tabert")
        if not osp.exists(log_dir):
            os.makedirs(log_dir)
        log_file_name = f"{data['dataset']}_tabert_t{threshold}_{time.time()}.json"
        metric_path = osp.join(log_dir, log_file_name)
        log_dir_log = "log"
        if not osp.exists(log_dir_log):
            os.makedirs(log_dir_log)
        log_file_path = osp.join(log_dir_log, log_file_name)

        json.dump(
            {"metrics": overall_metric.stat(), "threshold": threshold,
             "details": data},
            open(log_file_path, 'w', encoding='utf-8'),
            indent=4, ensure_ascii=False
        )
        json.dump(
            {"metrics": overall_metric.stat(), "threshold": threshold},
            open(metric_path, 'w', encoding='utf-8'),
            indent=4, ensure_ascii=False
        )

    def finalize(self, result: TaskResult):
        super().finalize(result)
