import os.path as osp
import os
import json
import pandas as pd
import time
from collections import defaultdict
from tqdm import tqdm
from itertools import product
from functools import partial
from task.BaseTask import BaseTask, TaskResult
# from profilling import profilling_tables, profilling_table
from task.schema_matching.schema_matching_prompt import get_prompt, get_system_msg
from utils import safe_json_loads, response_yes_or_no_parse
from llm_tool.call_llm import call_llm
from llm_tool.util import BinaryMetric

# 具体profilling的结果形式可在output\profilling_result.json中查看
# 记得传递好相关参数
CONFIG = {
    "prompt_version": "base_profilling",            # 选择使用的 prompt 模板
    "sample_size": 64,                              # 每张表最多采样多少行进行剖析
    "sample_step": 64,                              # 每次处理多少列进行剖析
    "profilling_model": "qwen2.5-72b-instruct",     # 使用的 LLM 模型
    "drop_empty_rows": True,                        # 是否剔除空行后再进行剖析
    "description": "使用 LLM 对表格数据进行剖析，生成数据字典。",
}

SUPPORTED_DATASETS = {"GDC", "MIMIC", "Wikidata", "HDXSM"}


class SchemaMatchingTask(BaseTask):
    """
    示例: 对指定表进行简单的剖析任务。
    """

    def prepare(self):
        super().prepare()
        assert "dataset_root" in self.args, "dataset_root is required."
        assert "dataset" in self.args, "dataset is required."
        assert "profile_path" in self.args, "profile_path is required."
        assert "model_code" in self.args, "model_code is required."
        assert "prompt_version" in self.args, "prompt_version is required."

    @staticmethod
    def _normalize_profile_key(key: str) -> str:
        return str(key).replace("\\", "/").strip("/")

    @classmethod
    def _build_profile_lookup(cls, schema_profile: dict) -> dict:
        normalized_profile = {}
        suffix_index = defaultdict(list)

        for key, value in schema_profile.items():
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

    # 若无特殊需求,只需要实现 execute 方法, profilling信息均在self.get_arg 中获取
    def execute(self):
        dataset_root = self.get_arg("dataset_root")
        dataset_name = self.get_arg("dataset")
        profile_path = self.get_arg("profile_path")
        prompt_version = self.get_arg("prompt_version")
        model_code = self.get_arg("model_code")
        run_only_pos = self.get_arg("run_only_pos", False)
        block_profile = self.get_arg("block_profile", False)
        if dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        schema_profile = json.load(open(profile_path, 'r', encoding='utf-8'))
        profile_lookup = self._build_profile_lookup(schema_profile)
        prompt_func = partial(get_prompt, version_str=prompt_version)
        system_msg = get_system_msg(prompt_version)
        results = {"dataset": dataset_name}

        if dataset_name in ["GDC", "MIMIC"]:
            # 枚举文件夹
            for table_pair_dir in os.listdir(dataset_root):
                if not osp.isdir(osp.join(dataset_root, table_pair_dir)):
                    continue
                results[table_pair_dir] = []
                table_pair_path = osp.join(dataset_root, table_pair_dir)
                gt_file_path = osp.join(table_pair_path, "groundtruth.csv")
                src_file_path = osp.join(table_pair_path, "source.csv")
                tgt_file_path = osp.join(table_pair_path, "target.csv")
                # 从groundtruth.csv中读入gt
                gt_frame = pd.read_csv(gt_file_path)
                labels = []
                for gt_case in gt_frame.iterrows():
                    gt_case = gt_case[1].to_dict()
                    src, tgt = gt_case.get(
                        "source", None), gt_case.get("target", None)
                    labels.append((src, tgt))
                print(f"{dataset_name}-{table_pair_dir}下共扫描到{len(labels)}条gt")
                # 枚举当前文件夹下source.scv和target.csv的所有column pair（schema-only）
                src_frame = pd.read_csv(src_file_path)
                tgt_frame = pd.read_csv(tgt_file_path)
                src_cols = list(src_frame.columns.values)
                tgt_cols = list(tgt_frame.columns.values)
                all_col_pairs = product(src_cols, tgt_cols)

                src_profile_dict = self._lookup_table_profile(
                    profile_lookup, dataset_name, table_pair_dir, "source.csv")
                tgt_profile_dict = self._lookup_table_profile(
                    profile_lookup, dataset_name, table_pair_dir, "target.csv")

                # 枚举所有列对，读取profile，构造prompt并进行推理
                for col_pair in tqdm(all_col_pairs, "分类所有列对"):
                    src_col, tgt_col = col_pair
                    is_positive = (src_col, tgt_col) in labels
                    if run_only_pos and (not is_positive):
                        continue
                    prompt = prompt_func(
                        src_col=src_col,
                        src_desc=None if block_profile else src_profile_dict.get(src_col, {}).get(
                            src_col),
                        tgt_col=tgt_col,
                        tgt_desc=None if block_profile else tgt_profile_dict.get(tgt_col, {}).get(
                            tgt_col)
                    )
                    response = call_llm(
                        model=model_code,
                        msgs=([] if system_msg is None else [{"role": "system", "content": system_msg}]) +
                        [{"role": "user", "content": prompt}],
                        max_tokens=2048
                    )
                    # print(response)
                    pred = response_yes_or_no_parse(response)
                    case = {"prompt": prompt, "response": response,
                            "pred": pred, "label": is_positive}
                    results[table_pair_dir].append(case)
        elif dataset_name in ["Wikidata"]:
            # 枚举文件夹
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
                # 从mapping.json中读入gt
                gt_json = json.load(open(gt_file_path, 'r', encoding='utf-8'))
                labels = []
                for gt_case in gt_json["matches"]:
                    src, tgt = gt_case.get("source_column", None), gt_case.get(
                        "target_column", None)
                    labels.append((src, tgt))
                print(f"{dataset_name}-{table_pair_dir}下共扫描到{len(labels)}条gt")
                # 枚举当前文件夹下source.scv和target.csv的所有column pair（schema-only）
                src_frame = pd.read_csv(src_file_path)
                tgt_frame = pd.read_csv(tgt_file_path)
                src_cols = list(src_frame.columns.values)
                tgt_cols = list(tgt_frame.columns.values)
                all_col_pairs = product(src_cols, tgt_cols)

                src_profile_dict = self._lookup_table_profile(
                    profile_lookup, dataset_name, table_pair_dir, f"{table_pair_dir.lower()}_source.csv")
                tgt_profile_dict = self._lookup_table_profile(
                    profile_lookup, dataset_name, table_pair_dir, f"{table_pair_dir.lower()}_target.csv")

                # 枚举所有列对，读取profile，构造prompt并进行推理
                for col_pair in tqdm(all_col_pairs, "分类所有列对"):
                    src_col, tgt_col = col_pair
                    is_positive = (src_col, tgt_col) in labels
                    if run_only_pos and (not is_positive):
                        continue
                    prompt = prompt_func(
                        src_col=src_col,
                        src_desc=None if block_profile else src_profile_dict.get(src_col, {}).get(
                            src_col),
                        tgt_col=tgt_col,
                        tgt_desc=None if block_profile else tgt_profile_dict.get(tgt_col, {}).get(
                            tgt_col)
                    )
                    response = call_llm(
                        model=model_code,
                        msgs=([] if system_msg is None else [{"role": "system", "content": system_msg}]) +
                        [{"role": "user", "content": prompt}],
                        max_tokens=2048
                    )
                    pred = response_yes_or_no_parse(response)
                    case = {"prompt": prompt, "response": response,
                            "pred": pred, "label": is_positive}
                    results[table_pair_dir].append(case)
        elif dataset_name in ["HDXSM"]:
            # 枚举文件夹
            for table_pair_dir in os.listdir(dataset_root):
                if not osp.isdir(osp.join(dataset_root, table_pair_dir)):
                    continue
                results[table_pair_dir] = []
                table_pair_path = osp.join(dataset_root, table_pair_dir)
                gt_file_path = osp.join(
                    table_pair_path, f"{table_pair_dir.lower()}_mapping.json")
                src_file_path = osp.join(table_pair_path, f"Table1.csv")
                tgt_file_path = osp.join(table_pair_path, f"Table2.csv")
                # 从mapping.json中读入gt
                gt_json = json.load(open(gt_file_path, 'r', encoding='utf-8'))
                labels = []
                for gt_case in gt_json["matches"]:
                    src, tgt = gt_case.get("source_column", None), gt_case.get(
                        "target_column", None)
                    labels.append((src, tgt))
                print(f"{dataset_name}-{table_pair_dir}下共扫描到{len(labels)}条gt")
                # 枚举当前文件夹下source.scv和target.csv的所有column pair（schema-only）
                src_frame = pd.read_csv(src_file_path)
                tgt_frame = pd.read_csv(tgt_file_path)
                src_cols = list(src_frame.columns.values)
                tgt_cols = list(tgt_frame.columns.values)
                all_col_pairs = product(src_cols, tgt_cols)

                src_profile_dict = self._lookup_table_profile(
                    profile_lookup, dataset_name, table_pair_dir, "Table1.csv")
                tgt_profile_dict = self._lookup_table_profile(
                    profile_lookup, dataset_name, table_pair_dir, "Table2.csv")

                # 枚举所有列对，读取profile，构造prompt并进行推理
                for col_pair in tqdm(all_col_pairs, "分类所有列对"):
                    src_col, tgt_col = col_pair
                    is_positive = (src_col, tgt_col) in labels
                    if run_only_pos and (not is_positive):
                        continue
                    prompt = prompt_func(
                        src_col=src_col,
                        src_desc=None if block_profile else src_profile_dict.get(src_col, {}).get(
                            src_col),
                        tgt_col=tgt_col,
                        tgt_desc=None if block_profile else tgt_profile_dict.get(tgt_col, {}).get(
                            tgt_col)
                    )
                    response = call_llm(
                        model=model_code,
                        msgs=([] if system_msg is None else [{"role": "system", "content": system_msg}]) +
                        [{"role": "user", "content": prompt}],
                        max_tokens=2048
                    )
                    pred = response_yes_or_no_parse(response)
                    case = {"prompt": prompt, "response": response,
                            "pred": pred, "label": is_positive}
                    results[table_pair_dir].append(case)

        return results

    def validate(self, result: TaskResult):
        data = result.data
        # 计算metrics
        overall_metric = BinaryMetric()
        for split in data.keys():
            if split in ["dataset"]:
                continue
            split_metric = BinaryMetric()
            print(f"统计{split}上的metrics")
            for case in data[split]:
                split_metric.update(x=case["pred"], y=case["label"])
                overall_metric.update(x=case["pred"], y=case["label"])
            print(split_metric.stat_str())
        log_dir = osp.join("output", "results", "schema_matching")
        if not osp.exists(log_dir):
            os.makedirs(log_dir)
        log_file_name = f"{data['dataset']}_{time.time()}.json"
        metric_path = osp.join(log_dir, log_file_name)
        log_dir_log = "log"
        if not osp.exists(log_dir_log):
            os.makedirs(log_dir_log)
        log_file_path = osp.join(log_dir_log, log_file_name)
        json.dump({"metrics": overall_metric.stat()}, open(metric_path, 'w',
                  encoding='utf-8'), indent=4, ensure_ascii=False)
        json.dump({"metrics": overall_metric.stat(), "details": data}, open(
            log_file_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

    def finalize(self, result: TaskResult):
        super().finalize(result)
