import time
import random
import subprocess
from typing import List, Optional
from .BaseTask import BaseTask, TaskResult
from profilling import profilling_tables, profilling_table
import os
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


class DatasetDiscoveryTask(BaseTask):
    """
    示例: 对指定表进行简单的剖析任务。
    """

    def prepare(self):
        super().prepare()
        # assert "target_table_path" in self.args, "target_table_path is required."
        # assert "target_table_paths" in self.args and type(
        #     self.args["target_table_paths"]) is list, "target_table_paths is required and must be a list."
        ml = 256
        ao = 'drop_col'
        sm = 'head'
        run_id = 0
        self.benchmark = self.get_arg("benchmark", "santos")
        self.sample_method = self.get_arg("sample_method", "head")
        self.augement_op = self.get_arg("augment_op", "drop_col")

        pretrain_args = self.get_arg("pretrain_args", [
            "python", "./task/dataset_discovery/starmie-main/run_pretrain.py",
            "--task", self.benchmark,
            "--batch_size", "64",
            "--lr", "5e-5",
            "--lm", "roberta",
            "--n_epochs", "3",
            "--max_len", str(ml),
            "--size", "10000",
            "--save_model",
            "--augment_op", self.augement_op,
            "--sample_meth", self.sample_method,
            "--save_model_path", self.get_arg("save_model_path"),
            "--run_id", str(run_id),
        ])
        if not self.get_arg("skip_pretrain", False):
            self._run_cmd(pretrain_args)

        extract_args = self.get_arg("extract_args", [
            "python", "./task/dataset_discovery/starmie-main/extractVectors.py",
            "--benchmark", self.benchmark,
            "--table_order", "column",
            "--save_model",
            "--model_path", self.get_arg("save_model_path"),
            "--query_vectors_path", f"{self.get_arg('vector_output_path')}/query_vectors.pickle",
            "--datalake_vectors_path", f"{self.get_arg('vector_output_path')}/datalake_vectors.pickle",
            "--run_id", str(run_id),
        ])
        if not self.get_arg("skip_extract", False):
            self._run_cmd(extract_args)

    # 若无特殊需求,只需要实现 execute 方法, profilling信息均在self.get_arg 中获取
    def execute(self):
        search_args = self.get_arg("search_args", [
            "python", "./task/dataset_discovery/starmie-main/test_naive_search.py",
            "--encoder", "cl",
            "--benchmark", self.benchmark,
            "--query_vectors_path", f"{self.get_arg('vector_output_path')}/query_vectors.pickle",
            "--datalake_vectors_path", f"{self.get_arg('vector_output_path')}/datalake_vectors.pickle",
            "--augment_op", self.augement_op,
            "--sample_meth", self.sample_method,
            "--matching", "linear",
            "--table_order", "column",
            "--run_id", "0",
            "--K", "10",
            "--threshold", "0.7",
        ])
        if not self.get_arg("skip_search", False):
            self._run_cmd(search_args)


if __name__ == "__main__":
    task = DatasetDiscoveryTask(
        task_name="DatasetDiscoveryTask",
        config={},
        save_model_path="./results/santos/target.pt",
        vector_output_path="./results/santos"
    )
    result = task.run()
    print(result.data)
