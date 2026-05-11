from BaseTask import BaseTask, TaskResult
import time
import subprocess
from typing import List, Optional
import os
import sys

# Add parent directory to path to import BaseTask
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


CONFIG = {
    "model_path": "./model/tabert_base_k3/model.bin",
    "benchmark": "SG",
    "file_type": ".csv",
    "profilling_path": None,
    "sample_rows": 1,
    "table_mapper": False,
    "K": 60,
    "scal": 1.0,
    "N": 10,
    "threshold": 0.7,
    "description": "使用 TaBERT 进行表格索引和查询的数据发现任务。",
}


class TabertIndexQueryTask(BaseTask):
    """
    TaBERT 数据发现任务: 先运行 index.py 生成表格嵌入，再运行 query.py 进行查询。
    """

    def _run_cmd(self, args: List[str], cwd: Optional[str] = None) -> None:
        """执行命令行指令"""
        cmd_str = " ".join(args)
        print(f"[TabertIndexQueryTask] Running: {cmd_str}")
        try:
            subprocess.run(args, cwd=cwd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[TabertIndexQueryTask] Command failed: {e}")
            raise

    def prepare(self):
        """准备阶段：运行 index.py 生成表格嵌入"""
        super().prepare()

        # 获取配置参数
        self.model_path = self.get_arg("model_path", CONFIG["model_path"])
        self.benchmark = self.get_arg("benchmark", CONFIG["benchmark"])
        self.file_type = self.get_arg("file_type", CONFIG["file_type"])
        self.profilling_path = self.get_arg(
            "profilling_path", CONFIG["profilling_path"])
        self.sample_rows = self.get_arg("sample_rows", CONFIG["sample_rows"])
        self.table_mapper = self.get_arg(
            "table_mapper", CONFIG["table_mapper"])

        # 构建 index.py 参数
        index_args = [
            "python", "./task/dataset_discovery/Tabert/index.py",
            "--model_path", self.model_path,
            "--benchmark", self.benchmark,
            "--file_type", self.file_type,
            "--sample_rows", str(self.sample_rows),
        ]

        # 添加可选参数
        if self.profilling_path:
            index_args.extend(["--profilling_path", self.profilling_path])

        if self.table_mapper:
            index_args.append("--table_mapper")

        if self.get_arg("shuffle", False):
            index_args.append("--shuffle")

        if self.get_arg("mask_header", False):
            index_args.append("--mask_header")

        # 运行 index.py
        if not self.get_arg("skip_index", False):
            print(
                "[TabertIndexQueryTask] Step 1: Running index.py to generate table embeddings...")
            self._run_cmd(index_args)
            print("[TabertIndexQueryTask] Index generation completed.")
        else:
            print("[TabertIndexQueryTask] Skipping index generation (skip_index=True)")

    def execute(self):
        """执行阶段：运行 query.py 进行查询"""
        # 获取查询参数
        K = self.get_arg("K", CONFIG["K"])
        scal = self.get_arg("scal", CONFIG["scal"])
        N = self.get_arg("N", CONFIG["N"])
        threshold = self.get_arg("threshold", CONFIG["threshold"])

        # 获取路径参数
        table_path = self.get_arg(
            "table_path", f"./results/{self.benchmark}/datasets_SG.pkl")
        query_path = self.get_arg(
            "query_path", f"./results/{self.benchmark}/opendata_query_datasets.pkl")
        output_path = self.get_arg(
            "output_path", f"./results/tabert_{self.benchmark}/")

        # 构建映射文件路径（如果使用了 table_mapper）
        mapping_file = None
        if self.table_mapper:
            mapping_file = f"./results/{self.benchmark}/table_name_mapping.json"

        # 构建 query.py 参数
        query_args = [
            "python", "./task/dataset_discovery/Tabert/query.py",
            "--encoder", "tabert",
            "--benchmark", self.benchmark,
            "--K", str(K),
            "--scal", str(scal),
            "--N", str(N),
            "--threshold", str(threshold),
            "--model_path", self.model_path,
            "--file_type", self.file_type,
            "--table_path", table_path,
            "--query_path", query_path,
            "--output_path", output_path,
        ]

        # 添加映射文件参数
        if mapping_file:
            query_args.extend(["--mapping_file", mapping_file])

        # 运行 query.py
        if not self.get_arg("skip_query", False):
            print(
                "[TabertIndexQueryTask] Step 2: Running query.py to perform search...")
            self._run_cmd(query_args)
            print("[TabertIndexQueryTask] Query completed.")
        else:
            print("[TabertIndexQueryTask] Skipping query (skip_query=True)")


if __name__ == "__main__":
    # 示例用法
    task = TabertIndexQueryTask(
        task_name="TabertIndexQueryTask",
        config={},
        # 通过修改benchmark来切换不同数据集
        benchmark="SG",
        # 使用的模型路径
        model_path="./model/tabert_base_k3/model.bin",
        profilling_path="./output/profilling_result.json",
        sample_rows=1,
        table_mapper=True,
        K=60,
        threshold=0.7,
        table_path="./results/SG/datasets_SG.pkl",
        query_path="./results/SG/opendata_query_datasets.pkl",
        output_path="./results/tabert_SG/",

    )
    result = task.run()
    print(result.data)
