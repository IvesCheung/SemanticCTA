from .BaseTask import BaseTask, TaskResult
import time
import subprocess
from typing import List, Optional
import os
import sys
from datetime import datetime
from utils import get_basename
# Add parent directory to path to import BaseTask
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


CONFIG = {
    # 编码器类型: 'tabert' 或 'qwen'
    "encoder_type": "tabert",

    # TaBERT 模型配置
    "model_path": "./model/tabert_base_k3/model.bin",

    # Qwen 模型配置
    "qwen_model": "qwen3-embedding-0.6b",
    "batch_size": 256,
    "local_model_path": None,  # 本地训练的模型路径
    "base_model_path": None,   # 基础模型路径

    # 数据集配置
    "benchmark": "SG",
    "file_type": ".csv",
    "profilling_path": None,
    "sample_rows": 1,

    # 索引配置
    "table_mapper": False,
    "mask_header": None,  # None 或字符串，如 "##"
    "shuffle": False,
    "shuffle_columns": False,

    # 查询配置
    "K": 60,
    "scal": 1.0,
    "N": 10,
    "threshold": 0.7,

    # Ground Truth 配置
    "gt_path": None,  # 自定义 Ground Truth 路径（覆盖 benchmark 默认路径，用于 join 任务等）

    # 控制选项
    "skip_index": False,
    "skip_query": False,
    "store_result": False,

    # 噪声增强 (0.0 = 不加噪声, 0.1 = 10% 概率对每张表做随机列操作)
    "noise_prob": 0.0,

    "description": "使用 TaBERT 或 Qwen 进行表格索引和查询的数据发现任务。",
}


class TableIndexQueryTask(BaseTask):
    """
    数据发现任务: 先运行 index 生成表格嵌入，再运行 query 进行查询。
    支持 TaBERT (index.py) 和 Qwen (index_qwen.py) 两种编码器。
    """

    def _generate_metrics_path(self) -> str:
        """生成有意义的 metrics 路径，包含实验配置信息"""
        encoder_name = self.encoder_type
        if self.encoder_type == "qwen":
            # 提取 qwen 模型的简称
            model_short = self.qwen_model.replace(
                "qwen", "q").replace("-embedding", "").replace(".", "")
            if self.local_model_path:
                encoder_name = f"qwen_local_{model_short}"
            else:
                encoder_name = f"qwen_{model_short}"
        elif self.encoder_type == "tabert":
            encoder_name = "tabert"

        # 构建文件名组件
        components = [
            f"sr{self.sample_rows}",  # sample_rows
            f"tm{int(self.table_mapper)}",  # table_mapper: 0 or 1
        ]

        # mask_header
        if self.mask_header:
            mask_str = self.mask_header.replace("#", "h")  # ## -> hh
            components.append(f"mh{mask_str}")
        else:
            components.append("mh0")

        # profilling_path
        if self.profilling_path:
            components.append(f"prof_{get_basename(self.profilling_path)}")
        else:
            components.append("prof_NONE")

        # shuffle_columns
        if self.shuffle_columns:
            components.append("sc1")
        else:
            components.append("sc0")

        # 添加时间戳以避免覆盖
        # 日期子文件夹（如 260119），便于折叠查看
        date_folder = datetime.now().strftime("%y%m%d")
        timestamp = datetime.now().strftime("%H%M")

        filename = f"{timestamp}_{encoder_name}_{'_'.join(components)}.csv"
        metrics_path = os.path.join(self.results_dir, "metrics", date_folder, filename)

        # 确保目录存在
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        return metrics_path

    def prepare(self):
        """准备阶段：运行 index 生成表格嵌入"""
        super().prepare()

        # 获取通用配置参数
        self.encoder_type = self.get_arg(
            "encoder_type", CONFIG["encoder_type"])
        self.benchmark = self.get_arg("benchmark", CONFIG["benchmark"])
        self.file_type = self.get_arg("file_type", CONFIG["file_type"])
        self.profilling_path = self.get_arg(
            "profilling_path", CONFIG["profilling_path"])
        self.sample_rows = self.get_arg("sample_rows", CONFIG["sample_rows"])
        self.table_mapper = self.get_arg(
            "table_mapper", CONFIG["table_mapper"])
        self.mask_header = self.get_arg("mask_header", CONFIG["mask_header"])
        self.shuffle = self.get_arg("shuffle", CONFIG["shuffle"])
        self.shuffle_columns = self.get_arg(
            "shuffle_columns", CONFIG["shuffle_columns"])
        self.noise_prob = self.get_arg("noise_prob", CONFIG["noise_prob"])
        # self.qwen_model = self.get_arg(
        #     "qwen_model", CONFIG["qwen_model"])

        # 自动生成路径
        self.results_dir = self.get_arg(
            "results_dir", f"./results/{self.benchmark}/")
        self.table_path = f"{self.results_dir}/datalake.pkl"
        self.query_path = f"{self.results_dir}/query.pkl"
        self.mapping_file = f"{self.results_dir}/table_name_mapping.json" if self.table_mapper else None

        # 生成 metrics 路径
        # 根据 encoder_type 选择不同的 index 脚本
        if self.encoder_type == "tabert":
            self._run_tabert_index()
        elif self.encoder_type == "qwen":
            self._run_qwen_index()
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

    def _run_tabert_index(self):
        """运行 TaBERT index.py"""
        self.model_path = self.get_arg("model_path", CONFIG["model_path"])

        index_args = [
            "python3", "./task/dataset_discovery/Tabert/index.py",
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

        if self.shuffle:
            index_args.append("--shuffle")

        if self.mask_header:
            index_args.extend(["--mask_header", self.mask_header])

        if self.shuffle_columns:
            index_args.append("--shuffle_columns")

        if self.noise_prob and self.noise_prob > 0.0:
            index_args.extend(["--noise_prob", str(self.noise_prob)])

        if self.results_dir:
            index_args.extend(["--result_dir", self.results_dir])

        # 运行 index.py
        if not self.get_arg("skip_index", False):
            print("[TableIndexQueryTask] Step 1: Running TaBERT index.py...")
            self._run_cmd(index_args)
            print("[TableIndexQueryTask] TaBERT index generation completed.")
        else:
            print("[TableIndexQueryTask] Skipping index generation (skip_index=True)")

    def _run_qwen_index(self):
        """运行 Qwen index_qwen.py"""
        self.qwen_model = self.get_arg("qwen_model", CONFIG["qwen_model"])
        self.batch_size = self.get_arg("batch_size", CONFIG["batch_size"])
        self.local_model_path = self.get_arg(
            "local_model_path", CONFIG["local_model_path"])
        self.base_model_path = self.get_arg(
            "base_model_path", CONFIG["base_model_path"])

        index_args = [
            "python3", "./task/dataset_discovery/Tabert/index_qwen.py",
            "--benchmark", self.benchmark,
            "--file_type", self.file_type,
            "--sample_rows", str(self.sample_rows),
            "--qwen_model", self.qwen_model,
            "--batch_size", str(self.batch_size),
        ]

        # 添加可选参数
        if self.profilling_path:
            index_args.extend(["--profilling_path", self.profilling_path])

        if self.table_mapper:
            index_args.append("--table_mapper")

        if self.shuffle:
            index_args.append("--shuffle")

        if self.mask_header:
            index_args.extend(["--mask_header", self.mask_header])

        if self.shuffle_columns:
            index_args.append("--shuffle_columns")

        # Qwen 特有参数
        if self.local_model_path:
            index_args.extend(["--local_model_path", self.local_model_path])

        if self.base_model_path:
            index_args.extend(["--base_model_path", self.base_model_path])

        if self.noise_prob and self.noise_prob > 0.0:
            index_args.extend(["--noise_prob", str(self.noise_prob)])

        if self.results_dir:
            index_args.extend(["--result_dir", self.results_dir])

        # 运行 index_qwen.py
        if not self.get_arg("skip_index", False):
            print("[TableIndexQueryTask] Step 1: Running Qwen index_qwen.py...")
            self._run_cmd(index_args)
            print("[TableIndexQueryTask] Qwen index generation completed.")
        else:
            print("[TableIndexQueryTask] Skipping index generation (skip_index=True)")

    def execute(self):
        """执行阶段：运行 query.py 进行查询"""
        # 获取查询参数
        K = self.get_arg("K", CONFIG["K"])
        scal = self.get_arg("scal", CONFIG["scal"])
        N = self.get_arg("N", CONFIG["N"])
        threshold = self.get_arg("threshold", CONFIG["threshold"])
        store_result = self.get_arg("store_result", CONFIG["store_result"])

        self.metrics_path = self.get_arg(
            "metrics_path", self._generate_metrics_path())
        print(
            f"[TableIndexQueryTask] Metrics will be saved to: {self.metrics_path}")
        # 输出路径
        output_path = f"{self.results_dir}{self.encoder_type}_results.jsonl"

        # 构建 query.py 参数
        query_args = [
            "python3", "./task/dataset_discovery/Tabert/query.py",
            # query.py 中 qwen 对应 cl
            "--encoder", self.encoder_type if self.encoder_type == "tabert" else "cl",
            "--benchmark", self.benchmark,
            "--K", str(K),
            "--scal", str(scal),
            "--N", str(N),
            "--threshold", str(threshold),
            "--file_type", self.file_type,
            "--table_path", self.table_path,
            "--query_path", self.query_path,
            # "--output_path", output_path,
            "--metrics_path", self.metrics_path,
        ]

        # 如果使用了 TaBERT，添加 model_path
        if self.encoder_type == "tabert":
            query_args.extend(["--model_path", self.model_path])

        # 添加映射文件参数
        if self.mapping_file:
            query_args.extend(["--mapping_file", self.mapping_file])

        # 是否存储结果
        if store_result:
            query_args.append("--store_result")
            query_args.extend(["--output_path", output_path])

        # 添加自定义 Ground Truth 路径
        gt_path = self.get_arg("gt_path", CONFIG["gt_path"])
        if gt_path:
            query_args.extend(["--gt_path", gt_path])

        # 运行 query.py
        if not self.get_arg("skip_query", False):
            print("[TableIndexQueryTask] Step 2: Running query.py to perform search...")
            self._run_cmd(query_args)
            print("[TableIndexQueryTask] Query completed.")
            print(f"[TableIndexQueryTask] Results saved to: {output_path}")
            print(
                f"[TableIndexQueryTask] Metrics saved to: {self.metrics_path}")
        else:
            print("[TableIndexQueryTask] Skipping query (skip_query=True)")


if __name__ == "__main__":
    # ============== 示例 1: TaBERT 编码器 ==============
    print("\n" + "="*60)
    print("Example 1: TaBERT Encoder")
    print("="*60)
    task_tabert = TableIndexQueryTask(
        task_name="TableIndexQueryTask_TaBERT",
        config={},
        encoder_type="tabert",
        benchmark="SG",
        model_path="./model/tabert_base_k3/model.bin",
        profilling_path="./output/profilling_result.json",
        sample_rows=1,
        table_mapper=True,
        mask_header=None,
        shuffle_columns=False,
        K=60,
        threshold=0.7,
    )
    result_tabert = task_tabert.run()
    print(result_tabert.data)

    # ============== 示例 2: Qwen 编码器 (API 模式) ==============
    print("\n" + "="*60)
    print("Example 2: Qwen Encoder (API Mode)")
    print("="*60)
    task_qwen_api = TableIndexQueryTask(
        task_name="TableIndexQueryTask_Qwen_API",
        config={},
        encoder_type="qwen",
        benchmark="SG",
        qwen_model="qwen3-embedding-0.6b",
        batch_size=256,
        profilling_path="./output/profilling_result.json",
        sample_rows=1,
        table_mapper=True,
        mask_header="##",
        shuffle_columns=False,
        K=60,
        threshold=0.7,
    )
    result_qwen_api = task_qwen_api.run()
    print(result_qwen_api.data)

    # ============== 示例 3: Qwen 编码器 (Local 模式) ==============
    print("\n" + "="*60)
    print("Example 3: Qwen Encoder (Local Trained Model)")
    print("="*60)
    task_qwen_local = TableIndexQueryTask(
        task_name="TableIndexQueryTask_Qwen_Local",
        config={},
        encoder_type="qwen",
        benchmark="SG",
        qwen_model="qwen3-embedding-0.6b",
        batch_size=256,
        local_model_path="./path/to/your/trained_model.pth",  # 你的本地训练模型
        base_model_path="./model/qwen3-embedding-0.6b",       # 基础模型路径
        profilling_path=None,  # 不使用 profilling
        sample_rows=2,
        table_mapper=False,
        mask_header=None,
        shuffle_columns=True,
        K=60,
        threshold=0.7,
    )
    result_qwen_local = task_qwen_local.run()
    print(result_qwen_local.data)
