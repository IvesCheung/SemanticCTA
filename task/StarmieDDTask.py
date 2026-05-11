import subprocess
from typing import Optional
from .BaseTask import BaseTask, TaskResult
import os
import sys
from datetime import datetime

CONFIG = {
    # ============== 基本配置 ==============
    # 数据集名称: santos, GDC, HDXSM, etc.
    "benchmark": "santos",
    "description": "使用 Starmie 进行数据集发现任务，包含预训练、向量提取和检索三个阶段。",

    # ============== 预训练参数 ==============
    "batch_size": 64,                               # 训练批次大小
    "lr": 5e-5,                                     # 学习率
    "lm": "roberta",                                # 语言模型: roberta, bert, distilbert
    "n_epochs": 3,                                  # 训练轮数
    "max_len": 256,                                 # 序列最大长度
    "size": 10000,                                  # 训练数据大小
    "augment_op": "drop_col",                       # 数据增强操作: drop_col, shuffle, None
    "sample_method": "head",                        # 采样方法: head, random, etc.
    "run_id": 0,                                    # 运行ID，用于区分不同实验

    # ============== 向量提取参数 ==============
    "table_order": "column",                        # 表格顺序: column, row

    # ============== 检索参数 ==============
    # 编码器类型: cl (contrastive learning)
    "encoder": "cl",
    "matching": "hnsw",                             # 匹配方法: 强制使用 hnsw（不允许修改）
    "K": 10,                                        # 返回 Top-K 结果
    "threshold": 0.7,                               # 相似度阈值
    # HNSW 专属参数
    "index_path": None,                             # HNSW 索引文件路径 (.bin)
    "N": 50,                                        # HNSW: 从索引检索的列数

    # ============== 路径配置 ==============
    "save_model_path": None,                        # 模型保存路径，默认自动生成
    "vector_output_path": None,                     # 向量输出路径，默认自动生成
    "results_dir": None,                            # 结果目录，默认自动生成
    "metrics_path": None,                           # 评估指标保存路径，默认自动生成

    # ============== 控制选项 ==============
    "skip_pretrain": False,                         # 跳过预训练阶段
    "skip_extract": False,                          # 跳过向量提取阶段
    "skip_search": False,                           # 跳过检索阶段
    "save_model": True,                             # 是否保存模型

    # ============== Profilling 增强 ==============
    "profilling_path": None,                        # 列描述 JSON 路径，传给 run_pretrain 和 extractVectors

    # ============== 实验记录 ==============
    "enable_mlflow": False,                         # 是否启用 MLflow 记录，默认关闭以避免重复初始化开销

    # ============== 采样行数 ==============
    "sample_rows": None,                            # 提取向量时每张表读取的行数 (None=全部, 0=仅header, N=N行)

    # ============== 噪声增强 ==============
    "noise_prob": 0.0,                              # 0.0=不加噪声, 0.1=10%概率对每张表做随机列操作

    # ============== Ground Truth ==============
    "gt_path": None,                                # 自定义 GT 路径（覆盖 benchmark 默认推断）
}


class StarmieDDTask(BaseTask):
    """
    Starmie 数据集发现任务：使用对比学习方法进行数据集发现。

    包含三个阶段：
    1. prepare(): 预训练阶段 + 向量提取阶段
    2. execute(): 检索阶段
    """

    def _generate_paths(self):
        """自动生成所有相关路径"""
        # 基础结果目录
        if not self.results_dir:
            self.results_dir = f"./results/{self.benchmark}/"

        # 模型保存路径
        if not self.save_model_path:
            # 构建有意义的模型名称
            model_components = [
                f"bs{self.batch_size}",
                f"lr{str(self.lr).replace('.', '')}",
                f"{self.lm}",
                f"ep{self.n_epochs}",
                f"ml{self.max_len}",
                f"{self.augment_op}",
                f"{self.sample_method}",
            ]
            model_name = "_".join(model_components)
            self.save_model_path = f"{self.results_dir}models/{model_name}_run{self.run_id}.pt"

        # 向量输出路径
        if not self.vector_output_path:
            self.vector_output_path = f"{self.results_dir}vectors/run{self.run_id}"

        # 确保目录存在
        os.makedirs(os.path.dirname(self.save_model_path), exist_ok=True)
        os.makedirs(self.vector_output_path, exist_ok=True)

        # 生成向量文件路径
        self.query_vectors_path = f"{self.vector_output_path}/query_vectors.pickle"
        self.datalake_vectors_path = f"{self.vector_output_path}/datalake_vectors.pickle"

    def _generate_metrics_path(self) -> str:
        """生成有意义的 metrics 路径，包含实验配置信息"""
        # 构建文件名组件
        components = [
            f"bs{self.batch_size}",
            f"ep{self.n_epochs}",
            f"ml{self.max_len}",
            f"{self.augment_op}",
            f"{self.sample_method}",
            f"K{self.K}",
        ]

        # 添加时间戳
        date_folder = datetime.now().strftime("%y%m%d")
        timestamp = datetime.now().strftime("%H%M")

        filename = f"{timestamp}_starmie_{'_'.join(components)}.csv"
        metrics_path = os.path.join(
            self.results_dir, "metrics", date_folder, filename)

        # 确保目录存在
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        return metrics_path

    def prepare(self):
        """准备阶段：预训练 + 向量提取"""
        if self._prepared:
            return
        super().prepare()

        # ============== 获取所有配置参数 ==============
        self.benchmark = self.get_arg("benchmark", CONFIG["benchmark"])
        self.batch_size = self.get_arg("batch_size", CONFIG["batch_size"])
        self.lr = self.get_arg("lr", CONFIG["lr"])
        self.lm = self.get_arg("lm", CONFIG["lm"])
        self.n_epochs = self.get_arg("n_epochs", CONFIG["n_epochs"])
        self.max_len = self.get_arg("max_len", CONFIG["max_len"])
        self.size = self.get_arg("size", CONFIG["size"])
        self.augment_op = self.get_arg("augment_op", CONFIG["augment_op"])
        self.sample_method = self.get_arg(
            "sample_method", CONFIG["sample_method"])
        self.run_id = self.get_arg("run_id", CONFIG["run_id"])
        self.table_order = self.get_arg("table_order", CONFIG["table_order"])
        self.save_model = self.get_arg("save_model", CONFIG["save_model"])
        self.profilling_path = self.get_arg("profilling_path", CONFIG["profilling_path"])
        self.enable_mlflow = self.get_arg(
            "enable_mlflow", CONFIG["enable_mlflow"])
        self.sample_rows = self.get_arg("sample_rows", CONFIG["sample_rows"])
        self.noise_prob = self.get_arg("noise_prob", CONFIG["noise_prob"])
        self.gt_path = self.get_arg("gt_path", CONFIG["gt_path"])

        # 路径配置
        self.results_dir = self.get_arg("results_dir", CONFIG["results_dir"])
        self.save_model_path = self.get_arg(
            "save_model_path", CONFIG["save_model_path"])
        self.vector_output_path = self.get_arg(
            "vector_output_path", CONFIG["vector_output_path"])

        # 自动生成路径
        self._generate_paths()

        print(f"[StarmieDDTask] Configuration:")
        print(f"  Benchmark: {self.benchmark}")
        print(f"  Model save path: {self.save_model_path}")
        print(f"  Vector output path: {self.vector_output_path}")

        # ============== 阶段1：预训练 ==============
        pretrain_args = [
            "python3", "./task/dataset_discovery/starmie-main/run_pretrain.py",
            "--task", self.benchmark,
            "--batch_size", str(self.batch_size),
            "--lr", str(self.lr),
            "--lm", self.lm,
            "--n_epochs", str(self.n_epochs),
            "--max_len", str(self.max_len),
            "--size", str(self.size),
            "--augment_op", self.augment_op,
            "--sample_meth", self.sample_method,
            "--save_model_path", self.save_model_path,
            "--run_id", str(self.run_id),
        ]

        if self.profilling_path:
            pretrain_args += ["--profilling_path", self.profilling_path]

        if self.enable_mlflow:
            pretrain_args.append("--enable_mlflow")

        if self.save_model:
            pretrain_args.append("--save_model")

        if not self.get_arg("skip_pretrain", CONFIG["skip_pretrain"]):
            print("[StarmieDDTask] Stage 1: Running pretraining...")
            self._run_cmd(pretrain_args)
            print("[StarmieDDTask] Pretraining completed.")
        else:
            print("[StarmieDDTask] Skipping pretraining (skip_pretrain=True)")

        # ============== 阶段2：向量提取 ==============
        extract_args = [
            "python3", "./task/dataset_discovery/starmie-main/extractVectors.py",
            "--benchmark", self.benchmark,
            "--table_order", self.table_order,
            "--model_path", self.save_model_path,
            "--query_vectors_path", self.query_vectors_path,
            "--datalake_vectors_path", self.datalake_vectors_path,
        ]

        if self.profilling_path:
            extract_args += ["--profilling_path", self.profilling_path]

        if self.sample_rows is not None:
            extract_args += ["--sample_rows", str(self.sample_rows)]

        if self.noise_prob and self.noise_prob > 0.0:
            extract_args += ["--noise_prob", str(self.noise_prob)]

        if self.save_model:
            extract_args.append("--save_model")

        if not self.get_arg("skip_extract", CONFIG["skip_extract"]):
            print("[StarmieDDTask] Stage 2: Extracting vectors...")
            self._run_cmd(extract_args)
            print("[StarmieDDTask] Vector extraction completed.")
        else:
            print("[StarmieDDTask] Skipping vector extraction (skip_extract=True)")

    def execute(self):
        """执行阶段：检索"""
        # 获取检索参数
        self.encoder = self.get_arg("encoder", CONFIG["encoder"])
        # 强制使用 HNSW，忽略用户传入的 matching 参数
        self.matching = "hnsw"
        self.K = self.get_arg("K", CONFIG["K"])
        self.threshold = self.get_arg("threshold", CONFIG["threshold"])
        self.index_path = self.get_arg("index_path", CONFIG["index_path"])
        self.N = self.get_arg("N", CONFIG["N"])

        # 生成 metrics 路径
        self.metrics_path = self.get_arg(
            "metrics_path", CONFIG["metrics_path"])
        if not self.metrics_path:
            self.metrics_path = self._generate_metrics_path()

        print(f"[StarmieDDTask] Metrics will be saved to: {self.metrics_path}")

        # ============== 阶段3：检索 ==============
        # 强制使用 HNSW 搜索
        search_script = "./task/dataset_discovery/starmie-main/test_hnsw_search.py"

        search_args = [
            "python3", search_script,
            "--encoder", self.encoder,
            "--benchmark", self.benchmark,
            "--query_vectors_path", self.query_vectors_path,
            "--datalake_vectors_path", self.datalake_vectors_path,
            "--augment_op", self.augment_op,
            "--sample_meth", self.sample_method,
            "--matching", "hnsw",
            "--table_order", self.table_order,
            "--run_id", str(self.run_id),
            "--K", str(self.K),
            "--threshold", str(self.threshold),
            "--metrics_path", self.metrics_path,
            "--N", str(self.N),
        ]

        # HNSW index_path（可选）
        if self.index_path:
            search_args += ["--index_path", self.index_path]

        if self.gt_path:
            search_args += ["--gt_path", self.gt_path]

        if self.enable_mlflow:
            search_args.append("--enable_mlflow")

        if not self.get_arg("skip_search", CONFIG["skip_search"]):
            print(f"[StarmieDDTask] Stage 3: Running search with HNSW...")
            self._run_cmd(search_args)
            print("[StarmieDDTask] Search completed.")
            print(f"[StarmieDDTask] Metrics saved to: {self.metrics_path}")
        else:
            print("[StarmieDDTask] Skipping search (skip_search=True)")


if __name__ == "__main__":
    # ============== 示例 1: 基础配置 - Santos 数据集 ==============
    print("\n" + "="*60)
    print("Example 1: Basic Configuration - Santos Dataset")
    print("="*60)
    task1 = StarmieDDTask(
        task_name="StarmieDDTask_Santos_Basic",
        config={},
        benchmark="santos",
        batch_size=64,
        lr=5e-5,
        n_epochs=3,
        max_len=256,
        augment_op="drop_col",
        sample_method="head",
        K=10,
        threshold=0.7,
    )
    result1 = task1.run()
    print(result1.data)

    # ============== 示例 2: 自定义路径配置 ==============
    print("\n" + "="*60)
    print("Example 2: Custom Paths Configuration")
    print("="*60)
    task2 = StarmieDDTask(
        task_name="StarmieDDTask_Custom_Paths",
        config={},
        benchmark="santos",
        save_model_path="./results/santos/custom_model.pt",
        vector_output_path="./results/santos/custom_vectors",
        results_dir="./results/santos/",
        batch_size=32,
        n_epochs=5,
        K=20,
    )
    result2 = task2.run()
    print(result2.data)

    # ============== 示例 3: 跳过某些阶段（用于调试） ==============
    print("\n" + "="*60)
    print("Example 3: Skip Stages (for debugging)")
    print("="*60)
    task3 = StarmieDDTask(
        task_name="StarmieDDTask_Skip_Pretrain",
        config={},
        benchmark="santos",
        skip_pretrain=True,    # 跳过预训练
        skip_extract=True,     # 跳过向量提取
        skip_search=False,     # 仅运行检索
        save_model_path="./results/santos/models/existing_model.pt",
        vector_output_path="./results/santos/vectors/run0",
        K=10,
    )
    result3 = task3.run()
    print(result3.data)

    # ============== 示例 4: 不同的数据增强方法对比 ==============
    print("\n" + "="*60)
    print("Example 4: Different Augmentation Methods")
    print("="*60)
    for augment_op in ["drop_col", "shuffle", "None"]:
        print(f"\nTesting augment_op: {augment_op}")
        task = StarmieDDTask(
            task_name=f"StarmieDDTask_Aug_{augment_op}",
            config={},
            benchmark="santos",
            augment_op=augment_op,
            sample_method="head",
            batch_size=64,
            n_epochs=3,
            K=10,
            run_id=0,
        )
        result = task.run()
        print(f"Result for {augment_op}: {result.data}")

    # ============== 示例 5: 多线程调用示例 ==============
    print("\n" + "="*60)
    print("Example 5: Multi-threading Setup")
    print("="*60)
    print("For multi-threading, you can create multiple tasks with different run_ids:")

    # 创建多个任务配置（不实际运行）
    configs = [
        {"run_id": 0, "batch_size": 32, "lr": 5e-5},
        {"run_id": 1, "batch_size": 64, "lr": 1e-4},
        {"run_id": 2, "batch_size": 128, "lr": 5e-4},
    ]

    tasks = []
    for config in configs:
        task = StarmieDDTask(
            task_name=f"StarmieDDTask_Run_{config['run_id']}",
            config={},
            benchmark="santos",
            **config
        )
        tasks.append(task)
        print(
            f"Created task with run_id={config['run_id']}, batch_size={config['batch_size']}, lr={config['lr']}")

    print("\nYou can now run these tasks in parallel using threading/multiprocessing:")
    print("Example:")
    print("  from concurrent.futures import ThreadPoolExecutor")
    print("  with ThreadPoolExecutor(max_workers=3) as executor:")
    print("      results = list(executor.map(lambda t: t.run(), tasks))")
