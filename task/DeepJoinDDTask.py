"""
DeepJoin 数据集发现任务封装

仿照 TableDDTask 对 Tabert 的调用方式，封装 DeepJoin 的索引构建和查询流程。
支持两种编码器模式:
1. sdd (BarlowTwinsSimCLR 对比学习模型)
2. sentence_transformers (all-mpnet-base-v2 等预训练句子编码模型)
"""

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

# Add Deepjoin dataprocess directory to path for process_table_sentense import
_DATAPROCESS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dataset_discovery", "Deepjoin", "dataprocess"
)
if _DATAPROCESS_DIR not in sys.path:
    sys.path.insert(0, _DATAPROCESS_DIR)


CONFIG = {
    # ============== 编码器配置 ==============
    # 编码器类型: 'sdd' (BarlowTwinsSimCLR) 或 'sentence_transformers'
    "encoder_type": "sdd",

    # SDD 模型配置
    "sdd_model_path": None,  # 预训练的 SDD 模型路径
    "augment_op": "drop_cell",  # 数据增强: drop_col, drop_cell
    "sample_method": "alphaHead",  # 采样方法: head, tfidf_entity, alphaHead

    # SentenceTransformer 模型配置
    "st_model_path": "./model/all-mpnet-base-v2",  # 句子编码模型路径

    # ============== 通用配置 ==============
    "batch_size": 528,  # 推理批次大小
    "single_column": False,  # 是否单列模式（无表格上下文）
    "table_order": "column",  # 表格顺序: column, row
    "run_id": 0,  # 实验运行 ID

    # ============== 数据集配置 ==============
    "benchmark": "santos-query",  # 基准数据集名称
    "data_path": None,  # 数据集根路径
    "file_type": ".csv",  # 文件类型

    # ============== 查询配置 ==============
    "K": 60,  # 返回 Top-K 结果
    "scal": 1.0,  # 数据湖规模缩放因子
    "N": 10,  # 从索引中检索的列数
    "threshold": 0.7,  # 相似度阈值
    "evaluation_mode": "column",  # 评估粒度: column 或 table

    # ============== 路径配置 ==============
    "results_dir": None,  # 结果目录，默认自动生成
    "table_path": None,  # datalake 嵌入文件路径
    "query_path": None,  # query 嵌入文件路径
    "index_path": None,  # HNSW 索引路径（可选）
    "metrics_path": None,  # 评估指标保存路径

    # ============== Ground Truth 配置 ==============
    "gt_path": None,  # Ground Truth 文件路径
    "query_folder": None,  # 查询表文件夹路径
    "datalake_folder": None,  # 数据湖表文件夹路径

    # ============== 控制选项 ==============
    "skip_index": False,  # 跳过索引构建阶段
    "skip_query": False,  # 跳过查询阶段
    "store_result": False,  # 是否存储详细结果

    # 噪声增强 (0.0 = 不加噪声, 0.1 = 10% 概率对每张表做随机列操作)
    "noise_prob": 0.0,

    # 表名匿名映射 (True = 将表名替换为匿名ID，降低表名语义对模型的影响)
    "table_mapper": False,
    # 随机打乱文件处理顺序
    "shuffle": False,

    "description": "使用 DeepJoin 进行表格索引和查询的数据发现任务。支持 SDD 和 SentenceTransformer 两种编码器。",
}


class DeepJoinDDTask(BaseTask):
    """
    DeepJoin 数据发现任务: 先运行 index 生成表格嵌入，再运行 query 进行查询。
    支持 SDD (BarlowTwinsSimCLR) 和 SentenceTransformer 两种编码器。
    """

    def _generate_metrics_path(self) -> str:
        """生成有意义的 metrics 路径，包含实验配置信息"""
        encoder_name = self.encoder_type
        if self.encoder_type == "sentence_transformers":
            # 提取模型名称的简称
            model_short = get_basename(self.st_model_path).replace("-", "_")
            encoder_name = f"st_{model_short}"
        elif self.encoder_type == "sdd":
            encoder_name = f"sdd_{self.augment_op}_{self.sample_method}"

        # 构建文件名组件
        components = [
            f"K{self.K}",
            f"N{self.N}",
            f"th{str(self.threshold).replace('.', '')}",
            f"sc{'1' if self.single_column else '0'}",
        ]

        # 添加时间戳
        date_folder = datetime.now().strftime("%y%m%d")
        timestamp = datetime.now().strftime("%H%M")

        filename = f"{timestamp}_deepjoin_{encoder_name}_{'_'.join(components)}.csv"
        metrics_path = os.path.join(
            self.results_dir, "metrics", date_folder, filename)

        # 确保目录存在
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        return metrics_path

    def _generate_paths(self):
        """自动生成所有相关路径"""
        # 基础结果目录
        if not self.results_dir:
            self.results_dir = f"./results/{self.benchmark}/"

        os.makedirs(self.results_dir, exist_ok=True)

        # 生成嵌入文件路径
        if self.encoder_type == "sdd":
            prefix = f"deepjoin_sdd_{self.augment_op}_{self.sample_method}"
        else:
            model_name = get_basename(self.st_model_path)
            prefix = f"deepjoin_st_{model_name}"

        if not self.table_path:
            self.table_path = os.path.join(
                self.results_dir, f"{prefix}_datalake.pkl")

        if not self.query_path:
            self.query_path = os.path.join(
                self.results_dir, f"{prefix}_query.pkl")

    def prepare(self):
        """准备阶段：运行 index 生成表格嵌入"""
        super().prepare()

        # 获取通用配置参数
        self.encoder_type = self.get_arg(
            "encoder_type", CONFIG["encoder_type"])
        self.benchmark = self.get_arg("benchmark", CONFIG["benchmark"])
        self.file_type = self.get_arg("file_type", CONFIG["file_type"])
        self.single_column = self.get_arg(
            "single_column", CONFIG["single_column"])
        self.table_order = self.get_arg("table_order", CONFIG["table_order"])
        self.run_id = self.get_arg("run_id", CONFIG["run_id"])
        self.batch_size = self.get_arg("batch_size", CONFIG["batch_size"])

        # 获取编码器特定参数（需要在 _generate_paths 之前设置）
        self.augment_op = self.get_arg("augment_op", CONFIG["augment_op"])
        self.sample_method = self.get_arg(
            "sample_method", CONFIG["sample_method"])
        self.st_model_path = self.get_arg(
            "st_model_path", CONFIG["st_model_path"])

        # 获取路径参数
        self.results_dir = self.get_arg(
            "results_dir", CONFIG["results_dir"])
        self.table_path = self.get_arg("table_path", CONFIG["table_path"])
        self.query_path = self.get_arg("query_path", CONFIG["query_path"])
        self.profilling_path = self.get_arg("profilling_path", None)
        self.sample_rows = self.get_arg("sample_rows", None)
        self.noise_prob = self.get_arg("noise_prob", CONFIG["noise_prob"])
        self.table_mapper = self.get_arg(
            "table_mapper", CONFIG["table_mapper"])
        self.shuffle = self.get_arg("shuffle", CONFIG["shuffle"])
        self._generate_paths()

        # 根据 encoder_type 选择不同的 index 脚本
        if self.encoder_type == "sdd":
            self._run_sdd_index()
        elif self.encoder_type == "sentence_transformers":
            self._run_st_index()
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

    def _run_sdd_index(self):
        """运行 SDD (BarlowTwinsSimCLR) index.py"""
        self.sdd_model_path = self.get_arg(
            "sdd_model_path", CONFIG["sdd_model_path"])
        self.data_path = self.get_arg("data_path", CONFIG["data_path"])

        if not self.sdd_model_path:
            raise ValueError(
                "sdd_model_path is required for SDD encoder type")

        index_args = [
            "python3", "./task/dataset_discovery/Deepjoin/index.py",
            "--benchmark", self.benchmark,
            "--table_order", self.table_order,
            "--run_id", str(self.run_id),
            "--model_path", self.sdd_model_path,
        ]

        # 添加可选参数
        if self.single_column:
            index_args.append("--single_column")

        if self.data_path:
            index_args.extend(["--data_path", self.data_path])

        # 设置输出路径（需要修改 index.py 支持这些参数，或者在这里硬编码处理）
        index_args.extend(["--output_path", self.table_path])

        if self.sample_rows is not None:
            index_args.extend(["--sample_rows", str(self.sample_rows)])

        if self.noise_prob and self.noise_prob > 0.0:
            index_args.extend(["--noise_prob", str(self.noise_prob)])

        if self.table_mapper:
            index_args.append("--table_mapper")

        if self.shuffle:
            index_args.append("--shuffle")

        # 运行 index.py
        if not self.get_arg("skip_index", False):
            print("[DeepJoinDDTask] Step 1: Running SDD index.py...")
            self._run_cmd(index_args)
            print("[DeepJoinDDTask] SDD index generation completed.")
        else:
            print("[DeepJoinDDTask] Skipping index generation (skip_index=True)")

    def _ensure_sentences_pickle(
        self,
        data_file: Optional[str],
        folder: Optional[str],
        default_pkl_name: str,
        split_num: int = 10,
        profilling_path: Optional[str] = None,
        noise_prob: float = 0.0,
    ) -> Optional[str]:
        """
        确保 sentences pickle 文件存在。
        - 若 data_file 已存在 → 直接返回。
        - 若 data_file 不存在（或未指定）但 folder 存在 → 自动生成。
        返回最终的 sentences pickle 路径，失败返回 None。
        """
        # 当 sample_rows 指定时，在文件名中加入 _rowN 后缀，避免不同 sample_rows 共用缓存
        def _inject_sample_rows(name: str) -> str:
            if self.sample_rows is None:
                return name
            stem, _, ext = name.rpartition('.')
            suffix = f"_row{self.sample_rows}"
            if stem:
                return f"{stem}{suffix}.{ext}"
            return f"{name}{suffix}"

        if data_file:
            data_file = os.path.join(
                os.path.dirname(data_file),
                _inject_sample_rows(os.path.basename(data_file))
            )
        default_pkl_name = _inject_sample_rows(default_pkl_name)

        if data_file and os.path.exists(data_file):
            return data_file

        if not folder or not os.path.isdir(folder):
            return data_file  # 无法生成，原样返回（后续会报错）

        # 确定输出路径
        if data_file:
            filepathstore = os.path.dirname(os.path.abspath(data_file))
            pkl_name = os.path.basename(data_file)
        else:
            filepathstore = os.path.join("./results", "data")
            pkl_name = default_pkl_name

        out_path = os.path.join(filepathstore, pkl_name)
        if os.path.exists(out_path):
            print(
                f"[DeepJoinDDTask] Sentences pickle already exists: {out_path}")
            return out_path

        print(
            f"[DeepJoinDDTask] Sentences pickle not found, auto-generating from: {folder}")
        print(f"[DeepJoinDDTask]   → {out_path}  (split_num={split_num})")

        tmppath = os.path.join(filepathstore, "_tmp_sentences")
        try:
            from process_table_tosentence import process_table_sentense
            process_table_sentense(
                filepathstore=filepathstore,
                datadir=folder,
                data_pkl_name=pkl_name,
                tmppath=tmppath,
                split_num=split_num,
                profilling_path=profilling_path,
                sample_rows=self.sample_rows,
                noise_prob=noise_prob,
            )
            print(f"[DeepJoinDDTask] Sentences pickle generated: {out_path}")
            return out_path
        except Exception as e:
            print(
                f"[DeepJoinDDTask] [ERROR] Failed to generate sentences pickle: {e}")
            raise

    def _run_st_index(self):
        """运行 SentenceTransformer deepjoin_infer.py（含自动预处理）"""
        skip_index = self.get_arg("skip_index", False)
        datalake_folder = self.get_arg("datalake_folder", None)
        query_folder = self.get_arg("query_folder", None)
        split_num = self.get_arg("preprocess_split_num", 10)

        # ── Step 1a: 确保 datalake sentences pickle 存在 ──────────────────
        datalake_data_file = self.get_arg("datalake_data_file", None)
        if not skip_index:
            benchmark_tag = getattr(
                self, "benchmark", "opendata").replace("/", "_")
            datalake_data_file = self._ensure_sentences_pickle(
                data_file=datalake_data_file,
                folder=datalake_folder,
                default_pkl_name=f"deepjoin_{benchmark_tag}_datalake.pkl",
                split_num=split_num,
                profilling_path=self.profilling_path,
                noise_prob=self.noise_prob,
            )

        # ── Step 1b: 确保 query sentences pickle 存在 ───────────────────────
        query_data_file = self.get_arg("query_data_file", None)
        if not skip_index:
            query_data_file = self._ensure_sentences_pickle(
                data_file=query_data_file,
                folder=query_folder,
                default_pkl_name=f"deepjoin_{benchmark_tag}_query.pkl",
                split_num=split_num,
                profilling_path=self.profilling_path,
                noise_prob=self.noise_prob,
            )

        # ── Step 1c: 运行 deepjoin_infer.py 生成 embedding pickle ─────────
        if not skip_index:
            if datalake_data_file and os.path.exists(datalake_data_file):
                # embedding 输出到 results_dir，文件名与 sentences pkl 相同
                self.table_path = os.path.join(
                    self.results_dir, os.path.basename(datalake_data_file))
                print(
                    "[DeepJoinDDTask] Step 1a: Encoding datalake with SentenceTransformer...")
                self._run_cmd([
                    "python3", "./task/dataset_discovery/Deepjoin/deepjoin_infer.py",
                    "--datafile", datalake_data_file,
                    "--storepath", self.results_dir,
                    "--model_name_or_path", self.st_model_path,
                ])
            else:
                print(
                    "[DeepJoinDDTask] WARNING: datalake sentences pickle unavailable, skipping datalake encoding.")

            if query_data_file and os.path.exists(query_data_file):
                self.query_path = os.path.join(
                    self.results_dir, os.path.basename(query_data_file))
                print(
                    "[DeepJoinDDTask] Step 1b: Encoding query with SentenceTransformer...")
                self._run_cmd([
                    "python3", "./task/dataset_discovery/Deepjoin/deepjoin_infer.py",
                    "--datafile", query_data_file,
                    "--storepath", self.results_dir,
                    "--model_name_or_path", self.st_model_path,
                ])
                print("[DeepJoinDDTask] Encoding completed.")
            else:
                print(
                    "[DeepJoinDDTask] WARNING: query sentences pickle unavailable, skipping query encoding.")
        else:
            # skip_index=True: 仍需同步 table_path/query_path 与实际文件
            if datalake_data_file:
                candidate = os.path.join(
                    self.results_dir, os.path.basename(datalake_data_file))
                if os.path.exists(candidate):
                    self.table_path = candidate
            if query_data_file:
                candidate = os.path.join(
                    self.results_dir, os.path.basename(query_data_file))
                if os.path.exists(candidate):
                    self.query_path = candidate
            print("[DeepJoinDDTask] Skipping index generation (skip_index=True)")

    def execute(self):
        """执行阶段：运行 query.py 进行查询"""
        # 获取查询参数 - 需要赋值给 self 以供 _generate_metrics_path 使用
        self.K = self.get_arg("K", CONFIG["K"])
        self.scal = self.get_arg("scal", CONFIG["scal"])
        self.N = self.get_arg("N", CONFIG["N"])
        self.threshold = self.get_arg("threshold", CONFIG["threshold"])
        self.evaluation_mode = self.get_arg(
            "evaluation_mode", CONFIG["evaluation_mode"])
        store_result = self.get_arg("store_result", CONFIG["store_result"])

        # Ground Truth 相关参数
        self.gt_path = self.get_arg("gt_path", CONFIG["gt_path"])
        self.query_folder = self.get_arg(
            "query_folder", CONFIG["query_folder"])
        self.datalake_folder = self.get_arg(
            "datalake_folder", CONFIG["datalake_folder"])
        self.index_path = self.get_arg("index_path", CONFIG["index_path"])

        self.metrics_path = self.get_arg(
            "metrics_path", self._generate_metrics_path())
        print(
            f"[DeepJoinDDTask] Metrics will be saved to: {self.metrics_path}")

        # 输出路径
        output_path = os.path.join(
            self.results_dir, f"deepjoin_{self.encoder_type}_results.csv")

        # 构建 query.py 参数
        query_args = [
            "python3", "./task/dataset_discovery/Deepjoin/query.py",
            "--encoder", "cl",  # DeepJoin query.py 中 cl 代表对比学习编码
            "--benchmark", self.benchmark,
            "--K", str(self.K),
            "--scal", str(self.scal),
            "--N", str(self.N),
            "--threshold", str(self.threshold),
            "--evaluation_mode", self.evaluation_mode,
            "--table_path", self.table_path,
            "--query_path", self.query_path,
            "--metrics_path", self.metrics_path,
        ]

        # 添加可选参数
        if self.single_column:
            query_args.append("--single_column")

        if self.index_path:
            query_args.extend(["--index_path", self.index_path])

        if self.gt_path:
            query_args.extend(["--gt_path", self.gt_path])

        if self.query_folder:
            query_args.extend(["--query_folder", self.query_folder])

        if self.datalake_folder:
            query_args.extend(["--datalake_folder", self.datalake_folder])

        # 是否存储结果
        if store_result:
            query_args.append("--store_result")
            query_args.extend(["--output_path", output_path])

        # 运行 query.py
        if not self.get_arg("skip_query", False):
            print("[DeepJoinDDTask] Step 2: Running query.py to perform search...")
            self._run_cmd(query_args)
            print("[DeepJoinDDTask] Query completed.")
            print(f"[DeepJoinDDTask] Results saved to: {output_path}")
            print(f"[DeepJoinDDTask] Metrics saved to: {self.metrics_path}")
        else:
            print("[DeepJoinDDTask] Skipping query (skip_query=True)")

        return {
            "metrics_path": self.metrics_path,
            "output_path": output_path if store_result else None,
            "table_path": self.table_path,
            "query_path": self.query_path,
        }


if __name__ == "__main__":
    # ============== 示例 1: SDD 编码器 (BarlowTwinsSimCLR) ==============
    print("\n" + "=" * 60)
    print("Example 1: SDD Encoder (BarlowTwinsSimCLR)")
    print("=" * 60)
    task_sdd = DeepJoinDDTask(
        task_name="DeepJoinDDTask_SDD",
        config={},
        encoder_type="sdd",
        benchmark="santos-query",
        sdd_model_path="./model/deepjoin_sdd_drop_cell_alphaHead.pt",
        augment_op="drop_cell",
        sample_method="alphaHead",
        single_column=False,
        K=60,
        N=10,
        threshold=0.7,
        gt_path="./task/dataset_discovery/santos_small/santosJoinBenchmark.csv",
        query_folder="./task/dataset_discovery/santos_small/query",
        datalake_folder="./task/dataset_discovery/santos_small/datalake",
    )
    # result_sdd = task_sdd.run()
    # print(result_sdd.data)

    # ============== 示例 2: SentenceTransformer 编码器 ==============
    print("\n" + "=" * 60)
    print("Example 2: SentenceTransformer Encoder")
    print("=" * 60)
    task_st = DeepJoinDDTask(
        task_name="DeepJoinDDTask_ST",
        config={},
        encoder_type="sentence_transformers",
        benchmark="santos-query",
        st_model_path="./model/all-mpnet-base-v2",
        datalake_data_file="./results/data/deepjoin_santos_datalake.pkl",
        query_data_file="./results/data/deepjoin_santos_query.pkl",
        single_column=False,
        K=60,
        N=10,
        threshold=0.7,
        gt_path="./task/dataset_discovery/santos_small/santosJoinBenchmark.csv",
        query_folder="./task/dataset_discovery/santos_small/query",
        datalake_folder="./task/dataset_discovery/santos_small/datalake",
    )
    # result_st = task_st.run()
    # print(result_st.data)

    # ============== 示例 3: 跳过索引直接查询 ==============
    print("\n" + "=" * 60)
    print("Example 3: Skip Index and Run Query Only")
    print("=" * 60)
    task_query_only = DeepJoinDDTask(
        task_name="DeepJoinDDTask_QueryOnly",
        config={},
        encoder_type="sdd",
        benchmark="santos-query",
        sdd_model_path="./model/deepjoin_sdd_drop_cell_alphaHead.pt",
        table_path="./results/santos-query/deepjoin_santos_datalake.pkl",
        query_path="./results/santos-query/deepjoin_santos_query.pkl",
        skip_index=True,
        K=60,
        N=10,
        threshold=0.7,
        store_result=True,
        gt_path="./task/dataset_discovery/santos_small/santosJoinBenchmark.csv",
        query_folder="./task/dataset_discovery/santos_small/query",
        datalake_folder="./task/dataset_discovery/santos_small/datalake",
    )
    # result_query_only = task_query_only.run()
    # print(result_query_only.data)

"""
python ./task/dataset_discovery/Deepjoin/deepjoin_train.py `
    --dataset opendata `
    --model_name ./model/all-mpnet-base-v2 `
    --model_save_path ./model/deepjoin_SG_model `
    --file_train_path ./datasets/opendata_SG `
    --tain_csv_file ./task/dataset_discovery/Deepjoin/sato_opendata_new.csv `
    --storepath ./results/SG/pretrain_data
参数说明：

参数	值	说明
--dataset	opendata	数据集类型
--model_name	all-mpnet-base-v2	基础预训练模型
--model_save_path	./model/deepjoin_SG_model	模型保存路径
--file_train_path	datasets_SG	训练表格所在文件夹
--tain_csv_file	.../sato_opendata_new.csv	训练数据标注CSV
--storepath	./results/SG/pretrain_data	中间数据存储路径
训练好的模型会保存在 ./model/deepjoin_SG_model 下（实际路径会加时间戳）
训练完成后，将模型路径传给 DeepJoinDDTask 的 sdd_model_path 参数即可


创建/修改的文件:

DeepJoinDDTask.py - 新建的任务封装类，支持：

sdd 编码器 (BarlowTwinsSimCLR 对比学习模型)
sentence_transformers 编码器 (all-mpnet-base-v2 等)
自动生成 metrics 路径
prepare() 阶段运行索引构建
execute() 阶段运行查询和评估
index.py - 添加命令行参数支持：

--model_path: 预训练模型路径
--output_path: 输出路径
--data_path: 自定义数据路径
--results_dir: 结果目录
query.py - 重写以支持命令行参数：

--table_path, --query_path: 嵌入文件路径
--gt_path: Ground Truth 路径
--metrics_path: 指标输出路径
--query_folder, --datalake_folder: 数据文件夹
deepjoin_infer.py - 添加命令行参数支持

"""
