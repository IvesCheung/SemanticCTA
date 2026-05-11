#!/usr/bin/env python3
"""
Join 任务超参数搜索脚本 - Hyperparameter Search for Join Discovery

功能：
1. 遍历超参数组合训练 DeepJoin 模型 (SentenceTransformer)
2. 对每个模型使用 DeepJoinDDTask 进行 Join 发现测试评估
3. 将训练参数、测试结果、模型文件存放到独立的子文件夹
4. 生成汇总报告
5. 支持断点续跑 (checkpoint resume)

使用示例：
    python hyperparam_search_join.py --config hyperparam_search_join_config.yaml
    python hyperparam_search_join.py --quick  # 快速测试模式
    python hyperparam_search_join.py --resume experiments_join/exp_20260310_120000 --config xxx.yaml  # 断点续跑
"""

from task.DeepJoinDDTask import DeepJoinDDTask
import torch
import os
import sys
import json
import argparse
import itertools
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
import csv
import traceback
from HYPERPARAMETERS import (
    get_dataset_paths, get_query_path, get_datalake_path,
    get_ground_truth_path, get_join_ground_truth_path,
)

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class JoinTrainConfig:
    """DeepJoin 训练配置"""
    # SentenceTransformer 基础模型
    model_name: str = "./model/all-mpnet-base-v2"
    # 模型保存路径（前缀，实际路径会自动加时间戳）
    model_save_path: str = "./model/deepjoin_model"
    # 训练表格文件夹
    file_train_path: str = "./datasets/opendata_SG"
    # 训练数据标注 CSV
    train_csv_file: str = "./task/dataset_discovery/Deepjoin/sato_opendata_new.csv"
    # 中间数据存储路径
    storepath: str = "./results/SG/pretrain_data"
    # 数据集类型 (opendata / webtable)
    dataset: str = "opendata"
    # 训练批次大小
    train_batch_size: int = 16
    # 训练轮数
    num_epochs: int = 4
    # GPU 设置
    gpu: int = 0
    # 模式: "train" (训练新模型) 或 "pretrained" (跳过训练，使用已有模型)
    mode: str = "train"
    # 预训练模型路径 (mode="pretrained" 时使用)
    pretrained_model_path: Optional[str] = None
    # region 数据路径映射 {前缀: 数据目录}，例如 {"SG": "./datasets/opendata_SG"}
    # 若提供，将替代 multi_preocess_csv.py 中的硬编码服务器路径
    region_data_paths: Optional[Dict[str, str]] = None


@dataclass
class JoinTestConfig:
    """DeepJoin 测试配置"""
    # 编码器类型: 'sdd' 或 'sentence_transformers'
    encoder_type: str = "sentence_transformers"
    # 基准数据集名称
    benchmark: str = "SG"
    # 数据发现任务类型: "union" 或 "join"
    dataset_discovery_type: str = "join"
    # Ground Truth 路径（留空则根据 benchmark + dataset_discovery_type 自动推断）
    gt_path: str = ""
    # 查询表文件夹（留空则根据 benchmark 自动推断）
    query_folder: str = ""
    # 数据湖文件夹（留空则根据 benchmark 自动推断）
    datalake_folder: str = ""
    # Top-K
    K: int = 60
    # N: 从索引中检索的列数
    N: int = 10
    # 相似度阈值列表 (搜索不同阈值)
    threshold_list: List[float] = field(default_factory=lambda: [0.7])
    # 评估粒度: column 或 table
    evaluation_mode: str = "column"
    # 数据湖规模
    scal: float = 1.0
    # 是否单列模式
    single_column: bool = False
    # batch_size
    batch_size: int = 528
    # 是否跳过索引构建
    skip_index: bool = False
    # 是否存储详细结果
    store_result: bool = False
    # datalake 预处理 pickle 路径 (sentence_transformers 模式用)
    datalake_data_file: Optional[str] = None
    # query 预处理 pickle 路径 (sentence_transformers 模式用)
    query_data_file: Optional[str] = None
    # 自动预处理时的并行进程数 (对应 process_table_sentense 的 split_num)
    preprocess_split_num: int = 10
    # Profilling JSON 路径 (可选)—若提供，将列描述/类型前置到句子中以丰富 join 语义
    profilling_path: Optional[str] = None
    # 读取每张表的行数（None=默认1000行, 0=空表/仅header, N=N行）
    sample_rows: Optional[int] = None
    # 噪声增强概率 (0.0=不加噪声, 0.1=10%概率, 以此类推)
    noise_prob: float = 0.0
    # 表名匿名映射 (True = 将表名替换为匿名ID)
    table_mapper: bool = False
    # 随机打乱文件处理顺序
    shuffle: bool = False


@dataclass
class JoinExperimentResult:
    """Join 实验结果"""
    experiment_id: str
    train_config: Dict[str, Any]
    test_config: Dict[str, Any]
    test_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    status: str = "pending"
    error_message: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    model_path: Optional[str] = None
    experiment_folder: Optional[str] = None


class JoinHyperparameterSearch:
    """Join 任务超参数搜索类"""

    def __init__(
        self,
        output_dir: str = "./experiments_join",
        experiment_name: Optional[str] = None
    ):
        self.output_dir = output_dir
        if experiment_name is None:
            experiment_name = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        self.experiment_root = os.path.join(output_dir, experiment_name)
        os.makedirs(self.experiment_root, exist_ok=True)
        self.results: List[JoinExperimentResult] = []

    def _load_completed_experiments(self) -> set:
        """扫描实验目录，加载已完成的实验结果"""
        completed = set()
        if not os.path.exists(self.experiment_root):
            return completed

        for entry in sorted(os.listdir(self.experiment_root)):
            exp_folder = os.path.join(self.experiment_root, entry)
            if not os.path.isdir(exp_folder):
                continue

            result_json = os.path.join(exp_folder, "experiment_result.json")
            if not os.path.exists(result_json):
                continue

            try:
                with open(result_json, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)

                if result_data.get("status") == "success":
                    completed.add(entry)
                    exp_result = JoinExperimentResult(
                        experiment_id=result_data.get("experiment_id", entry),
                        train_config=result_data.get("train_config", {}),
                        test_config=result_data.get("test_config", {}),
                        test_metrics=result_data.get("test_metrics", {}),
                        status=result_data.get("status", "success"),
                        error_message=result_data.get("error_message"),
                        start_time=result_data.get("start_time"),
                        end_time=result_data.get("end_time"),
                        model_path=result_data.get("model_path"),
                        experiment_folder=result_data.get(
                            "experiment_folder", exp_folder)
                    )
                    self.results.append(exp_result)
                else:
                    print(
                        f"  [INFO] {entry} has status '{result_data.get('status')}', will re-run")
            except (json.JSONDecodeError, KeyError, Exception) as e:
                print(
                    f"  [WARNING] Failed to load result from {result_json}: {e}")
                continue

        return completed

    def generate_search_space(
        self,
        model_name_options: List[str] = None,
        num_epochs_options: List[int] = None,
        train_batch_size_options: List[int] = None,
        threshold_options: List[float] = None,
        K_options: List[int] = None,
        N_options: List[int] = None,
        sample_rows_options: List[Optional[int]] = None,
        base_train_config: Optional[JoinTrainConfig] = None,
        base_test_config: Optional[JoinTestConfig] = None,
        table_mapper_options: List[bool] = None,
        shuffle_options: List[bool] = None,
    ) -> List[tuple]:
        """
        生成超参数搜索空间

        Returns:
            列表 [(JoinTrainConfig, JoinTestConfig), ...]
        """
        if base_train_config is None:
            base_train_config = JoinTrainConfig()
        if base_test_config is None:
            base_test_config = JoinTestConfig()

        if model_name_options is None:
            model_name_options = [base_train_config.model_name]
        if num_epochs_options is None:
            num_epochs_options = [base_train_config.num_epochs]
        if train_batch_size_options is None:
            train_batch_size_options = [base_train_config.train_batch_size]
        if threshold_options is None:
            threshold_options = base_test_config.threshold_list
        if K_options is None:
            K_options = [base_test_config.K]
        if N_options is None:
            N_options = [base_test_config.N]
        if sample_rows_options is None:
            sample_rows_options = [base_test_config.sample_rows]
        if table_mapper_options is None:
            table_mapper_options = [base_test_config.table_mapper]
        if shuffle_options is None:
            shuffle_options = [base_test_config.shuffle]

        configs = []
        for model_name, epochs, batch_size, threshold, K, N, sample_rows in itertools.product(
            model_name_options,
            num_epochs_options,
            train_batch_size_options,
            threshold_options,
            K_options,
            N_options,
            sample_rows_options,
        ):
            for table_mapper, shuffle in itertools.product(table_mapper_options, shuffle_options):
                train_cfg = JoinTrainConfig(
                    model_name=model_name,
                    model_save_path=base_train_config.model_save_path,
                    file_train_path=base_train_config.file_train_path,
                    train_csv_file=base_train_config.train_csv_file,
                    storepath=base_train_config.storepath,
                    dataset=base_train_config.dataset,
                    train_batch_size=batch_size,
                    num_epochs=epochs,
                    gpu=base_train_config.gpu,
                    mode=base_train_config.mode,
                    pretrained_model_path=base_train_config.pretrained_model_path,
                    region_data_paths=base_train_config.region_data_paths,
                )
                test_cfg = JoinTestConfig(
                    encoder_type=base_test_config.encoder_type,
                    benchmark=base_test_config.benchmark,
                    dataset_discovery_type=base_test_config.dataset_discovery_type,
                    gt_path=base_test_config.gt_path,
                    query_folder=base_test_config.query_folder,
                    datalake_folder=base_test_config.datalake_folder,
                    K=K,
                    N=N,
                    threshold_list=[threshold],
                    evaluation_mode=base_test_config.evaluation_mode,
                    scal=base_test_config.scal,
                    single_column=base_test_config.single_column,
                    batch_size=base_test_config.batch_size,
                    skip_index=base_test_config.skip_index,
                    store_result=base_test_config.store_result,
                    datalake_data_file=base_test_config.datalake_data_file,
                    query_data_file=base_test_config.query_data_file,
                    preprocess_split_num=base_test_config.preprocess_split_num,
                    sample_rows=sample_rows,
                    noise_prob=base_test_config.noise_prob,
                    table_mapper=table_mapper,
                    shuffle=shuffle,
                )
                configs.append((train_cfg, test_cfg))

        return configs

    def _create_experiment_folder(self, experiment_id: str) -> str:
        """创建单个实验的文件夹"""
        folder = os.path.join(self.experiment_root, experiment_id)
        os.makedirs(folder, exist_ok=True)
        os.makedirs(os.path.join(folder, "metrics"), exist_ok=True)
        return folder

    def _train_deepjoin_model(
        self,
        train_config: JoinTrainConfig,
        model_save_dir: str,
    ) -> str:
        """
        训练 DeepJoin 模型 (SentenceTransformer)

        Returns:
            训练完成的模型路径
        """
        train_args = [
            "python3", "./task/dataset_discovery/Deepjoin/deepjoin_train.py",
            "--dataset", train_config.dataset,
            "--model_name", train_config.model_name,
            "--model_save_path", model_save_dir,
            "--file_train_path", train_config.file_train_path,
            "--tain_csv_file", train_config.train_csv_file,
            "--storepath", train_config.storepath,
        ]
        if train_config.region_data_paths:
            train_args += ["--region_data_paths",
                           json.dumps(train_config.region_data_paths)]

        print(f"  Running: {' '.join(train_args)}")
        subprocess.run(train_args, check=True)

        # deepjoin_train.py 会在 model_save_dir 下生成带时间戳的目录
        # 找到最新的模型目录
        if os.path.exists(model_save_dir):
            # 如果 model_save_dir 本身就是模型目录
            if os.path.exists(os.path.join(model_save_dir, "config.json")):
                return model_save_dir

        # 查找 output/ 下面最新的 deepjoin 训练目录
        output_dir = "output"
        if os.path.exists(output_dir):
            deepjoin_dirs = [
                d for d in os.listdir(output_dir)
                if d.startswith("deepjoin_") and os.path.isdir(os.path.join(output_dir, d))
            ]
            if deepjoin_dirs:
                deepjoin_dirs.sort(reverse=True)
                return os.path.join(output_dir, deepjoin_dirs[0])

        return model_save_dir

    @staticmethod
    def _resolve_paths(test_config: "JoinTestConfig") -> tuple:
        """
        根据 benchmark + dataset_discovery_type 自动推断 gt_path / query_folder / datalake_folder。
        显式指定的值（非空）优先级最高。

        Returns:
            (gt_path, query_folder, datalake_folder)
        """
        benchmark = test_config.benchmark
        dd_type = test_config.dataset_discovery_type

        # gt_path
        gt_path = test_config.gt_path
        if not gt_path:
            if dd_type == "join":
                gt_path = get_join_ground_truth_path(benchmark)
            else:
                gt_path = get_ground_truth_path(benchmark)

        # query / datalake
        query_folder = test_config.query_folder or get_query_path(benchmark)
        datalake_folder = test_config.datalake_folder or get_datalake_path(benchmark)

        return gt_path, query_folder, datalake_folder

    def _test_model(
        self,
        model_path: str,
        test_config: JoinTestConfig,
        metrics_folder: str,
        experiment_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        使用 DeepJoinDDTask 测试模型

        Returns:
            字典 {threshold: metrics}
        """
        results = {}

        # 自动推断路径
        gt_path, query_folder, datalake_folder = self._resolve_paths(test_config)

        for threshold in test_config.threshold_list:
            metrics_path = os.path.join(
                metrics_folder,
                f"metrics_th{str(threshold).replace('.', '')}.csv"
            )

            try:
                task_kwargs = dict(
                    task_name=f"{experiment_id}_th{threshold}",
                    config={},
                    encoder_type=test_config.encoder_type,
                    benchmark=test_config.benchmark,
                    K=test_config.K,
                    N=test_config.N,
                    threshold=threshold,
                    scal=test_config.scal,
                    single_column=test_config.single_column,
                    batch_size=test_config.batch_size,
                    gt_path=gt_path,
                    query_folder=query_folder,
                    datalake_folder=datalake_folder,
                    metrics_path=metrics_path,
                    store_result=test_config.store_result,
                    skip_query=False,
                )

                if test_config.encoder_type == "sentence_transformers":
                    task_kwargs["st_model_path"] = model_path
                    if test_config.datalake_data_file:
                        task_kwargs["datalake_data_file"] = test_config.datalake_data_file
                    if test_config.query_data_file:
                        task_kwargs["query_data_file"] = test_config.query_data_file
                    task_kwargs["preprocess_split_num"] = test_config.preprocess_split_num
                    if test_config.profilling_path:
                        task_kwargs["profilling_path"] = test_config.profilling_path
                    task_kwargs["evaluation_mode"] = test_config.evaluation_mode
                    if test_config.sample_rows is not None:
                        task_kwargs["sample_rows"] = test_config.sample_rows
                elif test_config.encoder_type == "sdd":
                    task_kwargs["sdd_model_path"] = model_path
                    if test_config.sample_rows is not None:
                        task_kwargs["sample_rows"] = test_config.sample_rows

                # noise_prob / table_mapper / shuffle 对两种 encoder 都适用
                if test_config.noise_prob > 0.0:
                    task_kwargs["noise_prob"] = test_config.noise_prob
                if test_config.table_mapper:
                    task_kwargs["table_mapper"] = True
                if test_config.shuffle:
                    task_kwargs["shuffle"] = True

                # 第一个 threshold 才需要建索引，后续可以跳过
                if test_config.skip_index or threshold != test_config.threshold_list[0]:
                    task_kwargs["skip_index"] = True

                task = DeepJoinDDTask(**task_kwargs)
                task_result = task.run()

                if not task_result.success:
                    error_message = str(
                        task_result.error) if task_result.error else "task failed"
                    results[f"th{threshold}"] = {
                        "error": error_message,
                        "task_success": False,
                        "elapsed_time": task_result.elapsed
                    }
                    print(
                        f"  [WARNING] Test failed for threshold={threshold}: {error_message}")
                    continue

                metrics = self._parse_metrics_csv(metrics_path)
                metrics['task_success'] = task_result.success
                metrics['elapsed_time'] = task_result.elapsed

                results[f"th{threshold}"] = metrics

            except Exception as e:
                results[f"th{threshold}"] = {
                    "error": str(e),
                    "task_success": False
                }
                print(
                    f"  [WARNING] Test failed for threshold={threshold}: {e}")

        return results

    def _parse_metrics_csv(self, csv_path: str) -> Dict[str, Any]:
        """解析 metrics CSV 文件，并计算 MAP"""
        if not os.path.exists(csv_path):
            return {"error": "metrics file not found"}

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    precisions = []
                    max_k_row = None
                    max_k = 0

                    for row in rows:
                        try:
                            k = int(row.get('k', 0))
                            precision = float(row.get('precision', 0))
                            precisions.append(precision)
                            if k > max_k:
                                max_k = k
                                max_k_row = row
                        except (ValueError, TypeError):
                            continue

                    map_value = sum(precisions) / \
                        len(precisions) if precisions else 0.0

                    metrics = {"MAP": map_value}
                    if max_k_row:
                        for key, value in max_k_row.items():
                            try:
                                metrics[key] = float(value)
                            except (ValueError, TypeError):
                                metrics[key] = value

                    return metrics
            return {"error": "empty metrics file"}
        except Exception as e:
            return {"error": f"parse error: {str(e)}"}

    def run_single_experiment(
        self,
        train_config: JoinTrainConfig,
        test_config: JoinTestConfig,
        experiment_id: Optional[str] = None,
    ) -> JoinExperimentResult:
        """运行单个实验"""
        if experiment_id is None:
            experiment_id = f"exp_{len(self.results):04d}"

        exp_folder = self._create_experiment_folder(experiment_id)

        result = JoinExperimentResult(
            experiment_id=experiment_id,
            train_config=asdict(train_config),
            test_config=asdict(test_config),
            experiment_folder=exp_folder,
            start_time=datetime.now().isoformat()
        )

        is_pretrained = train_config.mode == "pretrained"

        print(f"\n{'='*60}")
        print(f"[Experiment {experiment_id}]")
        print(f"  mode: {train_config.mode}")
        print(f"  encoder_type: {test_config.encoder_type}")
        print(f"  dataset_discovery_type: {test_config.dataset_discovery_type}")
        if not is_pretrained:
            print(f"  model_name: {train_config.model_name}")
            print(f"  num_epochs: {train_config.num_epochs}")
            print(f"  train_batch_size: {train_config.train_batch_size}")
        else:
            print(f"  pretrained_model: {train_config.pretrained_model_path}")
        print(f"  K: {test_config.K}, N: {test_config.N}")
        print(f"  thresholds: {test_config.threshold_list}")
        print(f"{'='*60}")

        try:
            # Phase 1: 训练
            if is_pretrained:
                print(f"\n[Phase 1] Skipping training (pretrained mode)")
                model_path = train_config.pretrained_model_path or train_config.model_name
                result.model_path = model_path
            else:
                print(f"\n[Phase 1] Training DeepJoin model...")
                model_save_dir = os.path.join(exp_folder, "model")
                model_path = self._train_deepjoin_model(
                    train_config, model_save_dir)
                result.model_path = model_path
                print(f"  Training completed. Model at: {model_path}")

            # Phase 2: 测试
            print(f"\n[Phase 2] Testing model...")
            metrics_folder = os.path.join(exp_folder, "metrics")

            test_metrics = self._test_model(
                model_path=model_path,
                test_config=test_config,
                metrics_folder=metrics_folder,
                experiment_id=experiment_id,
            )

            result.test_metrics = test_metrics
            result.status = "success"

            # 打印摘要
            print(f"\n[Test Results Summary]")
            for th_key, metrics in test_metrics.items():
                if "error" not in metrics:
                    map_val = metrics.get("MAP", "N/A")
                    if isinstance(map_val, float):
                        print(f"  {th_key}: MAP={map_val:.4f}")
                    else:
                        print(f"  {th_key}: MAP={map_val}")
                else:
                    print(f"  {th_key}: ERROR - {metrics.get('error')}")

        except Exception as e:
            result.status = "failed"
            result.error_message = f"{str(e)}\n{traceback.format_exc()}"
            print(f"\n[ERROR] Experiment failed: {e}")

        result.end_time = datetime.now().isoformat()

        # 保存实验结果
        result_json_path = os.path.join(exp_folder, "experiment_result.json")
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)

        self.results.append(result)
        return result

    def run_search(
        self,
        configs: List[tuple],
        skip_on_error: bool = True,
        resume: bool = False,
    ) -> List[JoinExperimentResult]:
        """
        运行完整的超参数搜索

        Args:
            configs: [(JoinTrainConfig, JoinTestConfig), ...] 列表
            skip_on_error: 出错时是否跳过继续
            resume: 是否断点续跑
        """
        completed_ids = set()
        if resume:
            completed_ids = self._load_completed_experiments()
            print(
                f"\n[RESUME MODE] Found {len(completed_ids)} completed experiments, will skip them.")
            if completed_ids:
                print(f"  Completed: {sorted(completed_ids)}")

        remaining = len(configs) - len(completed_ids)

        print(f"\n{'#'*60}")
        print(f"# Join Hyperparameter Search: {self.experiment_name}")
        print(f"# Total experiments: {len(configs)}")
        if resume:
            print(f"# Already completed: {len(completed_ids)}")
            print(f"# Remaining to run:  {remaining}")
        print(f"# Output directory: {self.experiment_root}")
        print(f"{'#'*60}\n")

        skipped = 0
        for i, (train_config, test_config) in enumerate(configs):
            experiment_id = f"exp_{i:04d}"

            if experiment_id in completed_ids:
                skipped += 1
                print(
                    f"\n[SKIP] {experiment_id} - already completed (checkpoint)")
                continue

            try:
                self.run_single_experiment(
                    train_config=train_config,
                    test_config=test_config,
                    experiment_id=experiment_id,
                )
            except Exception as e:
                print(
                    f"\n[CRITICAL ERROR] Experiment {experiment_id} failed: {e}")
                if not skip_on_error:
                    raise

        self._generate_summary_report()

        if resume and skipped > 0:
            print(
                f"\n[RESUME SUMMARY] Skipped {skipped} completed, ran {len(configs) - skipped} new.")

        return self.results

    def _generate_summary_report(self):
        """生成汇总报告"""
        summary_path = os.path.join(self.experiment_root, "summary.json")

        summary = {
            "experiment_name": self.experiment_name,
            "total_experiments": len(self.results),
            "successful": sum(1 for r in self.results if r.status == "success"),
            "failed": sum(1 for r in self.results if r.status == "failed"),
            "generated_at": datetime.now().isoformat(),
            "experiments": []
        }

        for result in self.results:
            map_values = {}
            for th_key, metrics in result.test_metrics.items():
                if isinstance(metrics, dict) and "MAP" in metrics:
                    map_values[th_key] = metrics["MAP"]

            exp_summary = {
                "experiment_id": result.experiment_id,
                "status": result.status,
                "model_name": result.train_config.get("model_name"),
                "num_epochs": result.train_config.get("num_epochs"),
                "train_batch_size": result.train_config.get("train_batch_size"),
                "mode": result.train_config.get("mode"),
                "model_path": result.model_path,
                "MAP": map_values,
                "test_metrics": result.test_metrics,
                "error": result.error_message,
            }
            summary["experiments"].append(exp_summary)

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        csv_path = os.path.join(self.experiment_root, "summary.csv")
        self._generate_summary_csv(csv_path)

        print(f"\n{'='*60}")
        print(f"[Summary Report Generated]")
        print(f"  JSON: {summary_path}")
        print(f"  CSV:  {csv_path}")
        print(
            f"  Successful: {summary['successful']}/{summary['total_experiments']}")
        print(f"{'='*60}\n")

    def _generate_summary_csv(self, csv_path: str):
        """生成 CSV 格式的汇总"""
        if not self.results:
            return

        all_metrics_keys = set()
        for result in self.results:
            for th_key, metrics in result.test_metrics.items():
                if isinstance(metrics, dict):
                    all_metrics_keys.update(metrics.keys())

        fieldnames = ["experiment_id", "status"]

        # 先添加每个 threshold 的 MAP 列
        test_th_keys = sorted(set(
            th_key for r in self.results for th_key in r.test_metrics.keys()
        ))
        for th_key in test_th_keys:
            fieldnames.append(f"{th_key}_MAP")

        fieldnames.extend([
            "model_name", "num_epochs", "train_batch_size", "mode",
            "model_path",
        ])

        # 添加其他 metrics 列
        for th_key in test_th_keys:
            for metric_key in sorted(all_metrics_keys):
                if metric_key not in ["error", "task_success", "MAP"]:
                    fieldnames.append(f"{th_key}_{metric_key}")

        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.results:
                row = {
                    "experiment_id": result.experiment_id,
                    "status": result.status,
                    "model_name": result.train_config.get("model_name"),
                    "num_epochs": result.train_config.get("num_epochs"),
                    "train_batch_size": result.train_config.get("train_batch_size"),
                    "mode": result.train_config.get("mode"),
                    "model_path": result.model_path,
                }

                for th_key in test_th_keys:
                    metrics = result.test_metrics.get(th_key, {})
                    if isinstance(metrics, dict):
                        for metric_key in all_metrics_keys:
                            if metric_key not in ["error", "task_success"]:
                                col_name = f"{th_key}_{metric_key}"
                                if col_name in fieldnames:
                                    row[col_name] = metrics.get(
                                        metric_key, "")

                writer.writerow(row)


def build_quick_test_space() -> List[tuple]:
    """构建快速测试用的小规模搜索空间"""
    base_train = JoinTrainConfig(
        model_name="./model/all-mpnet-base-v2",
        mode="pretrained",
        pretrained_model_path="./model/all-mpnet-base-v2",
    )
    base_test = JoinTestConfig(
        encoder_type="sentence_transformers",
        benchmark="SG",
        dataset_discovery_type="join",
        K=10,
        N=10,
        threshold_list=[0.7],
    )

    searcher = JoinHyperparameterSearch(output_dir="./experiments_join")
    configs = searcher.generate_search_space(
        threshold_options=[0.6, 0.7],
        base_train_config=base_train,
        base_test_config=base_test,
    )

    return configs


def load_config_from_yaml(yaml_path: str) -> List[tuple]:
    """
    从 YAML 配置文件加载搜索空间

    Returns:
        [(JoinTrainConfig, JoinTestConfig), ...]
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required. Install it with: pip install pyyaml"
        )

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 解析训练配置
    train_section = config.get('train', {})
    base_train_config = JoinTrainConfig(
        model_name=train_section.get(
            'model_name', "./model/all-mpnet-base-v2"),
        model_save_path=train_section.get(
            'model_save_path', "./model/deepjoin_model"),
        file_train_path=train_section.get(
            'file_train_path', "./datasets/opendata_SG"),
        train_csv_file=train_section.get(
            'train_csv_file',
            "./task/dataset_discovery/Deepjoin/sato_opendata_new.csv"),
        storepath=train_section.get('storepath', "./results/SG/pretrain_data"),
        dataset=train_section.get('dataset', "opendata"),
        train_batch_size=train_section.get('train_batch_size', 16),
        num_epochs=train_section.get('num_epochs', 4),
        gpu=train_section.get('gpu', 0),
        mode=train_section.get('mode', 'train'),
        pretrained_model_path=train_section.get('pretrained_model_path'),
        region_data_paths=train_section.get('region_data_paths'),
    )

    # 解析测试配置
    test_section = config.get('test', {})
    base_test_config = JoinTestConfig(
        encoder_type=test_section.get('encoder_type', "sentence_transformers"),
        benchmark=test_section.get('benchmark', "SG"),
        dataset_discovery_type=test_section.get('dataset_discovery_type', 'join'),
        gt_path=test_section.get('gt_path', ""),
        query_folder=test_section.get('query_folder', ""),
        datalake_folder=test_section.get('datalake_folder', ""),
        K=test_section.get('K', 60),
        N=test_section.get('N', 10),
        threshold_list=test_section.get('threshold_list', [0.7]),
        evaluation_mode=test_section.get('evaluation_mode', 'table'),
        scal=test_section.get('scal', 1.0),
        single_column=test_section.get('single_column', False),
        batch_size=test_section.get('batch_size', 528),
        skip_index=test_section.get('skip_index', False),
        store_result=test_section.get('store_result', False),
        datalake_data_file=test_section.get('datalake_data_file'),
        query_data_file=test_section.get('query_data_file'),
        preprocess_split_num=test_section.get('preprocess_split_num', 10),
        profilling_path=test_section.get('profilling_path'),
        sample_rows=test_section.get('sample_rows'),
        noise_prob=test_section.get('noise_prob', 0.0),
        table_mapper=test_section.get('table_mapper', False),
        shuffle=test_section.get('shuffle', False),
    )

    # 解析搜索空间
    search_section = config.get('search_space', {})
    model_name_options = search_section.get('model_name')
    num_epochs_options = search_section.get('num_epochs')
    train_batch_size_options = search_section.get('train_batch_size')
    threshold_options = search_section.get('threshold')
    K_options = search_section.get('K')
    N_options = search_section.get('N')
    sample_rows_options = search_section.get('sample_rows')
    table_mapper_options = search_section.get('table_mapper')
    shuffle_options = search_section.get('shuffle')

    # 如果搜索空间中指定了 pretrained_model_paths，则使用 pretrained 模式
    pretrained_paths = search_section.get('pretrained_model_paths')
    if pretrained_paths:
        # 每个 pretrained_model_path 生成一组实验
        configs = []
        if threshold_options is None:
            threshold_options = base_test_config.threshold_list
        if K_options is None:
            K_options = [base_test_config.K]
        if N_options is None:
            N_options = [base_test_config.N]
        if sample_rows_options is None:
            sample_rows_options = [base_test_config.sample_rows]

        for model_path, threshold, K, N, sample_rows in itertools.product(
            pretrained_paths, threshold_options, K_options, N_options, sample_rows_options,
        ):
            train_cfg = JoinTrainConfig(
                model_name=model_path,
                mode="pretrained",
                pretrained_model_path=model_path,
                gpu=base_train_config.gpu,
            )
            test_cfg = JoinTestConfig(
                encoder_type=base_test_config.encoder_type,
                benchmark=base_test_config.benchmark,
                dataset_discovery_type=base_test_config.dataset_discovery_type,
                gt_path=base_test_config.gt_path,
                query_folder=base_test_config.query_folder,
                datalake_folder=base_test_config.datalake_folder,
                K=K,
                N=N,
                threshold_list=[threshold],
                evaluation_mode=base_test_config.evaluation_mode,
                scal=base_test_config.scal,
                single_column=base_test_config.single_column,
                batch_size=base_test_config.batch_size,
                skip_index=base_test_config.skip_index,
                store_result=base_test_config.store_result,
                datalake_data_file=base_test_config.datalake_data_file,
                query_data_file=base_test_config.query_data_file,
                preprocess_split_num=base_test_config.preprocess_split_num,
                profilling_path=base_test_config.profilling_path,
                sample_rows=sample_rows,
                table_mapper=base_test_config.table_mapper,
                shuffle=base_test_config.shuffle,
            )
            configs.append((train_cfg, test_cfg))
        return configs

    # 否则使用训练模式的搜索空间
    searcher = JoinHyperparameterSearch(output_dir="./experiments_join")
    configs = searcher.generate_search_space(
        model_name_options=model_name_options,
        num_epochs_options=num_epochs_options,
        train_batch_size_options=train_batch_size_options,
        threshold_options=threshold_options,
        K_options=K_options,
        N_options=N_options,
        sample_rows_options=sample_rows_options,
        table_mapper_options=table_mapper_options,
        shuffle_options=shuffle_options,
        base_train_config=base_train_config,
        base_test_config=base_test_config,
    )

    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter Search for Join Discovery (DeepJoin)"
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Run quick test with minimal search space'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./experiments_join',
        help='Output directory for experiments'
    )
    parser.add_argument(
        '--experiment_name', type=str, default=None,
        help='Name for this experiment run'
    )
    parser.add_argument(
        '--gpu', type=int, default=0,
        help='GPU index to use'
    )
    parser.add_argument(
        '--skip_on_error', action='store_true', default=True,
        help='Continue to next experiment if one fails'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Resume from an existing experiment directory '
             '(e.g., ./experiments_join/exp_20260310_120000). '
             'Automatically skips completed experiments.'
    )

    args = parser.parse_args()

    # 加载搜索空间
    if args.config:
        print(f"Loading configuration from: {args.config}")
        configs = load_config_from_yaml(args.config)
    elif args.quick:
        print("Using quick test search space (minimal)")
        configs = build_quick_test_space()
    else:
        print("No config specified. Use --config or --quick.")
        parser.print_help()
        sys.exit(1)

    # 更新 GPU 设置
    for train_config, _ in configs:
        train_config.gpu = args.gpu

    print(f"Total configurations to test: {len(configs)}")

    # 处理断点续跑
    resume_mode = False
    output_dir = args.output_dir
    experiment_name = args.experiment_name

    if args.resume:
        resume_mode = True
        resume_path = os.path.normpath(args.resume)
        experiment_name = os.path.basename(resume_path)
        output_dir = os.path.dirname(resume_path)
        if not output_dir:
            output_dir = '.'

        if not os.path.exists(resume_path):
            print(f"[ERROR] Resume directory not found: {resume_path}")
            sys.exit(1)

        print(f"\n[RESUME] Resuming from: {resume_path}")

    # 创建搜索实例并运行
    searcher = JoinHyperparameterSearch(
        output_dir=output_dir,
        experiment_name=experiment_name,
    )

    results = searcher.run_search(
        configs=configs,
        skip_on_error=args.skip_on_error,
        resume=resume_mode,
    )

    # 打印最终总结
    print("\n" + "=" * 60)
    print("JOIN HYPERPARAMETER SEARCH COMPLETED")
    print("=" * 60)
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r.status == 'success')}")
    print(f"Failed: {sum(1 for r in results if r.status == 'failed')}")
    print(f"Results saved to: {searcher.experiment_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
