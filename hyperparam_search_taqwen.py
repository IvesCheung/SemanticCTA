#!/usr/bin/env python3
"""
超参数搜索脚本 - Hyperparameter Search Script

功能：
1. 遍历超参数组合训练对比学习模型
2. 对每个模型进行测试评估
3. 将训练参数、测试结果、模型文件存放到独立的子文件夹
4. 生成汇总报告
5. 支持断点续跑 (checkpoint resume)

使用示例：
    python hyperparam_search.py --config hyperparam_config.yaml
    python hyperparam_search.py --quick  # 快速测试模式
    python hyperparam_search.py --resume experiments/exp_20260303_070958 --config hyperparam_config.yaml  # 断点续跑
"""

from task.TableDDTask import TableIndexQueryTask
import torch
import os
import sys
import json
import argparse
import itertools
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
import csv
import traceback
from HYPERPARAMETERS import get_join_ground_truth_path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class TrainConfig:
    """训练配置"""
    model_path: str = "./model/qwen3-0.6B-embedding"
    csv_dir: str = "./datasets/opendata_SG"
    anchor_profilling: str = "./output/SG/datasets_SG_single_column_and_table_profilling.json"
    positive_profillings: List[str] = field(default_factory=list)
    batch_size: int = 4
    epoch_num: int = 3
    sample_rows: int = 10
    temperature: float = 0.07
    use_aug: bool = False
    max_seq_length: int = 512
    sampling_ratio: float = 1.0
    sampling_seed: int = 42
    use_amp: bool = True
    gradient_checkpointing: bool = True
    gpu: int = 0
    mode: str = "contrastive"  # "contrastive" 或 "api"（跳过对比学习训练，直接用 API embedding 测试）


@dataclass
class TestConfig:
    """测试配置"""
    benchmark: str = "SG"
    profilling_path: Optional[str] = None
    sample_rows_list: List[int] = field(default_factory=lambda: [0, 1, 5, 10])
    K: int = 10
    threshold: float = 0.7
    table_mapper: bool = True
    shuffle: bool = True
    mask_header: Optional[str] = None
    shuffle_columns: bool = False
    # 数据发现任务类型: "union" 或 "join"
    dataset_discovery_type: str = "union"
    # 自定义 Ground Truth 路径（覆盖自动推断，优先级最高）
    gt_path: Optional[str] = None
    # 噪声增强概率 (0.0=不加噪声, 0.1=10%概率, 以此类推)
    noise_prob: float = 0.0


@dataclass
class ExperimentResult:
    """实验结果"""
    experiment_id: str
    train_config: Dict[str, Any]
    test_config: Dict[str, Any]
    train_history: List[float] = field(default_factory=list)
    val_history: List[float] = field(default_factory=list)
    test_metrics: Dict[str, Dict[str, Any]] = field(
        default_factory=dict)  # key: sample_rows
    best_val_loss: Optional[float] = None
    status: str = "pending"
    error_message: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    model_path: Optional[str] = None
    experiment_folder: Optional[str] = None


class HyperparameterSearch:
    """超参数搜索类"""

    def __init__(
        self,
        output_dir: str = "./experiments",
        experiment_name: Optional[str] = None
    ):
        """
        初始化超参数搜索

        Args:
            output_dir: 实验输出目录
            experiment_name: 实验名称，默认使用时间戳
        """
        self.output_dir = output_dir

        # 生成实验名称
        if experiment_name is None:
            experiment_name = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name

        # 创建主实验目录
        self.experiment_root = os.path.join(output_dir, experiment_name)
        os.makedirs(self.experiment_root, exist_ok=True)

        # 实验结果列表
        self.results: List[ExperimentResult] = []

        # 设备
        self.device = self._get_device()

    def _get_device(self, gpu: int = 0) -> torch.device:
        """获取训练设备"""
        if torch.cuda.is_available():
            if gpu < 0 or gpu >= torch.cuda.device_count():
                gpu = 0
            return torch.device(f'cuda:{gpu}')
        return torch.device('cpu')

    def _load_completed_experiments(self) -> set:
        """
        扫描实验目录，加载已完成的实验结果

        读取每个子实验文件夹中的 experiment_result.json，
        如果 status 为 "success"，则认为该实验已完成，跳过重跑。

        Returns:
            已完成实验ID的集合 (如 {"exp_0000", "exp_0001"})
        """
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
                    # 将已完成的结果加载到 self.results，用于最终汇总报告
                    exp_result = ExperimentResult(
                        experiment_id=result_data.get("experiment_id", entry),
                        train_config=result_data.get("train_config", {}),
                        test_config=result_data.get("test_config", {}),
                        train_history=result_data.get("train_history", []),
                        val_history=result_data.get("val_history", []),
                        test_metrics=result_data.get("test_metrics", {}),
                        best_val_loss=result_data.get("best_val_loss"),
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
        positive_profillings_options: List[List[str]],
        temperature_options: List[float] = [0.07],
        sample_rows_options: List[int] = [10],
        epoch_num_options: List[int] = [3],
        batch_size_options: List[int] = [4],
        use_aug_options: List[bool] = None,
        max_seq_length_options: List[int] = None,
        sampling_ratio_options: List[float] = None,
        use_amp_options: List[bool] = None,
        gradient_checkpointing_options: List[bool] = None,
        base_train_config: Optional[TrainConfig] = None
    ) -> List[TrainConfig]:
        """
        生成超参数搜索空间

        Args:
            positive_profillings_options: positive_profillings 的所有可能组合
            temperature_options: temperature 的可选值
            sample_rows_options: sample_rows 的可选值
            epoch_num_options: epoch_num 的可选值
            batch_size_options: batch_size 的可选值
            use_aug_options: use_aug 的可选值 (默认使用 base_train_config 中的值)
            max_seq_length_options: max_seq_length 的可选值
            sampling_ratio_options: sampling_ratio 的可选值
            use_amp_options: use_amp 的可选值
            gradient_checkpointing_options: gradient_checkpointing 的可选值
            base_train_config: 基础训练配置

        Returns:
            所有训练配置的列表
        """
        if base_train_config is None:
            base_train_config = TrainConfig()

        # 如果没有指定 use_aug_options，使用 base_train_config 中的单一值
        if use_aug_options is None:
            use_aug_options = [base_train_config.use_aug]
        if max_seq_length_options is None:
            max_seq_length_options = [base_train_config.max_seq_length]
        if sampling_ratio_options is None:
            sampling_ratio_options = [base_train_config.sampling_ratio]
        if use_amp_options is None:
            use_amp_options = [base_train_config.use_amp]
        if gradient_checkpointing_options is None:
            gradient_checkpointing_options = [
                base_train_config.gradient_checkpointing]

        configs = []

        # 生成所有可能的组合
        for pp, temp, sr, en, bs, aug, max_seq_length, sampling_ratio, use_amp, gradient_checkpointing in itertools.product(
            positive_profillings_options,
            temperature_options,
            sample_rows_options,
            epoch_num_options,
            batch_size_options,
            use_aug_options,
            max_seq_length_options,
            sampling_ratio_options,
            use_amp_options,
            gradient_checkpointing_options,
        ):
            config = TrainConfig(
                model_path=base_train_config.model_path,
                csv_dir=base_train_config.csv_dir,
                anchor_profilling=base_train_config.anchor_profilling,
                positive_profillings=pp,
                batch_size=bs,
                epoch_num=en,
                sample_rows=sr,
                temperature=temp,
                use_aug=aug,
                max_seq_length=max_seq_length,
                sampling_ratio=sampling_ratio,
                sampling_seed=base_train_config.sampling_seed,
                use_amp=use_amp,
                gradient_checkpointing=gradient_checkpointing,
                gpu=base_train_config.gpu,
                mode=base_train_config.mode,
            )
            configs.append(config)

        return configs

    def _create_experiment_folder(self, experiment_id: str) -> str:
        """创建单个实验的文件夹"""
        folder = os.path.join(self.experiment_root, experiment_id)
        os.makedirs(folder, exist_ok=True)
        os.makedirs(os.path.join(folder, "metrics"), exist_ok=True)
        return folder

    def _train_model(
        self,
        train_config: TrainConfig,
        save_path: str
    ) -> tuple:
        """
        训练模型

        Returns:
            (train_history, val_history, best_val_loss)
        """
        # 动态导入，避免循环导入问题
        from taqwen.contrastive_learning_profilling import train_column_contrastive_model

        train_history, val_history = train_column_contrastive_model(
            model_path=train_config.model_path,
            csv_dir=train_config.csv_dir,
            anchor_profilling=train_config.anchor_profilling,
            positive_profillings=train_config.positive_profillings,
            batch_size=train_config.batch_size,
            save_path=save_path,
            epoch_num=train_config.epoch_num,
            sample_rows=train_config.sample_rows,
            temperature=train_config.temperature,
            device=self._get_device(train_config.gpu),
            use_aug=train_config.use_aug,
            max_seq_length=train_config.max_seq_length,
            sampling_ratio=train_config.sampling_ratio,
            sampling_seed=train_config.sampling_seed,
            use_amp=train_config.use_amp,
            gradient_checkpointing=train_config.gradient_checkpointing,
            print_length_stats=False
        )

        best_val_loss = min(val_history) if val_history else None

        return train_history, val_history, best_val_loss

    def _test_model(
        self,
        model_path: str,
        base_model_path: str,
        test_config: TestConfig,
        metrics_folder: str,
        experiment_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        测试模型

        Returns:
            字典 {sample_rows: metrics}
        """
        results = {}

        for sample_rows in test_config.sample_rows_list:
            metrics_path = os.path.join(
                metrics_folder,
                f"metrics_sr{sample_rows}.csv"
            )

            try:
                # 根据 dataset_discovery_type 确定 gt_path
                gt_path = test_config.gt_path  # 优先级最高: 显式指定
                if not gt_path and test_config.dataset_discovery_type == "join":
                    gt_path = get_join_ground_truth_path(test_config.benchmark)
                # union 时 gt_path=None，由 query.py 内部走 get_ground_truth_path(benchmark)

                # 使用 TableIndexQueryTask 进行测试
                task = TableIndexQueryTask(
                    task_name=f"{experiment_id}_sr{sample_rows}",
                    config={},
                    encoder_type="qwen",
                    benchmark=test_config.benchmark,
                    profilling_path=test_config.profilling_path,
                    sample_rows=sample_rows,
                    shuffle=test_config.shuffle,
                    table_mapper=test_config.table_mapper,
                    mask_header=test_config.mask_header,
                    shuffle_columns=test_config.shuffle_columns,
                    noise_prob=test_config.noise_prob,
                    local_model_path=model_path,
                    base_model_path=base_model_path,
                    metrics_path=metrics_path,
                    K=test_config.K,
                    threshold=test_config.threshold,
                    gt_path=gt_path,
                )

                task_result = task.run()

                # 读取 metrics CSV 获取结果
                metrics = self._parse_metrics_csv(metrics_path)
                metrics['task_success'] = task_result.success
                metrics['elapsed_time'] = task_result.elapsed

                results[f"sr{sample_rows}"] = metrics

            except Exception as e:
                results[f"sr{sample_rows}"] = {
                    "error": str(e),
                    "task_success": False
                }
                print(
                    f"  [WARNING] Test failed for sample_rows={sample_rows}: {e}")

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
                    # 收集所有 precision 值来计算 MAP
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

                    # 计算 MAP (Mean Average Precision)
                    map_value = sum(precisions) / \
                        len(precisions) if precisions else 0.0

                    # 返回结果，包含 MAP 和最大 K 值的 metrics
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
        train_config: TrainConfig,
        test_config: TestConfig,
        experiment_id: Optional[str] = None
    ) -> ExperimentResult:
        """
        运行单个实验

        Args:
            train_config: 训练配置
            test_config: 测试配置
            experiment_id: 实验ID，默认自动生成

        Returns:
            实验结果
        """
        # 生成实验ID
        if experiment_id is None:
            experiment_id = f"exp_{len(self.results):04d}"

        # 创建实验文件夹
        exp_folder = self._create_experiment_folder(experiment_id)

        # 初始化结果
        result = ExperimentResult(
            experiment_id=experiment_id,
            train_config=asdict(train_config),
            test_config=asdict(test_config),
            experiment_folder=exp_folder,
            start_time=datetime.now().isoformat()
        )

        # 模型保存路径
        model_save_path = os.path.join(exp_folder, "model.pth")

        is_api_mode = getattr(train_config, 'mode', 'contrastive') == 'api'

        print(f"\n{'='*60}")
        print(f"[Experiment {experiment_id}]")
        print(f"  mode: {train_config.mode}")
        print(f"  dataset_discovery_type: {test_config.dataset_discovery_type}")
        if not is_api_mode:
            print(
                f"  positive_profillings: {train_config.positive_profillings}")
            print(f"  temperature: {train_config.temperature}")
            print(f"  sample_rows: {train_config.sample_rows}")
            print(f"  epoch_num: {train_config.epoch_num}")
            print(f"  batch_size: {train_config.batch_size}")
            print(f"  max_seq_length: {train_config.max_seq_length}")
            print(f"  use_amp: {train_config.use_amp}")
            print(
                f"  gradient_checkpointing: {train_config.gradient_checkpointing}")
        print(f"{'='*60}")

        try:
            if is_api_mode:
                # API 模式：跳过训练，直接使用基础模型 embedding
                print(f"\n[Phase 1] Skipping training (API mode)")
                result.model_path = None

                # API 模式下，用 positive_profillings 中的文件作为测试的 profilling_path
                if train_config.positive_profillings:
                    api_test_config = TestConfig(
                        benchmark=test_config.benchmark,
                        profilling_path=train_config.positive_profillings[0],
                        sample_rows_list=test_config.sample_rows_list,
                        K=test_config.K,
                        threshold=test_config.threshold,
                        table_mapper=test_config.table_mapper,
                        shuffle=test_config.shuffle,
                        mask_header=test_config.mask_header,
                        shuffle_columns=test_config.shuffle_columns,
                        dataset_discovery_type=test_config.dataset_discovery_type,
                        gt_path=test_config.gt_path,
                    )
                    print(
                        f"  Using profilling: {api_test_config.profilling_path}")
                else:
                    api_test_config = test_config
            else:
                # 对比学习模式：训练模型
                print(f"\n[Phase 1] Training model...")
                train_history, val_history, best_val_loss = self._train_model(
                    train_config, model_save_path
                )

                result.train_history = train_history
                result.val_history = val_history
                result.best_val_loss = best_val_loss
                result.model_path = model_save_path

                print(
                    f"  Training completed. Best val loss: {best_val_loss:.4f}" if best_val_loss else "  Training completed.")

            # 2. 测试模型
            print(f"\n[Phase 2] Testing model...")
            metrics_folder = os.path.join(exp_folder, "metrics")

            effective_test_config = api_test_config if is_api_mode else test_config

            test_metrics = self._test_model(
                model_path=None if is_api_mode else model_save_path,
                base_model_path=None if is_api_mode else train_config.model_path,
                test_config=effective_test_config,
                metrics_folder=metrics_folder,
                experiment_id=experiment_id
            )

            result.test_metrics = test_metrics
            result.status = "success"

            # 打印测试结果摘要
            print(f"\n[Test Results Summary]")
            for sr_key, metrics in test_metrics.items():
                if "error" not in metrics:
                    # 提取关键指标 (假设有 Precision@K, Recall@K 等)
                    p_key = [k for k in metrics.keys(
                    ) if 'precision' in k.lower() or 'P@' in k]
                    r_key = [k for k in metrics.keys(
                    ) if 'recall' in k.lower() or 'R@' in k]
                    if p_key:
                        print(f"  {sr_key}: {p_key[0]}={metrics[p_key[0]]:.4f}" if isinstance(metrics.get(
                            p_key[0]), (int, float)) else f"  {sr_key}: {p_key[0]}={metrics.get(p_key[0])}")
                else:
                    print(f"  {sr_key}: ERROR - {metrics.get('error')}")

        except Exception as e:
            result.status = "failed"
            result.error_message = f"{str(e)}\n{traceback.format_exc()}"
            print(f"\n[ERROR] Experiment failed: {e}")

        result.end_time = datetime.now().isoformat()

        # 保存实验配置和结果到 JSON
        result_json_path = os.path.join(exp_folder, "experiment_result.json")
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)

        self.results.append(result)

        return result

    def run_search(
        self,
        train_configs: List[TrainConfig],
        test_config: TestConfig,
        skip_on_error: bool = True,
        resume: bool = False
    ) -> List[ExperimentResult]:
        """
        运行完整的超参数搜索

        Args:
            train_configs: 训练配置列表
            test_config: 测试配置（所有实验共用）
            skip_on_error: 出错时是否跳过继续
            resume: 是否从断点续跑，自动跳过已完成的实验

        Returns:
            所有实验结果
        """
        # 加载已完成的实验（断点续跑）
        completed_ids = set()
        if resume:
            completed_ids = self._load_completed_experiments()
            print(
                f"\n[RESUME MODE] Found {len(completed_ids)} completed experiments, will skip them.")
            if completed_ids:
                print(f"  Completed: {sorted(completed_ids)}")

        remaining = len(train_configs) - len(completed_ids)

        print(f"\n{'#'*60}")
        print(f"# Hyperparameter Search: {self.experiment_name}")
        print(f"# Total experiments: {len(train_configs)}")
        if resume:
            print(f"# Already completed: {len(completed_ids)}")
            print(f"# Remaining to run:  {remaining}")
        print(f"# Output directory: {self.experiment_root}")
        print(f"{'#'*60}\n")

        skipped = 0
        for i, train_config in enumerate(train_configs):
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
                    experiment_id=experiment_id
                )
            except Exception as e:
                print(
                    f"\n[CRITICAL ERROR] Experiment {experiment_id} failed: {e}")
                if not skip_on_error:
                    raise

        # 生成汇总报告（包含所有实验，含之前已完成的）
        self._generate_summary_report()

        if resume and skipped > 0:
            print(
                f"\n[RESUME SUMMARY] Skipped {skipped} completed experiments, ran {len(train_configs) - skipped} new experiments.")

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

        # 提取每个实验的关键信息
        for result in self.results:
            # 提取各个 sample_rows 配置下的 MAP 值
            map_values = {}
            for sr_key, metrics in result.test_metrics.items():
                if isinstance(metrics, dict) and "MAP" in metrics:
                    map_values[sr_key] = metrics["MAP"]

            exp_summary = {
                "experiment_id": result.experiment_id,
                "status": result.status,
                "positive_profillings": result.train_config.get("positive_profillings", []),
                "temperature": result.train_config.get("temperature"),
                "train_sample_rows": result.train_config.get("sample_rows"),
                "epoch_num": result.train_config.get("epoch_num"),
                "batch_size": result.train_config.get("batch_size"),
                "use_aug": result.train_config.get("use_aug"),
                "max_seq_length": result.train_config.get("max_seq_length"),
                "sampling_ratio": result.train_config.get("sampling_ratio"),
                "use_amp": result.train_config.get("use_amp"),
                "gradient_checkpointing": result.train_config.get("gradient_checkpointing"),
                "best_val_loss": result.best_val_loss,
                "MAP": map_values,
                "test_metrics": result.test_metrics,
                "error": result.error_message
            }
            summary["experiments"].append(exp_summary)

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 生成 CSV 格式的汇总
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

        # 收集所有可能的 test metrics 键
        all_metrics_keys = set()
        for result in self.results:
            for sr_key, metrics in result.test_metrics.items():
                if isinstance(metrics, dict):
                    all_metrics_keys.update(metrics.keys())

        # 构建表头 - 把 MAP 放在前面更显眼的位置
        fieldnames = [
            "experiment_id", "status",
        ]

        # 先添加每个 sample_rows 的 MAP 列
        test_sr_keys = sorted(set(
            sr_key for r in self.results for sr_key in r.test_metrics.keys()
        ))
        for sr_key in test_sr_keys:
            fieldnames.append(f"{sr_key}_MAP")

        # 再添加训练参数列
        fieldnames.extend([
            "positive_profillings",
            "temperature", "train_sample_rows", "epoch_num", "batch_size", "use_aug",
            "max_seq_length", "sampling_ratio", "use_amp", "gradient_checkpointing",
            "best_val_loss"
        ])

        # 为每个 test sample_rows 添加对应的其他 metrics 列
        for sr_key in test_sr_keys:
            for metric_key in sorted(all_metrics_keys):
                if metric_key not in ["error", "task_success", "MAP"]:
                    fieldnames.append(f"{sr_key}_{metric_key}")

        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.results:
                row = {
                    "experiment_id": result.experiment_id,
                    "status": result.status,
                    "positive_profillings": ";".join(result.train_config.get("positive_profillings", [])),
                    "temperature": result.train_config.get("temperature"),
                    "train_sample_rows": result.train_config.get("sample_rows"),
                    "epoch_num": result.train_config.get("epoch_num"),
                    "batch_size": result.train_config.get("batch_size"),
                    "use_aug": result.train_config.get("use_aug", False),
                    "max_seq_length": result.train_config.get("max_seq_length"),
                    "sampling_ratio": result.train_config.get("sampling_ratio"),
                    "use_amp": result.train_config.get("use_amp"),
                    "gradient_checkpointing": result.train_config.get("gradient_checkpointing"),
                    "best_val_loss": result.best_val_loss
                }

                # 添加 test metrics
                for sr_key in test_sr_keys:
                    metrics = result.test_metrics.get(sr_key, {})
                    if isinstance(metrics, dict):
                        for metric_key in all_metrics_keys:
                            if metric_key not in ["error", "task_success"]:
                                col_name = f"{sr_key}_{metric_key}"
                                if col_name in fieldnames:
                                    row[col_name] = metrics.get(metric_key, "")

                writer.writerow(row)


def build_default_search_space() -> tuple:
    """
    构建默认的超参数搜索空间

    Returns:
        (train_configs, test_config)
    """
    # 定义 positive_profillings 的搜索空间
    # 这里是重点：不同的 profilling 组合
    base_path = "./output/SG/"

    profilling_files = [
        f"{base_path}datasets_SG_single_column_and_table_profilling.json",
        f"{base_path}datasets_SG_single_column_profilling.json",
        f"{base_path}datasets_SG_single_column_profilling_another.json",
        f"{base_path}datasets_SG_table_only_profilling.json",
    ]

    # 生成不同的 profilling 组合
    positive_profillings_options = [
        # 单个 profilling
        [profilling_files[0]],
        [profilling_files[1]],
        [profilling_files[2]],
        [profilling_files[3]],
        # 组合多个 profilling
        [profilling_files[0], profilling_files[1]],
        [profilling_files[0], profilling_files[2]],
        [profilling_files[0], profilling_files[3]],
        # 全部组合
        profilling_files,
    ]

    # 基础训练配置
    base_train_config = TrainConfig(
        model_path="./model/qwen3-0.6B-embedding",
        csv_dir="./datasets/opendata_SG",
        anchor_profilling=f"{base_path}datasets_SG_single_column_and_table_profilling.json",
        max_seq_length=512,
        sampling_ratio=1.0,
        use_amp=True,
        gradient_checkpointing=True,
    )

    # 创建搜索实例来生成配置
    searcher = HyperparameterSearch(output_dir="./experiments")

    train_configs = searcher.generate_search_space(
        positive_profillings_options=positive_profillings_options,
        temperature_options=[0.05, 0.07, 0.1],
        sample_rows_options=[5, 10],
        epoch_num_options=[3],
        batch_size_options=[4],
        max_seq_length_options=[512],
        sampling_ratio_options=[1.0],
        base_train_config=base_train_config
    )

    # 测试配置
    test_config = TestConfig(
        benchmark="SG",
        profilling_path=f"{base_path}datasets_SG_single_column_and_table_profilling.json",
        sample_rows_list=[0, 1, 5, 10],
        K=10,
        threshold=0.7,
        table_mapper=True,
        shuffle=True
    )

    return train_configs, test_config


def build_quick_test_space() -> tuple:
    """
    构建快速测试用的小规模搜索空间

    Returns:
        (train_configs, test_config)
    """
    base_path = "./output/SG/"

    # 仅测试两种 profilling 组合
    positive_profillings_options = [
        [f"{base_path}datasets_SG_single_column_and_table_profilling.json"],
        [f"{base_path}datasets_SG_single_column_profilling.json"],
    ]

    base_train_config = TrainConfig(
        model_path="./model/qwen3-0.6B-embedding",
        csv_dir="./datasets/opendata_SG",
        anchor_profilling=f"{base_path}datasets_SG_single_column_and_table_profilling.json",
        max_seq_length=512,
        sampling_ratio=1.0,
        use_amp=True,
        gradient_checkpointing=True,
    )

    searcher = HyperparameterSearch(output_dir="./experiments")

    train_configs = searcher.generate_search_space(
        positive_profillings_options=positive_profillings_options,
        temperature_options=[0.07],
        sample_rows_options=[10],
        epoch_num_options=[1],  # 快速测试只训练1轮
        batch_size_options=[4],
        max_seq_length_options=[512],
        sampling_ratio_options=[1.0],
        base_train_config=base_train_config
    )

    test_config = TestConfig(
        benchmark="SG",
        profilling_path=f"{base_path}datasets_SG_single_column_and_table_profilling.json",
        sample_rows_list=[0, 1],  # 快速测试只测试两种 sample_rows
        K=10,
        threshold=0.7,
        table_mapper=True,
        shuffle=True
    )

    return train_configs, test_config


def load_config_from_yaml(yaml_path: str) -> tuple:
    """
    从 YAML 配置文件加载搜索空间

    Returns:
        (train_configs, test_config)
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for loading YAML config. "
            "Install it with: pip install pyyaml"
        )

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 解析训练配置
    train_section = config.get('train', {})
    base_train_config = TrainConfig(
        model_path=train_section.get(
            'model_path', "./model/qwen3-0.6B-embedding"),
        csv_dir=train_section.get('csv_dir', "./datasets/opendata_SG"),
        anchor_profilling=train_section.get('anchor_profilling', ""),
        max_seq_length=train_section.get('max_seq_length', 512),
        sampling_ratio=train_section.get('sampling_ratio', 1.0),
        sampling_seed=train_section.get('sampling_seed', 42),
        use_amp=train_section.get('use_amp', True),
        gradient_checkpointing=train_section.get(
            'gradient_checkpointing', True),
        gpu=train_section.get('gpu', 0),
        use_aug=train_section.get('use_aug', False)
    )

    # 解析模式
    mode = train_section.get('mode', 'contrastive')
    base_train_config.mode = mode

    # 解析搜索空间
    search_section = config.get('search_space', {})
    positive_profillings_options = search_section.get(
        'positive_profillings', [[]])
    temperature_options = search_section.get('temperature', [0.07])
    sample_rows_options = search_section.get('sample_rows', [10])
    epoch_num_options = search_section.get('epoch_num', [3])
    batch_size_options = search_section.get('batch_size', [16])
    max_seq_length_options = search_section.get('max_seq_length', None)
    sampling_ratio_options = search_section.get('sampling_ratio', None)
    use_amp_options = search_section.get('use_amp', None)
    gradient_checkpointing_options = search_section.get(
        'gradient_checkpointing', None)

    # 解析 use_aug 搜索空间（支持列表或单个值）
    use_aug_raw = search_section.get('use_aug', None)
    if use_aug_raw is None:
        # 没有在搜索空间中指定，使用 base_train_config 中的值
        use_aug_options = None
    elif isinstance(use_aug_raw, list):
        use_aug_options = use_aug_raw
    else:
        # 单个值转换为列表
        use_aug_options = [use_aug_raw]

    # 生成训练配置
    searcher = HyperparameterSearch(output_dir="./experiments")

    if mode == 'api':
        # API 模式：不需要训练，只需生成不同 profilling 组合用于测试
        train_configs = []
        for pp in positive_profillings_options:
            config_item = TrainConfig(
                model_path=base_train_config.model_path,
                csv_dir=base_train_config.csv_dir,
                anchor_profilling=base_train_config.anchor_profilling,
                positive_profillings=pp,
                max_seq_length=base_train_config.max_seq_length,
                sampling_ratio=base_train_config.sampling_ratio,
                sampling_seed=base_train_config.sampling_seed,
                use_amp=base_train_config.use_amp,
                gradient_checkpointing=base_train_config.gradient_checkpointing,
                gpu=base_train_config.gpu,
                mode='api'
            )
            train_configs.append(config_item)
    else:
        train_configs = searcher.generate_search_space(
            positive_profillings_options=positive_profillings_options,
            temperature_options=temperature_options,
            sample_rows_options=sample_rows_options,
            epoch_num_options=epoch_num_options,
            batch_size_options=batch_size_options,
            use_aug_options=use_aug_options,
            max_seq_length_options=max_seq_length_options,
            sampling_ratio_options=sampling_ratio_options,
            use_amp_options=use_amp_options,
            gradient_checkpointing_options=gradient_checkpointing_options,
            base_train_config=base_train_config
        )

    # 解析测试配置
    test_section = config.get('test', {})
    test_config = TestConfig(
        benchmark=test_section.get('benchmark', "SG"),
        profilling_path=test_section.get('profilling_path'),
        sample_rows_list=test_section.get('sample_rows_list', [0, 1, 5, 10]),
        K=test_section.get('K', 10),
        threshold=test_section.get('threshold', 0.7),
        table_mapper=test_section.get('table_mapper', True),
        shuffle=test_section.get('shuffle', True),
        mask_header=test_section.get('mask_header'),
        shuffle_columns=test_section.get('shuffle_columns', False),
        dataset_discovery_type=test_section.get('dataset_discovery_type', 'union'),
        gt_path=test_section.get('gt_path'),
        noise_prob=test_section.get('noise_prob', 0.0),
    )

    return train_configs, test_config


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter Search for Contrastive Learning Model"
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
        '--output_dir', type=str, default='./experiments',
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
        '--max_seq_length', type=int, default=None,
        help='Override training max sequence length for all experiments'
    )
    parser.add_argument(
        '--sampling_ratio', type=float, default=None,
        help='Override CSV file sampling ratio for all experiments'
    )
    parser.add_argument(
        '--disable_amp', action='store_true',
        help='Disable AMP for all experiments'
    )
    parser.add_argument(
        '--disable_gradient_checkpointing', action='store_true',
        help='Disable gradient checkpointing for all experiments'
    )
    parser.add_argument(
        '--skip_on_error', action='store_true', default=True,
        help='Continue to next experiment if one fails'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Resume from an existing experiment directory '
             '(e.g., ./experiments/exp_20260303_070958). '
             'Automatically skips completed experiments and only runs remaining ones.'
    )
    parser.add_argument(
        '--mode', type=str, default=None, choices=['contrastive', 'api'],
        help='Running mode: "contrastive" (default, train contrastive model then test) '
             'or "api" (skip training, directly use base model API embeddings for testing)'
    )

    args = parser.parse_args()

    # 加载搜索空间
    if args.config:
        print(f"Loading configuration from: {args.config}")
        train_configs, test_config = load_config_from_yaml(args.config)
    elif args.quick:
        print("Using quick test search space (minimal)")
        train_configs, test_config = build_quick_test_space()
    else:
        print("Using default search space")
        train_configs, test_config = build_default_search_space()

    # 更新 GPU 设置和模式
    for config in train_configs:
        config.gpu = args.gpu
        if args.mode:
            config.mode = args.mode
        if args.max_seq_length is not None:
            config.max_seq_length = args.max_seq_length
        if args.sampling_ratio is not None:
            config.sampling_ratio = args.sampling_ratio
        if args.disable_amp:
            config.use_amp = False
        if args.disable_gradient_checkpointing:
            config.gradient_checkpointing = False

    print(f"Total configurations to test: {len(train_configs)}")

    # 处理断点续跑逻辑
    resume_mode = False
    output_dir = args.output_dir
    experiment_name = args.experiment_name

    if args.resume:
        resume_mode = True
        resume_path = os.path.normpath(args.resume)
        # 从 resume 路径中提取 output_dir 和 experiment_name
        # e.g., "./experiments/exp_20260303_070958" -> output_dir="./experiments", name="exp_20260303_070958"
        experiment_name = os.path.basename(resume_path)
        output_dir = os.path.dirname(resume_path)
        if not output_dir:
            output_dir = '.'

        if not os.path.exists(resume_path):
            print(f"[ERROR] Resume directory not found: {resume_path}")
            sys.exit(1)

        print(f"\n[RESUME] Resuming from: {resume_path}")

    # 创建搜索实例并运行
    searcher = HyperparameterSearch(
        output_dir=output_dir,
        experiment_name=experiment_name
    )

    results = searcher.run_search(
        train_configs=train_configs,
        test_config=test_config,
        skip_on_error=args.skip_on_error,
        resume=resume_mode
    )

    # 打印最终总结
    print("\n" + "=" * 60)
    print("HYPERPARAMETER SEARCH COMPLETED")
    print("=" * 60)
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r.status == 'success')}")
    print(f"Failed: {sum(1 for r in results if r.status == 'failed')}")
    print(f"Results saved to: {searcher.experiment_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
