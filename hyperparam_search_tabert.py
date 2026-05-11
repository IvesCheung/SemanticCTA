#!/usr/bin/env python3
"""
TaBERT 超参数搜索脚本 - Hyperparameter Search Script for TaBERT

功能：
1. 遍历超参数组合（profilling、sample_rows、mask_header 等），使用预训练 TaBERT 模型直接做 index+query
2. 对每个配置进行测试评估
3. 将测试参数、结果存放到独立的子文件夹
4. 生成汇总报告
5. 支持断点续跑 (checkpoint resume)
6. 支持 union 和 join 两种数据发现任务

使用示例：
    python hyperparam_search_tabert.py --config hyperparam_search_tabert_config.yaml
    python hyperparam_search_tabert.py --quick  # 快速测试模式
    python hyperparam_search_tabert.py --resume experiments_tabert/exp_20260310_120000 --config xxx.yaml  # 断点续跑
"""

from task.TableDDTask import TableIndexQueryTask
import os
import sys
import json
import argparse
import itertools
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
import csv
import traceback
from HYPERPARAMETERS import get_join_ground_truth_path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class TabertConfig:
    """TaBERT 实验配置（无训练阶段，直接使用预训练模型）"""
    # TaBERT 模型路径
    model_path: str = "./model/tabert_base_k3/model.bin"
    # Profilling 路径（可选，不同 profilling 是核心搜索维度）
    profilling_path: Optional[str] = None
    # 采样行数
    sample_rows: int = 1
    # 是否使用表名映射
    table_mapper: bool = True
    # 是否打乱数据
    shuffle: bool = True
    # 是否 mask header
    mask_header: Optional[str] = None
    # 是否打乱列顺序
    shuffle_columns: bool = False
    # 噪声增强概率 (0.0=不加噪声, 0.1=10%概率, 以此类推)
    noise_prob: float = 0.0


@dataclass
class TestConfig:
    """测试配置"""
    benchmark: str = "SG"
    K: int = 10
    threshold: float = 0.7
    # 数据发现任务类型: "union" 或 "join"
    dataset_discovery_type: str = "union"
    # 自定义 Ground Truth 路径（覆盖自动推断，优先级最高）
    gt_path: Optional[str] = None


@dataclass
class ExperimentResult:
    """实验结果"""
    experiment_id: str
    tabert_config: Dict[str, Any]
    test_config: Dict[str, Any]
    test_metrics: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    error_message: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    experiment_folder: Optional[str] = None


class TabertHyperparameterSearch:
    """TaBERT 超参数搜索类"""

    def __init__(
        self,
        output_dir: str = "./experiments_tabert",
        experiment_name: Optional[str] = None
    ):
        self.output_dir = output_dir
        if experiment_name is None:
            experiment_name = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        self.experiment_root = os.path.join(output_dir, experiment_name)
        os.makedirs(self.experiment_root, exist_ok=True)
        self.results: List[ExperimentResult] = []

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
                    exp_result = ExperimentResult(
                        experiment_id=result_data.get("experiment_id", entry),
                        tabert_config=result_data.get("tabert_config", {}),
                        test_config=result_data.get("test_config", {}),
                        test_metrics=result_data.get("test_metrics", {}),
                        status=result_data.get("status", "success"),
                        error_message=result_data.get("error_message"),
                        start_time=result_data.get("start_time"),
                        end_time=result_data.get("end_time"),
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
        profilling_path_options: List[Optional[str]] = None,
        sample_rows_options: List[int] = None,
        model_path_options: List[str] = None,
        table_mapper_options: List[bool] = None,
        mask_header_options: List[Optional[str]] = None,
        shuffle_columns_options: List[bool] = None,
        shuffle_options: List[bool] = None,
        base_config: Optional[TabertConfig] = None,
    ) -> List[TabertConfig]:
        """
        生成超参数搜索空间

        Returns:
            所有 TaBERT 配置的列表
        """
        if base_config is None:
            base_config = TabertConfig()

        if profilling_path_options is None:
            profilling_path_options = [base_config.profilling_path]
        if sample_rows_options is None:
            sample_rows_options = [base_config.sample_rows]
        if model_path_options is None:
            model_path_options = [base_config.model_path]
        if table_mapper_options is None:
            table_mapper_options = [base_config.table_mapper]
        if mask_header_options is None:
            mask_header_options = [base_config.mask_header]
        if shuffle_columns_options is None:
            shuffle_columns_options = [base_config.shuffle_columns]
        if shuffle_options is None:
            shuffle_options = [base_config.shuffle]

        configs = []
        for model_path, profilling, sr, tm, mh, sc, sh in itertools.product(
            model_path_options,
            profilling_path_options,
            sample_rows_options,
            table_mapper_options,
            mask_header_options,
            shuffle_columns_options,
            shuffle_options,
        ):
            config = TabertConfig(
                model_path=model_path,
                profilling_path=profilling,
                sample_rows=sr,
                table_mapper=tm,
                shuffle=sh,
                mask_header=mh,
                shuffle_columns=sc,
                noise_prob=base_config.noise_prob,
            )
            configs.append(config)

        return configs

    def _create_experiment_folder(self, experiment_id: str) -> str:
        """创建单个实验的文件夹"""
        folder = os.path.join(self.experiment_root, experiment_id)
        os.makedirs(folder, exist_ok=True)
        os.makedirs(os.path.join(folder, "metrics"), exist_ok=True)
        return folder

    def _test_model(
        self,
        tabert_config: TabertConfig,
        test_config: TestConfig,
        metrics_folder: str,
        experiment_id: str,
    ) -> Dict[str, Any]:
        """
        使用 TaBERT 进行 index + query 测试

        Returns:
            metrics 字典
        """
        metrics_path = os.path.join(metrics_folder, "metrics.csv")

        # 根据 dataset_discovery_type 确定 gt_path
        gt_path = test_config.gt_path  # 优先级最高: 显式指定
        if not gt_path and test_config.dataset_discovery_type == "join":
            gt_path = get_join_ground_truth_path(test_config.benchmark)
        # union 时 gt_path=None，由 query.py 内部走 get_ground_truth_path(benchmark)

        # results_dir 加入 encoder+task_type，避免与 taqwen/join 并发时共用同一 pkl
        results_dir = (
            f"./results/{test_config.benchmark}/"
            f"tabert_{test_config.dataset_discovery_type}/"
        )
        task = TableIndexQueryTask(
            task_name=f"{experiment_id}",
            config={},
            encoder_type="tabert",
            benchmark=test_config.benchmark,
            model_path=tabert_config.model_path,
            profilling_path=tabert_config.profilling_path,
            sample_rows=tabert_config.sample_rows,
            shuffle=tabert_config.shuffle,
            table_mapper=tabert_config.table_mapper,
            mask_header=tabert_config.mask_header,
            shuffle_columns=tabert_config.shuffle_columns,
            noise_prob=tabert_config.noise_prob,
            metrics_path=metrics_path,
            K=test_config.K,
            threshold=test_config.threshold,
            gt_path=gt_path,
            results_dir=results_dir,
        )

        task_result = task.run()

        metrics = self._parse_metrics_csv(metrics_path)
        metrics['task_success'] = task_result.success
        metrics['elapsed_time'] = task_result.elapsed

        return metrics

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
        tabert_config: TabertConfig,
        test_config: TestConfig,
        experiment_id: Optional[str] = None
    ) -> ExperimentResult:
        """运行单个实验"""
        if experiment_id is None:
            experiment_id = f"exp_{len(self.results):04d}"

        exp_folder = self._create_experiment_folder(experiment_id)

        result = ExperimentResult(
            experiment_id=experiment_id,
            tabert_config=asdict(tabert_config),
            test_config=asdict(test_config),
            experiment_folder=exp_folder,
            start_time=datetime.now().isoformat()
        )

        print(f"\n{'='*60}")
        print(f"[Experiment {experiment_id}]")
        print(f"  model_path: {tabert_config.model_path}")
        print(f"  profilling_path: {tabert_config.profilling_path}")
        print(f"  sample_rows: {tabert_config.sample_rows}")
        print(f"  table_mapper: {tabert_config.table_mapper}")
        print(f"  mask_header: {tabert_config.mask_header}")
        print(f"  shuffle_columns: {tabert_config.shuffle_columns}")
        print(f"  benchmark: {test_config.benchmark}")
        print(f"  dataset_discovery_type: {test_config.dataset_discovery_type}")
        print(f"{'='*60}")

        try:
            print(f"\n[Running] TaBERT index + query...")
            metrics_folder = os.path.join(exp_folder, "metrics")

            test_metrics = self._test_model(
                tabert_config=tabert_config,
                test_config=test_config,
                metrics_folder=metrics_folder,
                experiment_id=experiment_id,
            )

            result.test_metrics = test_metrics
            result.status = "success"

            # 打印测试结果摘要
            print(f"\n[Test Results Summary]")
            if "error" not in test_metrics:
                map_val = test_metrics.get("MAP")
                if map_val is not None:
                    print(f"  MAP: {map_val:.4f}")
                p_keys = [k for k in test_metrics.keys()
                          if 'precision' in k.lower()]
                r_keys = [k for k in test_metrics.keys()
                          if 'recall' in k.lower()]
                for pk in p_keys:
                    v = test_metrics[pk]
                    print(f"  {pk}: {v:.4f}" if isinstance(v, (int, float))
                          else f"  {pk}: {v}")
                for rk in r_keys:
                    v = test_metrics[rk]
                    print(f"  {rk}: {v:.4f}" if isinstance(v, (int, float))
                          else f"  {rk}: {v}")
            else:
                print(f"  ERROR: {test_metrics.get('error')}")

        except Exception as e:
            result.status = "failed"
            result.error_message = f"{str(e)}\n{traceback.format_exc()}"
            print(f"\n[ERROR] Experiment failed: {e}")

        result.end_time = datetime.now().isoformat()

        result_json_path = os.path.join(exp_folder, "experiment_result.json")
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)

        self.results.append(result)
        return result

    def run_search(
        self,
        tabert_configs: List[TabertConfig],
        test_config: TestConfig,
        skip_on_error: bool = True,
        resume: bool = False,
    ) -> List[ExperimentResult]:
        """运行完整的超参数搜索"""
        completed_ids = set()
        if resume:
            completed_ids = self._load_completed_experiments()
            print(
                f"\n[RESUME MODE] Found {len(completed_ids)} completed experiments, will skip them.")
            if completed_ids:
                print(f"  Completed: {sorted(completed_ids)}")

        remaining = len(tabert_configs) - len(completed_ids)

        print(f"\n{'#'*60}")
        print(f"# TaBERT Hyperparameter Search: {self.experiment_name}")
        print(f"# Total experiments: {len(tabert_configs)}")
        if resume:
            print(f"# Already completed: {len(completed_ids)}")
            print(f"# Remaining to run:  {remaining}")
        print(f"# Output directory: {self.experiment_root}")
        print(f"{'#'*60}\n")

        skipped = 0
        for i, tabert_config in enumerate(tabert_configs):
            experiment_id = f"exp_{i:04d}"

            if experiment_id in completed_ids:
                skipped += 1
                print(
                    f"\n[SKIP] {experiment_id} - already completed (checkpoint)")
                continue

            try:
                self.run_single_experiment(
                    tabert_config=tabert_config,
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
                f"\n[RESUME SUMMARY] Skipped {skipped} completed experiments, ran {len(tabert_configs) - skipped} new experiments.")

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
            map_value = None
            if isinstance(result.test_metrics, dict) and "MAP" in result.test_metrics:
                map_value = result.test_metrics["MAP"]

            exp_summary = {
                "experiment_id": result.experiment_id,
                "status": result.status,
                "model_path": result.tabert_config.get("model_path"),
                "profilling_path": result.tabert_config.get("profilling_path"),
                "sample_rows": result.tabert_config.get("sample_rows"),
                "table_mapper": result.tabert_config.get("table_mapper"),
                "mask_header": result.tabert_config.get("mask_header"),
                "shuffle_columns": result.tabert_config.get("shuffle_columns"),
                "MAP": map_value,
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

        # 收集所有可能的 test metrics 键
        all_metrics_keys = set()
        for result in self.results:
            if isinstance(result.test_metrics, dict):
                all_metrics_keys.update(result.test_metrics.keys())

        fieldnames = [
            "experiment_id", "status", "MAP",
            "model_path", "profilling_path", "sample_rows",
            "table_mapper", "mask_header", "shuffle_columns",
        ]

        # 添加其他 metrics 列
        for metric_key in sorted(all_metrics_keys):
            if metric_key not in ["error", "task_success", "MAP"]:
                fieldnames.append(metric_key)

        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.results:
                row = {
                    "experiment_id": result.experiment_id,
                    "status": result.status,
                    "MAP": result.test_metrics.get("MAP", "")
                    if isinstance(result.test_metrics, dict) else "",
                    "model_path": result.tabert_config.get("model_path"),
                    "profilling_path": result.tabert_config.get("profilling_path"),
                    "sample_rows": result.tabert_config.get("sample_rows"),
                    "table_mapper": result.tabert_config.get("table_mapper"),
                    "mask_header": result.tabert_config.get("mask_header"),
                    "shuffle_columns": result.tabert_config.get("shuffle_columns"),
                }

                if isinstance(result.test_metrics, dict):
                    for metric_key in sorted(all_metrics_keys):
                        if metric_key not in ["error", "task_success", "MAP"]:
                            if metric_key in fieldnames:
                                row[metric_key] = result.test_metrics.get(
                                    metric_key, "")

                writer.writerow(row)


def build_quick_test_space() -> tuple:
    """
    构建快速测试用的小规模搜索空间

    Returns:
        (tabert_configs, test_config)
    """
    base_path = "./output/SG/"

    profilling_path_options = [
        None,
        f"{base_path}datasets_SG_single_column_and_table_profilling.json",
    ]

    base_config = TabertConfig(
        model_path="./model/tabert_base_k3/model.bin",
        table_mapper=True,
        shuffle=True,
    )

    searcher = TabertHyperparameterSearch(output_dir="./experiments_tabert")
    tabert_configs = searcher.generate_search_space(
        profilling_path_options=profilling_path_options,
        sample_rows_options=[0, 1],
        base_config=base_config,
    )

    test_config = TestConfig(
        benchmark="SG",
        K=10,
        threshold=0.7,
    )

    return tabert_configs, test_config


def load_config_from_yaml(yaml_path: str) -> tuple:
    """
    从 YAML 配置文件加载搜索空间

    Returns:
        (tabert_configs, test_config)
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required. Install it with: pip install pyyaml"
        )

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 解析 TaBERT 基础配置
    tabert_section = config.get('tabert', {})
    base_config = TabertConfig(
        model_path=tabert_section.get(
            'model_path', "./model/tabert_base_k3/model.bin"),
        profilling_path=tabert_section.get('profilling_path'),
        sample_rows=tabert_section.get('sample_rows', 1),
        table_mapper=tabert_section.get('table_mapper', True),
        shuffle=tabert_section.get('shuffle', True),
        mask_header=tabert_section.get('mask_header'),
        shuffle_columns=tabert_section.get('shuffle_columns', False),
        noise_prob=tabert_section.get('noise_prob', 0.0),
    )

    # 解析搜索空间
    search_section = config.get('search_space', {})
    profilling_path_options = search_section.get('profilling_path')
    sample_rows_options = search_section.get('sample_rows')
    model_path_options = search_section.get('model_path')
    table_mapper_options = search_section.get('table_mapper')
    mask_header_options = search_section.get('mask_header')
    shuffle_columns_options = search_section.get('shuffle_columns')
    shuffle_options = search_section.get('shuffle')

    searcher = TabertHyperparameterSearch(output_dir="./experiments_tabert")
    tabert_configs = searcher.generate_search_space(
        profilling_path_options=profilling_path_options,
        sample_rows_options=sample_rows_options,
        model_path_options=model_path_options,
        table_mapper_options=table_mapper_options,
        mask_header_options=mask_header_options,
        shuffle_columns_options=shuffle_columns_options,
        shuffle_options=shuffle_options,
        base_config=base_config,
    )

    # 解析测试配置
    test_section = config.get('test', {})
    test_config = TestConfig(
        benchmark=test_section.get('benchmark', "SG"),
        K=test_section.get('K', 10),
        threshold=test_section.get('threshold', 0.7),
        dataset_discovery_type=test_section.get(
            'dataset_discovery_type', 'union'),
        gt_path=test_section.get('gt_path'),
    )

    return tabert_configs, test_config


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter Search for TaBERT Model"
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
        '--output_dir', type=str, default='./experiments_tabert',
        help='Output directory for experiments'
    )
    parser.add_argument(
        '--experiment_name', type=str, default=None,
        help='Name for this experiment run'
    )
    parser.add_argument(
        '--skip_on_error', action='store_true', default=True,
        help='Continue to next experiment if one fails'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Resume from an existing experiment directory '
             '(e.g., ./experiments_tabert/exp_20260310_120000). '
             'Automatically skips completed experiments.'
    )

    args = parser.parse_args()

    # 加载搜索空间
    if args.config:
        print(f"Loading configuration from: {args.config}")
        tabert_configs, test_config = load_config_from_yaml(args.config)
    elif args.quick:
        print("Using quick test search space (minimal)")
        tabert_configs, test_config = build_quick_test_space()
    else:
        print("Using quick test search space (no default config provided)")
        tabert_configs, test_config = build_quick_test_space()

    print(f"Total configurations to test: {len(tabert_configs)}")

    # 处理断点续跑逻辑
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
    searcher = TabertHyperparameterSearch(
        output_dir=output_dir,
        experiment_name=experiment_name,
    )

    results = searcher.run_search(
        tabert_configs=tabert_configs,
        test_config=test_config,
        skip_on_error=args.skip_on_error,
        resume=resume_mode,
    )

    # 打印最终总结
    print("\n" + "=" * 60)
    print("TABERT HYPERPARAMETER SEARCH COMPLETED")
    print("=" * 60)
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r.status == 'success')}")
    print(f"Failed: {sum(1 for r in results if r.status == 'failed')}")
    print(f"Results saved to: {searcher.experiment_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
