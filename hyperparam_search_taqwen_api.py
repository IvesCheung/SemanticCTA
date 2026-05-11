#!/usr/bin/env python3
"""
Qwen API Embedding 超参数搜索脚本 - 无需对比学习训练

功能：
1. 直接使用 Qwen 基础模型（API embedding）进行表格索引和查询
2. 遍历不同的 profilling、sample_rows、threshold 等超参数组合
3. 无需训练阶段，仅做索引 + 查询评估
4. 将测试结果存放到独立的子文件夹
5. 生成汇总报告（JSON + CSV）
6. 支持断点续跑 (checkpoint resume)
7. 支持 union / join 双模式

使用示例：
    python hyperparam_search_taqwen_api.py --config hyperparam_search_taqwen_api_config.yaml
    python hyperparam_search_taqwen_api.py --quick
    python hyperparam_search_taqwen_api.py --resume experiments_qwen_api/exp_20260317_120000 --config xxx.yaml
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# 配置数据类
# ============================================================

@dataclass
class QwenApiTestConfig:
    """Qwen API 模式测试配置（无训练）"""
    # Qwen 模型名称（用于 index_qwen.py）
    qwen_model: str = "qwen3-embedding-0.6b"
    # embedding 批次大小
    batch_size: int = 256

    # 基准数据集
    benchmark: str = "SG"
    # 数据发现任务类型: "union" 或 "join"
    dataset_discovery_type: str = "union"
    # 自定义 Ground Truth 路径（覆盖自动推断，优先级最高）
    gt_path: Optional[str] = None

    # profilling 路径（核心搜索维度）
    profilling_path: Optional[str] = None
    # 测试时的采样行数列表
    sample_rows_list: List[int] = field(default_factory=lambda: [0, 1, 5, 10])
    # Top-K
    K: int = 10
    # 相似度阈值
    threshold: float = 0.7
    # 是否使用 table_mapper
    table_mapper: bool = True
    # 是否 shuffle 文件顺序
    shuffle: bool = True
    # 是否擦除 header
    mask_header: Optional[str] = None
    # 是否 shuffle 列顺序
    shuffle_columns: bool = False
    # 噪声增强概率 (0.0=不加噪声, 0.1=10%概率, 以此类推)
    noise_prob: float = 0.0


@dataclass
class QwenApiExperimentResult:
    """单次实验结果"""
    experiment_id: str
    test_config: Dict[str, Any]
    test_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    status: str = "pending"
    error_message: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    experiment_folder: Optional[str] = None


# ============================================================
# 核心搜索类
# ============================================================

class QwenApiHyperparameterSearch:
    """Qwen API Embedding 超参数搜索（无训练）"""

    def __init__(
        self,
        output_dir: str = "./experiments_qwen_api",
        experiment_name: Optional[str] = None,
    ):
        self.output_dir = output_dir
        if experiment_name is None:
            experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        self.experiment_root = os.path.join(output_dir, experiment_name)
        os.makedirs(self.experiment_root, exist_ok=True)
        self.results: List[QwenApiExperimentResult] = []

    # ------ 断点续跑 ------

    def _load_completed_experiments(self) -> set:
        completed = set()
        if not os.path.exists(self.experiment_root):
            return completed

        for entry in sorted(os.listdir(self.experiment_root)):
            entry_path = os.path.join(self.experiment_root, entry)
            if not os.path.isdir(entry_path):
                continue
            result_json = os.path.join(entry_path, "experiment_result.json")
            if not os.path.exists(result_json):
                continue
            try:
                with open(result_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("status") == "success":
                    completed.add(data["experiment_id"])
                    result = QwenApiExperimentResult(
                        experiment_id=data["experiment_id"],
                        test_config=data.get("test_config", {}),
                        test_metrics=data.get("test_metrics", {}),
                        status=data.get("status", "success"),
                        experiment_folder=entry_path,
                        start_time=data.get("start_time"),
                        end_time=data.get("end_time"),
                    )
                    self.results.append(result)
            except Exception:
                pass

        return completed

    # ------ 搜索空间生成 ------

    def generate_search_space(
        self,
        profilling_path_options: Optional[List[Optional[str]]] = None,
        sample_rows_options: Optional[List[int]] = None,
        threshold_options: Optional[List[float]] = None,
        K_options: Optional[List[int]] = None,
        table_mapper_options: Optional[List[bool]] = None,
        mask_header_options: Optional[List[Optional[str]]] = None,
        shuffle_columns_options: Optional[List[bool]] = None,
        base_config: Optional[QwenApiTestConfig] = None,
    ) -> List[QwenApiTestConfig]:
        """
        生成 QwenApiTestConfig 笛卡尔积搜索空间。

        Returns:
            [QwenApiTestConfig, ...]
        """
        if base_config is None:
            base_config = QwenApiTestConfig()

        profilling_path_options = profilling_path_options or [
            base_config.profilling_path]
        sample_rows_options = sample_rows_options or [
            base_config.sample_rows_list]
        threshold_options = threshold_options or [base_config.threshold]
        K_options = K_options or [base_config.K]
        table_mapper_options = table_mapper_options or [
            base_config.table_mapper]
        mask_header_options = mask_header_options or [base_config.mask_header]
        shuffle_columns_options = shuffle_columns_options or [
            base_config.shuffle_columns]

        configs = []
        for prof, sr_list, threshold, K, tm, mh, sc in itertools.product(
            profilling_path_options,
            sample_rows_options,
            threshold_options,
            K_options,
            table_mapper_options,
            mask_header_options,
            shuffle_columns_options,
        ):
            cfg = QwenApiTestConfig(
                qwen_model=base_config.qwen_model,
                batch_size=base_config.batch_size,
                benchmark=base_config.benchmark,
                dataset_discovery_type=base_config.dataset_discovery_type,
                gt_path=base_config.gt_path,
                profilling_path=prof,
                sample_rows_list=sr_list if isinstance(
                    sr_list, list) else [sr_list],
                K=K,
                threshold=threshold,
                table_mapper=tm,
                shuffle=base_config.shuffle,
                mask_header=mh,
                shuffle_columns=sc,
                noise_prob=base_config.noise_prob,
            )
            configs.append(cfg)

        return configs

    # ------ 辅助方法 ------

    def _create_experiment_folder(self, experiment_id: str) -> str:
        folder = os.path.join(self.experiment_root, experiment_id)
        os.makedirs(folder, exist_ok=True)
        os.makedirs(os.path.join(folder, "metrics"), exist_ok=True)
        return folder

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

    # ------ 单次实验 ------

    def run_single_experiment(
        self,
        test_config: QwenApiTestConfig,
        experiment_id: Optional[str] = None,
    ) -> QwenApiExperimentResult:
        """运行单次实验（索引 + 查询，无训练）"""
        if experiment_id is None:
            experiment_id = f"exp_{len(self.results):04d}"

        exp_folder = self._create_experiment_folder(experiment_id)

        result = QwenApiExperimentResult(
            experiment_id=experiment_id,
            test_config=asdict(test_config),
            experiment_folder=exp_folder,
            start_time=datetime.now().isoformat(),
        )

        # 简短的 profilling 显示名
        prof_display = os.path.basename(
            test_config.profilling_path) if test_config.profilling_path else "None"

        print(f"\n{'='*60}")
        print(f"[Experiment {experiment_id}]")
        print(f"  qwen_model : {test_config.qwen_model}")
        print(f"  benchmark  : {test_config.benchmark}")
        print(f"  dd_type    : {test_config.dataset_discovery_type}")
        print(f"  profilling : {prof_display}")
        print(f"  sample_rows: {test_config.sample_rows_list}")
        print(f"  K={test_config.K}  threshold={test_config.threshold}  "
              f"table_mapper={test_config.table_mapper}  mask_header={test_config.mask_header}  "
              f"shuffle_columns={test_config.shuffle_columns}")
        print(f"{'='*60}")

        try:
            # 根据 dataset_discovery_type 解析 gt_path
            gt_path = test_config.gt_path
            if not gt_path and test_config.dataset_discovery_type == "join":
                gt_path = get_join_ground_truth_path(test_config.benchmark)

            metrics_folder = os.path.join(exp_folder, "metrics")

            test_metrics = {}
            for sample_rows in test_config.sample_rows_list:
                metrics_path = os.path.join(
                    metrics_folder, f"metrics_sr{sample_rows}.csv"
                )

                try:
                    # API 模式: local_model_path=None, base_model_path=None
                    # results_dir 加入 encoder+task_type，避免与 tabert/join 并发时共用同一 pkl
                    results_dir = (
                        f"./results/{test_config.benchmark}/"
                        f"qwen_api_{test_config.dataset_discovery_type}/"
                    )
                    task = TableIndexQueryTask(
                        task_name=f"{experiment_id}_sr{sample_rows}",
                        config={},
                        encoder_type="qwen",
                        benchmark=test_config.benchmark,
                        qwen_model=test_config.qwen_model,
                        batch_size=test_config.batch_size,
                        profilling_path=test_config.profilling_path,
                        sample_rows=sample_rows,
                        shuffle=test_config.shuffle,
                        table_mapper=test_config.table_mapper,
                        mask_header=test_config.mask_header,
                        shuffle_columns=test_config.shuffle_columns,
                        noise_prob=test_config.noise_prob,
                        local_model_path=None,
                        base_model_path=None,
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
                    test_metrics[f"sr{sample_rows}"] = metrics

                except Exception as e:
                    test_metrics[f"sr{sample_rows}"] = {
                        "error": str(e),
                        "task_success": False,
                    }
                    print(
                        f"  [WARNING] Test failed for sample_rows={sample_rows}: {e}")

            result.test_metrics = test_metrics
            result.status = "success"

            # 打印摘要
            print(f"\n[Test Results Summary]")
            for sr_key, metrics in test_metrics.items():
                if "error" not in metrics:
                    map_val = metrics.get("MAP", "N/A")
                    if isinstance(map_val, float):
                        print(f"  {sr_key}: MAP={map_val:.4f}")
                    else:
                        print(f"  {sr_key}: MAP={map_val}")
                else:
                    print(f"  {sr_key}: ERROR - {metrics.get('error')}")

        except Exception as e:
            result.status = "failed"
            result.error_message = f"{str(e)}\n{traceback.format_exc()}"
            print(f"\n[ERROR] Experiment failed: {e}")

        result.end_time = datetime.now().isoformat()

        result_json_path = os.path.join(exp_folder, "experiment_result.json")
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)

        self.results.append(result)
        return result

    # ------ 全量搜索 ------

    def run_search(
        self,
        configs: List[QwenApiTestConfig],
        skip_on_error: bool = True,
        resume: bool = False,
    ) -> List[QwenApiExperimentResult]:
        completed_ids: set = set()
        if resume:
            completed_ids = self._load_completed_experiments()
            print(
                f"[Resume] Found {len(completed_ids)} completed experiments, skipping them.")

        print(f"\n{'#'*60}")
        print(f"# Qwen API Hyperparameter Search: {self.experiment_name}")
        print(f"# Total experiments: {len(configs)}")
        if resume:
            print(f"# Already completed: {len(completed_ids)}")
            print(f"# Remaining: {len(configs) - len(completed_ids)}")
        print(f"# Output directory: {self.experiment_root}")
        print(f"{'#'*60}\n")

        for i, test_config in enumerate(configs):
            experiment_id = f"exp_{i:04d}"

            if resume and experiment_id in completed_ids:
                print(f"[{i+1}/{len(configs)}] Skipping completed: {experiment_id}")
                continue

            print(f"\n[{i+1}/{len(configs)}] Starting: {experiment_id}")

            try:
                self.run_single_experiment(
                    test_config=test_config,
                    experiment_id=experiment_id,
                )
            except Exception as e:
                print(f"[{i+1}/{len(configs)}] ERROR: {e}")
                if not skip_on_error:
                    raise

        self._generate_summary_report()
        return self.results

    # ------ 汇总报告 ------

    def _generate_summary_report(self):
        summary_path = os.path.join(self.experiment_root, "summary.json")
        summary = {
            "experiment_name": self.experiment_name,
            "total_experiments": len(self.results),
            "successful": sum(1 for r in self.results if r.status == "success"),
            "failed": sum(1 for r in self.results if r.status == "failed"),
            "generated_at": datetime.now().isoformat(),
            "experiments": [],
        }

        for result in self.results:
            map_values = {}
            for sr_key, metrics in result.test_metrics.items():
                if isinstance(metrics, dict) and "MAP" in metrics:
                    map_values[sr_key] = metrics["MAP"]

            exp_entry = {
                "experiment_id": result.experiment_id,
                "status": result.status,
                "profilling_path": result.test_config.get("profilling_path"),
                "qwen_model": result.test_config.get("qwen_model"),
                "benchmark": result.test_config.get("benchmark"),
                "dataset_discovery_type": result.test_config.get("dataset_discovery_type"),
                "sample_rows_list": result.test_config.get("sample_rows_list"),
                "K": result.test_config.get("K"),
                "threshold": result.test_config.get("threshold"),
                "table_mapper": result.test_config.get("table_mapper"),
                "mask_header": result.test_config.get("mask_header"),
                "shuffle_columns": result.test_config.get("shuffle_columns"),
                "MAP": map_values,
                "test_metrics": result.test_metrics,
                "error": result.error_message,
                "start_time": result.start_time,
                "end_time": result.end_time,
            }
            summary["experiments"].append(exp_entry)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        csv_path = os.path.join(self.experiment_root, "summary.csv")
        self._generate_summary_csv(csv_path)

        print(f"\n{'='*60}")
        print(f"[Summary Report Generated]")
        print(f"  JSON : {summary_path}")
        print(f"  CSV  : {csv_path}")
        print(
            f"  Successful: {summary['successful']}/{summary['total_experiments']}")
        print(f"{'='*60}\n")

    def _generate_summary_csv(self, csv_path: str):
        if not self.results:
            return

        # 收集所有 sample_rows keys
        sr_keys = sorted(set(
            sr_key for r in self.results for sr_key in r.test_metrics.keys()
        ))

        # 收集所有 metric keys
        all_metric_keys: set = set()
        for r in self.results:
            for metrics in r.test_metrics.values():
                if isinstance(metrics, dict):
                    all_metric_keys.update(metrics.keys())

        fieldnames = [
            "experiment_id", "status",
        ]
        # MAP 列优先
        for sr_key in sr_keys:
            fieldnames.append(f"{sr_key}_MAP")

        fieldnames.extend([
            "profilling_path", "qwen_model", "benchmark",
            "dataset_discovery_type", "K", "threshold",
            "table_mapper", "mask_header", "shuffle_columns",
            "start_time", "end_time",
        ])

        # 其他 metric 列
        for sr_key in sr_keys:
            for mk in sorted(all_metric_keys):
                if mk not in ("error", "task_success", "MAP"):
                    fieldnames.append(f"{sr_key}_{mk}")

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for result in self.results:
                row = {
                    "experiment_id": result.experiment_id,
                    "status": result.status,
                    "profilling_path": result.test_config.get("profilling_path"),
                    "qwen_model": result.test_config.get("qwen_model"),
                    "benchmark": result.test_config.get("benchmark"),
                    "dataset_discovery_type": result.test_config.get("dataset_discovery_type"),
                    "K": result.test_config.get("K"),
                    "threshold": result.test_config.get("threshold"),
                    "table_mapper": result.test_config.get("table_mapper"),
                    "mask_header": result.test_config.get("mask_header"),
                    "shuffle_columns": result.test_config.get("shuffle_columns"),
                    "start_time": result.start_time,
                    "end_time": result.end_time,
                }
                for sr_key in sr_keys:
                    metrics = result.test_metrics.get(sr_key, {})
                    if isinstance(metrics, dict):
                        for mk in all_metric_keys:
                            if mk not in ("error", "task_success"):
                                row[f"{sr_key}_{mk}"] = metrics.get(mk, "")
                writer.writerow(row)


# ============================================================
# 快速测试
# ============================================================

def build_quick_test_space() -> List[QwenApiTestConfig]:
    base = QwenApiTestConfig(
        benchmark="SG",
        dataset_discovery_type="union",
        sample_rows_list=[0, 1],
        K=10,
        threshold=0.7,
    )
    searcher = QwenApiHyperparameterSearch()
    return searcher.generate_search_space(
        profilling_path_options=[None],
        base_config=base,
    )


# ============================================================
# YAML 配置加载
# ============================================================

def load_config_from_yaml(yaml_path: str) -> List[QwenApiTestConfig]:
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required. Install it with: pip install pyyaml")

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ---- 基础测试配置 ----
    test_section = config.get("test", {})
    base_config = QwenApiTestConfig(
        qwen_model=test_section.get("qwen_model", "qwen3-embedding-0.6b"),
        batch_size=test_section.get("batch_size", 256),
        benchmark=test_section.get("benchmark", "SG"),
        dataset_discovery_type=test_section.get(
            "dataset_discovery_type", "union"),
        gt_path=test_section.get("gt_path"),
        profilling_path=test_section.get("profilling_path"),
        sample_rows_list=test_section.get("sample_rows_list", [0, 1, 5, 10]),
        K=test_section.get("K", 10),
        threshold=test_section.get("threshold", 0.7),
        table_mapper=test_section.get("table_mapper", True),
        shuffle=test_section.get("shuffle", True),
        mask_header=test_section.get("mask_header"),
        shuffle_columns=test_section.get("shuffle_columns", False),
        noise_prob=test_section.get("noise_prob", 0.0),
    )

    # ---- 搜索空间 ----
    search_section = config.get("search_space", {})
    profilling_path_options = search_section.get("profilling_path")
    sample_rows_options = search_section.get("sample_rows_list")
    threshold_options = search_section.get("threshold")
    K_options = search_section.get("K")
    table_mapper_options = search_section.get("table_mapper")
    mask_header_options = search_section.get("mask_header")
    shuffle_columns_options = search_section.get("shuffle_columns")

    searcher = QwenApiHyperparameterSearch()
    configs = searcher.generate_search_space(
        profilling_path_options=profilling_path_options,
        sample_rows_options=sample_rows_options,
        threshold_options=threshold_options,
        K_options=K_options,
        table_mapper_options=table_mapper_options,
        mask_header_options=mask_header_options,
        shuffle_columns_options=shuffle_columns_options,
        base_config=base_config,
    )
    return configs


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter Search for Qwen API Embedding (no training)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick test with minimal search space"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./experiments_qwen_api",
        help="Output directory for experiments"
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None,
        help="Name for this experiment run (default: auto timestamp)"
    )
    parser.add_argument(
        "--skip_on_error", action="store_true", default=True,
        help="Continue to next experiment if one fails"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help=(
            "Resume from an existing experiment directory "
            "(e.g., ./experiments_qwen_api/exp_20260317_120000). "
            "Automatically skips completed experiments."
        ),
    )
    args = parser.parse_args()

    # 加载搜索空间
    if args.config:
        print(f"Loading config from: {args.config}")
        configs = load_config_from_yaml(args.config)
    elif args.quick:
        print("Running in quick test mode ...")
        configs = build_quick_test_space()
    else:
        parser.print_help()
        print("\nError: Please provide --config or --quick.")
        sys.exit(1)

    print(f"Total configurations to test: {len(configs)}")

    # 处理断点续跑
    resume_mode = False
    output_dir = args.output_dir
    experiment_name = args.experiment_name

    if args.resume:
        resume_path = os.path.normpath(args.resume)
        resume_mode = True
        output_dir = os.path.dirname(resume_path) or "."
        experiment_name = os.path.basename(resume_path)
        print(f"Resuming experiment: {experiment_name} in {output_dir}")

    # 运行搜索
    searcher = QwenApiHyperparameterSearch(
        output_dir=output_dir,
        experiment_name=experiment_name,
    )

    results = searcher.run_search(
        configs=configs,
        skip_on_error=args.skip_on_error,
        resume=resume_mode,
    )

    print("\n" + "=" * 60)
    print("QWEN API HYPERPARAMETER SEARCH COMPLETED")
    print("=" * 60)
    print(f"Total experiments : {len(results)}")
    print(
        f"Successful        : {sum(1 for r in results if r.status == 'success')}")
    print(
        f"Failed            : {sum(1 for r in results if r.status == 'failed')}")
    print(f"Results saved to  : {searcher.experiment_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
