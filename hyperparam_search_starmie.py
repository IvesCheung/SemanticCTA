#!/usr/bin/env python3
"""
Starmie 超参数搜索脚本 - Hyperparameter Search for Starmie Dataset Discovery

功能：
1. 遍历超参数组合训练 Starmie 模型（对比学习，RoBERTa/BERT 等）
2. 对每个模型使用 StarmieDDTask 进行数据集发现评估
3. 将训练参数、测试结果、模型文件存放到独立的子文件夹
4. 生成汇总报告（JSON + CSV）
5. 支持断点续跑 (checkpoint resume)
6. 支持直接用已有模型跳过训练（pretrained 模式）

使用示例：
    python hyperparam_search_starmie.py --config hyperparam_search_config_starmie.yaml
    python hyperparam_search_starmie.py --quick                          # 快速测试模式
    python hyperparam_search_starmie.py --resume experiments_starmie/exp_20260317_120000 \\
        --config hyperparam_search_config_starmie.yaml                   # 断点续跑
"""

from task.StarmieDDTask import StarmieDDTask
import os
import sys
import json
import argparse
import itertools
import csv
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from HYPERPARAMETERS import (
    get_ground_truth_path, get_join_ground_truth_path,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# 配置数据类
# ============================================================

@dataclass
class StarmieTrainConfig:
    """Starmie 预训练阶段配置"""
    # 语言模型基座: roberta, bert, distilbert
    lm: str = "roberta"
    # 训练批次大小
    batch_size: int = 64
    # 学习率
    lr: float = 5e-5
    # 训练轮数
    n_epochs: int = 3
    # 序列最大长度
    max_len: int = 256
    # 训练数据大小
    size: int = 10000
    # 数据增强操作: drop_col, drop_col,sample_row, None
    augment_op: str = "drop_col"
    # 采样方法: head, tfidf_entity, random
    sample_method: str = "head"
    # 表格列顺序: column, row
    table_order: str = "column"
    # 是否保存模型
    save_model: bool = True
    # 模式: "train" 正常训练, "pretrained" 跳过训练使用已有模型
    mode: str = "train"
    # pretrained 模式时使用的模型路径
    pretrained_model_path: Optional[str] = None
    # Profilling JSON 路径（可选，为列添加语义描述）
    profilling_path: Optional[str] = None
    # 提取向量时每张表读取的行数（None=全部, 0=仅header, N=N行）
    sample_rows: Optional[int] = None
    # 噪声增强概率 (0.0=不加噪声, 0.1=10%概率, 以此类推)
    noise_prob: float = 0.0
    # 表名匿名映射 (True = 将表名替换为匿名ID)
    table_mapper: bool = False
    # 随机打乱文件处理顺序
    shuffle: bool = False


@dataclass
class StarmieTestConfig:
    """Starmie 检索阶段配置"""
    # 基准数据集: santos, GDC, HDXSM, SG, USA, UK, CAN, etc.
    benchmark: str = "santos"
    # 数据发现任务类型: "union" 或 "join"
    dataset_discovery_type: str = "union"
    # 自定义 Ground Truth 路径（覆盖自动推断，优先级最高）
    gt_path: Optional[str] = None
    # 编码器类型: cl
    encoder: str = "cl"
    # 匹配方法: 强制使用 hnsw（不允许修改）
    matching: str = "hnsw"
    # Top-K 返回数量
    K: int = 10
    # 相似度阈值列表
    threshold_list: List[float] = field(default_factory=lambda: [0.6, 0.7, 0.8])
    # 数据湖规模
    scal: float = 1.0
    # HNSW 专属参数
    index_path: Optional[str] = None               # HNSW 索引文件路径 (.bin)
    N: int = 50                                     # HNSW: 从索引检索的列数


@dataclass
class StarmieExperimentResult:
    """单次实验结果"""
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


# ============================================================
# 核心搜索类
# ============================================================

class StarmieHyperparameterSearch:
    """Starmie 超参数搜索"""

    def __init__(
        self,
        output_dir: str = "./experiments_starmie",
        experiment_name: Optional[str] = None,
    ):
        self.output_dir = output_dir
        if experiment_name is None:
            experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        self.experiment_root = os.path.join(output_dir, experiment_name)
        os.makedirs(self.experiment_root, exist_ok=True)
        self.results: List[StarmieExperimentResult] = []

    # ------ 断点续跑 ------

    def _load_completed_experiments(self) -> set:
        """扫描实验目录，加载已完成的实验 ID"""
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
                    # 加载到 self.results 以便生成完整汇总报告
                    result = StarmieExperimentResult(
                        experiment_id=data["experiment_id"],
                        train_config=data.get("train_config", {}),
                        test_config=data.get("test_config", {}),
                        test_metrics=data.get("test_metrics", {}),
                        status=data.get("status", "success"),
                        model_path=data.get("model_path"),
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
        lm_options: Optional[List[str]] = None,
        batch_size_options: Optional[List[int]] = None,
        lr_options: Optional[List[float]] = None,
        n_epochs_options: Optional[List[int]] = None,
        max_len_options: Optional[List[int]] = None,
        augment_op_options: Optional[List[str]] = None,
        sample_method_options: Optional[List[str]] = None,
        sample_rows_options: Optional[List[Optional[int]]] = None,
        K_options: Optional[List[int]] = None,
        threshold_options: Optional[List[float]] = None,
        matching_options: Optional[List[str]] = None,
        table_mapper_options: Optional[List[bool]] = None,
        shuffle_options: Optional[List[bool]] = None,
        base_train_config: Optional[StarmieTrainConfig] = None,
        base_test_config: Optional[StarmieTestConfig] = None,
    ) -> List[tuple]:
        """
        生成 (StarmieTrainConfig, StarmieTestConfig) 笛卡尔积搜索空间。

        Returns:
            [(StarmieTrainConfig, StarmieTestConfig), ...]
        """
        if base_train_config is None:
            base_train_config = StarmieTrainConfig()
        if base_test_config is None:
            base_test_config = StarmieTestConfig()

        lm_options = lm_options or [base_train_config.lm]
        batch_size_options = batch_size_options or [base_train_config.batch_size]
        lr_options = lr_options or [base_train_config.lr]
        n_epochs_options = n_epochs_options or [base_train_config.n_epochs]
        max_len_options = max_len_options or [base_train_config.max_len]
        augment_op_options = augment_op_options or [base_train_config.augment_op]
        sample_method_options = sample_method_options or [base_train_config.sample_method]
        sample_rows_options = sample_rows_options if sample_rows_options is not None else [base_train_config.sample_rows]
        K_options = K_options or [base_test_config.K]
        threshold_options = threshold_options or [base_test_config.threshold_list]
        matching_options = matching_options or [base_test_config.matching]
        table_mapper_options = table_mapper_options if table_mapper_options is not None else [
            base_train_config.table_mapper]
        shuffle_options = shuffle_options if shuffle_options is not None else [
            base_train_config.shuffle]

        configs = []
        run_id = 0
        for lm, batch_size, lr, n_epochs, max_len, augment_op, sample_method, sample_rows, K, thresholds, matching, table_mapper, shuffle in itertools.product(
            lm_options,
            batch_size_options,
            lr_options,
            n_epochs_options,
            max_len_options,
            augment_op_options,
            sample_method_options,
            sample_rows_options,
            K_options,
            threshold_options if (threshold_options and isinstance(
                threshold_options[0], list)) else [threshold_options],
            matching_options,
            table_mapper_options,
            shuffle_options,
        ):
            train_cfg = StarmieTrainConfig(
                lm=lm,
                batch_size=batch_size,
                lr=lr,
                n_epochs=n_epochs,
                max_len=max_len,
                size=base_train_config.size,
                augment_op=augment_op,
                sample_method=sample_method,
                table_order=base_train_config.table_order,
                save_model=base_train_config.save_model,
                mode=base_train_config.mode,
                pretrained_model_path=base_train_config.pretrained_model_path,
                profilling_path=base_train_config.profilling_path,
                sample_rows=sample_rows,
                noise_prob=base_train_config.noise_prob,
                table_mapper=table_mapper,
                shuffle=shuffle,
            )
            test_cfg = StarmieTestConfig(
                benchmark=base_test_config.benchmark,
                dataset_discovery_type=base_test_config.dataset_discovery_type,
                gt_path=base_test_config.gt_path,
                encoder=base_test_config.encoder,
                matching=matching,
                K=K,
                threshold_list=thresholds if isinstance(thresholds, list) else [thresholds],
                scal=base_test_config.scal,
                index_path=base_test_config.index_path,
                N=base_test_config.N,
            )
            configs.append((train_cfg, test_cfg, run_id))
            run_id += 1

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
            with open(csv_path, "r", encoding="utf-8") as f:
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
            print(f"  [Warning] Failed to parse metrics CSV {csv_path}: {e}")
            return {"error": f"parse error: {str(e)}"}

    # ------ 单次实验 ------

    def run_single_experiment(
        self,
        train_config: StarmieTrainConfig,
        test_config: StarmieTestConfig,
        run_id: int = 0,
        experiment_id: Optional[str] = None,
    ) -> StarmieExperimentResult:
        """运行单次实验（训练 + 多阈值检索）"""
        if experiment_id is None:
            if train_config.mode == "pretrained" and train_config.pretrained_model_path:
                safe_model_name = os.path.basename(
                    train_config.pretrained_model_path).replace(",", "_")
                experiment_id = (
                    f"exp_{run_id:04d}_pretrained_{safe_model_name}"
                    f"_K{test_config.K}_{test_config.matching}"
                )
            else:
                experiment_id = (
                    f"exp_{run_id:04d}_lm{train_config.lm}"
                    f"_bs{train_config.batch_size}"
                    f"_ep{train_config.n_epochs}"
                    f"_aug{train_config.augment_op.replace(',', '+')}"
                )

        exp_folder = self._create_experiment_folder(experiment_id)

        result = StarmieExperimentResult(
            experiment_id=experiment_id,
            train_config=asdict(train_config),
            test_config=asdict(test_config),
            experiment_folder=exp_folder,
            start_time=datetime.now().isoformat(),
        )

        is_pretrained = train_config.mode == "pretrained"

        print(f"\n{'='*60}")
        print(f"[Experiment {experiment_id}]")
        print(f"  mode       : {train_config.mode}")
        if not is_pretrained:
            print(f"  lm         : {train_config.lm}")
            print(f"  batch_size : {train_config.batch_size}")
            print(f"  lr         : {train_config.lr}")
            print(f"  n_epochs   : {train_config.n_epochs}")
            print(f"  augment_op : {train_config.augment_op}")
        else:
            print(f"  model      : {train_config.pretrained_model_path}")
        print(f"  benchmark  : {test_config.benchmark}")
        print(f"  dataset_discovery_type: {test_config.dataset_discovery_type}")
        print(f"  K          : {test_config.K}")
        print(f"  matching   : {test_config.matching}")
        print(f"  thresholds : {test_config.threshold_list}")
        print(f"{'='*60}")

        try:
            # ====== 确定模型路径 ======
            if is_pretrained:
                model_path = train_config.pretrained_model_path
            else:
                model_path = os.path.join(exp_folder, "model.pt")

            result.model_path = model_path
            vectors_dir = os.path.join(exp_folder, "vectors")

            # ====== 根据 dataset_discovery_type 解析 gt_path ======
            gt_path = test_config.gt_path  # 显式指定优先
            if not gt_path:
                if test_config.dataset_discovery_type == "join":
                    gt_path = get_join_ground_truth_path(test_config.benchmark)
                # union 时 gt_path=None，由 test_naive_search.py 内部走 get_ground_truth_path(benchmark)

            # ====== 对每个 threshold 运行检索 ======
            for threshold in test_config.threshold_list:
                th_key = f"threshold_{threshold}"
                metrics_filename = f"metrics_K{test_config.K}_th{threshold:.2f}.csv"
                metrics_path = os.path.join(exp_folder, "metrics", metrics_filename)

                print(f"\n  [Threshold={threshold}] Running StarmieDDTask ...")
                task = StarmieDDTask(
                    task_name=f"{experiment_id}_th{threshold}",
                    config={},
                    # 训练参数
                    benchmark=test_config.benchmark,
                    lm=train_config.lm,
                    batch_size=train_config.batch_size,
                    lr=train_config.lr,
                    n_epochs=train_config.n_epochs,
                    max_len=train_config.max_len,
                    size=train_config.size,
                    augment_op=train_config.augment_op,
                    sample_method=train_config.sample_method,
                    table_order=train_config.table_order,
                    save_model=train_config.save_model,
                    save_model_path=model_path,
                    vector_output_path=vectors_dir,
                    profilling_path=train_config.profilling_path,
                    sample_rows=train_config.sample_rows,
                    noise_prob=train_config.noise_prob,
                    table_mapper=train_config.table_mapper,
                    shuffle=train_config.shuffle,
                    # 检索参数
                    encoder=test_config.encoder,
                    # 强制使用 HNSW，忽略 test_config.matching
                    matching="hnsw",
                    K=test_config.K,
                    threshold=threshold,
                    metrics_path=metrics_path,
                    gt_path=gt_path,
                    index_path=test_config.index_path,
                    N=test_config.N,
                    # 控制: pretrained 模式跳过训练和提取（但每个 threshold 只需要训练一次）
                    skip_pretrain=is_pretrained or threshold != test_config.threshold_list[0],
                    skip_extract=is_pretrained or threshold != test_config.threshold_list[0],
                    skip_search=False,
                    run_id=run_id,
                )
                task_result = task.run()

                if task_result.success:
                    metrics = self._parse_metrics_csv(metrics_path)
                    result.test_metrics[th_key] = {
                        "threshold": threshold,
                        "metrics_path": metrics_path,
                        **metrics,
                    }
                    map_val = metrics.get("MAP")
                    if isinstance(map_val, float):
                        print(
                            f"  [Threshold={threshold}] Done. MAP={map_val:.4f}")
                    else:
                        print(
                            f"  [Threshold={threshold}] Done. Metrics: {metrics}")
                else:
                    error_str = str(task_result.error)
                    result.test_metrics[th_key] = {
                        "threshold": threshold,
                        "error": error_str,
                    }
                    print(f"  [Threshold={threshold}] FAILED: {error_str}")

            result.status = "success"

        except Exception as e:
            result.status = "failed"
            result.error_message = traceback.format_exc()
            print(f"[Experiment {experiment_id}] FAILED: {e}")

        result.end_time = datetime.now().isoformat()

        # 保存单次实验结果
        result_json_path = os.path.join(exp_folder, "experiment_result.json")
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)

        self.results.append(result)
        return result

    # ------ 全量搜索 ------

    def run_search(
        self,
        configs: List[tuple],
        skip_on_error: bool = True,
        resume: bool = False,
    ) -> List[StarmieExperimentResult]:
        """
        遍历所有 (train_config, test_config, run_id) 组合运行实验。

        Args:
            configs: generate_search_space() 返回的列表
            skip_on_error: 出错时是否跳过继续
            resume: 是否断点续跑（跳过已完成的实验）
        """
        completed_ids: set = set()
        if resume:
            completed_ids = self._load_completed_experiments()
            print(f"[Resume] Found {len(completed_ids)} completed experiments, skipping them.")

        print(f"\n{'#'*60}")
        print(f"# Starmie Hyperparameter Search: {self.experiment_name}")
        print(f"# Total experiments: {len(configs)}")
        if resume:
            print(f"# Already completed: {len(completed_ids)}")
            print(f"# Remaining: {len(configs) - len(completed_ids)}")
        print(f"# Output directory: {self.experiment_root}")
        print(f"{'#'*60}\n")

        skipped = 0
        for i, (train_config, test_config, run_id) in enumerate(configs):
            if train_config.mode == "pretrained" and train_config.pretrained_model_path:
                safe_model_name = os.path.basename(
                    train_config.pretrained_model_path).replace(",", "_")
                experiment_id = (
                    f"exp_{run_id:04d}_pretrained_{safe_model_name}"
                    f"_K{test_config.K}_{test_config.matching}"
                )
            else:
                experiment_id = (
                    f"exp_{run_id:04d}_lm{train_config.lm}"
                    f"_bs{train_config.batch_size}"
                    f"_ep{train_config.n_epochs}"
                    f"_aug{train_config.augment_op.replace(',', '+')}"
                )

            if resume and experiment_id in completed_ids:
                print(f"[{i+1}/{len(configs)}] Skipping completed: {experiment_id}")
                skipped += 1
                continue

            print(f"\n[{i+1}/{len(configs)}] Starting: {experiment_id}")

            try:
                self.run_single_experiment(
                    train_config=train_config,
                    test_config=test_config,
                    run_id=run_id,
                    experiment_id=experiment_id,
                )
            except Exception as e:
                print(f"[{i+1}/{len(configs)}] ERROR: {e}")
                if not skip_on_error:
                    raise

        self._generate_summary_report()

        if resume and skipped > 0:
            print(f"[Resume] Skipped {skipped} already-completed experiments.")

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
            exp_entry = {
                "experiment_id": result.experiment_id,
                "status": result.status,
                "model_path": result.model_path,
                "test_metrics": result.test_metrics,
                "lm": result.train_config.get("lm"),
                "batch_size": result.train_config.get("batch_size"),
                "lr": result.train_config.get("lr"),
                "n_epochs": result.train_config.get("n_epochs"),
                "augment_op": result.train_config.get("augment_op"),
                "K": result.test_config.get("K"),
                "matching": result.test_config.get("matching"),
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
        print(f"  Successful: {summary['successful']}/{summary['total_experiments']}")
        print(f"{'='*60}\n")

    def _get_th_metrics_with_map(self, th_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """确保 th_metrics 中含有 MAP；如果缺失则从 metrics_path 重新计算。"""
        if "MAP" in th_metrics:
            return th_metrics
        metrics_path = th_metrics.get("metrics_path", "")
        if metrics_path and os.path.exists(metrics_path):
            computed = self._parse_metrics_csv(metrics_path)
            if "MAP" in computed:
                return {**th_metrics, **{k: v for k, v in computed.items() if k not in th_metrics}}
        return th_metrics

    def _generate_summary_csv(self, csv_path: str):
        if not self.results:
            return

        fieldnames = [
            "experiment_id", "status", "best_MAP",
            "lm", "batch_size", "lr", "n_epochs", "max_len",
            "augment_op", "sample_method", "K", "matching", "mode",
            "model_path", "start_time", "end_time",
        ]
        # 为每个 threshold 添加 metrics 列（MAP 优先拿到后才能统计列名）
        metric_keys_seen: set = set()
        for r in self.results:
            for th_key, th_metrics in r.test_metrics.items():
                th_metrics = self._get_th_metrics_with_map(th_metrics)
                for mk in th_metrics.keys():
                    if mk not in ("threshold", "metrics_path", "error"):
                        metric_keys_seen.add(f"{th_key}_{mk}")
        for col in sorted(metric_keys_seen):
            fieldnames.append(col)

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for result in self.results:
                row = {
                    "experiment_id": result.experiment_id,
                    "status": result.status,
                    "lm": result.train_config.get("lm"),
                    "batch_size": result.train_config.get("batch_size"),
                    "lr": result.train_config.get("lr"),
                    "n_epochs": result.train_config.get("n_epochs"),
                    "max_len": result.train_config.get("max_len"),
                    "augment_op": result.train_config.get("augment_op"),
                    "sample_method": result.train_config.get("sample_method"),
                    "K": result.test_config.get("K"),
                    "matching": result.test_config.get("matching"),
                    "mode": result.train_config.get("mode"),
                    "model_path": result.model_path,
                    "start_time": result.start_time,
                    "end_time": result.end_time,
                }
                map_values = []
                for th_key, th_metrics in result.test_metrics.items():
                    th_metrics = self._get_th_metrics_with_map(th_metrics)
                    for mk, mv in th_metrics.items():
                        if mk not in ("threshold", "metrics_path", "error"):
                            row[f"{th_key}_{mk}"] = mv
                    if isinstance(th_metrics.get("MAP"), float):
                        map_values.append(th_metrics["MAP"])
                row["best_MAP"] = max(map_values) if map_values else ""
                writer.writerow(row)


# ============================================================
# 快速测试搜索空间
# ============================================================

def build_quick_test_space() -> List[tuple]:
    """快速测试用的小规模搜索空间（跳过训练，直接用预训练模型）"""
    base_train = StarmieTrainConfig(
        mode="pretrained",
        pretrained_model_path=None,  # 需要手动设置或通过 YAML 配置
        lm="roberta",
    )
    base_test = StarmieTestConfig(
        benchmark="santos",
        K=10,
        threshold_list=[0.6, 0.7],
    )
    if not base_train.pretrained_model_path:
        raise ValueError(
            "--quick 模式需要预训练模型路径，请修改 build_quick_test_space() 中的 "
            "pretrained_model_path，或改用 --config 指定 YAML 配置文件。"
        )
    searcher = StarmieHyperparameterSearch(output_dir="./experiments_starmie")
    configs = searcher.generate_search_space(
        augment_op_options=["drop_col"],
        base_train_config=base_train,
        base_test_config=base_test,
    )
    return configs


# ============================================================
# YAML 配置加载
# ============================================================

def load_config_from_yaml(yaml_path: str) -> List[tuple]:
    """
    从 YAML 配置文件加载搜索空间。

    Returns:
        [(StarmieTrainConfig, StarmieTestConfig, run_id), ...]
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install it with: pip install pyyaml")

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ---- 训练基础配置 ----
    train_section = config.get("train", {})
    base_train_config = StarmieTrainConfig(
        lm=train_section.get("lm", "roberta"),
        batch_size=train_section.get("batch_size", 64),
        lr=train_section.get("lr", 5e-5),
        n_epochs=train_section.get("n_epochs", 3),
        max_len=train_section.get("max_len", 256),
        size=train_section.get("size", 10000),
        augment_op=train_section.get("augment_op", "drop_col"),
        sample_method=train_section.get("sample_method", "head"),
        table_order=train_section.get("table_order", "column"),
        save_model=train_section.get("save_model", True),
        mode=train_section.get("mode", "train"),
        pretrained_model_path=train_section.get("pretrained_model_path"),
        profilling_path=train_section.get("profilling_path"),
        sample_rows=train_section.get("sample_rows"),
        noise_prob=train_section.get("noise_prob", 0.0),
        table_mapper=train_section.get("table_mapper", False),
        shuffle=train_section.get("shuffle", False),
    )

    # ---- 测试基础配置 ----
    test_section = config.get("test", {})
    base_test_config = StarmieTestConfig(
        benchmark=test_section.get("benchmark", "santos"),
        dataset_discovery_type=test_section.get("dataset_discovery_type", "union"),
        gt_path=test_section.get("gt_path"),
        encoder=test_section.get("encoder", "cl"),
        matching=test_section.get("matching", "exact"),
        K=test_section.get("K", 10),
        threshold_list=test_section.get("threshold_list", [0.6, 0.7, 0.8]),
        scal=test_section.get("scal", 1.0),
    )

    # ---- 搜索空间 ----
    search_section = config.get("search_space", {})
    lm_options = search_section.get("lm")
    batch_size_options = search_section.get("batch_size")
    lr_options = search_section.get("lr")
    n_epochs_options = search_section.get("n_epochs")
    max_len_options = search_section.get("max_len")
    augment_op_options = search_section.get("augment_op")
    sample_method_options = search_section.get("sample_method")
    sample_rows_options = search_section.get("sample_rows")
    K_options = search_section.get("K")
    threshold_options = search_section.get("threshold")
    matching_options = search_section.get("matching")
    table_mapper_options = search_section.get("table_mapper")
    shuffle_options = search_section.get("shuffle")

    # pretrained 模式：每个模型路径生成一组实验
    pretrained_paths = search_section.get("pretrained_model_paths")
    if pretrained_paths:
        configs = []
        th_opts = threshold_options or [base_test_config.threshold_list]
        # 确保 th_opts 是 list of lists
        if th_opts and not isinstance(th_opts[0], list):
            th_opts = [th_opts]
        K_opts = K_options or [base_test_config.K]
        matching_opts = matching_options or [base_test_config.matching]
        sr_opts = sample_rows_options if sample_rows_options is not None else [base_train_config.sample_rows]
        run_id = 0
        for model_path, thresholds, K, matching, sample_rows in itertools.product(
            pretrained_paths, th_opts, K_opts, matching_opts, sr_opts
        ):
            train_cfg = StarmieTrainConfig(
                mode="pretrained",
                pretrained_model_path=model_path,
                lm=base_train_config.lm,
                profilling_path=base_train_config.profilling_path,
                sample_rows=sample_rows,
            )
            test_cfg = StarmieTestConfig(
                benchmark=base_test_config.benchmark,
                dataset_discovery_type=base_test_config.dataset_discovery_type,
                gt_path=base_test_config.gt_path,
                encoder=base_test_config.encoder,
                matching=matching,
                K=K,
                threshold_list=thresholds if isinstance(thresholds, list) else [thresholds],
                scal=base_test_config.scal,
                index_path=base_test_config.index_path,
                N=base_test_config.N,
            )
            configs.append((train_cfg, test_cfg, run_id))
            run_id += 1
        return configs

    # 普通训练模式
    th_opts = threshold_options or None
    searcher = StarmieHyperparameterSearch(output_dir="./experiments_starmie")
    configs = searcher.generate_search_space(
        lm_options=lm_options,
        batch_size_options=batch_size_options,
        lr_options=lr_options,
        n_epochs_options=n_epochs_options,
        max_len_options=max_len_options,
        augment_op_options=augment_op_options,
        sample_method_options=sample_method_options,
        sample_rows_options=sample_rows_options,
        K_options=K_options,
        threshold_options=th_opts,
        matching_options=matching_options,
        table_mapper_options=table_mapper_options,
        shuffle_options=shuffle_options,
        base_train_config=base_train_config,
        base_test_config=base_test_config,
    )
    return configs


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter Search for Starmie Dataset Discovery"
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
        "--output_dir", type=str, default="./experiments_starmie",
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
            "(e.g., ./experiments_starmie/exp_20260317_120000). "
            "Automatically skips completed experiments."
        )
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
        output_dir = os.path.dirname(resume_path)
        experiment_name = os.path.basename(resume_path)
        print(f"Resuming experiment: {experiment_name} in {output_dir}")

    # 创建搜索实例并运行
    searcher = StarmieHyperparameterSearch(
        output_dir=output_dir,
        experiment_name=experiment_name,
    )

    results = searcher.run_search(
        configs=configs,
        skip_on_error=args.skip_on_error,
        resume=resume_mode,
    )

    print("\n" + "=" * 60)
    print("STARMIE HYPERPARAMETER SEARCH COMPLETED")
    print("=" * 60)
    print(f"Total experiments : {len(results)}")
    print(f"Successful        : {sum(1 for r in results if r.status == 'success')}")
    print(f"Failed            : {sum(1 for r in results if r.status == 'failed')}")
    print(f"Results saved to  : {searcher.experiment_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
