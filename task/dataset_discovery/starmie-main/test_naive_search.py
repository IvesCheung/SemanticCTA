from mlflow_logging import get_mlflow_logger
from utils import get_basename, transform_dict, loadDictionaryFromPickleFile
from HYPERPARAMETERS import get_ground_truth_path, DATASET_PATHS
import numpy as np
import random
import pickle
import argparse
import os
import sys
import json
import csv as csv_module
from datetime import datetime

# 添加项目根目录到 sys.path
root_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


from naive_search import NaiveSearcher
from checkPrecisionRecall import saveDictionaryAsPickleFile, calcMetrics
import time


def generate_random_table(nrow, ncol):
    return np.random.rand(nrow, ncol)


def generate_test_data(num, ndim):
    # for test only: randomly generate tables and 2 queries
    # num: the number of tables in the dataset; ndim: dimension of column vectors
    tables = []
    queries = []
    for i in range(num):
        ncol = random.randint(2, 9)
        tbl = generate_random_table(ncol, ndim)
        tables.append((i, tbl))
    for j in range(2):
        ncol = random.randint(2, 9)
        tbl = generate_random_table(ncol, ndim)
        queries.append((j+num, tbl))
    return tables, queries


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="cl",
                        choices=['sherlock', 'sato', 'cl', 'tapex'])
    parser.add_argument("--benchmark", type=str, default='santos',
                        help=f"数据集名称，可选: {', '.join(DATASET_PATHS.keys())}")
    parser.add_argument("--augment_op", type=str, default="drop_col")
    parser.add_argument("--sample_meth", type=str, default="tfidf_entity")
    # matching is the type of matching: exact or bounds
    parser.add_argument("--matching", type=str, default='exact')
    parser.add_argument("--table_order", type=str, default="column")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--single_column",
                        dest="single_column", action="store_true")
    # For error analysis
    parser.add_argument("--bucket", type=int, default=0)
    parser.add_argument("--analysis", type=str, default='col')
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.6)
    # For Scalability experiments
    parser.add_argument("--scal", type=float, default=1.00)
    # mlflow tag
    parser.add_argument("--mlflow_tag", type=str, default=None)
    parser.add_argument("--enable_mlflow", action="store_true",
                        help="Enable MLflow logging for this search run")
    parser.add_argument("--datalake_vectors_path",
                        dest="datalake_vectors_path", type=str, default=None, required=True)
    parser.add_argument("--query_vectors_path",
                        dest="query_vectors_path", type=str, default=None, required=True)
    parser.add_argument("--metrics_path", type=str, default=None,
                        help="存储实验 metrics 的 CSV 路径")
    parser.add_argument("--gt_path", type=str, default=None,
                        help="自定义 Ground Truth 路径（覆盖 benchmark 默认推断，支持 .json/.pickle/.csv）")

    hp = parser.parse_args()
    mlflow_logger = get_mlflow_logger(hp.enable_mlflow)

    # mlflow logging
    for variable in ["encoder", "benchmark", "augment_op", "sample_meth", "matching", "table_order", "run_id", "single_column", "K", "threshold", "scal"]:
        mlflow_logger.log_param(variable, getattr(hp, variable))

    if hp.mlflow_tag:
        mlflow_logger.set_tag("tag", hp.mlflow_tag)

    dataFolder = hp.benchmark

    # 使用命令行参数中的向量路径
    query_path = hp.query_vectors_path
    table_path = hp.datalake_vectors_path

    print(f"[test_naive_search] Benchmark: {dataFolder}")
    print(f"[test_naive_search] Query vectors path: {query_path}")
    print(f"[test_naive_search] Datalake vectors path: {table_path}")

    # Load the query file
    qfile = open(query_path, "rb")
    queries = pickle.load(qfile)
    qfile.close()
    print(f"[test_naive_search] Number of queries: {len(queries)}")

    # Call NaiveSearcher, which has linear search and bounds search, from naive_search.py
    searcher = NaiveSearcher(table_path, hp.scal)
    returnedResults = {}
    start_time = time.time()

    queries.sort(key=lambda x: x[0])
    query_times = []
    qCount = 0

    for query in queries:
        qCount += 1
        if qCount % 10 == 0:
            print(
                f"Processing query {qCount} of {len(queries)} total queries.")

        query_start_time = time.time()
        try:
            if hp.matching == 'exact':
                qres = searcher.topk(hp.encoder, query, hp.K,
                                     threshold=hp.threshold)
            else:  # Bounds matching
                qres = searcher.topk_bounds(
                    hp.encoder, query, hp.K, threshold=hp.threshold)
        except Exception as e:
            print(f"\nQuery '{query[0]}' failed, skipping: {e}")
            query_times.append(time.time() - query_start_time)
            continue
        res = []
        for tpl in qres:
            tmp = (tpl[0], tpl[1])
            res.append(tmp)
        returnedResults[query[0]] = [r[1] for r in res]
        query_times.append(time.time() - query_start_time)

    print(
        f"Average QUERY TIME: {sum(query_times)/len(query_times):.4f} seconds")
    print(f"10th percentile: {np.percentile(query_times, 10):.4f}, "
          f"90th percentile: {np.percentile(query_times, 90):.4f}")
    print(f"--- Total Query Time: {time.time() - start_time:.2f} seconds ---")

    # 确定 ground truth 路径：命令行 --gt_path 优先，否则根据 benchmark 推断
    try:
        if hp.gt_path:
            gtPath = hp.gt_path
        else:
            gtPath = get_ground_truth_path(hp.benchmark)
        print(f"[test_naive_search] Ground truth path: {gtPath}")

        # 加载 ground truth
        if gtPath.endswith('.pickle') or gtPath.endswith('.pkl'):
            groundtruth = loadDictionaryFromPickleFile(gtPath)
        elif gtPath.endswith('.json'):
            with open(gtPath, mode='r', encoding='utf-8-sig') as f:
                groundtruth = json.load(f)
        elif gtPath.endswith('.csv'):
            # CSV 格式: 每行包含 query_table 和 candidate_table 列
            groundtruth = {}
            with open(gtPath, mode='r', encoding='utf-8-sig') as f:
                reader = csv_module.DictReader(f)
                for row in reader:
                    qt = row.get('query_table', '').strip()
                    ct = row.get('candidate_table', '').strip()
                    if qt and ct:
                        groundtruth.setdefault(qt, []).append(ct)
        else:
            raise ValueError(f"Unsupported ground truth file format: {gtPath}")

        # 根据数据集设置 k_range
        if 'santos' in hp.benchmark:
            k_range = 1
        else:
            k_range = 10

        print(f"[test_naive_search] Calculating effectiveness scores...")
        calcMetrics(hp.K, k_range, returnedResults, gt=groundtruth,
                    output_csv=hp.metrics_path,
                    record=hp.enable_mlflow,
                    mlflow_logger=mlflow_logger)

        if hp.metrics_path:
            print(f"[test_naive_search] Metrics saved to: {hp.metrics_path}")

    except ValueError as e:
        print(f"[test_naive_search] Warning: {e}")
        print(
            f"[test_naive_search] No ground truth available for benchmark '{hp.benchmark}'")
