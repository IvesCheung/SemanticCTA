"""
DeepJoin Query Script

支持命令行参数进行查询，可被 DeepJoinDDTask 调用。
"""
import pickle
import argparse
import time
import numpy as np
from hnsw_search import HNSWSearcher
import csv
import sys
import os
from tqdm import tqdm
from metrics import calcMetrics, evaluate_joinable


def run_query(encoder, table_path, query_path, index_path, scal, K, N, threshold,
              output_path, gt_path, query_folder, datalake_folder, metrics_path,
              store_result=False, experiment_name=None, evaluation_mode="column"):
    """
    执行查询并评估结果

    Args:
        encoder: 编码器类型 ('cl', 'sherlock', 'starmie', 'tapex')
        table_path: datalake 嵌入文件路径
        query_path: query 嵌入文件路径
        index_path: HNSW 索引路径（可选）
        scal: 数据湖规模缩放因子
        K: 返回 Top-K 结果
        N: 从索引中检索的列数
        threshold: 相似度阈值
        output_path: 结果输出路径
        gt_path: Ground Truth 文件路径
        query_folder: 查询表文件夹路径
        datalake_folder: 数据湖表文件夹路径
        metrics_path: 评估指标保存路径
        store_result: 是否存储详细结果
        experiment_name: 实验名称
        evaluation_mode: 评估粒度，"column" 或 "table"
    """
    # Call HNSWSearcher from hnsw_search.py
    searcher = HNSWSearcher(table_path, index_path, scal)
    print(f"table_path: {table_path}")
    print(f"query_path: {query_path}")

    queries = pickle.load(open(query_path, "rb"))

    start_time = time.time()
    returnedResults = {}
    avgNumResults = []
    query_times = []

    dic = {}
    for qu in queries:
        str_q = qu[0].split('__')[0]
        if str_q not in dic:
            dic[str_q] = []
        dic[str_q].append(qu)

    # Remove old output file if exists
    if output_path and os.path.exists(output_path):
        os.remove(output_path)

    # Store results with column pairs for joinable evaluation
    returnedResultsWithColumns = {}

    # Open CSV file for writing all results (if store_result is True)
    if store_result and output_path:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8', newline='') as file_writer:
            csv_writer = csv.writer(file_writer, delimiter=',')
            # Write header
            csv_writer.writerow(['query_table', 'candidate_table',
                                'query_column', 'candidate_column', 'similarity_score'])

            for q in tqdm(queries, desc="Processing queries"):
                query_start_time = time.time()
                try:
                    res, scoreLength = searcher.topk(
                        encoder, q, K, N=N, threshold=threshold)
                except Exception as e:
                    print(f"\nQuery '{q[0]}' failed, skipping: {e}")
                    query_times.append(time.time() - query_start_time)
                    continue
                # Store table-level results (for backward compatibility)
                returnedResults[q[0]] = [r[2] for r in res]
                # Store column-level results (for joinable evaluation)
                returnedResultsWithColumns[q[0]] = res
                avgNumResults.append(scoreLength)
                query_times.append(time.time() - query_start_time)

                # Write column pairs to CSV for this query
                for i in range(0, len(res)):
                    candidate_table = res[i][2].split('/')[-1]

                    # j = (query_col_idx, cand_col_idx, score)
                    for j in res[i][1]:
                        out_data = []
                        out_data.append(q[0])
                        out_data.append(candidate_table)

                        # Read query column name
                        if query_folder:
                            query_file = os.path.join(
                                query_folder, q[0].replace("query-", "").replace("datalake-", ""))
                            try:
                                with open(query_file, 'r', encoding='utf-8') as csvfile:
                                    csvreader = csv.reader(csvfile)
                                    first_row = next(csvreader)
                                    out_data.append(first_row[j[0]])
                            except (FileNotFoundError, IndexError, StopIteration, IOError):
                                out_data.append(f"col_{j[0]}")
                        else:
                            out_data.append(f"col_{j[0]}")

                        # Read candidate column name
                        if datalake_folder:
                            datalake_file = os.path.join(
                                datalake_folder, res[i][2].replace("query-", "").replace("datalake-", ""))
                            try:
                                with open(datalake_file, 'r', encoding='utf-8') as csvfile:
                                    csvreader = csv.reader(csvfile)
                                    first_row = next(csvreader)
                                    out_data.append(first_row[j[1]])
                            except (FileNotFoundError, IndexError, StopIteration, IOError):
                                out_data.append(f"col_{j[1]}")
                        else:
                            out_data.append(f"col_{j[1]}")

                        # Add similarity score
                        out_data.append(j[2])
                        csv_writer.writerow(out_data)
    else:
        # Just run queries without storing detailed results
        for q in tqdm(queries, desc="Processing queries"):
            query_start_time = time.time()
            try:
                res, scoreLength = searcher.topk(
                    encoder, q, K, N=N, threshold=threshold)
            except Exception as e:
                print(f"\nQuery '{q[0]}' failed, skipping: {e}")
                query_times.append(time.time() - query_start_time)
                continue
            returnedResults[q[0]] = [r[2] for r in res]
            returnedResultsWithColumns[q[0]] = res
            avgNumResults.append(scoreLength)
            query_times.append(time.time() - query_start_time)

    total_time = time.time() - start_time
    print(f"\n--- Total Query Time: {total_time:.2f} seconds ---")
    print(f"--- Average Query Time: {np.mean(query_times)*1000:.2f} ms ---")
    print(f"--- Average Results per Query: {np.mean(avgNumResults):.2f} ---")

    evaluation_mode = (evaluation_mode or "column").lower()

    if gt_path:
        print("\n" + "=" * 80)
        if evaluation_mode == "table":
            print("Evaluating Joinable Table Discovery (Table-Level Retrieval)")
            print("=" * 80)
            calcMetrics(
                max_k=K,
                k_range=1,
                resultFile=returnedResults,
                gt=gt_path,
                output_csv=metrics_path,
                experiment_name=experiment_name
            )
        else:
            print("Evaluating Joinable Table Discovery (Column-Level Matching)")
            print("=" * 80)
            evaluate_joinable(
                max_k=K,
                k_range=1,
                resultFile=returnedResultsWithColumns,
                gt=gt_path,
                query_folder=query_folder,
                datalake_folder=datalake_folder,
                output_csv=metrics_path,
                experiment_name=experiment_name
            )
    else:
        print("\nWarning: No ground truth file provided, skipping evaluation.")

    return {
        "returnedResults": returnedResults,
        "returnedResultsWithColumns": returnedResultsWithColumns,
        "total_time": total_time,
        "avg_query_time": np.mean(query_times),
        "avg_results_per_query": np.mean(avgNumResults)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeepJoin Query Script")
    parser.add_argument("--encoder", type=str, default="cl",
                        choices=['sherlock', 'starmie', 'cl', 'tapex'],
                        help="Encoder type")
    parser.add_argument("--benchmark", type=str, default='test',
                        help="Benchmark name")
    parser.add_argument("--run_id", type=int, default=0,
                        help="Run ID")
    parser.add_argument("--single_column", dest="single_column",
                        action="store_true", default=False,
                        help="Single column mode")
    parser.add_argument("--K", type=int, default=60,
                        help="Number of top results to return")
    parser.add_argument("--scal", type=float, default=1.0,
                        help="Scale factor for data lake size")
    parser.add_argument("--mlflow_tag", type=str, default=None,
                        help="MLflow tag")
    parser.add_argument("--N", type=int, default=10,
                        help="Number of columns to retrieve from index")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Similarity threshold")
    parser.add_argument("--evaluation_mode", type=str, default="column",
                        choices=["column", "table"],
                        help="Evaluation granularity: column-level or table-level")

    # Path arguments
    parser.add_argument("--table_path", type=str, required=True,
                        help="Path to datalake embeddings pickle file")
    parser.add_argument("--query_path", type=str, required=True,
                        help="Path to query embeddings pickle file")
    parser.add_argument("--index_path", type=str, default=None,
                        help="Path to HNSW index file (optional)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save output CSV file")
    parser.add_argument("--metrics_path", type=str, default=None,
                        help="Path to save metrics CSV file")

    # Ground truth and evaluation arguments
    parser.add_argument("--gt_path", type=str, default=None,
                        help="Path to ground truth file")
    parser.add_argument("--query_folder", type=str, default=None,
                        help="Path to query tables folder")
    parser.add_argument("--datalake_folder", type=str, default=None,
                        help="Path to datalake tables folder")

    # Control arguments
    parser.add_argument("--store_result", action="store_true", default=False,
                        help="Store detailed results to CSV")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name for metrics")

    hp = parser.parse_args()

    # Set default experiment name if not provided
    if hp.experiment_name is None:
        hp.experiment_name = f"deepjoin_{hp.benchmark}"

    # Run query
    results = run_query(
        encoder=hp.encoder,
        table_path=hp.table_path,
        query_path=hp.query_path,
        index_path=hp.index_path,
        scal=hp.scal,
        K=hp.K,
        N=hp.N,
        threshold=hp.threshold,
        output_path=hp.output_path,
        gt_path=hp.gt_path,
        query_folder=hp.query_folder,
        datalake_folder=hp.datalake_folder,
        metrics_path=hp.metrics_path,
        store_result=hp.store_result,
        experiment_name=hp.experiment_name,
        evaluation_mode=hp.evaluation_mode
    )

    print("\nQuery completed successfully!")
