import pickle
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
import importlib
import csv
from datetime import datetime

# ==========================================
# 路径与动态导入设置
# ==========================================
root_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
utils_path = os.path.join(root_dir, 'utils.py')

spec = importlib.util.spec_from_file_location("root_utils", utils_path)
root_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(root_utils)

list_files_by_extension = root_utils.list_files_by_extension
get_basename = root_utils.get_basename
transform_dict = root_utils.transform_dict
loadDictionaryFromPickleFile = root_utils.loadDictionaryFromPickleFile
saveDictionaryAsPickleFile = root_utils.saveDictionaryAsPickleFile


# ==========================================
# Unionable 表发现评估 (表级别)
# ==========================================
def calcMetrics(max_k, k_range, resultFile, gt, output_csv=None, experiment_name=None):
    ''' Calculate and log the performance metrics: MAP, Precision@k, Recall@k, F1@k
    Args:
        max_k: the maximum K value
        k_range: step size for the K's to display in console
        resultFile: dictionary of results {table_name: [retrieved_tables]}
        gt: groundtruth dictionary or file path
        output_csv: (Optional) path to save the metrics to a CSV file (append mode)
        experiment_name: (Optional) name/identifier for this experiment run
    Return: MAP, P@max_k, R@max_k
    '''
    # 1. 加载并预处理数据
    if isinstance(gt, str):
        if gt.endswith('.pickle') or gt.endswith('.pkl'):
            groundtruth = loadDictionaryFromPickleFile(gt)
        elif gt.endswith('.json'):
            import json
            with open(gt, mode='r', encoding='utf-8-sig') as f:
                groundtruth = json.load(f)
        elif gt.endswith('.csv'):
            df_gt = pd.read_csv(gt)
            groundtruth = {}
            for _, row in df_gt.iterrows():
                query_table = get_basename(str(row['query_table'])).replace(
                    "query-", "").replace("datalake-", "")
                candidate_table = get_basename(str(row['candidate_table'])).replace(
                    "query-", "").replace("datalake-", "")
                if query_table not in groundtruth:
                    groundtruth[query_table] = []
                if candidate_table not in groundtruth[query_table]:
                    groundtruth[query_table].append(candidate_table)
        else:
            raise ValueError("Unsupported file format")
    else:
        groundtruth = gt

    groundtruth = transform_dict(groundtruth, get_basename)
    resultFile = transform_dict(resultFile, get_basename)
    resultFile = transform_dict(resultFile, lambda x: x.replace("query-", '').replace("datalake-", ''))

    print(f"Number of queries in groundtruth: {len(groundtruth)}")
    print(f"Number of queries in result: {len(resultFile)}")

    # 2. 初始化统计数组 (索引 0 不使用，从 1 到 max_k)
    global_tp_at_k = np.zeros(max_k + 1)
    global_fp_at_k = np.zeros(max_k + 1)
    sum_recall_at_k = np.zeros(max_k + 1)

    # 3. 遍历每个查询表进行计算
    valid_query_count = 0

    for table, retrieved_list in resultFile.items():
        if table not in groundtruth:
            continue

        valid_query_count += 1
        gt_set = set(groundtruth[table])
        gt_size = len(gt_set)

        # 截取前 max_k 个结果
        current_results = retrieved_list[:max_k]

        # 计算累积 TP
        hits = [1 if x in gt_set else 0 for x in current_results]
        if len(hits) < max_k:
            hits += [0] * (max_k - len(hits))

        cum_tp = np.cumsum(hits)

        # 更新全局统计量
        for k in range(1, max_k + 1):
            tp = cum_tp[k-1]
            fp = k - tp

            if gt_size > 0:
                sum_recall_at_k[k] += tp / gt_size

            global_tp_at_k[k] += tp
            global_fp_at_k[k] += fp

    # 4. 计算最终指标
    metrics_data = []
    precision_array = []
    recall_array = []

    for k in range(1, max_k + 1):
        denom = global_tp_at_k[k] + global_fp_at_k[k]
        precision = global_tp_at_k[k] / denom if denom > 0 else 0.0
        recall = sum_recall_at_k[k] / \
            valid_query_count if valid_query_count > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision +
                                         recall) if (precision + recall) > 0 else 0.0

        precision_array.append(precision)
        recall_array.append(recall)

        metrics_data.append({
            "experiment": experiment_name if experiment_name else "deepjoin_union_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
            "k": k,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    # 5. 输出到 CSV (追加模式)
    if output_csv:
        df = pd.DataFrame(metrics_data)
        output_dir = os.path.dirname(os.path.abspath(output_csv))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(output_csv):
            df.to_csv(output_csv, mode='a', header=False, index=False)
            print(f"Metrics appended to {output_csv}")
        else:
            df.to_csv(output_csv, index=False)
            print(f"Metrics saved to {output_csv}")

    # 6. 控制台输出摘要
    print("-" * 30)
    used_k = [k_range]
    if max_k > k_range:
        for i in range(k_range * 2, max_k + 1, k_range):
            used_k.append(i)

    for k in used_k:
        m = metrics_data[k - 1]
        print(
            f"K = {k}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f}")

    print("-" * 30)

    map_val = sum(precision_array) / max_k
    print(f"The mean average precision is: {map_val:.4f}")

    return map_val, precision_array[max_k-1], recall_array[max_k-1]


# ==========================================
# Joinable 表发现评估 (列级别)
# ==========================================
def evaluate_joinable(max_k, k_range, resultFile, gt, query_folder=None, datalake_folder=None,
                      output_csv=None, experiment_name=None):
    ''' Calculate and log the performance metrics for Joinable Table Discovery: MAP, Precision@k, Recall@k, F1@k
    This method evaluates column-level matches (column pairs) instead of table-level matches.

    Args:
        max_k: the maximum K value (e.g. 10, 60)
        k_range: step size for the K's up to max_k
        resultFile: dictionary {query_table: [(score, union_column, table_path), ...]}
        gt: groundtruth - dict or file path (CSV/pickle)
            CSV format columns: query_table, candidate_table, query_column, candidate_column
        query_folder: folder path to query CSV files (for reading column names)
        datalake_folder: folder path to datalake CSV files (for reading column names)
        output_csv: (Optional) path to save the metrics to a CSV file (append mode)
        experiment_name: (Optional) name/identifier for this experiment run
    Return: MAP, P@K, R@K
    '''
    # 1. 加载 Ground Truth
    if isinstance(gt, str):
        if gt.endswith('.pickle') or gt.endswith('.pkl'):
            groundtruth_raw = loadDictionaryFromPickleFile(gt)
            groundtruth = {}
            for query_table, pairs in groundtruth_raw.items():
                query_table_clean = get_basename(query_table).replace(
                    "query-", "").replace("datalake-", "")
                groundtruth[query_table_clean] = []
                for pair in pairs:
                    if isinstance(pair, tuple) and len(pair) >= 3:
                        cand_table_clean = get_basename(pair[0]).replace(
                            "query-", "").replace("datalake-", "")
                        groundtruth[query_table_clean].append(
                            (cand_table_clean, pair[1], pair[2]))
        elif gt.endswith('.csv'):
            df_gt = pd.read_csv(gt)
            groundtruth = {}
            for _, row in df_gt.iterrows():
                query_table = get_basename(str(row['query_table'])).replace(
                    "query-", "").replace("datalake-", "")
                candidate_table = get_basename(str(row['candidate_table'])).replace(
                    "query-", "").replace("datalake-", "")
                query_col = row['query_column']
                cand_col = row['candidate_column']
                if query_table not in groundtruth:
                    groundtruth[query_table] = []
                groundtruth[query_table].append(
                    (candidate_table, query_col, cand_col))
        else:
            raise ValueError("Unsupported file format for groundtruth")
    else:
        groundtruth = gt

    print(f"Number of queries in groundtruth: {len(groundtruth)}")
    print(f"Number of queries in results: {len(resultFile)}")

    # 2. 预处理：提取所有查询的列对结果
    all_result_pairs = {}
    for query_table in resultFile:
        query_table_clean = get_basename(query_table).replace(
            "query-", "").replace("datalake-", "")
        result_pairs = []

        for result_item in resultFile[query_table]:
            score, union_column, table_path = result_item
            candidate_table = get_basename(table_path).replace(
                "query-", "").replace("datalake-", "")

            query_csv_path = os.path.join(
                query_folder, query_table_clean + ".csv") if query_folder else ""
            cand_csv_path = os.path.join(
                datalake_folder, candidate_table + ".csv") if datalake_folder else ""

            query_cols = None
            cand_cols = None

            try:
                if query_csv_path and os.path.exists(query_csv_path):
                    with open(query_csv_path, 'r', encoding='utf-8') as f:
                        query_cols = next(csv.reader(f))
                if cand_csv_path and os.path.exists(cand_csv_path):
                    with open(cand_csv_path, 'r', encoding='utf-8') as f:
                        cand_cols = next(csv.reader(f))
            except (FileNotFoundError, StopIteration, IOError):
                pass

            for query_col_idx, cand_col_idx, sim in union_column:
                if query_cols and cand_cols and query_col_idx < len(query_cols) and cand_col_idx < len(cand_cols):
                    result_pairs.append(
                        (candidate_table, query_cols[query_col_idx], cand_cols[cand_col_idx]))
                else:
                    result_pairs.append(
                        (candidate_table, str(query_col_idx), str(cand_col_idx)))

        all_result_pairs[query_table_clean] = result_pairs

    # 3. 初始化统计数组
    global_tp_at_k = np.zeros(max_k + 1)
    global_fp_at_k = np.zeros(max_k + 1)
    sum_recall_at_k = np.zeros(max_k + 1)
    valid_query_count = 0

    # 4. 遍历每个查询计算指标
    for query_table_clean, result_pairs in all_result_pairs.items():
        if query_table_clean not in groundtruth:
            continue

        valid_query_count += 1
        gt_pairs = set(groundtruth[query_table_clean])
        gt_size = len(gt_pairs)

        # 截取前 max_k 个结果
        current_results = result_pairs[:max_k]

        # 计算累积 TP
        hits = [1 if pair in gt_pairs else 0 for pair in current_results]
        if len(hits) < max_k:
            hits += [0] * (max_k - len(hits))

        cum_tp = np.cumsum(hits)

        for k in range(1, max_k + 1):
            tp = cum_tp[k-1]
            fp = k - tp

            if gt_size > 0:
                sum_recall_at_k[k] += tp / gt_size

            global_tp_at_k[k] += tp
            global_fp_at_k[k] += fp

    # 5. 计算最终指标
    metrics_data = []
    precision_array = []
    recall_array = []

    for k in range(1, max_k + 1):
        denom = global_tp_at_k[k] + global_fp_at_k[k]
        precision = global_tp_at_k[k] / denom if denom > 0 else 0.0
        recall = sum_recall_at_k[k] / \
            valid_query_count if valid_query_count > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision +
                                         recall) if (precision + recall) > 0 else 0.0

        precision_array.append(precision)
        recall_array.append(recall)

        metrics_data.append({
            "experiment": experiment_name if experiment_name else "deepjoin_join_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
            "k": k,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    # 6. 输出到 CSV
    if output_csv:
        df = pd.DataFrame(metrics_data)
        output_dir = os.path.dirname(os.path.abspath(output_csv))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(output_csv):
            df.to_csv(output_csv, mode='a', header=False, index=False)
            print(f"Metrics appended to {output_csv}")
        else:
            df.to_csv(output_csv, index=False)
            print(f"Metrics saved to {output_csv}")

    # 7. 控制台输出摘要
    print("-" * 30)
    print("Joinable Table Discovery - Column-Level Evaluation")
    print("-" * 30)

    used_k = [k_range]
    if max_k > k_range:
        for i in range(k_range * 2, max_k + 1, k_range):
            used_k.append(i)

    for k in used_k:
        m = metrics_data[k - 1]
        print(
            f"K = {k}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f}")

    print("-" * 30)

    map_val = sum(precision_array) / max_k
    print(f"The mean average precision is: {map_val:.4f}")

    return map_val, precision_array[max_k-1], recall_array[max_k-1]


# ==========================================
# 工具函数
# ==========================================
def csv_to_json(csv_path, json_path=None, orient='records'):
    ''' Convert CSV metrics file to JSON format '''
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if json_path is None:
        json_path = csv_path.rsplit('.', 1)[0] + '.json'

    output_dir = os.path.dirname(os.path.abspath(json_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_json(json_path, orient=orient, indent=2)
    print(f"JSON file saved to {json_path}")
    return json_path


def load_metrics_as_dict(csv_path, group_by_experiment=True):
    ''' Load metrics from CSV and return as nested dictionary '''
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if not group_by_experiment:
        return df.to_dict('records')

    result = {}
    for exp_name in df['experiment'].unique():
        exp_df = df[df['experiment'] == exp_name]
        result[exp_name] = {}

        for _, row in exp_df.iterrows():
            k = int(row['k'])
            result[exp_name][k] = {
                'precision': float(row['precision']),
                'recall': float(row['recall']),
                'f1': float(row['f1'])
            }

    return result
