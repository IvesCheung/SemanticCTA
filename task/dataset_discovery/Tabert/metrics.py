import pickle
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
import importlib
import json
from datetime import datetime
# ==========================================
# 路径与动态导入设置 (保持不变)
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
# 核心计算函数
# ==========================================


def calcMetrics(max_k, k_range, resultFile, gt, output_csv=None, experiment_name=None):
    ''' Calculate and log the performance metrics: MAP, Precision@k, Recall@k, F1@k
    Args:
        max_k: the maximum K value
        k_range: step size for the K's to display in console
        resultFile: dictionary of results {table_name: [retrieved_tables]}
        gtPath: file path to the groundtruth
        output_csv: (Optional) path to save the metrics to a CSV file (append mode)
        experiment_name: (Optional) name/identifier for this experiment run
    Return: MAP, P@max_k, R@max_k
    '''
    # 1. 加载并预处理数据
    # if gtPath.rsplit('.')[-1] == 'pickle':
    #     groundtruth = loadDictionaryFromPickleFile(gtPath)
    # elif gtPath.rsplit('.')[-1] == 'json':
    #     import json
    #     with open(gtPath, mode='r', encoding='utf-8-sig') as f:
    #         groundtruth = json.load(f)
    # else:
    #     raise ValueError("No support file")
    groundtruth = transform_dict(gt, get_basename)
    resultFile = transform_dict(resultFile, get_basename)
    import json
    # json.dump(groundtruth, open("./results/groundtruth.json", mode='w',
    #           encoding='utf-8-sig'), ensure_ascii=False, indent=2)
    # json.dump(resultFile, open("./results/result.json", mode='w',
    #                            encoding='utf-8-sig'), ensure_ascii=False, indent=2)
    print(f"Number of queries in groundtruth: {len(groundtruth)}")
    print(f"Number of queries in result: {len(resultFile)}")

    # 2. 初始化统计数组 (索引 0 不使用，从 1 到 max_k)
    # Precision 相关
    global_tp_at_k = np.zeros(max_k + 1)
    global_fp_at_k = np.zeros(max_k + 1)

    # Recall 相关
    sum_recall_at_k = np.zeros(max_k + 1)

    # 3. 遍历每个查询表进行计算 (O(N) 复杂度)
    valid_query_count = 0

    for table, retrieved_list in resultFile.items():
        if table not in groundtruth:
            continue

        valid_query_count += 1
        gt_set = set(groundtruth[table])
        gt_size = len(gt_set)

        # 截取前 max_k 个结果，不足的补齐或切片
        current_results = retrieved_list[:max_k]

        # 计算累积 TP (Cumulative True Positives)
        # hits[i] = 1 if 第 i 个结果是相关的 else 0
        # 这个是看看前 max_k 个结果中哪些是命中的
        hits = [1 if x in gt_set else 0 for x in current_results]

        # 如果结果少于 max_k，需要补 0 以便进行数组运算
        if len(hits) < max_k:
            hits += [0] * (max_k - len(hits))

        # cum_tp[k-1] 就是 @k 时的 TP 数量
        cum_tp = np.cumsum(hits)

        # 更新全局统计量
        for k in range(1, max_k + 1):
            tp = cum_tp[k-1]
            fp = k - tp

            # Recall 累加 (Macro-Average logic)
            if gt_size > 0:
                sum_recall_at_k[k] += tp / gt_size

            # Precision 累加 (保留原代码逻辑：只有当 GT 数量 >= k 时才计入 Precision 的 TP/FP)
            # if gt_size >= k:
            global_tp_at_k[k] += tp
            global_fp_at_k[k] += fp

    # 4. 计算最终指标
    metrics_data = []
    precision_array = []
    recall_array = []

    # 这里的 range 对应数组索引 1 到 max_k
    for k in range(1, max_k + 1):
        # Calculate Precision (Micro-average style based on filtered queries)
        denom = global_tp_at_k[k] + global_fp_at_k[k]
        if denom > 0:
            precision = global_tp_at_k[k] / denom
        else:
            precision = 0.0

        # Calculate Recall (Macro-average)
        if valid_query_count > 0:
            recall = sum_recall_at_k[k] / valid_query_count
        else:
            recall = 0.0

        # Calculate F1
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        precision_array.append(precision)
        recall_array.append(recall)

        metrics_data.append({
            "experiment": experiment_name if experiment_name else "default_"+datetime.now().strftime("%Y%m%d_%H%M%S"),
            "k": k,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    # 5. 输出到 CSV (追加模式)
    if output_csv:
        df = pd.DataFrame(metrics_data)
        # 确保目录存在
        output_dir = os.path.dirname(os.path.abspath(output_csv))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 如果文件存在则追加，否则创建新文件
        if os.path.exists(output_csv):
            df.to_csv(output_csv, mode='a', header=False, index=False)
            print(f"Metrics appended to {output_csv}")
        else:
            df.to_csv(output_csv, index=False)
            print(f"Metrics saved to {output_csv}")

    # 6. 控制台输出摘要
    print("-" * 30)

    # 确定要显示的 K 值
    used_k = [k_range]
    if max_k > k_range:
        for i in range(k_range * 2, max_k + 1, k_range):
            used_k.append(i)

    for k in used_k:
        idx = k - 1  # 数组索引
        # 从 metrics_data 获取数据更方便
        m = metrics_data[idx]
        print(
            f"K = {k}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f}")

    print("-" * 30)

    # 计算 MAP (Mean Average Precision)
    map_val = sum(precision_array) / max_k
    print(f"The mean average precision is: {map_val:.4f}")

    return map_val, precision_array[max_k-1], recall_array[max_k-1]


# ==========================================
# 工具函数：CSV 转 JSON
# ==========================================

def csv_to_json(csv_path, json_path=None, orient='records'):
    ''' Convert CSV metrics file to JSON format
    Args:
        csv_path: path to the CSV file
        json_path: (Optional) output JSON file path. If None, replaces .csv with .json
        orient: JSON orientation, options are:
                - 'records': list of dicts [{col1: val1, col2: val2}, ...]
                - 'split': dict with 'index', 'columns', 'data' keys
                - 'index': dict with index as keys
                - 'columns': dict with columns as keys
    Return: path to the saved JSON file
    '''
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # 读取 CSV
    df = pd.read_csv(csv_path)

    # 确定输出路径
    if json_path is None:
        json_path = csv_path.rsplit('.', 1)[0] + '.json'

    # 确保目录存在
    output_dir = os.path.dirname(os.path.abspath(json_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 转换并保存
    df.to_json(json_path, orient=orient, indent=2)
    print(f"JSON file saved to {json_path}")

    return json_path


def load_metrics_as_dict(csv_path, group_by_experiment=True):
    ''' Load metrics from CSV and return as nested dictionary for easier computation
    Args:
        csv_path: path to the CSV file
        group_by_experiment: if True, returns {experiment: {k: {metric: value}}}
                           if False, returns list of all records
    Return: dictionary or list of metrics
    '''
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if not group_by_experiment:
        return df.to_dict('records')

    # 按实验分组
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
