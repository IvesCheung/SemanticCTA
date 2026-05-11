from utils import get_basename, transform_dict
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
import json
from datetime import datetime
import sys

# 添加项目根目录到 sys.path
root_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


def loadDictionaryFromPickleFile(dictionaryPath):
    ''' Load the pickle file as a dictionary
    Args:
        dictionaryPath: path to the pickle file
    Return: dictionary from the pickle file
    '''
    filePointer = open(dictionaryPath, 'rb')
    dictionary = pickle.load(filePointer)
    filePointer.close()
    return dictionary


def saveDictionaryAsPickleFile(dictionary, dictionaryPath):
    ''' Save dictionary as a pickle file
    Args:
        dictionary to be saved
        dictionaryPath: filepath to which the dictionary will be saved
    '''
    filePointer = open(dictionaryPath, 'wb')
    pickle.dump(dictionary, filePointer, protocol=pickle.HIGHEST_PROTOCOL)
    filePointer.close()


def calcMetrics(max_k, k_range, resultFile, gt, output_csv=None,
                experiment_name=None, record=True, mlflow_logger=None):
    ''' Calculate and log the performance metrics: MAP, Precision@k, Recall@k, F1@k
    Args:
        max_k: the maximum K value (e.g. for SANTOS benchmark, max_k = 10. For TUS benchmark, max_k = 60)
        k_range: step size for the K's to display in console
        resultFile: dictionary of results {table_name: [retrieved_tables]}
        gt: ground truth dictionary (already loaded)
        output_csv: (Optional) path to save the metrics to a CSV file (append mode)
        experiment_name: (Optional) name/identifier for this experiment run
        record (boolean): to log in MLFlow or not
    Return: MAP, P@max_k, R@max_k
    '''
    # 1. 标准化键名（使用 basename）
    groundtruth = transform_dict(gt, get_basename)
    resultFile = transform_dict(resultFile, get_basename)

    print(f"Number of queries in groundtruth: {len(groundtruth)}")
    print(f"Number of queries in result: {len(resultFile)}")

    # 3. 初始化统计数组
    global_tp_at_k = np.zeros(max_k + 1)
    global_fp_at_k = np.zeros(max_k + 1)
    sum_recall_at_k = np.zeros(max_k + 1)

    # 4. 遍历每个查询表进行计算
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

        # 如果结果少于 max_k，需要补 0
        if len(hits) < max_k:
            hits += [0] * (max_k - len(hits))

        cum_tp = np.cumsum(hits)

        # 更新全局统计量
        for k in range(1, max_k + 1):
            tp = cum_tp[k-1]
            fp = k - tp

            # Recall 累加 (Macro-Average)
            if gt_size > 0:
                sum_recall_at_k[k] += tp / gt_size

            # Precision 累加
            global_tp_at_k[k] += tp
            global_fp_at_k[k] += fp

    # 5. 计算最终指标
    metrics_data = []
    precision_array = []
    recall_array = []

    for k in range(1, max_k + 1):
        # Calculate Precision
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
            "experiment": experiment_name if experiment_name else "starmie_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
            "k": k,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    # 6. 输出到 CSV (追加模式)
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

    # 7. 控制台输出摘要
    print("-" * 30)

    # 确定要显示的 K 值
    used_k = [k_range]
    if max_k > k_range:
        for i in range(k_range * 2, max_k + 1, k_range):
            used_k.append(i)

    for k in used_k:
        idx = k - 1
        m = metrics_data[idx]
        print(
            f"K = {k}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f}")

    print("-" * 30)

    # 计算 MAP (Mean Average Precision)
    map_val = sum(precision_array) / max_k
    print(f"The mean average precision is: {map_val:.4f}")

    # logging to mlflow
    if record and mlflow_logger is not None:
        mlflow_logger.log_metric("mean_avg_precision", map_val)
        mlflow_logger.log_metric("prec_k", precision_array[max_k-1])
        mlflow_logger.log_metric("recall_k", recall_array[max_k-1])

    return map_val, precision_array[max_k-1], recall_array[max_k-1]
