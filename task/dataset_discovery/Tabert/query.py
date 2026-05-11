from HYPERPARAMETERS import get_ground_truth_path
import pickle
# import mlflow
import argparse
import time
import numpy as np
from hnsw_search import HNSWSearcher
# import tqdm.auto
import csv
import sys
import os
from tqdm import tqdm
import importlib
from metrics import calcMetrics
from utils import list_files_by_extension, get_basename, transform_dict, loadDictionaryFromPickleFile, saveDictionaryAsPickleFile

# 导入 HYPERPARAMETERS 配置
root_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# root_dir = os.path.dirname(os.path.dirname(
#     os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# utils_path = os.path.join(root_dir, 'utils.py')

# spec = importlib.util.spec_from_file_location("root_utils", utils_path)
# root_utils = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(root_utils)

# list_files_by_extension = root_utils.list_files_by_extension
# get_basename = root_utils.get_basename
# transform_dict = root_utils.transform_dict
# loadDictionaryFromPickleFile = root_utils.loadDictionaryFromPickleFile
# saveDictionaryAsPickleFile = root_utils.saveDictionaryAsPickleFile

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="cl",
                        choices=['sherlock', 'tabert', 'cl', 'tapex'])
    parser.add_argument("--benchmark", type=str, default='test')
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--single_column", dest="single_column",
                        action="store_true", default=False)
    parser.add_argument("--K", type=int, default=60)
    parser.add_argument("--scal", type=float, default=1.0)
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--model_path", type=str,
                        default="./model/tabert_base_k3/model.bin")
    parser.add_argument("--file_type", type=str, default='.csv')

    parser.add_argument("--table_path", type=str,
                        default="./results/testdatalake.pkl")
    parser.add_argument("--query_path", type=str,
                        default="results/testquery.pkl")
    parser.add_argument("--output_path", type=str, default="./results/SG/tabert_result.jsonl")
    parser.add_argument("--mapping_file", type=str, default=None,
                        help="表名映射文件路径")
    parser.add_argument("--metrics_path", type=str, default=None,
                        help="存储实验metrics的路径")
    parser.add_argument("--gt_path", type=str, default=None,
                        help="Ground Truth 文件路径 (覆盖 benchmark 默认路径，支持 json/pickle/csv)")
    parser.add_argument("--store_result", action='store_true', help="是否记录结果到文件")
    hp = parser.parse_args()

    encoder = hp.encoder
    singleCol = False
    dataFolder = hp.benchmark
    K = hp.K
    threshold = hp.threshold
    N = hp.N

    # Set augmentation operators, sampling methods, K, and threshold values according to the benchmark
    if 'santos' in dataFolder or dataFolder == 'opendata':
        sampAug = "drop_cell_alphaHead"

    elif dataFolder == 'opendata' or dataFolder == 'test':
        sampAug = "drop_cell_alphaHead"
    singSampAug = "drop_col,sample_row_head"

    table_id = hp.run_id
    # table path 就是 datalake的pkl
    table_path = hp.table_path
    query_path = hp.query_path
    # index_path = "/data/final_result/tabert/"+dataFolder + \
    #     "/hnsw_opendata_small_"+str(table_id)+"_"+str(hp.scal)+".bin"

    # Call HNSWSearcher from hnsw_search.py
    searcher = HNSWSearcher(table_path, None, hp.scal)
    print(f"table_path: {table_path}")
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
            # dic[str_q].append(qu)
        # else:
            # continue
        dic[str_q].append(qu)

for q in tqdm(queries):
    query_start_time = time.time()
    try:
        res, scoreLength = searcher.topk(
            encoder, q, K, N=N, threshold=threshold)
        # res: [(score, [(col1, col2, sim_score)], table_name)]
        returnedResults[q[0]] = [r[2] for r in res]
        avgNumResults.append(scoreLength)
    except Exception as e:
        print(f"\nQuery '{q[0]}' failed, skipping: {e}")
    query_times.append(time.time() - query_start_time)

table_name_mapping = None
reverse_mapping = None  # 添加反向映射
if hp.mapping_file:
    import json
    if os.path.exists(hp.mapping_file):
        with open(hp.mapping_file, 'r', encoding='utf-8') as f:
            table_name_mapping = json.load(f)
        # 创建反向映射：TABLE_000001 -> 原始表名
        reverse_mapping = {v: k for k, v in table_name_mapping.items()}
        print(f"Loaded table name mapping from {hp.mapping_file}")
        print(f"Total mappings: {len(table_name_mapping)}")
    else:
        print(f"Warning: Mapping file {hp.mapping_file} not found!")

# 从 HYPERPARAMETERS 获取 ground truth 路径（支持 --gt_path 覆盖）
if hp.gt_path:
    gtPath = hp.gt_path
    print(f"Using custom ground truth path: {gtPath}")
else:
    gtPath = get_ground_truth_path(hp.benchmark)

if gtPath.rsplit('.')[-1] == 'pickle':
    groundtruth = loadDictionaryFromPickleFile(gtPath)
elif gtPath.rsplit('.')[-1] == 'json':
    import json
    with open(gtPath, mode='r', encoding='utf-8-sig') as f:
        groundtruth = json.load(f)
elif gtPath.rsplit('.')[-1] == 'csv':
    import pandas as pd
    df_gt = pd.read_csv(gtPath)
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
    raise ValueError(f"Unsupported ground truth file format: {gtPath}")
groundtruth = transform_dict(groundtruth, get_basename)

# 对查询结果进行反向映射转换（从编码映射回原始表名）
if reverse_mapping:
    print("Reverse mapping query results to original table names...")

    def strip_prefix(name):
        # 去除 index.py 中添加的前缀，以便与 groundtruth 匹配
        for prefix in ['query_', 'datalake_']:
            if name.startswith(prefix):
                return name[len(prefix):]
        return name

    mapped_returnedResults = {}
    for query_name, result_tables in returnedResults.items():
        # 映射查询名（从编码到原始名），并去除前缀
        original_query_name = reverse_mapping.get(query_name, query_name)
        original_query_name = strip_prefix(original_query_name)

        # 映射结果表名列表（从编码到原始名），并去除前缀
        original_result_tables = []
        for result_table in result_tables:
            original_table = reverse_mapping.get(result_table, result_table)
            original_table = strip_prefix(original_table)
            original_result_tables.append(original_table)

        mapped_returnedResults[original_query_name] = original_result_tables

    returnedResults = mapped_returnedResults
    print(
        f"Query results mapping completed. Total queries: {len(returnedResults)}")

returnedResults = transform_dict(returnedResults, get_basename)

path_output = hp.output_path
if hp.store_result:
    print(f"Storing results to {path_output}")
    os.makedirs(os.path.dirname(path_output), exist_ok=True)
    # Store results in JSONL format
    output_file = path_output
    with open(output_file, 'w', encoding='utf-8') as f:
        for query_name, result_tables in returnedResults.items():
            json_line = json.dumps({
                'query': query_name,
                'results': result_tables
            }, ensure_ascii=False)
            f.write(json_line + '\n')
    print(f"Results saved to {output_file}")

print("Calculating metrics..., the output path is:", hp.metrics_path)
calcMetrics(hp.N, 1, returnedResults, groundtruth,
            output_csv=hp.metrics_path)
"""
    with open(path_output, 'a', encoding='utf-8', newline='') as file_writer:
        # 遍历返回的结果
        for i in range(0, len(res)):
            out_data = []
            out_data.append(q[0])
            # 第一行是query 表名
            out_data.append(res[i][2].split('/')[-1])
            # 第二行是datalake 表名
            for j in res[i][1]:
                # j是 (col1, col2, sim_score)
                out_data = []
                out_data.append(q[0])
                out_data.append(res[i][2].split('/')[-1])
                query_path = "./task/dataset_discovery/santos_small/query/" + \
                    q[0]+'.csv'
                with open(query_path, 'r') as csvfile:
                    csvreader = csv.reader(csvfile)
                    first_row = next(csvreader)
                    out_data.append(first_row[j[0]])
                if 'query' in res[i][2]:
                    path = "./task/dataset_discovery/santos_small/datalake/" + \
                        res[i][2] + '.csv'
                    with open(path, 'r') as csvfile:
                        csvreader = csv.reader(csvfile)
                        first_row = next(csvreader)
                        out_data.append(first_row[j[1]])
                else:
                    path = "./task/dataset_discovery/santos_small/datalake/" + \
                        res[i][2]+'.csv'
                    with open(path, 'r') as csvfile:
                        csvreader = csv.reader(csvfile)
                        first_row = next(csvreader)
                        out_data.append(first_row[j[1]])

                out_data.append(j[2])
                w = csv.writer(file_writer, delimiter=',')
                w.writerow(out_data)
"""

# for q in queries:
#     if len(returnedResults[q[0]]) < K:
#         print(returnedResults[q[0]])
