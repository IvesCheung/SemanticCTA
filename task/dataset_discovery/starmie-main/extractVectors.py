from add_profilling import read_table, print_noise_stats
from HYPERPARAMETERS import get_query_path, get_datalake_path, DATASET_PATHS
from sdd.pretrain import load_checkpoint, inference_on_tables
import torch
import pandas as pd
import numpy as np
import glob
import pickle
import time
import sys
import argparse
from tqdm import tqdm
import os
import json

# 添加项目根目录到 sys.path
root_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


torch.serialization.add_safe_globals([argparse.Namespace])


def extractVectors(dfs, model_path, ds_path=None, singleCol=False):
    ''' Get model inference on tables
    Args:
        dfs (list of DataFrames): tables to get model inference on
        model_path (str): path to the trained model
        ds_path (str, optional): datalake path passed to load_checkpoint for
            PretrainTableDataset initialization (e.g. from HYPERPARAMETERS)
        singleCol (boolean): is this for single column baseline
    Return:
        list of features for the dataframe
    '''
    ckpt = torch.load(model_path, map_location=torch.device('cuda'))
    model, trainset = load_checkpoint(ckpt, ds_path)
    return inference_on_tables(dfs, model, trainset, batch_size=1024)


def get_df(dataFolder, profilling_path=None, sample_rows=None, noise_prob=0.0):
    ''' Get the DataFrames of each table in a folder
    Args:
        dataFolder: filepath to the folder with all tables
        profilling_path: optional path to profilling JSON for column enrichment
        sample_rows: number of rows to read (None = all, 0 = header only, N = N rows)
        noise_prob: probability of applying random column augmentation (0.0 = no noise)
    Return:
        dataDfs (dict): key is the filename, value is the dataframe of that table
    '''
    dataFiles = glob.glob(dataFolder+"/*.csv")
    dataDFs = {}
    for file in dataFiles:
        nrows = sample_rows if sample_rows is not None else 1000
        df = read_table(file, profilling_path=profilling_path,
                        sample_rows=nrows, noise_prob=noise_prob)
        filename = file.split("/")[-1]
        dataDFs[filename] = df
    return dataDFs


def _add_column_info(table, csv_file_path, profilling_path):
    """Add column types as first row and descriptions as second row"""
    with open(profilling_path, 'r', encoding='utf-8') as f:
        descriptions_data = json.load(f)

    normalized_csv_path = os.path.normpath(csv_file_path)
    matching_key = None
    for json_path in descriptions_data.keys():
        if os.path.normpath(json_path) == normalized_csv_path or \
                os.path.basename(normalized_csv_path) == os.path.basename(json_path):
            matching_key = json_path
            break

    if matching_key and matching_key in descriptions_data.keys():
        column_info = descriptions_data[matching_key]
        type_row = {}
        description_row = {}
        for col in table.columns:
            if col in column_info.keys():
                col_data = column_info[col]
                if isinstance(col_data, dict):
                    type_row[col] = col_data.get("__type__", "unknown")
                    description_row[col] = ""
                    for related_col in col_data:
                        if related_col != "__type__" and col in related_col:
                            description = col_data.get(related_col, f"No description for {related_col}")
                            description_row[col] += f"{related_col}: {description} "
                else:
                    type_row[col] = "unknown"
                    description_row[col] = str(col_data)
            else:
                type_row[col] = "unknown"
                description_row[col] = f"No description available for {col}"
        type_df = pd.DataFrame([type_row])
        description_df = pd.DataFrame([description_row])
        table = pd.concat([type_df, description_df, table], ignore_index=True)
    else:
        print(f"No column information found for {csv_file_path}")
    return table


if __name__ == '__main__':
    ''' Get the model features by calling model inference from sdd/pretrain
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="santos",
                        help=f"数据集名称，可选: {', '.join(DATASET_PATHS.keys())}")
    # single-column mode without table context
    parser.add_argument("--single_column",
                        dest="single_column", action="store_true")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--table_order", type=str, default='column')
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--profilling_path", type=str, default=None,
                        help="profilling JSON 路径（可选）")
    parser.add_argument("--sample_rows", type=int, default=None,
                        help="每张表读取的行数（None=全部，0=仅header，N=N行）")
    parser.add_argument("--noise_prob", type=float, default=0.0,
                        help="应用随机列增强噪声的概率：0.0=不加噪声，0.1=10%%概率，以此类推")
    parser.add_argument("--datalake_vectors_path",
                        dest="datalake_vectors_path", type=str, default=None, required=True)
    parser.add_argument("--query_vectors_path",
                        dest="query_vectors_path", type=str, default=None, required=True)
    parser.add_argument("--model_path",
                        dest="model_path", type=str, default=None, required=True,
                        help="训练好的模型路径")

    hp = parser.parse_args()

    dataFolder = hp.benchmark
    isSingleCol = hp.single_column
    profilling_path = hp.profilling_path
    model_path = hp.model_path

    # 从 HYPERPARAMETERS 获取数据路径
    query_data_path = get_query_path(dataFolder)
    datalake_data_path = get_datalake_path(dataFolder)

    print(f"[extractVectors] Benchmark: {dataFolder}")
    print(f"[extractVectors] Query path: {query_data_path}")
    print(f"[extractVectors] Datalake path: {datalake_data_path}")
    print(f"[extractVectors] Model path: {model_path}")

    # 定义要处理的目录: (数据路径, 输出向量路径, 目录类型)
    data_dirs = [
        (query_data_path, hp.query_vectors_path, 'query'),
        (datalake_data_path, hp.datalake_vectors_path, 'datalake'),
    ]

    inference_times = 0

    for data_path, output_path, dir_type in data_dirs:
        print(f"\n//==== Processing {dir_type}")
        print(f"Data folder: {data_path}")

        dfs = get_df(data_path, profilling_path, hp.sample_rows, hp.noise_prob)
        print(f"Number of tables: {len(dfs)}")

        dataEmbeds = []

        # Extract model vectors, and measure model inference time
        start_time = time.time()
        cl_features = extractVectors(
            list(dfs.values()), model_path, ds_path=datalake_data_path, singleCol=isSingleCol)
        inference_times += time.time() - start_time
        print(
            f"{dataFolder} {dir_type} inference time: {time.time() - start_time:.2f} seconds")

        for i, file in enumerate(dfs):
            # get features for this file / dataset
            cl_features_file = np.array(cl_features[i])
            dataEmbeds.append((file, cl_features_file))

        # 保存向量
        if hp.save_model:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pickle.dump(dataEmbeds, open(output_path, "wb"))
            print(f"Vectors saved to: {output_path}")

    print(f"\n[extractVectors] Benchmark: {dataFolder}")
    print(f"--- Total Inference Time: {inference_times:.2f} seconds ---")
    print_noise_stats()
