from HYPERPARAMETERS import get_query_path, get_datalake_path
import json
import os
import pickle
import time
from typing import List

import numpy
import pandas as pd
import torch
from table_bert import Table, Column, TableBertModel
from torch import nn
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import logging as log
import argparse
import importlib.util
import sys
from utils import get_basename, list_files_by_extension
from add_profilling import read_table, get_table_profilling, print_noise_stats

# 导入 HYPERPARAMETERS 配置
root_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# root_dir = os.path.dirname(os.path.dirname(
#     os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# utils_path = os.path.join(root_dir, 'utils.py')
# add_profilling_utils_path = os.path.join(root_dir, 'add_profilling.py')
# # 动态导入
# spec = importlib.util.spec_from_file_location("root_utils", utils_path)
# root_utils = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(root_utils)
# list_files_by_extension = root_utils.list_files_by_extension
# get_basename = root_utils.get_basename
# spec = importlib.util.spec_from_file_location(
#     "add_profilling_utils", add_profilling_utils_path)
# add_profilling_utils = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(add_profilling_utils)
# read_table = add_profilling_utils.read_table
# get_table_profilling = add_profilling_utils.get_table_profilling

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str,
                    default="./model/tabert_base_k3/model.bin")
parser.add_argument("--benchmark", type=str, default='SG')
parser.add_argument("--file_type", type=str, default='.csv')
parser.add_argument("--profilling_path", type=str, default=None)
parser.add_argument("--sample_rows", type=int, default=1,
                    help="每张表读取的行数（0=仅header，N=N行）")
parser.add_argument("--noise_prob", type=float, default=0.0,
                    help="应用随机列增强噪声的概率：0.0=不加噪声，0.1=10%%概率，以此类推")
parser.add_argument("--table_mapper", action='store_true',
                    help="是否使用table_mapper")
parser.add_argument("--mask_header", type=str, default=None,
                    help="是否擦除header,以及用什么字符擦除")
parser.add_argument("--mask_tablename", type=str, default=None,
                    help="是否擦除tablename,以及用什么字符擦除")
parser.add_argument("--shuffle", action='store_true',
                    help="是否shuffle数据")
parser.add_argument("--shuffle_columns", action='store_true',
                    help="是否shuffle列")
parser.add_argument('--result_dir', type=str, default=None, help="result_dir")
hp = parser.parse_args()


class OurTable:
    def __init__(self, id, header, data, context):
        self.id: str = id
        self.header: List[str] = header
        self.data: List[List[str]] = data
        self.context: str = context

    def __str__(self):  # for debug
        heading = "\t|||\t".join(self.header)
        body = ''
        for row in self.data:
            body += "\t|||\t".join(row) + '\n'
        return (f"______________________________________\n"
                f"Table ID:{self.id}\n"
                f"Context:{self.context}\n"
                f"{heading}\n"
                f"_____\n"
                f"{body}")


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Qmodel = BertModel.from_pretrained(BERT_MODEL)
        self.Tmodel = TableBertModel.from_pretrained(
            TABERT_MODEL_PATH
        )

    def forward(self, tp_table, tp_context, tn_table=None, tn_context=None):
        context_encoding, column_encoding, info = self.Tmodel.encode(
            contexts=[tp_context],
            tables=[tp_table]
        )
        return column_encoding


def build_dataset_predict(tablePosList, bertTokenizer, _tabertModel):
    tp_table_list = []
    tp_context_list = []
    for tp in tablePosList:
        column_list = []
        head_list = tp.header
        value_1 = tp.data[0] if len(tp.data) >= 1 else [''] * len(tp.header)
        # print(type(value_1))
        for i in range(len(head_list)):
            column_list.append(
                Column(head_list[i].strip() if not hp.mask_header else hp.mask_header, 'text', value_1[i]))
        table_p = Table(
            id=tp.id,
            header=column_list,
            data=tp.data if len(tp.data) >= 1 else [['']*len(tp.header)]
        ).tokenize(_tabertModel.tokenizer)
        tp_table_list.append(table_p)
        tp_context_list.append(_tabertModel.tokenizer.tokenize(tp.context))
        # tp_context_list.append(_tabertModel.tokenizer.tokenize("###"))
    return list(zip(tp_table_list, tp_context_list))


def get_now() -> str:
    now = time.localtime()
    return "%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)


def csv_to_list(csv_file, table_name, profilling_path=None, sample_rows=1, noise_prob=0.0):
    # Read CSV file and convert to DataFrame
    table_pos_list = []
    data_source = read_table(csv_file, profilling_path,
                             sample_rows=sample_rows, noise_prob=noise_prob)
    if hp.shuffle_columns:
        data_source = data_source.sample(
            frac=1, axis=1, random_state=42).reset_index(drop=True)
    column_names = data_source.columns.tolist()
    data = data_source.values.tolist()
    num_rows, num_cols = data_source.values.shape[0], data_source.values.shape[1]
    profile = get_table_profilling(csv_file, profilling_path) if profilling_path else None
    if profile:
        for index in range(len(column_names)):
            col = column_names[index]
            if col in profile.keys():
                col_profile = profile[col]
                description_parts = []
                for key, value in col_profile.items():
                    if key != "__type__":
                        description_parts.append(f"{key}: {value}")
                description = " | ".join(description_parts)
                column_names[index] = f"{col} ({description})"
        # print(f"Added profilling info for {csv_file}")
        # print(column_names)
    num_rows, num_cols = data_source.values.shape[0], data_source.values.shape[1]
    jsonStr = {
        "id": table_name,
        "header": list(column_names),
        "data": data,
        "context": table_name,
        "num_cols": num_cols,
        "num_rows": num_rows
    }
    id = jsonStr['id']
    header = jsonStr['header']
    data = jsonStr['data']
    context = jsonStr['context']
    row = jsonStr['num_rows']
    col = jsonStr['num_cols']

    # if col == 0:
    #     if DEBUG:
    #         print('col len error', col, id)
    # elif row == 0:
    #     if DEBUG:
    #         print('row len error', row, id)
    table_pos_list.append(OurTable(id, header, data, context))
    return table_pos_list


def convert_all_csv_to_jsonl(folder_path, output_path):
    # Get the paths of all CSV files in a folder
    file_paths = [os.path.join(folder_path, file) for file in os.listdir(
        folder_path) if file.endswith('.csv')]
    total_file_num = len(file_paths)
    current_index = 1

    # Iterate through each .csv file and convert it, writing the results to a .json file
    with open(output_path, 'w') as jsonl_file:
        for csv_file in file_paths:
            table_name = os.path.splitext(os.path.basename(csv_file))[0]
            print(table_name)
            json_data = csv_to_list(csv_file, table_name)
            json.dump(json_data, jsonl_file)
            jsonl_file.write('\n')
            print('当前进度' + str(current_index) + '/' + str(total_file_num))
            current_index = current_index + 1


def get_now_str() -> str:
    now = time.localtime()
    return "%04d_%02d_%02d_%02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)


dataFolder = hp.benchmark
numpy.set_printoptions(suppress=True)
TABERT_MODEL_PATH = hp.model_path
BERT_MODEL = 'bert-base-uncased'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 从 HYPERPARAMETERS 获取数据集路径配置
FILE_PATHs = {
    'query': get_query_path(hp.benchmark),
    'datalake': get_datalake_path(hp.benchmark)
}
RESULT_DIR = hp.result_dir if hp.result_dir else f"./results/{dataFolder}/"
os.makedirs(RESULT_DIR, exist_ok=True)
BATCH_SIZE = 20*1024

if hp.table_mapper:
    # 构建全局表名到无语义编码的映射
    # print("")
    print("Building global table name mapping...")
    global_table_name_mapping = {}
    for FILE_TYPE_KEY, FILE_PATH in FILE_PATHs.items():
        table_counter = 0
        file_paths = list_files_by_extension(FILE_PATH, hp.file_type)
        if hp.shuffle:
            numpy.random.shuffle(file_paths)
        for csv_file in file_paths:
            table_name = get_basename(csv_file)
            key = f"{FILE_TYPE_KEY}_{table_name}"
            if key not in global_table_name_mapping:
                global_table_name_mapping[key] = f"{FILE_TYPE_KEY}_{table_counter}"
                table_counter += 1

    # 保存映射关系到文件,这个最好固定路径，以后常用
    mapping_output_path = os.path.join(RESULT_DIR, 'table_name_mapping.json')
    os.makedirs(os.path.dirname(mapping_output_path), exist_ok=True)
    with open(mapping_output_path, 'w', encoding='utf-8') as f:
        json.dump(global_table_name_mapping, f,
                  indent=2, ensure_ascii=False)
    print(f"Table name mapping saved to {mapping_output_path}")
    print(f"Total tables mapped: {len(global_table_name_mapping)}")

for FILE_TYPE_KEY, FILE_PATH in FILE_PATHs.items():
    # FILE_PATH = './task/dataset_discovery/santos_small/datalake'
    FILE_TYPE = hp.file_type
    BATCH = 2
    # BERT Model Load & Create model
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    model = Model()
    model.to(device)
    model.eval()
    # read csv file
    file_paths = list_files_by_extension(FILE_PATH, FILE_TYPE)
    # Shuffle files
    if hp.shuffle:
        numpy.random.shuffle(file_paths)

    total_file_num = len(file_paths)
    skipnums = 0
    dataEmbeds = []
    trainDatasets = []
    for csv_file in file_paths:
        table_name = get_basename(csv_file)
        if hp.table_mapper:
            # 使用映射后的表名
            key = f"{FILE_TYPE_KEY}_{table_name}"
            table_name = global_table_name_mapping.get(key, table_name)
        tablePosList = csv_to_list(
            csv_file, table_name, hp.profilling_path, hp.sample_rows, hp.noise_prob)
        trainDataset = build_dataset_predict(
            tablePosList, tokenizer, model.Tmodel)
        trainDatasets.extend(trainDataset)

    for trainDataset in tqdm(trainDatasets, desc="Processing tables"):
        try:
            with torch.no_grad():  # Disable gradient calculation for inference
                tp_table, tp_context = trainDataset
                q_tp_embedding = model(tp_table, tp_context)
                table_name = tp_table.id
                dim0, dim1, _ = q_tp_embedding.shape
                for i in range(dim0):
                    a_cpu = q_tp_embedding[i].cpu()
                    item = a_cpu.numpy()
                    dataEmbeds.append((table_name, item))
        except Exception as e:
            skipnums += 1
            print("skip a table:", table_name,
                  "reason", e, "already", skipnums)

    temp_output_path = os.path.join(RESULT_DIR, f"{FILE_TYPE_KEY}.pkl")
    pickle.dump(dataEmbeds, open(temp_output_path, "wb"))

print_noise_stats()
