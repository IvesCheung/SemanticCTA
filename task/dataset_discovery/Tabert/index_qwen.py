"""
使用 Qwen Embedding 模型的索引构建脚本
完全替代 index.py，使用 Qwen embedding 代替 Tabert 进行列编码

输入输出格式与 index.py 保持一致，生成相同格式的 pickle 文件
可以直接用 query.py 进行查询
"""

from EmbeddingModel import EmbeddingModelFactory
import json
import os
import pickle
import time
from typing import List

import numpy
import pandas as pd
from tqdm import tqdm
import logging as log
import argparse
import importlib.util
import sys
# 导入通用工具
root_dir = os.getcwd()
sys.path.insert(0, root_dir)

try:
    from utils import list_files_by_extension, get_basename
    from add_profilling import read_table, get_table_profilling, encode_column, print_noise_stats
    from HYPERPARAMETERS import get_query_path, get_datalake_path
except ImportError:
    utils_path = os.path.join(root_dir, 'utils.py')
    add_profilling_utils_path = os.path.join(root_dir, 'add_profilling.py')
    exit(0)

# 导入 EmbeddingModel

# ===================== 命令行参数配置 =====================
parser = argparse.ArgumentParser()
parser.add_argument("--benchmark", type=str, default='SG',
                    help="基准数据集名称")
parser.add_argument("--file_type", type=str, default='.csv',
                    help="文件类型")
parser.add_argument("--profilling_path", type=str, default=None,
                    help="profilling 数据路径")
parser.add_argument("--sample_rows", type=int, default=1,
                    help="采样行数")
parser.add_argument("--noise_prob", type=float, default=0.0,
                    help="应用随机列增强噪声的概率：0.0=不加噪声，0.1=10%%概率，以此类推")
parser.add_argument("--qwen_model", type=str, default='qwen3-embedding-0.6b',
                    help="Qwen 模型名称")
parser.add_argument("--batch_size", type=int, default=256,
                    help="Qwen embedding 批次大小")
parser.add_argument("--table_mapper", action='store_true',
                    help="是否使用table_mapper")
parser.add_argument("--shuffle", action='store_true',
                    help="是否shuffle数据")
parser.add_argument("--mask_header", type=str, default=None,
                    help="是否擦除header,以及用什么字符擦除")
parser.add_argument("--mask_tablename", type=str, default=None,
                    help="是否擦除tablename,以及用什么字符擦除")
parser.add_argument("--shuffle_columns", action='store_true',
                    help="是否shuffle列")
parser.add_argument('--local_model_path', type=str, default=None,
                    help='Path to local trained model weights file (.pth).')
parser.add_argument('--base_model_path', type=str, default=None,
                    help='Base model path for initializing ContrastiveEmbeddingModel structure.')
parser.add_argument('--result_dir', type=str, default=None, help="result_dir")

hp = parser.parse_args()


class OurTable:
    """表的数据结构"""

    def __init__(self, id, header, data, context, profilling_data={}):
        self.id: str = id
        self.header: List[str] = header
        self.data: List[List[str]] = data
        self.context: str = context
        self.profilling_data = profilling_data

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


def get_now_str() -> str:
    """获取当前时间字符串"""
    now = time.localtime()
    return "%04d_%02d_%02d_%02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)


def csv_to_table(csv_file, table_name, profilling_path=None, sample_rows=1):
    """将 CSV 文件转换为 OurTable 对象列表

    Args:
        csv_file: CSV 文件路径
        table_name: 表名
        profilling_path: profilling 数据路径
        sample_rows: 采样行数

    Returns:
        OurTable 对象列表
    """
    # table_pos_list = []
    # 对于qwen系列而言, profilling数据的增加, 应该利用专门的结构才可以,所以这里直接加载源表,这是最理想的
    data_source = read_table(
        csv_file, sample_rows=sample_rows, noise_prob=hp.noise_prob)

    # 如果需要，对列进行 shuffle
    if hp.shuffle_columns:
        data_source = data_source.sample(
            frac=1, axis=1, random_state=42).reset_index(drop=True)

    column_names = data_source.columns.tolist()
    data = data_source.values.tolist()
    num_rows, num_cols = data_source.values.shape[0], data_source.values.shape[1]

    # 添加 profilling 信息
    profile = get_table_profilling(
        csv_file, profilling_path) if profilling_path else {}

    # if profile:
    #     for index in range(len(column_names)):
    #         col = column_names[index]
    #         if col in profile.keys():
    #             col_profile = profile[col]
    #             description_parts = []
    #             for key, value in col_profile.items():
    #                 if key != "__type__":
    #                     description_parts.append(f"{key}: {value}")
    #             description = " | ".join(description_parts)
    #             column_names[index] = f"{col} ({description})"

    jsonStr = {
        "id": table_name,
        "header": list(column_names),
        "data": data,
        "context": table_name,
        "num_cols": num_cols,
        "num_rows": num_rows
    }

    if hp.mask_header:
        print(f"Masking header of table {table_name} with '{hp.mask_header}'")

    id = jsonStr['id']
    header = jsonStr['header']
    data = jsonStr['data']
    context = jsonStr['context']
    row = jsonStr['num_rows']
    col = jsonStr['num_cols']
    # if col == 0:
    #     print(f'Warning: col len error, col={col}, id={id}')
    # elif row == 0:
    #     print(f'Warning: row len error, row={row}, id={id}')
    return OurTable(id, header, data, context, profilling_data=profile)


def main():
    """主函数：构建索引"""
    numpy.set_printoptions(suppress=True)

    # Qwen 模型配置
    # 从 HYPERPARAMETERS 获取数据集路径配置
    FILE_PATHs = {
        'query': get_query_path(hp.benchmark),
        'datalake': get_datalake_path(hp.benchmark)
    }
    # 创建 Qwen embedding 模型
    print(f"Initializing Qwen Embedding Model: {hp.qwen_model}")
    RESULT_DIR = hp.result_dir if hp.result_dir else f"./results/{hp.benchmark}/"
    os.makedirs(RESULT_DIR, exist_ok=True)
    if hp.local_model_path:
        print(f"Using local trained model from: {hp.local_model_path}")
        if hp.base_model_path:
            print(f"Base model path: {hp.base_model_path}")

    embedding_model = EmbeddingModelFactory.create(
        'qwen',
        model_name=hp.qwen_model,
        batch_size=hp.batch_size,
        local_model_path=hp.local_model_path,
        base_model_path=hp.base_model_path
    )
    # print(f"Embedding dimension: {embedding_model.embedding_dim}")

    # 如果使用 table_mapper，构建全局表名到无语义编码的映射
    global_table_name_mapping = {}
    if hp.table_mapper:
        print("Building global table name mapping...")
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

        # 保存映射关系到文件
        mapping_output_path = os.path.join(RESULT_DIR, 'table_name_mapping.json')
        os.makedirs(os.path.dirname(mapping_output_path), exist_ok=True)
        with open(mapping_output_path, 'w', encoding='utf-8') as f:
            json.dump(global_table_name_mapping, f,
                      indent=2, ensure_ascii=False)
        print(f"Table name mapping saved to {mapping_output_path}")
        print(f"Total tables mapped: {len(global_table_name_mapping)}")

    # 处理每个数据源
    for FILE_TYPE_KEY, FILE_PATH in FILE_PATHs.items():
        FILE_TYPE = hp.file_type

        print(f"\n{'='*60}")
        print(f"Processing {FILE_TYPE_KEY}: {FILE_PATH}")
        print(f"{'='*60}")

        # 获取文件列表
        file_paths = list_files_by_extension(FILE_PATH, FILE_TYPE)

        # 如果需要，shuffle 文件列表
        if hp.shuffle:
            numpy.random.shuffle(file_paths)

        total_file_num = len(file_paths)
        print(f"Total files: {total_file_num}")

        skipnums = 0
        dataEmbeds = []

        # 处理每个表文件
        for file_idx, csv_file in tqdm(enumerate(file_paths, 1), total=total_file_num, desc="Indexing tables"):
            table_name = get_basename(csv_file)
            # 如果使用 table_mapper，使用映射后的表名
            if hp.table_mapper:
                key = f"{FILE_TYPE_KEY}_{table_name}"
                table_name = global_table_name_mapping.get(key, table_name)
            try:
                # 读取表
                tablepos = csv_to_table(
                    csv_file, table_name, hp.profilling_path, hp.sample_rows)
                # print("csv_totable", tablepos)
                # 编码所有列
                column_embeddings = embedding_model.encode_columns(
                    tablepos.id,
                    tablepos.header,
                    tablepos.data,
                    sample_rows=hp.sample_rows,
                    mask_header=hp.mask_header,
                    profilling_data=tablepos.profilling_data
                )

                # 保存结果：(table_name, column_embeddings_matrix)
                # column_embeddings shape: (num_columns, embedding_dim)
                dataEmbeds.append((tablepos.id, column_embeddings))

            except Exception as e:
                skipnums += 1
                print(f"\nError processing {csv_file}: {e}")
                continue

        print()  # 换行
        print()  # 换行
        print(f"Processed: {total_file_num - skipnums}/{total_file_num}")
        print(f"Skipped: {skipnums}")
        print(f"Total tables with embeddings: {len(dataEmbeds)}")

        # 保存结果到 pickle 文件
        if len(dataEmbeds) > 0:
            temp_output_path = os.path.join(
                RESULT_DIR, f"{FILE_TYPE_KEY}.pkl")
            pickle.dump(dataEmbeds, open(temp_output_path, "wb"))
            print(f"Saved embeddings to {temp_output_path}")
        else:
            print(f"Warning: No embeddings generated for {FILE_TYPE_KEY}")

    print(f"\n{'='*60}")
    print("Indexing completed!")
    print(f"{'='*60}")
    print_noise_stats()


if __name__ == '__main__':
    main()
