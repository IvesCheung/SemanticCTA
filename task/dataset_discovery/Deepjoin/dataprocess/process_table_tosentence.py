import os
import pickle
import random
import shutil
import sys
import nltk

import pandas as pd
import multiprocessing
from multiprocessing import Process, Queue
from tqdm import tqdm
import multiprocessing
import time
import torch.multiprocessing

# 设置要使用的 CPU 核心编号

cpu_cores = [i for i in range(70)]  # 设置为你想使用的核心编号列表

# 设置进程的 CPU 亲和性
os.sched_setaffinity(os.getpid(), cpu_cores)  # 将 0 替换为你的进程 ID


torch.multiprocessing.set_sharing_strategy('file_system')


def analyze_column_values(df, column_name, description=None, col_type=None):

    # 获取指定列的所有不同的列值数据和它们的频率
    value_counts = df[column_name].astype(str).value_counts()

    # 按照频率由高到低对列值进行排序
    sorted_values = value_counts.index.tolist()

    n = len(sorted_values)

    # 空表（如 sample_rows=0）时无值可统计，退化为仅描述列名
    if n == 0:
        tokens = f"{column_name} contains 0 values"
        prefix = ""
        if col_type:
            prefix += f"Type: {col_type}. "
        if description:
            prefix += f"Description: {description}. "
        if prefix:
            tokens = prefix + tokens
        tokens = nltk.word_tokenize(tokens)
        return ' '.join(tokens[:512])

    # 以逗号分隔列值
    col = ', '.join(sorted_values)

    # 统计列值的最大、最小和平均长度
    lengths = [len(str(value)) for value in sorted_values]
    max_len = max(lengths)
    min_len = min(lengths)
    avg_len = sum(lengths) / len(lengths)
    tokens = f"{column_name} contains {str(n)} values ({str(max_len)}, {str(min_len)}, {str(avg_len)}): {col}"

    # 若提供了 profilling 信息，前置类型和描述以丰富句子语义
    prefix = ""
    if col_type:
        prefix += f"Type: {col_type}. "
    if description:
        prefix += f"Description: {description}. "
    if prefix:
        tokens = prefix + tokens

    tokens = nltk.word_tokenize(tokens)
    truncated_tokens = tokens[:512]
    truncated_sentence = ' '.join(truncated_tokens)
    return truncated_sentence

# 输入一个df 文件句柄，输出 是一个字符串list 的，每一个字符串表示这一列 的数据的串行


def evaluate4(df, profilling_data=None, csv_file_path=None):
    columns = df.columns.tolist()
    sentens_list = []

    # 查找该表格的 profilling 信息（按完整路径或文件名匹配）
    col_profilling = {}
    if profilling_data and csv_file_path:
        normalized_path = os.path.normpath(csv_file_path)
        for json_key in profilling_data:
            if os.path.normpath(json_key) == normalized_path or \
                    os.path.basename(normalized_path) == os.path.basename(json_key):
                col_profilling = profilling_data[json_key]
                break

    for column in columns:
        col_info = col_profilling.get(column, {})
        col_type = None
        description = None
        if isinstance(col_info, dict):
            col_type = col_info.get("__type__")
            desc_parts = [str(v)
                          for k, v in col_info.items() if k != "__type__"]
            description = " ".join(desc_parts) if desc_parts else None
        s = analyze_column_values(
            df, column, description=description, col_type=col_type)
        sentens_list.append(s)
    return sentens_list


def get_file_columns(file_path):
    df = pd.read_csv(file_path, engine='python', nrows=1)
    columns = len(df.columns)
    return columns


def partition_files(file_paths, m):
    random.shuffle(file_paths)
    file_info = []  # 存储文件列表及其对应的列数的元组列表

    current_group = []  # 当前文件组
    current_columns = 0  # 当前文件组的列数和
    stime = time.time()
    for file_path in file_paths:
        columns = get_file_columns(file_path)
        # 如果当前文件组的列数和加上当前文件的列数超过m，则将当前文件组加入结果列表，并创建新的文件组
        if current_columns + columns > m:
            file_info.append(current_group)
            current_group = [file_path + "_" + str(columns)]
            current_columns = columns
        else:
            current_group.append(file_path + "_" + str(columns))
            current_columns += columns
    endtime = time.time()
    # print(f"after partioning : {endtime - stime}")
    # 添加最后一个文件组
    if current_group:
        file_info.append(current_group)
    return file_info


def create_folder(path):
    if os.path.exists(path):
        # 如果路径存在，删除文件夹及其下的所有文件
        shutil.rmtree(path)
        print("Folder and its content deleted.")

    # 创建新的文件夹
    os.makedirs(path)
    print("Folder created.")


def read_pkl_files(folder_path):
    # 获取文件夹中的文件列表
    file_list = os.listdir(folder_path)

    re_dict = {}

    # 遍历文件列表
    for file_name in file_list:
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, file_name)

        # 检查文件是否为.pkl文件
        if file_name.endswith(".pkl"):
            try:
                # 打开.pkl文件并加载变量
                with open(file_path, 'rb') as file:
                    obj = pickle.load(file)

                re_dict.update(obj)
            except Exception as e:
                print("Error occurred while reading", file_name, ":", str(e))
    return re_dict


def process_task4(i, input_values, queue, queue_inforgather, file_dic_path, profilling_data=None, sample_rows=None, noise_prob=0.0):

    dict = {}
    for input_value in input_values:
        k = struct_dic_key(input_value)
        try:
            # sample_rows=None → all rows; sample_rows=0 → header only (0 data rows); N → N rows
            df = pd.read_csv(input_value, low_memory=False, nrows=sample_rows)
        except Exception as e:
            print("error filename:", input_value)
            continue
        # Apply column-level noise augmentation if requested
        if noise_prob > 0.0 and random.random() < noise_prob:
            try:
                _root = os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
                if _root not in sys.path:
                    sys.path.insert(0, _root)
                from taqwen.augment import column_augment
                import random as _random
                ops = [
                    'shuffle_col', 'drop_random_cols', 'drop_single_col', 'sample_col',
                    'drop_num_col', 'drop_text_col', 'drop_nan_col', 'keep_top_cols',
                    'reverse_col', 'clear_col_values', 'swap_two_cols',
                ]
                op = _random.choice(ops)
                df = column_augment(df, op)
            except Exception as _e:
                pass  # silently skip if augment unavailable
        embdings = evaluate4(
            df, profilling_data=profilling_data, csv_file_path=input_value)
        dict[k] = embdings
        queue.put(1)
    filename = os.path.join(file_dic_path, str(i)+".pkl")
    with open(filename, 'wb') as file:
        pickle.dump(dict, file)
    queue.put((-1, "test-pid"))


def struct_dic_key(filepath):
    elelist = filepath.split(os.sep)
    return elelist[-2] + "-" + elelist[-1]


def split_list(lst, num_parts):
    avg = len(lst) // num_parts
    remainder = len(lst) % num_parts

    result = []
    start = 0
    for i in range(num_parts):
        if i < remainder:
            end = start + avg + 1
        else:
            end = start + avg
        result.append(lst[start:end])
        start = end

    return result


def process_table_sentense(filepathstore, datadir, data_pkl_name, tmppath="/data/lijiajun/webtable_tmp", split_num=10, profilling_path=None, sample_rows=None, noise_prob=0.0):
    """
    以下是各参数的具体含义：

    filepathstore

    含义: 最终结果的存储目录路径。
    作用: 代码会检查该目录是否存在，如果不存在会通过 os.makedirs 创建。最终合并后的 pickle 文件会保存在这个目录下。
    
    datadir
    含义: 原始数据的输入目录路径。
    作用: 代码会使用 os.walk(dir) 遍历这个目录及其子目录，查找所有的 CSV 文件（排除特定的文件名如 small_join.csv 等）作为待处理的输入文件列表。
   
    data_pkl_name
    含义: 最终输出的文件名。
    作用: 处理完成并合并所有子进程的结果后，数据会被保存为 pickle 格式，文件名为此参数指定的值（例如 "output.pkl"），完整路径为 filepathstore + data_pkl_name。
    
    tmppath (默认值: "/data/lijiajun/webtable_tmp")
    含义: 临时文件夹路径。
    作用:
    用于存放多进程处理过程中的中间结果。
    函数开始运行时会调用 create_folder 清空并重新创建此目录。
    每个子进程处理完分配的任务后，会将结果保存为一个小的 .pkl 文件在这个目录下。
    最后主进程会从这个目录读取所有临时文件进行合并。
    
    split_num (默认值: 10)
    含义: 并行进程的数量（分片数）。
    作用:
    决定了将输入文件列表切分成多少份。
    决定了启动多少个 Process
    """
    # filepathstore = "/data/lijiajun/opendata/large"
    # dir = "/data/opendata/large/query/"
    # list_of_tuples_name = "opendata_large_query_new.pkl"
    # file_dic_path = "/data/lijiajun/webtable_tmp"
    # split_num = 10
    list_of_tuples_name = data_pkl_name
    dir = datadir
    file_dic_path = tmppath

    os.makedirs(filepathstore, exist_ok=True)
    create_folder(file_dic_path)

    # 加载 profilling 数据（若提供）
    profilling_data = None
    if profilling_path and os.path.exists(profilling_path):
        import json
        with open(profilling_path, 'r', encoding='utf-8') as f:
            profilling_data = json.load(f)
        print(
            f"[Profilling] Loaded {len(profilling_data)} table entries from: {profilling_path}")
    elif profilling_path:
        print(
            f"[Profilling] WARNING: profilling_path not found, skipping: {profilling_path}")

    filelist = []

    for root, dirs, files in os.walk(dir):
        for file in files:
            if file == 'small_join.csv' or file == 'large_join.csv':
                continue
            filepath = os.path.join(root, file)
            if os.path.isfile(filepath):
                # print(filepath)
                filelist.append(filepath)

            else:
                print(f"file: {filepath} is not a file ,pass")
    print(f"split1 all file ,filelistlen: {len(filelist)} added to filelist")

    inputs = filelist

    # 指定包含查询表的文件夹路径

    # 获取文件夹中的所有文件名

    sub_file_ls = split_list(inputs, split_num)
    process_list = []

    #####
    # 为每个进程创建一个队列
    queues = [Queue() for i in range(split_num)]
    # queue = Queue()
    # 一个用于标识所有进程已结束的数组
    finished = [False for i in range(split_num)]

    # 为每个进程创建一个进度条
    bars = [tqdm(total=len(sub_file_ls[i]),
                 desc=f"bar-{i}", position=i) for i in range(split_num)]
    # 用于保存每个进程的返回结果
    results = [None for i in range(split_num)]
    queue_inforgather = multiprocessing.Manager().Queue()

    for i in range(split_num):
        process = Process(target=process_task4, args=(
            i, sub_file_ls[i], queues[i], queue_inforgather, file_dic_path, profilling_data, sample_rows, noise_prob))
        process_list.append(process)
        process.start()

    while True:
        for i in range(split_num):
            queue = queues[i]
            bar = bars[i]
            try:
                # 从队列中获取数据
                # 这里需要用非阻塞的get_nowait或get(True)
                # 如果用get()，当某个进程在某一次处理的时候花费较长时间的话，会把后面的进程的进度条阻塞着
                # 一定要try捕捉错误，get_nowait读不到数据时会跑出错误
                res = queue.get_nowait()
                if isinstance(res, tuple) and res[0] == -1:
                    # 某个进程已经处理完毕
                    finished[i] = True
                    results[i] = res[1]
                    continue
                bar.update(res)
            except Exception as e:
                continue

                # 所有进程处理完毕
        if all(finished):
            break

    for process in process_list:
        process.join()

    result_dict = read_pkl_files(file_dic_path)

    list_of_tuples = list(result_dict.items())

    with open(os.path.join(filepathstore, list_of_tuples_name), 'wb') as file:
        pickle.dump(list_of_tuples, file)
    print("pickle sucesss")
