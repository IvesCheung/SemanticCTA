from add_profilling import read_table, print_noise_stats
from HYPERPARAMETERS import DATASET_PATHS, get_datalake_path, get_query_path
from sdd.pretrain import load_checkpoint, inference_on_tables
import torch
import pandas as pd
import numpy as np
import glob
import os
import pickle
import time
import sys
import argparse
from tqdm import tqdm
torch.cuda.empty_cache()
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# 添加项目根目录到 sys.path，以便导入 HYPERPARAMETERS
_root_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)



def extractVectors(dfs, ds_path, dataFolder, augment, sample, table_order, run_id, model_path, singleCol=False):
    # if singleCol:
    #     model_path = "/home/benchmark/starmie-main/" \
    #                  "model_%s_%s_%s_%dsingleCol.pt" % (
    #                      augment, sample, table_order, run_id)
    # else:
    #     model_path = "/home/benchmark/starmie-main/" \
    #                  "model_%s_%s_%s_%d.pt" % (
    #                      augment, sample, table_order, run_id)
    print(f"model_path: {model_path}")
    ckpt = torch.load(model_path, map_location=torch.device('cuda'))
    # load_checkpoint from sdd/pretain

    model, trainset = load_checkpoint(ckpt, ds_path)

    return inference_on_tables(dfs, model, trainset, batch_size=528)


def get_df(dataFolder, sample_rows=None, noise_prob=0.0):
    '''
    Get the DataFrames of each table in a folder
    Args:
        dataFolder: filepath to the folder with all tables
        sample_rows: number of rows to read per table (None = up to 1000, 0 = zero rows/empty, N = N rows)
        noise_prob: probability of applying random column augmentation (0.0 = no noise)
    Return:
        dataDFs (dict): key is the filename, value is the dataframe of that table
    '''

    # dataFiles = glob.glob(dataFolder+"/*.csv")
    dataFiles = os.listdir(dataFolder)

    dataDFs = {}
    columns = 0

    nrows = sample_rows if sample_rows is not None else 1000
    ind = 0
    for file in sorted(dataFiles):
        ind += 1
        if file == "CSV0000000000000435.csv":
            continue
        df = read_table(
            os.path.join(dataFolder, file),
            sample_rows=nrows,
            noise_prob=noise_prob,
            encoding="ISO-8859-1",
            low_memory=False,
        )
        filename = file.split("/")[-1]
        dataDFs[filename] = df

        print(f"ind: {ind}, file: {file}, columns: {df.shape[1]}")
        columns += df.shape[1]

    return dataDFs, columns


if __name__ == '__main__':
    ''' Get the model features by calling model inference from sdd/pretrain
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="webtable")
    # single-column mode without table context
    parser.add_argument("--single_column", dest="single_column",
                        action="store_true", default=False)
    parser.add_argument("--run_id", type=int, default=0)
    # column-ordered or row-ordered (always use column)
    parser.add_argument("--table_order", type=str, default='column')
    parser.add_argument("--save_model", dest="save_model",
                        action="store_true", default=True)
    # Model path for loading pre-trained SDD model
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the pre-trained SDD model")
    # Output path for saving embeddings
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output path for saving embeddings pickle file")
    # Data path for custom data location
    parser.add_argument("--data_path", type=str, default=None,
                        help="Custom data path for tables")
    # Results directory
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Results directory for output files")
    parser.add_argument("--sample_rows", type=int, default=None,
                        help="每张表读取的行数（None=全部，0=仅header，N=N行）")
    parser.add_argument("--noise_prob", type=float, default=0.0,
                        help="应用随机列增强噪声的概率：0.0=不加噪声，0.1=10%%概率，以此类推")
    parser.add_argument("--table_mapper", action='store_true',
                        help="是否使用table_mapper（将表名映射为匿名ID）")
    parser.add_argument("--shuffle", action='store_true',
                        help="是否shuffle数据（随机打乱文件处理顺序）")

    hp = parser.parse_args()
    print(f'hp.save_model: {hp.save_model}, hp.single_column: {hp.single_column}')

    # START PARAMETER: defining the benchmark (dataFolder), if it is a single column baseline,
    # run_id, table_order, and augmentation operators and sampling method if they are different from default
    dataFolder = hp.benchmark
    isSingleCol = hp.single_column
    if dataFolder == 'webtable':
        ao = 'drop_col'
        sm = 'tfidf_entity'
        if isSingleCol:
            ao = 'drop_cell'
    ao = 'drop_cell'
    sm = 'alphaHead'

    run_id = hp.run_id
    table_order = hp.table_order
    # END PARAMETER

    # Change the data paths to where the benchmarks are stored
    # Support custom data path via --data_path argument
    if hp.data_path:
        DATAPATH = hp.data_path
        dataDir = ['']  # Use the path directly
    elif dataFolder in DATASET_PATHS:
        # Use real paths from HYPERPARAMETERS for known benchmarks (SG, USA, santos, UK, CAN)
        DATAPATH = ""  # resolved per-dir below
        dataDir = ['query', 'datalake']
    elif dataFolder == 'opendata':
        DATAPATH = "/data/opendata/small/"
        dataDir = ['datasets_UK', 'datasets_SG',
                   'datasets_CAN', 'datasets_USA']
    elif dataFolder == 'opendata_large':
        DATAPATH = "/data/opendata/large/"
        dataDir = ['UK', 'SG', 'CAN', 'USA']
    elif dataFolder == 'webtable':
        DATAPATH = '/data/webtable/small/'
        dataDir = ['split_1']
    elif dataFolder == 'webtable_large':
        DATAPATH = '/data/webtable/large/'
        dataDir = ['split_1', 'split_2', 'split_3',
                   'split_4', 'split_5', 'split_6']
    elif dataFolder == 'webtable_small_query':
        DATAPATH = "/data/webtable/small/"
        dataDir = ['small_query']
    elif dataFolder == 'webtable_large_query':
        DATAPATH = "/data/webtable/large/"
        dataDir = ['large_query']
    elif dataFolder == 'opendata_small_query':
        DATAPATH = "/data/webtable/small/"
        dataDir = ['small_query']
    elif dataFolder == 'opendata_large_query':
        DATAPATH = "/data/webtable/large/"
        dataDir = ['large_query']
    else:
        # Default: treat benchmark as a path
        DATAPATH = f"./task/dataset_discovery/{dataFolder}/"
        dataDir = ['datalake', 'query']

    # 如果使用 table_mapper，构建全局表名到匿名编码的映射
    global_table_name_mapping = {}
    if hp.table_mapper:
        print("Building global table name mapping...")
        for dir in dataDir:
            if DATAPATH == "" and dataFolder in DATASET_PATHS:
                _df = get_query_path(
                    dataFolder) if dir == 'query' else get_datalake_path(dataFolder)
            else:
                _df = DATAPATH + dir
            file_list = sorted(
                [f for f in os.listdir(_df) if f.endswith('.csv')])
            if hp.shuffle:
                numpy.random.shuffle(file_list)
            for counter, filename in enumerate(file_list):
                key = f"{dir}_{filename}"
                if key not in global_table_name_mapping:
                    global_table_name_mapping[key] = f"{dir}_{counter}"
        print(
            f"Table name mapping built. Total tables mapped: {len(global_table_name_mapping)}")

    inference_times = 0
    # dataDir is the query and data lake
    for dir in tqdm(dataDir):
        print("//==== ", dir)
        # For known benchmarks, resolve each dir's actual path from HYPERPARAMETERS
        if DATAPATH == "" and dataFolder in DATASET_PATHS:
            DATAFOLDER = get_query_path(
                dataFolder) if dir == 'query' else get_datalake_path(dataFolder)
        else:
            DATAFOLDER = DATAPATH + dir
        print(f"Datafloder: {DATAFOLDER}")

        # 获取文件夹下面所有csv文件,并且统计所有列的总数目
        dfs, col_count = get_df(DATAFOLDER, hp.sample_rows, hp.noise_prob)

        # 如果需要，shuffle 文件顺序
        if hp.shuffle:
            dfs_items = list(dfs.items())
            numpy.random.shuffle(dfs_items)
            dfs = dict(dfs_items)

        dataEmbeds = []
        # 有多少张表
        dfs_totalCount = len(dfs)
        print(f"The number of total tables {dfs_totalCount} in {dir}")
        print(f"The number of total columns {col_count} in {dir}")

        # Extract model vectors, and measure model inference time
        start_time = time.time()
        # 获取所有表的特征向量；传入 DATAFOLDER 作为 ds_path 供 PretrainTableDataset 使用
        cl_features = extractVectors(list(dfs.values(
        )), DATAFOLDER, dataFolder, ao, sm, table_order, run_id, model_path=hp.model_path, singleCol=isSingleCol)
        inference_times += time.time() - start_time
        print("%s %s inference time: %d seconds" %
              (dataFolder, dir, time.time() - start_time))
        for i, file in enumerate(dfs):
            # get features for this file / dataset
            cl_features_file = np.array(cl_features[i])
            # 如果使用 table_mapper，使用映射后的匿名表名
            if hp.table_mapper:
                key = f"{dir}_{file}"
                file = global_table_name_mapping.get(key, file)
            dataEmbeds.append((file, cl_features_file))
            # 格式是[(文件名, 特征向量的np数组), (), ...]

        saveDir = dir

        # Determine output path: use --output_path if provided, otherwise auto-generate
        if hp.output_path:
            final_output_path = hp.output_path
        elif hp.results_dir:
            os.makedirs(hp.results_dir, exist_ok=True)
            if isSingleCol:
                final_output_path = os.path.join(
                    hp.results_dir, f"deepjoin_{saveDir}_singleCol.pkl")
            else:
                final_output_path = os.path.join(
                    hp.results_dir, f"deepjoin_{saveDir}.pkl")
        else:
            if isSingleCol:
                final_output_path = f"/data/final_result/starmie/webtable/webtable_{saveDir}_singleCol.pkl"
            else:
                final_output_path = f"/data/final_result/starmie/webtable/webtable_{saveDir}.pkl"

        if hp.save_model:
            # Ensure output directory exists
            output_dir = os.path.dirname(final_output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            pickle.dump(dataEmbeds, open(final_output_path, "wb"))
        print("Benchmark: ", dataFolder)
        print(f"output: {final_output_path}")
        print("--- Total Inference Time: %s seconds ---" % (inference_times))
    print_noise_stats()
