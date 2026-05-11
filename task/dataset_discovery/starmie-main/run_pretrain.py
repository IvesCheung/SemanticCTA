from HYPERPARAMETERS import get_datalake_path, DATASET_PATHS
import argparse
import numpy as np
import random
import torch
import sys
import os

# 添加项目根目录到 sys.path
root_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from sdd.dataset import PretrainTableDataset
from sdd.pretrain import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="santos",
                        help=f"数据集名称，可选: {', '.join(DATASET_PATHS.keys())}")
    parser.add_argument("--logdir", type=str, default="results/")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lm", type=str, default='roberta')
    parser.add_argument("--projector", type=int, default=768)
    parser.add_argument("--augment_op", type=str, default='drop_col,sample_row')
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    # single-column mode without table context
    parser.add_argument("--single_column", dest="single_column", action="store_true")
    # row / column-ordered for preprocessing
    parser.add_argument("--table_order", type=str, default='column')
    # for sampling
    parser.add_argument("--sample_meth", type=str, default='head')
    parser.add_argument("--profilling_path", type=str, default=None)
    parser.add_argument("--save_model_path", type=str, default=None)
    # mlflow tag
    # parser.add_argument("--mlflow_tag", type=str, default=None)

    hp = parser.parse_args()

    # mlflow logging
    # for variable in ["task", "batch_size", "lr", "n_epochs", "augment_op", "sample_meth", "table_order"]:
    #     mlflow.log_param(variable, getattr(hp, variable))

    # if hp.mlflow_tag:
    #     mlflow.set_tag("tag", hp.mlflow_tag)

    # set seed
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 从 HYPERPARAMETERS 获取数据集路径
    path = get_datalake_path(hp.task)
    print(f"[run_pretrain] Using datalake path: {path}")

    trainset = PretrainTableDataset.from_hp(path, hp)

    train(trainset, hp)
