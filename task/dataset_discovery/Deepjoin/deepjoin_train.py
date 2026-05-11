import json as _json
import argparse

from torch.utils.data import DataLoader
import torch
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.nn.parallel import DataParallel
import logging
from datetime import datetime
import os
from dataprocess.multi_preocess_csv import process_before_train, transform_train_dev_toInput
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(train_samples, dev_samples, model_save_path, model_name='all-mpnet-base-v2',
          train_batch_size=16, num_epochs=4, cpuid=3):
    # 使用传入的 model_save_path，不再硬编码覆盖
    os.makedirs(model_save_path, exist_ok=True)

    # load model
    model = SentenceTransformer(model_name)

    # set cuda
    if cpuid == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    elif cpuid == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif cpuid == 2:
        device_ids = [0, 1]
        torch.cuda.set_device(device_ids[0])
        model = DataParallel(model, device_ids=[0, 1])
        model = model.module  # 获取原始模型
    else:
        pass

    # load data
    train_dataloader = DataLoader(
        train_samples, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples)
    # warmup_steps 按实际训练步数的 10% 计算，避免小数据集 warmup 过长
    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = max(100, int(total_steps * 0.1))
    print(f"[Train] total_steps={total_steps}, warmup_steps={warmup_steps}")
    # train model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              weight_decay=0.01,
              output_path=model_save_path)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="dataset")
parser.add_argument("--model_name", help="model_name")
parser.add_argument("--model_save_path", help="model_save_path")
parser.add_argument("--file_train_path", help="file_train_path")
parser.add_argument("--tain_csv_file", help="tain_csv_file")
parser.add_argument("--storepath", help="storepath")
parser.add_argument("--region_data_paths",
                    help="JSON 字符串，格式 '{\"SG\": \"./datasets/opendata_SG\", ...}'，用于替换硬编码 region 路径", default=None)
args = parser.parse_args()

# 获取特定的参数
dataset_para = args.dataset
model_name = args.model_name
model_save_path = args.model_save_path
file_train_path = args.file_train_path
tain_csv_file = args.tain_csv_file
storepath = args.storepath
region_data_paths = _json.loads(
    args.region_data_paths) if args.region_data_paths else None

if dataset_para == "opendata":

    abspath_cav = os.path.abspath(tain_csv_file)
    process_before_train(file_train_path, abspath_cav,
                         filepath=storepath,
                         name="train_opendata_list.pkl", name2="evluate_opendata_list.pkl",
                         region_paths=region_data_paths)

    # opendata train size ：129823 test_siez: 32443；splitnumn=1 使用全量数据
    train_samples, dev_samples = transform_train_dev_toInput(storepath, name="train_opendata_list.pkl",
                                                             name2="evluate_opendata_list.pkl", splitnumn=1)

    train(train_samples, dev_samples, model_save_path)
else:
    abspath_cav = os.path.abspath(tain_csv_file)
    process_before_train(file_train_path, abspath_cav)
    train_samples, dev_samples = transform_train_dev_toInput(splitnumn=1)

    train(train_samples, dev_samples, model_save_path, cpuid=0)
