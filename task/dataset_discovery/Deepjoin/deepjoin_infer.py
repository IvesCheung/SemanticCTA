"""
DeepJoin Inference Script

使用 SentenceTransformer 模型对表格数据生成嵌入向量。
支持命令行参数调用，可被 DeepJoinDDTask 调用。
"""
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def process_onedataset(dataset_file, model_name_or_path, storepath):
    """
    处理单个数据集，生成对应的特征向量并保存

    Args:
        dataset_file: pickle 文件路径，里面存储了数据集的信息
        model_name_or_path: 模型的名称或者路径
        storepath: 输出目录路径
    """
    path, filename_dataset = os.path.split(dataset_file)
    model = SentenceTransformer(model_name_or_path)
    os.makedirs(storepath, exist_ok=True)
    storedata = []
    if os.path.isfile(dataset_file):
        print("Processing data:", dataset_file)
        try:
            # 加载数据
            with open(dataset_file, "rb") as f:
                data = pickle.load(f)

            for ele in tqdm(data, desc="Generating embeddings"):
                key, value = ele
                sentence_embeddings = model.encode(value)
                sentence_embeddings_np = np.array(sentence_embeddings)
                tu1 = (key, sentence_embeddings_np)
                storedata.append(tu1)
        except Exception as e:
            print(f"Error processing file: {e}")
            raise

    storefilename = os.path.join(storepath, filename_dataset)
    with open(storefilename, "wb") as f:
        pickle.dump(storedata, f)
    print("Data processed successfully:", storefilename)
    return storefilename


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeepJoin Inference Script")
    parser.add_argument("--datafile", type=str, required=True,
                        help="Path to input pickle file containing table data")
    parser.add_argument("--storepath", type=str, required=True,
                        help="Output directory path for embeddings")
    parser.add_argument("--model_name_or_path", type=str,
                        default='./model/all-mpnet-base-v2',
                        help="SentenceTransformer model name or path")

    args = parser.parse_args()

    output_file = process_onedataset(
        args.datafile,
        args.model_name_or_path,
        args.storepath
    )
    print(f"\nEmbeddings saved to: {output_file}")
