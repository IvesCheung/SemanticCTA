
from time import sleep

from process_table_tosentence import process_table_sentense
import os
import nltk
nltk.set_proxy('https://hub.fastgit.org/nltk/nltk.git')
print(nltk.data.path)
# 添加以下代码以自动下载缺失的 NLTK 数据
# try:
#     nltk.data.find('tokenizers/punkt_tab')
# except LookupError:
#     print("Downloading punkt_tab...")
#     nltk.download('punkt_tab')

# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     print("Downloading punkt...")
#     nltk.download('punkt')

# 先处理相关的小数据集
filepathstore = "/home/project/schema_profiling/results/data/"
dirlist = [
    "/home/project/schema_profiling/task/dataset_discovery/santos_small/datalake"]
data_names_list = ["deepjoin_santos_datalake.pkl"]
file_dic_path = "/home/project/schema_profiling/results/data/tmp"
split_num = 10
for i in range(len(data_names_list)):
    process_table_sentense(filepathstore=filepathstore,
                           datadir=dirlist[i], data_pkl_name=data_names_list[i], tmppath=file_dic_path, split_num=10)
    print("process sucess", data_names_list[i])
