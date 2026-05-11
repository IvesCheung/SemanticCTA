from argparse import Namespace
import torch
import random
import pandas as pd
import os
import json

from torch.utils import data
from transformers import AutoTokenizer
from .augment import augment
from typing import List
from .preprocessor import computeTfIdf, tfidfRowSample, preprocess

# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'bert': 'bert-base-uncased',
         'distilbert': 'distilbert-base-uncased'}


class TableDataset(data.Dataset):
    """Table dataset"""

    def __init__(self,
                 path,
                 max_len=256,
                 lm='roberta'):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])
        self.pairs = []
        self.max_len = max_len
        self.samples = pd.read_csv(path)
        self.labels = self.samples['match']
        self.table_path = os.path.join(os.path.split(path)[0], "tables")
        self.table_cache = {}

    def _read_table(self, table_id):
        """Read a table"""
        if table_id in self.table_cache:
            table = self.table_cache[table_id]
        else:
            table = pd.read_csv(os.path.join(self.table_path,
                                             "table_%d.csv" % table_id))
            self.table_cache[table_id] = table

        return table

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the two entities combined
            int: the label of the pair (0: unmatch, 1: match)
        """
        # idx = random.randint(0, len(self.pairs)-1)
        l_table_id = self.samples['l_table_id'][idx]
        r_table_id = self.samples['r_table_id'][idx]
        l_column_id = self.samples['l_column_id'][idx]
        r_column_id = self.samples['r_column_id'][idx]

        l_table = self._read_table(l_table_id)
        r_table = self._read_table(r_table_id)

        l_column = l_table[l_table.columns[l_column_id]].astype(str)
        r_column = r_table[r_table.columns[r_column_id]].astype(str)

        # baseline: simple concatenation
        left = ' '.join(l_column)
        right = ' '.join(r_column)

        x = self.tokenizer.encode(text=left,
                                  text_pair=right,
                                  max_length=self.max_len,
                                  truncation=True)
        return x, self.labels[idx]

    def pad(self, batch):
        """Merge a list of dataset items into a train/test batch

        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: x2 of shape (batch_size, seq_len).
                        Elements of x1 and x2 are padded to the same length
            LongTensor: x12 of shape (batch_size, seq_len').
                        Elements of x12 are padded to the same length
            LongTensor: a batch of labels, (batch_size,)
        """
        if len(batch[0]) == 4:
            # em
            x1, x2, x12, y = zip(*batch)

            maxlen = max([len(x) for x in x1+x2])

            x1 = [xi + [self.tokenizer.pad_token_id]
                  * (maxlen - len(xi)) for xi in x1]
            x2 = [xi + [self.tokenizer.pad_token_id]
                  * (maxlen - len(xi)) for xi in x2]

            maxlen = max([len(x) for x in x12])
            x12 = [xi + [self.tokenizer.pad_token_id]
                   * (maxlen - len(xi)) for xi in x12]

            return torch.LongTensor(x1), \
                torch.LongTensor(x2), \
                torch.LongTensor(x12), \
                torch.LongTensor(y)
        else:
            # cleaning
            x1, y = zip(*batch)
            maxlen = max([len(x) for x in x1])
            x1 = [xi + [self.tokenizer.pad_token_id]
                  * (maxlen - len(xi)) for xi in x1]
            return torch.LongTensor(x1), torch.LongTensor(y)


class PretrainTableDataset(data.Dataset):
    """Table dataset for pre-training"""

    def __init__(self,
                 path,
                 augment_op,
                 max_len=256,
                 size=None,
                 lm='roberta',
                 single_column=False,
                 sample_meth='wordProb',
                 table_order='column',
                 profilling_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])
        self.max_len = max_len
        self.path = path
        self.profilling_path = profilling_path

        # assuming tables are in csv format
        self.tables = [fn for fn in os.listdir(
            path) if '.csv' in fn] if os.path.exists(path) else []

        # only keep the first n tables
        if size is not None:
            self.tables = self.tables[:size]

        self.table_cache = {}

        # augmentation operators
        self.augment_op = augment_op

        # logging counter
        self.log_cnt = 0

        # sampling method
        self.sample_meth = sample_meth

        # single-column mode
        self.single_column = single_column

        # row or column order for preprocessing
        self.table_order = table_order

        # tokenizer cache
        self.tokenizer_cache = {}

    @staticmethod
    def from_hp(path: str, hp: Namespace):
        """Construct a PretrainTableDataset from hyperparameters

        Args:
            path (str): the path to the table directory
            hp (Namespace): the hyperparameters

        Returns:
            PretrainTableDataset: the constructed dataset
        """
        return PretrainTableDataset(path,
                                    augment_op=hp.augment_op,
                                    lm=hp.lm,
                                    max_len=hp.max_len,
                                    size=hp.size,
                                    single_column=hp.single_column,
                                    sample_meth=hp.sample_meth,
                                    table_order=hp.table_order,
                                    profilling_path=hp.profilling_path if hasattr(hp, 'profilling_path') else None)

    def _read_table(self, table_id, profilling_path=None):
        """Read a table and optionally add column types and descriptions as first two rows"""
        if table_id in self.table_cache:
            table = self.table_cache[table_id]
        else:
            fn = os.path.join(self.path, self.tables[table_id])
            table = pd.read_csv(fn, lineterminator='\n')

            # 如果提供了描述JSON文件路径，则添加类型和描述信息
            if profilling_path and os.path.exists(profilling_path):
                # print("Adding column info from:", profilling_path)
                table = self._add_column_info(table, fn, profilling_path)

            self.table_cache[table_id] = table

        return table

    def _add_column_info(self, table, csv_file_path, profilling_path):
        """Add column types as first row and descriptions as second row"""
        # 读取JSON文件
        with open(profilling_path, 'r', encoding='utf-8') as f:
            descriptions_data = json.load(f)

        # 标准化文件路径（处理路径分隔符差异）
        normalized_csv_path = os.path.normpath(csv_file_path)

        # 查找匹配的文件路径
        matching_key = None
        for json_path in descriptions_data.keys():
            if os.path.normpath(json_path) == normalized_csv_path or \
                    os.path.basename(normalized_csv_path) == os.path.basename(json_path):
                matching_key = json_path
                break

        if matching_key and matching_key in descriptions_data.keys():
            column_info = descriptions_data[matching_key]

            # 创建类型行（第一行）
            type_row = {}
            # 创建描述行（第二行）
            description_row = {}
            for col in table.columns:
                # column info 类似于
                """
                {
                "Department": {
                "__type__": "object",
                "Department": "Type: String; Example Values: HMRC; Distribution: Unique Count: 1 / Total Rows: 5; Null Presence: False; Possible Business Meaning: The government department responsible for the case."
                },
                "Business Stream": {
                "__type__": "float64",
                "Business Stream": "Type: String; Example Values: ; Distribution: Unique Count: 1 / Total Rows: 5; Null Presence: True; Possible Business Meaning: A specific business area or stream within the department (all values are null in this sample)."
                },
                }
                """
                if col in column_info.keys():
                    # 获取列的信息
                    col_data = column_info[col]
                    if isinstance(col_data, dict):
                        # 获取类型信息
                        col_type = col_data.get("__type__", "unknown")
                        type_row[col] = col_type
                        description_row[col] = ""

                        # 获取描述信息（列名对应的描述）
                        for related_col in col_data:
                            if related_col != "__type__" and col in related_col:
                                description = col_data.get(
                                    related_col, f"No description available for {related_col}")
                                description_row[col] += f"{related_col}: {description} "
                    else:
                        type_row[col] = "unknown"
                        description_row[col] = str(col_data)
                else:
                    type_row[col] = "unknown"
                    description_row[col] = f"No description available for {col}"

            # 创建包含类型和描述的DataFrame
            type_df = pd.DataFrame([type_row])
            description_df = pd.DataFrame([description_row])

            # 将类型行和描述行插入到表格的开头
            table = pd.concat(
                [type_df, description_df, table], ignore_index=True)

            # print(
            #     f"Added column types and descriptions for {csv_file_path}")
        else:
            print(f"No column information found for {csv_file_path}")

        return table

    def _tokenize(self, table: pd.DataFrame) -> List[int]:
        """Tokenize a DataFrame table

        Args:
            table (DataFrame): the input table

        Returns:
            List of int: list of token ID's with special tokens inserted
            Dictionary: a map from column names to special tokens
        """
        res = []
        max_tokens = self.max_len * 2 // len(table.columns)
        budget = max(1, self.max_len // len(table.columns) - 1)
        # from preprocessor.py
        tfidfDict = computeTfIdf(
            table) if "tfidf" in self.sample_meth else None

        # a map from column names to special token indices
        column_mp = {}

        # column-ordered preprocessing
        if self.table_order == 'column':
            if 'row' in self.sample_meth:
                table = tfidfRowSample(table, tfidfDict, max_tokens)
            for column in table.columns:
                # from preprocessor.py
                tokens = preprocess(
                    table[column], tfidfDict, max_tokens, self.sample_meth)
                col_text = self.tokenizer.cls_token + " " + \
                    ' '.join(tokens[:max_tokens]) + " "

                column_mp[column] = len(res)
                res += self.tokenizer.encode(text=col_text,
                                             max_length=budget,
                                             add_special_tokens=False,
                                             truncation=True)
        else:
            # row-ordered preprocessing
            reached_max_len = False
            for rid in range(len(table)):
                row = table.iloc[rid:rid+1]
                for column in table.columns:
                    # from preprocessor.py
                    tokens = preprocess(
                        row[column], tfidfDict, max_tokens, self.sample_meth)
                    if rid == 0:
                        column_mp[column] = len(res)
                        col_text = self.tokenizer.cls_token + " " + \
                            ' '.join(tokens[:max_tokens]) + " "
                    else:
                        col_text = self.tokenizer.pad_token + " " + \
                            ' '.join(tokens[:max_tokens]) + " "

                    tokenized = self.tokenizer.encode(text=col_text,
                                                      max_length=budget,
                                                      add_special_tokens=False,
                                                      truncation=True)

                    if len(tokenized) + len(res) <= self.max_len:
                        res += tokenized
                    else:
                        reached_max_len = True
                        break

                if reached_max_len:
                    break

        self.log_cnt += 1
        if self.log_cnt % 5000 == 0:
            print(self.tokenizer.decode(res))

        return res, column_mp

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.tables)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the first view
            List of int: token ID's of the second view
        """
        table_ori = self._read_table(
            idx, None)

        # single-column mode: only keep one random column
        if self.single_column:
            col = random.choice(table_ori.columns)
            table_ori = table_ori[[col]]

        # apply the augmentation operator
        if ',' in self.augment_op:
            op1, op2 = self.augment_op.split(',')
            table_tmp = table_ori
            table_ori = augment(table_tmp, op1)
            table_aug = augment(table_tmp, op2)
        else:
            table_aug = augment(table_ori, self.augment_op)

        if self.profilling_path:
            table_ori = self._add_column_info(
                table_ori, os.path.join(self.path, self.tables[idx]), self.profilling_path)
            table_aug = self._add_column_info(
                table_aug, os.path.join(self.path, self.tables[idx]), self.profilling_path)
        # convert table into string
        x_ori, mp_ori = self._tokenize(table_ori)
        x_aug, mp_aug = self._tokenize(table_aug)

        # make sure that x_ori and x_aug has the same number of cls tokens
        # x_ori_cnt = sum([int(x == self.tokenizer.cls_token_id) for x in x_ori])
        # x_aug_cnt = sum([int(x == self.tokenizer.cls_token_id) for x in x_aug])
        # assert x_ori_cnt == x_aug_cnt

        # insertsect the two mappings
        cls_indices = []
        for col in mp_ori:
            if col in mp_aug:
                cls_indices.append((mp_ori[col], mp_aug[col]))

        return x_ori, x_aug, cls_indices

    def pad(self, batch):
        """Merge a list of dataset items into a training batch

        Args:
            batch (list of tuple): a list of sequences

        Returns:
            LongTensor: x_ori of shape (batch_size, seq_len)
            LongTensor: x_aug of shape (batch_size, seq_len)
            tuple of List: the cls indices
        """
        x_ori, x_aug, cls_indices = zip(*batch)
        max_len_ori = max([len(x) for x in x_ori])
        max_len_aug = max([len(x) for x in x_aug])
        maxlen = max(max_len_ori, max_len_aug)
        x_ori_new = [xi + [self.tokenizer.pad_token_id]
                     * (maxlen - len(xi)) for xi in x_ori] 
        x_aug_new = [xi + [self.tokenizer.pad_token_id]
                     * (maxlen - len(xi)) for xi in x_aug]

        # decompose the column alignment
        cls_ori = []
        cls_aug = []
        for item in cls_indices:
            cls_ori.append([])
            cls_aug.append([])

            for idx1, idx2 in item:
                cls_ori[-1].append(idx1)
                cls_aug[-1].append(idx2)

        return torch.LongTensor(x_ori_new), torch.LongTensor(x_aug_new), (cls_ori, cls_aug)
