from task.TableDDTask import TableIndexQueryTask
from typing import List, Optional
import argparse
import os
from datetime import datetime


if __name__ == "__main__":
    print("\n" + "="*60)
    # print("Example 1: TaBERT Encoder")
    print("="*60)

    # 获取当前日期和时间
    now = datetime.now()
    date_folder = now.strftime("%Y%m%d")  # 年月日作为文件夹
    time_prefix = now.strftime("%H%M%S")  # 时分秒作为文件前缀
    os.makedirs(f"./results/SG/CL/{date_folder}", exist_ok=True)
    for row in [0, 1, 5, 10]:
        print(f"Sample Rows: {row}")
        task_tabert = TableIndexQueryTask(
            task_name="TableIndexQueryTask",
            config={},
            encoder_type="qwen",
            benchmark="SG",
            profilling_path="./output/SG/datasets_SG_single_column_and_table_profilling.json",
            sample_rows=row,
            shuffle=True,
            table_mapper=True,
            mask_header=None,
            shuffle_columns=False,
            local_model_path="./model/cl/qwen3-0.6B-embedding/column_contrastive_0222_temp0.07_ep3.pth",
            base_model_path="./model/qwen3-0.6B-embedding",
            metrics_path=f"./results/SG/CL/{date_folder}/{time_prefix}_r{row}column_contrastive.csv",
            # results_dir="./results/SG/CL",
            K=10,
            threshold=0.7,
        )
        result_tabert = task_tabert.run()
        # print(result_tabert.data)
        print("\n" + "="*60)
        print("\n\n\n\n\n")
