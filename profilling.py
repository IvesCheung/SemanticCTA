from llm_tool.call_llm import call_llm
from llm_tool.prompt import get_prompt
from csv_tool import get_csv_column_groups, get_csv_schema
from utils import get_basename, safe_json_loads, list_csv_files, file_path_to_key
import os
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import argparse
import glob


def build_profilling_parser():
    parser = argparse.ArgumentParser(
        description="使用 LLM 对表格数据进行剖析，生成数据字典。")
    parser.add_argument(
        "--log", action='store_true', help="是否记录日志到文件")
    parser.add_argument(
        "--ckpt", type=str, default=None, help="从哪个文件继续生成profilling结果")
    parser.add_argument(
        "-o", "--output_file", type=str, default=None, help="输出结果文件路径")
    parser.add_argument(
        "-r", "--root_dir", type=str, required=True, help="待剖析表格文件夹路径")
    parser.add_argument(
        "--sleep_time", type=int, default=0, help="组间休息时间，单位秒")
    parser.add_argument(
        "-p", "--prompt_version", type=str, default="multi_column", help="使用的prompt版本")
    parser.add_argument(
        "-s", "--sample_size", type=int, default=10, help="每张表最多采样多少行进行剖析")
    parser.add_argument(
        "--max_workers", type=int, default=16, help="并发处理的最大线程数")
    parser.add_argument(
        "--save_interval", type=int, default=16, help="每处理多少个文件保存一次结果")
    parser.add_argument(
        "--model", type=str, default=None, help="使用的 LLM 模型名称，默认使用 CONFIG 中配置的模型")
    return parser


CONFIG = {
    "prompt_version": "multi_column",               # 选择使用的 prompt 模板
    "sample_size": 10,                              # 每张表最多采样多少行进行剖析
    "sample_step": 64,                              # 每次处理多少列进行剖析
    "profilling_model": "qwen2.5-72b-instruct",     # 使用的 LLM 模型
    "drop_empty_rows": True,                        # 是否剔除空行后再进行剖析
    "description": "使用 LLM 对表格数据进行剖析，生成数据字典。",
}


def profilling_table(file_path: str,
                     prompt_version: str = CONFIG.get(
                         "prompt_version", "base_profilling"),
                     sample_size: int = CONFIG.get("sample_size"),
                     sample_step: int = CONFIG.get("sample_step", 64),
                     profilling_model: str = CONFIG.get(
                         "profilling_model", "qwen2.5-72b-instruct"),
                     drop_empty_rows: bool = CONFIG.get(
                         "drop_empty_rows", True),
                     **args
                     ) -> dict:
    """
    使用 LLM 对 CSV 文件进行数据剖析, 返回文本结果。
    """
    tablename = get_basename(file_path)
    table_profiling_results = {col: {"__type__": value}
                               for col, value in get_csv_schema(file_path).items()}
    raw_responses = {}
    # pbar = tqdm(desc=f"列剖析: {tablename}", unit="group")
    for cols, chunk_csv in get_csv_column_groups(
            file_path,
            step=sample_step,
            drop_empty_rows=drop_empty_rows,
            sample_size=sample_size
    ):
        prompt = get_prompt(prompt_version,
                            table_name=tablename,
                            headers=cols,
                            csv_encoded=chunk_csv,
                            max_rows_hint=len(chunk_csv.split('\n'))-1)
        response = call_llm(model=profilling_model,
                            msgs=[
                                {"role": "system",
                                    "content": "You are a data analysis and data dictionary assistant"},
                                {"role": "user", "content": prompt}]
                            )
        raw_responses[f"{','.join(cols)}"] = response
        response = safe_json_loads(response)
        if isinstance(response, list):
            response = {k: v for item in response if isinstance(item, dict) for k, v in item.items()}
        if not isinstance(response, dict):
            continue
        response_col_keys = response.keys()
        for col in cols:
            for response_key in response_col_keys:
                if col in [key.strip() for key in response_key.split(',')]:
                    table_profiling_results[col][response_key] = response[response_key]
    #     pbar.update(1)
    # pbar.close()

    return table_profiling_results, raw_responses


def profilling_tables(file_paths: list[str], **args):
    """
    对指定 CSV 文件进行数据剖析。
    """
    final_results = {}
    for file_path in file_paths:
        # table_name = get_basename(file_path)
        print(f"开始剖析表格: {file_path}")
        result, _ = profilling_table(file_path, **args)
        # files_profilling_results[table_name] = result
        final_results[file_path_to_key(file_path)] = result
    return final_results


def profilling_csv_files(file_paths: list[str], **args):
    """
    对指定 CSV 文件进行数据剖析。
    """
    iterator = tqdm(file_paths, desc="文件剖析", unit="file")
    for file_path in iterator:
        # table_name = get_basename(file_path)
        iterator.write(f"开始剖析表格: {file_path}")
        try:
            result, raw_response = profilling_table(file_path, **args)
            # files_profilling_results[table_name] = result
            yield file_path_to_key(file_path), result, raw_response
        except Exception as e:
            print(f"剖析表格 {file_path} 时出错: {e}")
            continue


def process_single_file(file_path: str, **kwargs):
    """
    处理单个文件的包装函数，用于多线程处理
    """
    print(f"开始处理: {file_path}", flush=True)
    try:
        result, raw_response = profilling_table(file_path, **kwargs)
        return file_path_to_key(file_path), result, raw_response, None
    except Exception as e:
        return file_path_to_key(file_path), None, None, str(e)


def profilling_csv_files_parallel(file_paths: list[str], max_workers: int = 5,
                                  save_callback=None, save_interval: int = 10, **args):
    """
    使用多线程并行处理CSV文件剖析

    Args:
        file_paths: 待处理的文件路径列表
        max_workers: 最大并发线程数
        save_callback: 定期保存的回调函数
        save_interval: 每处理多少个文件调用一次save_callback
        **args: 传递给profilling_table的其他参数
    """
    results = []
    completed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(process_single_file, file_path, **args): file_path
                          for file_path in file_paths}

        # 使用tqdm显示进度
        with tqdm(total=len(file_paths), desc="文件剖析", unit="file") as pbar:
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    file_key, result, raw_response, error = future.result()

                    if error:
                        pbar.write(f"❌ 剖析表格 {file_path} 时出错: {error}")
                    else:
                        # pbar.write(f"✓ 完成剖析表格: {file_path}")
                        results.append((file_key, result, raw_response))

                    # 每处理save_interval个文件（不论成败）调用保存回调
                    completed_count += 1
                    if save_callback and completed_count % save_interval == 0:
                        if results:
                            save_callback(results)
                            results = []  # 清空已保存的结果

                except Exception as e:
                    pbar.write(f"❌ 处理 {file_path} 时发生未预期的错误: {e}")

                pbar.update(1)

    # 返回剩余未保存的结果
    return results


if __name__ == "__main__":
    args = build_profilling_parser().parse_args()
    root_dir = args.root_dir
    os.makedirs("./output", exist_ok=True)
    files_profilling_results = {}

    if args.log:
        from utils import log_file_path
        log_path = log_file_path(__file__, suffix=".json")
        raw_responses = {}

    import json

    # 使用多线程并行处理
    model_name = args.model or CONFIG["profilling_model"]

    # 去掉模型名中的特殊字符，只保留字母、数字、下划线和连字符
    import re
    model_name_safe = re.sub(r'[^\w\-]', '_', model_name)

    # 输出文件路径
    output_path = args.output_file or f"./output/{get_basename(root_dir)}_{args.prompt_version}_profilling_row{args.sample_size}_{model_name_safe}.json"

    # 分片目录
    shard_dir = output_path + ".shards"

    # 加载checkpoint：优先从分片目录恢复，其次从 --ckpt 文件恢复
    if os.path.isdir(shard_dir):
        shard_files = sorted(glob.glob(os.path.join(shard_dir, "part_*[!_raw].json")))
        for sf in shard_files:
            with open(sf, "r", encoding="utf-8") as f:
                files_profilling_results.update(json.load(f))
        print(f"从分片目录恢复: 已加载 {len(shard_files)} 个分片，共 {len(files_profilling_results)} 个文件")
    elif args.ckpt is not None and os.path.exists(args.ckpt):
        files_profilling_results = json.load(open(args.ckpt, "r", encoding="utf-8"))
        print(f"从checkpoint文件恢复: 已加载 {len(files_profilling_results)} 个文件")

    all_csvfiles = list_csv_files(root_dir)
    unprocessed_files = [f for f in all_csvfiles if file_path_to_key(
        f) not in files_profilling_results.keys()]
    print(
        f"总共 {len(all_csvfiles)} 个文件，已处理 {len(files_profilling_results)} 个文件，剩余 {len(unprocessed_files)} 个文件待处理。")
    print(f"使用 {args.max_workers} 个线程并行处理，每 {args.save_interval} 个文件保存一次结果。")

    if not unprocessed_files:
        print("所有文件已处理完毕，无需继续。")
        exit(0)

    # 分片目录：追加模式，已有的分片保留
    os.makedirs(shard_dir, exist_ok=True)
    existing_shards = glob.glob(os.path.join(shard_dir, "part_*[!_raw].json"))
    shard_index = [len(existing_shards)]  # 从已有分片数量继续编号
    save_lock = threading.Lock()

    # 定义保存回调函数 —— 每次写入独立的分片文件
    def save_results(new_results):
        with save_lock:
            shard_index[0] += 1
            shard_path = os.path.join(shard_dir, f"part_{shard_index[0]:04d}.json")
            shard_data = {}
            for file_key, result, raw in new_results:
                shard_data[str(file_key)] = result
                if args.log:
                    raw_responses[str(file_key)] = raw

            with open(shard_path, "w", encoding="utf-8") as f:
                json.dump(shard_data, f, ensure_ascii=False, indent=2)

            if args.log:
                shard_raw_path = os.path.join(shard_dir, f"part_{shard_index[0]:04d}_raw.json")
                with open(shard_raw_path, "w", encoding="utf-8") as f:
                    json.dump(raw_responses, f, ensure_ascii=False, indent=2)

            print(f"\n💾 已保存分片 {shard_index[0]}，包含 {len(shard_data)} 个文件")

    print(f"使用模型: {model_name}")
    remaining_results = profilling_csv_files_parallel(
        unprocessed_files,
        max_workers=args.max_workers,
        save_callback=save_results,
        save_interval=args.save_interval,
        prompt_version=args.prompt_version,
        sample_size=args.sample_size,
        profilling_model=model_name
    )

    # 保存最后剩余的结果
    if remaining_results:
        save_results(remaining_results)

    # 合并所有分片
    print(f"\n正在合并分片文件...")
    merged = {}
    for shard_file in sorted(glob.glob(os.path.join(shard_dir, "part_*[!_raw].json"))):
        with open(shard_file, "r", encoding="utf-8") as f:
            merged.update(json.load(f))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # 合并 raw 日志
    if args.log:
        merged_raw = {}
        for shard_file in sorted(glob.glob(os.path.join(shard_dir, "part_*_raw.json"))):
            with open(shard_file, "r", encoding="utf-8") as f:
                merged_raw.update(json.load(f))
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(merged_raw, f, ensure_ascii=False, indent=2)

    # 清理分片目录
    import shutil
    shutil.rmtree(shard_dir, ignore_errors=True)

    print(f"\n✅ 所有文件处理完成！总共处理了 {len(merged)} 个文件。")
    print(f"结果已保存到: {output_path}")
