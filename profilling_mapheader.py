from llm_tool.call_llm import call_llm
from llm_tool.prompt import get_prompt
from csv_tool import get_csv_column_groups, get_csv_schema
from utils import get_basename, safe_json_loads, list_csv_files, file_path_to_key
import os
from tqdm import tqdm
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import argparse


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
        "-p", "--prompt_version", type=str, default="single_column", help="使用的prompt版本")
    parser.add_argument(
        "--max_workers", type=int, default=16, help="并发处理的最大线程数")
    parser.add_argument(
        "--save_interval", type=int, default=16, help="每处理多少个文件保存一次结果")
    return parser


CONFIG = {
    "prompt_version": "single_column",               # 选择使用的 prompt 模板
    "sample_size": 64,                              # 每张表最多采样多少行进行剖析
    "sample_step": 64,                              # 每次处理多少列进行剖析
    "profilling_model": "qwen2.5-72b-instruct",     # 使用的 LLM 模型
    "drop_empty_rows": True,                        # 是否剔除空行后再进行剖析
    # 随机编码列名的比例 (0.0-1.0)，0表示不编码
    "encode_header_ratio": 0.5,
    "description": "使用 LLM 对表格数据进行剖析，生成数据字典。",
}


def profilling_table(file_path: str,
                     prompt_version: str = CONFIG.get(
                         "prompt_version", "base_profilling"),
                     sample_size: int = CONFIG.get("sample_size", 64),
                     sample_step: int = CONFIG.get("sample_step", 64),
                     profilling_model: str = CONFIG.get(
                         "profilling_model", "qwen2.5-72b-instruct"),
                     drop_empty_rows: bool = CONFIG.get(
                         "drop_empty_rows", True),
                     encode_header_ratio: float = CONFIG.get(
                         "encode_header_ratio", 0.0),
                     **args
                     ) -> dict:
    """
    使用 LLM 对 CSV 文件进行数据剖析, 返回文本结果。

    Args:
        encode_header_ratio: 随机编码列名的比例 (0.0-1.0)，0表示不编码
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
        # 随机选择一部分列名进行编码
        encoded_cols = cols.copy()
        header_mapping = {}  # 编码名 -> 原始名
        reverse_mapping = {}  # 原始名 -> 编码名

        if encode_header_ratio > 0:
            num_to_encode = max(1, int(len(cols) * encode_header_ratio))
            cols_to_encode = random.sample(cols, num_to_encode)

            for idx, col in enumerate(cols_to_encode):
                encoded_name = f"header{idx + 1}"
                header_mapping[encoded_name] = col
                reverse_mapping[col] = encoded_name

            # 替换列名
            encoded_cols = [reverse_mapping.get(col, col) for col in cols]
            # print(encoded_cols)

            # 替换 CSV 内容中的列名（第一行）
            csv_lines = chunk_csv.split('\n')
            if csv_lines:
                header_line = csv_lines[0]
                for original, encoded in reverse_mapping.items():
                    header_line = header_line.replace(
                        f'"{original}"', f'"{encoded}"')
                    header_line = header_line.replace(original, encoded)
                csv_lines[0] = header_line
                chunk_csv = '\n'.join(csv_lines)

        prompt = get_prompt(prompt_version,
                            table_name=tablename,
                            headers=encoded_cols,
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

        # 将编码的列名映射回原始列名
        if header_mapping:
            decoded_response = {}
            for response_key, response_value in response.items():
                # 按逗号分割键，逐个映射后重新组合
                key_parts = [part.strip() for part in response_key.split(',')]
                decoded_parts = [header_mapping.get(
                    part, part) for part in key_parts]
                decoded_key = ', '.join(decoded_parts)
                decoded_response[decoded_key] = response_value
            response = decoded_response

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
                        pbar.write(f"✓ 完成剖析表格: {file_path}")
                        results.append((file_key, result, raw_response))

                        # 每处理save_interval个文件，调用保存回调
                        completed_count += 1
                        if save_callback and completed_count % save_interval == 0:
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

    # 加载checkpoint
    if args.ckpt is not None and os.path.exists(args.ckpt):
        files_profilling_results = json.load(
            open(args.ckpt, "r", encoding="utf-8"))

    all_csvfiles = list_csv_files(root_dir)
    unprocessed_files = [f for f in all_csvfiles if file_path_to_key(
        f) not in files_profilling_results.keys()]

    print(
        f"总共 {len(all_csvfiles)} 个文件，已处理 {len(files_profilling_results)} 个文件，剩余 {len(unprocessed_files)} 个文件待处理。")
    print(f"使用 {args.max_workers} 个线程并行处理，每 {args.save_interval} 个文件保存一次结果。")

    # 线程锁，保证文件写入的线程安全
    save_lock = threading.Lock()

    # 输出文件路径
    output_path = args.output_file or f"./output/{get_basename(root_dir)}_{args.prompt_version}_profilling_mapheader.json"

    # 定义保存回调函数
    def save_results(new_results):
        with save_lock:
            for file_key, result, raw in new_results:
                files_profilling_results[str(file_key)] = result
                if args.log:
                    raw_responses[str(file_key)] = raw

            # 保存结果到文件
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(files_profilling_results, f,
                          ensure_ascii=False, indent=2)

            if args.log:
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(raw_responses, f, ensure_ascii=False, indent=2)

            print(f"\n💾 已保存 {len(files_profilling_results)} 个文件的剖析结果")

    # 使用多线程并行处理
    remaining_results = profilling_csv_files_parallel(
        unprocessed_files,
        max_workers=args.max_workers,
        save_callback=save_results,
        save_interval=args.save_interval,
        prompt_version=args.prompt_version
    )

    # 保存最后剩余的结果
    if remaining_results:
        save_results(remaining_results)

    print(f"\n✅ 所有文件处理完成！总共处理了 {len(files_profilling_results)} 个文件。")
    print(f"结果已保存到: {output_path}")
