import argparse
import copy
import json
import os
import time
from pathlib import Path

from tqdm import tqdm

from csv_tool import get_csv_column_groups, get_csv_schema
from llm_tool.call_llm import call_llm
from llm_tool.prompt import get_prompt
from utils import (
    file_path_to_key,
    get_basename,
    list_csv_files,
    log_file_path,
    safe_json_loads,
)


CONFIG = {
    "sample_size": 64,
    "sample_step": 64,
    "rephrase_model": "qwen2.5-72b-instruct",
    "drop_empty_rows": True,
    "description": "使用 LLM 对已有 profiling 结果进行改写（重写每列 description，保留 type 和嵌套结构）。",
}


def rephrase_table_profile(
    file_path: str | Path,
    original_table_profile: dict,
    sample_size: int = CONFIG.get("sample_size", 64),
    sample_step: int = CONFIG.get("sample_step", 64),
    rephrase_model: str = CONFIG.get("rephrase_model", "qwen2.5-72b-instruct"),
    drop_empty_rows: bool = CONFIG.get("drop_empty_rows", True),
    **args,
):
    """Rephrase per-column descriptions for one CSV table.

    Strategy:
    - Process columns in groups (sample_step) to control prompt size.
    - Call prompt version: rephrase_columns.
    - Merge back by updating ONLY the per-column description at key `<col>`.
      Preserve `__type__` and all other keys/values.
    """
    tablename = get_basename(str(file_path))

    # Start from a deep copy so we don't mutate input.
    new_table_profile: dict = copy.deepcopy(original_table_profile)

    # Track one table-level description for consistency across column groups.
    # Try to get existing table description from original profile
    table_desc: str | None = None
    for _col, _obj in original_table_profile.items():
        if isinstance(_obj, dict):
            _t = _obj.get("__table__")
            if isinstance(_t, str) and _t.strip():
                table_desc = _t.strip()
                break

    raw_responses: dict[str, str] = {}

    # Use schema headers to drive grouping (keeps consistent ordering).
    try:
        schema_cols = list(get_csv_schema(str(file_path)).keys())
    except Exception as e:
        print(f"[WARN] Failed to get schema for {file_path}: {e}")
        return new_table_profile, raw_responses

    if not schema_cols:
        print(f"[WARN] Empty schema for {file_path}")
        return new_table_profile, raw_responses

    for cols, chunk_csv in get_csv_column_groups(
        str(file_path),
        step=sample_step,
        drop_empty_rows=drop_empty_rows,
        sample_size=sample_size,
    ):
        # Only rephrase columns that exist in the original profiling.
        cols = [c for c in cols if c in original_table_profile]
        if not cols:
            continue

        profile_subset = {c: original_table_profile[c] for c in cols}

        prompt = get_prompt(
            "rephrase_columns",
            table_name=tablename,
            headers=cols,
            csv_encoded=chunk_csv,
            max_rows_hint=len(chunk_csv.split("\n")) - 1,
            profile=profile_subset,
        )

        response = call_llm(
            model=rephrase_model,
            msgs=[
                {"role": "system",
                    "content": "You are a data analysis and data dictionary assistant"},
                {"role": "user", "content": prompt},
            ],
        )

        raw_responses[",".join(cols)] = response

        parsed = safe_json_loads(response)
        if not isinstance(parsed, dict):
            continue

        # Update table_desc from the first valid one we get from model
        # (Keep the first one for consistency across multiple column groups)
        if table_desc is None:
            for _col in cols:
                _obj = parsed.get(_col)
                if isinstance(_obj, dict):
                    _t = _obj.get("__table__")
                    if isinstance(_t, str) and _t.strip():
                        table_desc = _t.strip()
                        break

        for col in cols:
            new_col_obj = parsed.get(col)
            if not isinstance(new_col_obj, dict):
                continue

            # Get the rewritten description for this column
            new_desc = new_col_obj.get(col)
            if not isinstance(new_desc, str) or not new_desc.strip():
                continue

            # Ensure the column exists in new_table_profile with proper structure
            if col not in new_table_profile or not isinstance(new_table_profile[col], dict):
                new_table_profile[col] = {}

            # Update the column description
            new_table_profile[col][col] = new_desc.strip()

            # Preserve __type__ from original (mandatory)
            if isinstance(original_table_profile.get(col), dict) and "__type__" in original_table_profile[col]:
                new_table_profile[col]["__type__"] = original_table_profile[col]["__type__"]

            # Update __table__ (use rewritten one if available, otherwise keep original)
            if table_desc is not None:
                new_table_profile[col]["__table__"] = table_desc
            elif isinstance(original_table_profile.get(col), dict) and "__table__" in original_table_profile[col]:
                new_table_profile[col]["__table__"] = original_table_profile[col]["__table__"]

    return new_table_profile, raw_responses


def main():
    parser = argparse.ArgumentParser(description=CONFIG["description"])
    parser.add_argument("--root_dir", type=str,
                        required=True, help="CSV 文件根目录")
    parser.add_argument("--profilling_file", type=str,
                        required=True, help="原始 profilling 结果 JSON 文件路径")
    parser.add_argument("--output_file", type=str,
                        default=None, help="输出结果文件路径")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="从哪个输出文件继续（断点续跑）")
    parser.add_argument("--log", action="store_true", help="是否记录每次调用的原始返回")
    parser.add_argument("--sleep_time", type=int,
                        default=0, help="表与表之间休息时间（秒）")
    parser.add_argument("--sample_size", type=int,
                        default=CONFIG["sample_size"], help="每张表最多采样多少行")
    parser.add_argument("--sample_step", type=int,
                        default=CONFIG["sample_step"], help="每次处理多少列")
    parser.add_argument("--rephrase_model", type=str,
                        default=CONFIG["rephrase_model"], help="使用的 LLM 模型")
    parser.add_argument("--no_drop_empty_rows", action="store_true",
                        help="不剔除空行（默认会剔除空行）")

    args = parser.parse_args()

    # Set drop_empty_rows based on the flag (inverted logic)
    drop_empty_rows = not args.no_drop_empty_rows

    os.makedirs("./output", exist_ok=True)

    with open(args.profilling_file, "r", encoding="utf-8") as f:
        original_profilling = json.load(f)

    # Resume from checkpoint (output file).
    rephrased_results: dict = {}
    if args.ckpt is not None and os.path.exists(args.ckpt):
        with open(args.ckpt, "r", encoding="utf-8") as f:
            rephrased_results = json.load(f)

    # Default output path.
    if args.output_file:
        output_path = args.output_file
    else:
        output_path = os.path.join(
            "./output", f"rephrased_{os.path.basename(args.profilling_file)}")

    # Optional raw log path.
    raw_log_path = None
    raw_responses_all: dict = {}
    if args.log:
        raw_log_path = log_file_path(__file__, suffix=".json")

    all_csvfiles = list_csv_files(args.root_dir)
    iterator = tqdm(all_csvfiles, desc="Rephrase profiling", unit="file")

    for csv_path in iterator:
        key = file_path_to_key(str(csv_path))
        key = os.path.basename(key)  # Use basename as key for matching
        if key in rephrased_results:
            continue

        original_table_profile = original_profilling.get(key)
        if not isinstance(original_table_profile, dict):
            # No original profiling for this table.
            continue

        iterator.write(f"Rephrasing: {csv_path}")
        try:
            new_profile, raw = rephrase_table_profile(
                csv_path,
                original_table_profile,
                sample_size=args.sample_size,
                sample_step=args.sample_step,
                rephrase_model=args.rephrase_model,
                drop_empty_rows=drop_empty_rows,
            )
            rephrased_results[key] = new_profile

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(rephrased_results, f, ensure_ascii=False, indent=2)

            if args.log and raw_log_path:
                raw_responses_all[key] = raw
                with open(raw_log_path, "w", encoding="utf-8") as f:
                    json.dump(raw_responses_all, f,
                              ensure_ascii=False, indent=2)

        except Exception as e:
            iterator.write(f"[ERROR] Failed rephrasing {csv_path}")
            iterator.write(f"        Error: {type(e).__name__}: {e}")
            continue

        time.sleep(args.sleep_time)


if __name__ == "__main__":
    main()
