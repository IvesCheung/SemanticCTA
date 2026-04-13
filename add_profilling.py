import os
import random
import pandas as pd
import json
from functools import lru_cache
from collections import Counter
# from HYPERPARAMETERS import TableSampleRows
# from utils import get_basename
# TableSampleRows = 1
SPLIT = ","
SEPARATOR = "\n"

# ---------------------------------------------------------------------------
# Noise augmentation support
# ---------------------------------------------------------------------------
# All column-level augmentation operation names
_NOISE_OPS = [
    # ----- gentle ops -----
    'shuffle_col',
    # 'drop_random_cols',
    # 'drop_single_col',
    # 'sample_col',
    # 'drop_num_col',
    # 'drop_text_col',
    # 'drop_nan_col',
    # 'keep_top_cols',
    # 'reverse_col',
    # 'clear_col_values',
    # 'swap_two_cols',
    # ----- brutal ops -----
    'drop_half_cols',
    'null_half_values',
    'corrupt_half_values',
    'shuffle_col_values',
    'overwrite_col_with_other',
    'keep_minimal_cols',
    'corrupt_col_entirely',
    'mask_cell_values',
]

try:
    from taqwen.augment import column_augment as _column_augment
    _AUGMENT_AVAILABLE = True
except ImportError:
    _AUGMENT_AVAILABLE = False
    _column_augment = None

# Statistics counter: maps operation_name -> number of times applied
_noise_stats: Counter = Counter()


def get_noise_stats() -> dict:
    """Return a copy of the current noise operation statistics."""
    return dict(_noise_stats)


def reset_noise_stats():
    """Reset the noise operation statistics counter."""
    _noise_stats.clear()


def print_noise_stats():
    """Print a summary of how many times each noise operation was triggered."""
    total = sum(_noise_stats.values())
    if total == 0:
        print("[Noise Stats] No noise operations applied.")
        return
    print(f"[Noise Stats] Total noise operations applied: {total}")
    for op, count in sorted(_noise_stats.items(), key=lambda x: -x[1]):
        print(f"  {op}: {count}")


def encode_column(table_name: str, header_name: str, col_type, profile, column_samples=None, split=SPLIT, separator=SEPARATOR) -> str:
    """Encode a column with its profilling information and samples"""
    # Do not mutate profile in-place; it may be shared across samples.
    if not isinstance(profile, dict):
        profile = {}

    # description = split.join(
    #     [f"{k}:{v}" for k, v in profile.items()]) if profile else None
    description_values = [v for k, v in profile.items() if k not in [
        '__type__', '__table__']]
    description = "\n+ ".join([f"{v}" for v in description_values if v]
                              ) if description_values else ''

    samples_text = split.join([str(sample) for sample in column_samples]
                              ) if column_samples else None

    _encoded_column = [
        f"Column name: {header_name}",
        f"Column type: {col_type}" if col_type else "",
        f"Description of this column: {description}",
        f"From table: {table_name}, {profile.get('__table__', '')}",
        f"Example data: {samples_text}" if samples_text else "",
    ]
    encoded_column = [_ for _ in _encoded_column if _ != ""]
    return separator.join(encoded_column)


def _encode_column(table_name: str, header_name: str, col_type, profilling_data, column_samples=None, split=SPLIT, separator=SEPARATOR) -> str:
    """Encode a column with its profilling information and samples"""
    # profilling_info = profilling_data
    # col_type = profilling_info.get("__type__", "unknown")
    if '__type__' in profilling_data:
        del profilling_data['__type__']

    description = split.join(
        [f"{k}:{v}" for k, v in profilling_data.items()]) if profilling_data else "No description available."

    samples_text = split.join([str(sample) for sample in column_samples]
                              ) if column_samples else "No samples available."

    encoded_column = [
        f"Table: {table_name}",
        f"Column: {header_name}",
        f"Type: {col_type}",
        f"Description: {description}",
        f"Samples: {samples_text}",
    ]
    return separator.join(encoded_column)


def get_basename(file_path: str) -> str:
    """
    获取文件名（不含路径和扩展名）
    """
    # if os.path.exists(file_path) is False:
    #     raise FileNotFoundError(f"文件不存在: {file_path}")
    base = os.path.basename(file_path)
    name = os.path.splitext(base)[0]
    return name


def get_table_profilling(csv_file_path, profilling_path):
    descriptions_data, norm_map, base_map, keys = _get_profilling_index(
        profilling_path)

    # 标准化文件路径（处理路径分隔符差异）
    normalized_csv_path = os.path.normpath(csv_file_path)

    # Fast paths: exact normalized match, then basename match
    if normalized_csv_path in norm_map:
        return norm_map[normalized_csv_path]

    base = get_basename(normalized_csv_path)
    candidates = base_map.get(base)
    if candidates:
        if len(candidates) == 1:
            return candidates[0]
        # Handle rare basename collisions by falling back to substring match
        for json_path in keys:
            if get_basename(json_path) == base and os.path.normpath(json_path) in normalized_csv_path:
                return descriptions_data.get(json_path)
        return candidates[0]

    # Slow path: scan keys (kept for compatibility with old matching rules)
    for json_path in keys:
        if os.path.normpath(json_path) == normalized_csv_path or \
                get_basename(normalized_csv_path) == get_basename(json_path) or \
                os.path.normpath(json_path) in normalized_csv_path:
            return descriptions_data.get(json_path)
    return None


@lru_cache(maxsize=128)
def _load_profilling_json(profilling_path: str):
    with open(profilling_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@lru_cache(maxsize=128)
def _get_profilling_index(profilling_path: str):
    """Build fast lookup maps for a profiling json.

    Returns:
        (descriptions_data, norm_map, base_map, keys)
    """
    descriptions_data = _load_profilling_json(profilling_path)
    norm_map = {}
    base_map = {}
    keys = list(descriptions_data.keys())
    for json_path, value in descriptions_data.items():
        try:
            norm = os.path.normpath(json_path)
        except Exception:
            norm = json_path
        norm_map[norm] = value
        base = get_basename(json_path)
        base_map.setdefault(base, []).append(value)
    return descriptions_data, norm_map, base_map, keys


def add_column_info(table, csv_file_path, profilling_path):
    """Add column types as first row and descriptions as second row"""
    # 读取JSON文件
    with open(profilling_path, 'r', encoding='utf-8') as f:
        descriptions_data = json.load(f)

    # 标准化文件路径（处理路径分隔符差异）
    normalized_csv_path = os.path.normpath(csv_file_path)

    # 查找匹配的文件路径
    matching_key = None
    # 尝试通过标准化路径匹配目标表
    for json_path in descriptions_data.keys():
        if os.path.normpath(json_path) == normalized_csv_path or \
                get_basename(normalized_csv_path) == get_basename(json_path):
            matching_key = json_path
            break

    if matching_key and matching_key in descriptions_data.keys():
        column_info = descriptions_data[matching_key]
        # 创建类型行（第一行）
        type_row = {}
        # 创建描述行（第二行）
        description_row = {}
        for col in table.columns:
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


_table_cache: dict = {}


def read_table(table_path, profilling_path=None, sample_rows=None, profiling_mode=None,
               noise_prob: float = 0.0, encoding: str = None, low_memory: bool = False):
    """Read a table and optionally add column types/descriptions and random noise.

    Args:
        table_path:      Path to the CSV file.
        profilling_path: Optional path to a profilling JSON file; adds type/description
                         rows to the table when provided.
        sample_rows:     Number of rows to read (None = all).
        profiling_mode:  Reserved for future profiling layout variants.
        noise_prob:      Probability of applying a random column-level augmentation.
                         0.0  = no noise (default).
                         0.1  = 10 % probability, etc.
                         Requires taqwen.augment to be importable.
        encoding:        File encoding passed to pd.read_csv (default: None / utf-8).
        low_memory:      Passed to pd.read_csv; set False to suppress dtype warnings.

    Returns:
        pd.DataFrame
    """
    # Only cache when noise is disabled (noise implies fresh randomness each call)
    use_cache = noise_prob == 0.0
    if use_cache:
        key = (table_path, profilling_path, sample_rows, profiling_mode)
        if key in _table_cache:
            return _table_cache[key]

    read_kwargs = {'nrows': sample_rows,
                   'lineterminator': '\n', 'low_memory': low_memory}
    if encoding:
        read_kwargs['encoding'] = encoding
    table = pd.read_csv(table_path, **read_kwargs)

    # Apply noise *before* profilling so metadata rows correctly reflect the
    # noised column structure (dropped / shuffled columns are handled consistently).
    if noise_prob > 0.0:
        if not _AUGMENT_AVAILABLE:
            import warnings
            warnings.warn(
                "noise_prob > 0 but taqwen.augment could not be imported — "
                "noise augmentation is disabled. Check that taqwen/ is on sys.path.",
                RuntimeWarning, stacklevel=2)
        elif random.random() < noise_prob:
            # 优先从环境变量 NOISE_OP 读取固定算子，否则随机选
            env_op = os.environ.get("NOISE_OP", "").strip()
            op = env_op if env_op in _NOISE_OPS else random.choice(_NOISE_OPS)
            table = _column_augment(table, op)
            _noise_stats[op] += 1

    # 如果提供了描述JSON文件路径，则添加类型和描述信息
    if profilling_path:
        if not os.path.exists(profilling_path):
            raise FileNotFoundError(
                f"Profilling file not found: {profilling_path}")
        table = add_column_info(table, table_path, profilling_path)

    if use_cache:
        _table_cache[key] = table
    return table
