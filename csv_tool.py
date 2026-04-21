import pandas as pd


def _load_csv(file_path: str,
              drop_empty_rows: bool = True,
              sample_size: int | None = None,
              sample_frac: float | None = None,
              random_state: int = 42) -> pd.DataFrame:
    """
    统一读取 + 清洗 + 采样.
    drop_empty_rows: 是否删除全空行(仅逗号或空白).
    sample_size: 采样固定行数(优先于 sample_frac).
    sample_frac: 按比例采样(0-1).
    """
    if sample_size is not None and sample_frac is not None:
        raise ValueError("sample_size 与 sample_frac 只能二选一")
    df = pd.read_csv(file_path, skip_blank_lines=True, low_memory=False)
    if drop_empty_rows:
        # 将纯空白单元格置为 NA, 删除整行全 NA
        df = df.replace(r'^\s*$', pd.NA, regex=True).dropna(how='all')
    if sample_size is not None:
        sample_size = min(sample_size, len(df))
        df = df.sample(n=sample_size, random_state=random_state)
    elif sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=random_state)
    return df.reset_index(drop=True)


def get_csv_schema(file_path: str) -> dict:
    df = _load_csv(file_path, drop_empty_rows=True)
    schema = {}
    for column in df.columns:
        schema[column] = str(df[column].dtype)
    return schema


def summarize_csv(file_path: str) -> str:
    df = _load_csv(file_path, drop_empty_rows=True)
    summary = df.describe(include='all').to_string()
    return summary


def raw_csv(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def get_csv_column_iterator(file_path: str,
                            drop_empty_rows: bool = True,
                            sample_size: int | None = None,
                            sample_frac: float | None = None):
    """
    逐列生成 (列名, 列值列表).
    """
    df = _load_csv(file_path, drop_empty_rows,
                   sample_size, sample_frac)
    for column in df.columns:
        yield column, df[column].tolist()


def get_csv_column_groups(file_path: str,
                          step: int,
                          drop_empty_rows: bool = True,
                          sample_size: int | None = None,
                          sample_frac: float | None = None,
                          max_cell_length: int = 256):
    """
    按列步长生成子集数据的 CSV 字符串, 已清洗与(可选)采样。
    max_cell_length: 单个单元格最大字符数，超长截断。
    """
    if step <= 0:
        raise ValueError("step 必须为正整数")
    df = _load_csv(file_path, drop_empty_rows,
                   sample_size, sample_frac)
    # 截断过长的单元格
    if max_cell_length and max_cell_length > 0:
        df = df.map(lambda x: str(x)[:max_cell_length] if isinstance(
            x, str) and len(str(x)) > max_cell_length else x)
    cols = df.columns.tolist()
    for i in range(0, len(cols), step):
        group_cols = cols[i:i + step]
        group_df = df[group_cols]
        yield group_cols, group_df.to_csv(index=False)


if __name__ == "__main__":
    file_path = './task/dataset_discovery/santos_small/datalake/HMRC_exceptions_to_spending_controls_October_to_December_2017_property.csv'
    # 列组迭代(步长5), 采样前100行
    print(get_csv_schema(file_path))
    for chunk_csv in get_csv_column_groups(file_path, 5, sample_size=100):
        print("一组列的文本:")
        print(chunk_csv)
    # 逐列迭代(按 30% 行采样)
    for column, data in get_csv_column_iterator(file_path, sample_frac=1):
        print(f"列: {column}, 数据样本: {data[:5]}")
