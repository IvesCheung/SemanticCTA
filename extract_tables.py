import pandas as pd
import os
from collections import defaultdict

# 读取Excel文件
xlsx_path = 'datasets/SM/omop_synthea_data.xlsx'
df = pd.read_excel(xlsx_path)

print(f'=== 原始数据统计 ===')
print(f'总行数: {len(df)}')

def extract_tables(df, column_name):
    """从指定列提取表结构"""
    df_copy = df.copy()
    df_copy['table_name'] = df_copy[column_name].apply(lambda x: x.split('-')[0])
    df_copy['col_name'] = df_copy[column_name].apply(lambda x: '-'.join(x.split('-')[1:]))

    tables_dict = {}
    for table_name in sorted(df_copy['table_name'].unique()):
        columns = df_copy[df_copy['table_name'] == table_name]['col_name'].unique().tolist()
        tables_dict[table_name] = columns

    return tables_dict

# 提取omop表和table表
omop_tables = extract_tables(df, 'omop')
table_tables = extract_tables(df, 'table')

# 输出目录
omop_output_dir = 'datasets/SM/extracted_tables/omop'
table_output_dir = 'datasets/SM/extracted_tables/synthea'
os.makedirs(omop_output_dir, exist_ok=True)
os.makedirs(table_output_dir, exist_ok=True)

# 生成OMOP表的CSV文件
print(f'\n=== OMOP Tables ({len(omop_tables)}个表) ===')
omop_total_cols = 0
for table_name, columns in omop_tables.items():
    output_path = os.path.join(omop_output_dir, f'{table_name}.csv')
    pd.DataFrame(columns=columns).to_csv(output_path, index=False)
    omop_total_cols += len(columns)
    print(f'  {table_name}.csv: {len(columns)} columns')

print(f'\n=== Synthea Tables ({len(table_tables)}个表) ===')
table_total_cols = 0
for table_name, columns in table_tables.items():
    output_path = os.path.join(table_output_dir, f'{table_name}.csv')
    pd.DataFrame(columns=columns).to_csv(output_path, index=False)
    table_total_cols += len(columns)
    print(f'  {table_name}.csv: {len(columns)} columns')

print(f'\n=== 总结 ===')
print(f'OMOP表数: {len(omop_tables)}, 总列数: {omop_total_cols}')
print(f'Synthea表数: {len(table_tables)}, 总列数: {table_total_cols}')
print(f'总计表数: {len(omop_tables) + len(table_tables)}, 总列数: {omop_total_cols + table_total_cols}')
print(f'\n输出目录:')
print(f'  OMOP: {omop_output_dir}')
print(f'  Synthea: {table_output_dir}')
