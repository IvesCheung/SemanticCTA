"""
Convert sota-cpa dataset to column_type_annotation (CTA) format.

Source format (sota-cpa CSVs):
    table_id, column_index, label, data
    - one row per column
    - data: space-separated cell values for that column
    - label: integer class ID (-1 = background/unlabeled column)

Target format (CTA pipeline):
    1. Fold CSVs: table_id, col_idx, class_id
       - train.csv / valid.csv / test.csv
    2. Individual table CSV files: one CSV per table_id
       - headers  = column indices (0, 1, 2, ...)
       - rows     = cell values

Usage:
    python convert_to_cta.py
    python convert_to_cta.py --input_dir . --output_dir ../sota-cpa-cta
"""

import os
import re
import csv
import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm


def sanitize_filename(name: str) -> str:
    """Replace characters that are illegal in Windows/Linux filenames."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)


def write_table_csv(args):
    """Worker function: reconstruct and write one table CSV.

    Args:
        args: (table_id, col_records, output_dir)
              col_records: list of (column_index, data_str)

    Returns:
        True on success, False on skip.
    """
    table_id, col_records, output_dir = args

    # Build column → value list mapping
    columns: dict[int, list] = {}
    for col_idx, raw_data in col_records:
        values = raw_data.split() if isinstance(raw_data, str) else []
        columns[col_idx] = values

    if not columns:
        return False

    max_len = max(len(v) for v in columns.values())
    if max_len == 0:
        return False

    ordered = sorted(columns.keys())
    # Pad all columns to max_len
    rows_data = []
    for r in range(max_len):
        rows_data.append([columns[c][r] if r < len(
            columns[c]) else '' for c in ordered])

    safe_name = sanitize_filename(table_id)
    out_path = os.path.join(output_dir, f'{safe_name}.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(ordered)           # header = column indices
        writer.writerows(rows_data)
    return True


def convert(input_dir: str, output_dir: str, workers: int = 8) -> None:
    os.makedirs(output_dir, exist_ok=True)

    splits = ['train', 'valid', 'test']

    # ── 1. Load all splits ──────────────────────────────────────────────────
    dfs: dict[str, pd.DataFrame] = {}
    # Collect unique tables: table_id → list of (col_idx, data)
    all_tables: dict[str, list] = {}

    for split in splits:
        path = os.path.join(input_dir, f'{split}.csv')
        if not os.path.exists(path):
            print(f'[skip] {path} not found')
            continue
        df = pd.read_csv(path)
        dfs[split] = df
        for tid, grp in df.groupby('table_id', sort=False):
            if tid not in all_tables:
                records = list(zip(
                    grp['column_index'].tolist(),
                    grp['data'].tolist()
                ))
                all_tables[tid] = records
        print(
            f'Loaded {split}: {len(df):,} rows, {df["table_id"].nunique():,} tables')

    # ── 2. Reconstruct and write individual table CSV files (parallel) ──────
    print(
        f'\nWriting {len(all_tables):,} table CSV files (workers={workers}) ...')
    tasks = [(tid, records, output_dir) for tid, records in all_tables.items()]

    success = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(write_table_csv, t): t[0] for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc='Tables'):
            if fut.result():
                success += 1
            else:
                failed += 1

    print(f'  Written: {success:,}  Skipped (empty): {failed:,}')

    # ── 3. Generate fold_0.csv ~ fold_4.csv (required by train_cta.py) ──────
    # Strategy: split train tables into 5 folds (by table_id, so all columns
    # of the same table stay in the same fold).  valid+test remain separate.
    print('\nGenerating fold_0.csv ~ fold_4.csv from train data ...')
    if 'train' in dfs:
        train_df = dfs['train'].copy()
        train_df['table_id'] = train_df['table_id'].apply(sanitize_filename)
        train_df = train_df.rename(columns={
            'column_index': 'col_idx', 'label': 'class_id'})

        unique_train_tables = train_df['table_id'].unique().tolist()
        import random
        random.seed(42)
        random.shuffle(unique_train_tables)

        n = len(unique_train_tables)
        fold_size = (n + 4) // 5          # ceiling division
        for k in range(5):
            fold_tables = set(
                unique_train_tables[k * fold_size: (k + 1) * fold_size])
            fold_df = train_df[train_df['table_id'].isin(fold_tables)][
                ['table_id', 'col_idx', 'class_id']]
            out_path = os.path.join(output_dir, f'fold_{k}.csv')
            fold_df.to_csv(out_path, index=False)
            print(f'  fold_{k}.csv: {len(fold_df):,} rows  '
                  f'({len(fold_tables):,} tables)')
    else:
        print('  [skip] train.csv not found, fold files not generated')

    # ── 4. Copy type_vocab.txt ───────────────────────────────────────────────
    vocab_src = os.path.join(input_dir, 'type_vocab.txt')
    if os.path.exists(vocab_src):
        shutil.copy(vocab_src, os.path.join(output_dir, 'type_vocab.txt'))
        print('\nCopied type_vocab.txt')

    print(f'\nDone → {output_dir}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert sota-cpa → CTA format')
    parser.add_argument('--input_dir', type=str, default='.',
                        help='Directory containing train.csv, valid.csv, test.csv')
    parser.add_argument('--output_dir', type=str, default='../sota-cpa-cta',
                        help='Output directory for converted dataset')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel file-writing threads (default 8)')
    parser.add_argument('--num_folds', type=int, default=5,
                        help='Number of folds to generate from train data (default 5)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    convert(input_dir=args.input_dir,
            output_dir=args.output_dir, workers=args.workers)
