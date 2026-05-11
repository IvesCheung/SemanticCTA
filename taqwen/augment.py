import pandas as pd
import random


def augment(table: pd.DataFrame, op: str):
    """Apply an augmentation operator on a table.

    Args:
        table (DataFrame): the input table
        op (str): operator name

    Return:
        DataFrame: the augmented table
    """

    if op == 'drop_col':
        # set values of a random column to 0
        col = random.choice(table.columns)
        table = table.copy()
        table[col] = ""
    elif op == 'sample_row':
        # sample 50% of rows
        if len(table) > 0:
            table = table.sample(frac=0.5)
    elif op == 'sample_row_ordered':
        # sample 50% of rows
        if len(table) > 0:
            table = table.sample(frac=0.5).sort_index()
    elif op == 'shuffle_col':
        # shuffle the column orders
        new_columns = list(table.columns)
        random.shuffle(new_columns)
        table = table[new_columns]
    elif op == 'drop_cell':
        # drop a random cell
        table = table.copy()
        row_idx = random.randint(0, len(table) - 1)
        col_idx = random.randint(0, len(table.columns) - 1)
        table.iloc[row_idx, col_idx] = ""
    elif op == 'sample_cells':
        # sample half of the cells randomly
        table = table.copy()
        col_idx = random.randint(0, len(table.columns) - 1)
        sampleRowIdx = []
        for _ in range(len(table) // 2 - 1):
            sampleRowIdx.append(random.randint(0, len(table) - 1))
        for ind in sampleRowIdx:
            table.iloc[ind, col_idx] = ""
    elif op == 'replace_cells':
        # replace half of the cells randomly with the first values after sorting
        table = table.copy()
        col_idx = random.randint(0, len(table.columns) - 1)
        sortedCol = table[table.columns[col_idx]].sort_values().tolist()
        sampleRowIdx = []
        for _ in range(len(table) // 2 - 1):
            sampleRowIdx.append(random.randint(0, len(table) - 1))
        for ind in sampleRowIdx:
            table.iloc[ind, col_idx] = sortedCol[ind]
    elif op == 'drop_head_cells':
        # drop the first quarter of cells
        table = table.copy()
        col_idx = random.randint(0, len(table.columns) - 1)
        sortedCol = table[table.columns[col_idx]].sort_values().tolist()
        sortedHead = sortedCol[:len(table)//4]
        for ind in range(0, len(table)):
            if table.iloc[ind, col_idx] in sortedHead:
                table.iloc[ind, col_idx] = ""
    elif op == 'drop_num_cells':
        # drop numeric cells
        table = table.copy()
        tableCols = list(table.columns)
        numTable = table.select_dtypes(include=['number'])
        numCols = numTable.columns.tolist()
        if numCols == []:
            col_idx = random.randint(0, len(table.columns) - 1)
        else:
            col = random.choice(numCols)
            col_idx = tableCols.index(col)
        sampleRowIdx = []
        for _ in range(len(table) // 2 - 1):
            sampleRowIdx.append(random.randint(0, len(table) - 1))
        for ind in sampleRowIdx:
            table.iloc[ind, col_idx] = ""
    elif op == 'swap_cells':
        # randomly swap two cells
        table = table.copy()
        row_idx = random.randint(0, len(table) - 1)
        row2_idx = random.randint(0, len(table) - 1)
        while row2_idx == row_idx:
            row2_idx = random.randint(0, len(table) - 1)
        col_idx = random.randint(0, len(table.columns) - 1)
        cell1 = table.iloc[row_idx, col_idx]
        cell2 = table.iloc[row2_idx, col_idx]
        table.iloc[row_idx, col_idx] = cell2
        table.iloc[row2_idx, col_idx] = cell1
    elif op == 'drop_num_col':  # number of columns is not preserved
        # remove numeric columns
        numTable = table.select_dtypes(include=['number'])
        numCols = numTable.columns.tolist()
        textTable = table.select_dtypes(exclude=['number'])
        textCols = textTable.columns.tolist()
        addedCols = 0
        while addedCols <= len(numCols) // 2 and len(numCols) > 0:
            numRandCol = numCols.pop(random.randrange(len(numCols)))
            textCols.append(numRandCol)
            addedCols += 1
        textCols = sorted(textCols, key=list(table.columns).index)
        table = table[textCols]
    elif op == 'drop_nan_col':  # number of columns is not preserved
        # remove a half of the number of columns that contain nan values
        newCols, nanSums = [], {}
        for column in table.columns:
            if table[column].isna().sum() != 0:
                nanSums[column] = table[column].isna().sum()
            else:
                newCols.append(column)
        nanSums = {k: v for k, v in sorted(
            nanSums.items(), key=lambda item: item[1], reverse=True)}
        nanCols = list(nanSums.keys())
        newCols += random.sample(nanCols, len(nanCols) // 2)
        table = table[newCols]
    elif op == 'shuffle_row':
        # shuffle the rows
        table = table.sample(frac=1)
    return table


def column_augment(table: pd.DataFrame, op: str):
    """Apply column-level augmentation operators on a table.

    Args:
        table (DataFrame): the input table
        op (str): operator name
            - 'shuffle_col': shuffle the column orders randomly
            - 'drop_random_cols': randomly drop 30-50% of columns
            - 'drop_single_col': randomly drop one column
            - 'sample_col': sample 50-70% of columns randomly
            - 'drop_num_col': drop numeric columns (keep at least half)
            - 'drop_text_col': drop text columns (keep at least half)
            - 'drop_nan_col': drop columns with most NaN values
            - 'keep_top_cols': keep only the first N columns
            - 'reverse_col': reverse the column order
            - 'clear_col_values': clear values of a random column

    Return:
        DataFrame: the augmented table with column-level changes
    """

    if len(table.columns) == 0:
        return table

    if op == 'shuffle_col':
        # shuffle the column orders randomly
        new_columns = list(table.columns)
        random.shuffle(new_columns)
        table = table[new_columns]

    elif op == 'drop_random_cols':
        # randomly drop 30-50% of columns, keep at least 1 column
        if len(table.columns) > 1:
            drop_ratio = random.uniform(0.3, 0.5)
            n_keep = max(1, int(len(table.columns) * (1 - drop_ratio)))
            kept_cols = random.sample(list(table.columns), n_keep)
            # maintain original order
            kept_cols = sorted(kept_cols, key=list(table.columns).index)
            table = table[kept_cols]

    elif op == 'drop_single_col':
        # randomly drop one column, keep at least 1 column
        if len(table.columns) > 1:
            drop_col = random.choice(table.columns)
            table = table.drop(columns=[drop_col])

    elif op == 'sample_col':
        # sample 50-70% of columns randomly
        if len(table.columns) > 1:
            sample_ratio = random.uniform(0.5, 0.7)
            n_sample = max(1, int(len(table.columns) * sample_ratio))
            sampled_cols = random.sample(list(table.columns), n_sample)
            # maintain original order
            sampled_cols = sorted(sampled_cols, key=list(table.columns).index)
            table = table[sampled_cols]

    elif op == 'drop_num_col':
        # remove numeric columns, keep at least half or at least 1 column
        num_table = table.select_dtypes(include=['number'])
        num_cols = num_table.columns.tolist()
        text_table = table.select_dtypes(exclude=['number'])
        text_cols = text_table.columns.tolist()

        if len(num_cols) > 0 and len(text_cols) > 0:
            # drop some numeric columns
            n_drop = max(1, len(num_cols) // 2)
            dropped_num_cols = random.sample(num_cols, n_drop)
            kept_cols = [
                col for col in table.columns if col not in dropped_num_cols]
            table = table[kept_cols]

    elif op == 'drop_text_col':
        # remove text columns, keep at least half or at least 1 column
        num_table = table.select_dtypes(include=['number'])
        num_cols = num_table.columns.tolist()
        text_table = table.select_dtypes(exclude=['number'])
        text_cols = text_table.columns.tolist()

        if len(text_cols) > 0 and len(num_cols) > 0:
            # drop some text columns
            n_drop = max(1, len(text_cols) // 2)
            dropped_text_cols = random.sample(text_cols, n_drop)
            kept_cols = [
                col for col in table.columns if col not in dropped_text_cols]
            table = table[kept_cols]

    elif op == 'drop_nan_col':
        # remove columns with most NaN values, keep at least half
        if len(table.columns) > 1:
            nan_counts = {}
            for column in table.columns:
                nan_counts[column] = table[column].isna().sum()

            # sort by NaN count in descending order
            sorted_cols = sorted(nan_counts.items(),
                                 key=lambda x: x[1], reverse=True)

            # drop columns with most NaNs, keep at least half
            n_drop = max(1, len(table.columns) // 2)
            cols_to_drop = [col for col, _ in sorted_cols[:n_drop]]

            # only drop if there are NaN values
            if sorted_cols[0][1] > 0:
                kept_cols = [
                    col for col in table.columns if col not in cols_to_drop]
                if len(kept_cols) > 0:
                    table = table[kept_cols]

    elif op == 'keep_top_cols':
        # keep only the first 50-70% of columns
        if len(table.columns) > 1:
            keep_ratio = random.uniform(0.5, 0.7)
            n_keep = max(1, int(len(table.columns) * keep_ratio))
            table = table.iloc[:, :n_keep]

    elif op == 'reverse_col':
        # reverse the column order
        table = table[table.columns[::-1]]

    elif op == 'clear_col_values':
        # clear all values of a random column (set to empty string)
        col = random.choice(table.columns)
        table = table.copy()
        table[col] = ""

    elif op == 'swap_two_cols':
        # randomly swap positions of two columns
        if len(table.columns) >= 2:
            col_indices = list(range(len(table.columns)))
            idx1, idx2 = random.sample(col_indices, 2)
            cols = list(table.columns)
            cols[idx1], cols[idx2] = cols[idx2], cols[idx1]
            table = table[cols]

    # ------------------------------------------------------------------ brutal
    elif op == 'drop_half_cols':
        # drop exactly half the columns (at least 1 kept)
        if len(table.columns) > 1:
            n_drop = max(1, len(table.columns) // 2)
            drop_cols = random.sample(list(table.columns), n_drop)
            table = table.drop(columns=drop_cols)

    elif op == 'null_half_values':
        # set ~50% of cell values to NaN across ~half of the columns
        import string  # noqa: F401 (unused here but keeps import block consistent)
        table = table.copy()
        for col in table.columns:
            if random.random() < 0.5:
                mask = [random.random() < 0.5 for _ in range(len(table))]
                table.loc[[i for i, m in enumerate(mask) if m], col] = None

    elif op == 'corrupt_half_values':
        # overwrite ~50% of values in ~half of the columns with random garbage strings
        import string
        table = table.copy()
        for col in table.columns:
            if random.random() < 0.5:
                for i in range(len(table)):
                    if random.random() < 0.5:
                        garbage = ''.join(random.choices(
                            string.ascii_letters + string.digits,
                            k=random.randint(4, 12)))
                        table.iat[i, table.columns.get_loc(col)] = garbage

    elif op == 'shuffle_col_values':
        # shuffle values within each column independently (destroys row-level correlation)
        table = table.copy()
        for col in table.columns:
            if random.random() < 0.5:
                vals = table[col].tolist()
                random.shuffle(vals)
                table[col] = vals

    elif op == 'overwrite_col_with_other':
        # replace all values of a random column with values from another column
        if len(table.columns) >= 2:
            table = table.copy()
            col1, col2 = random.sample(list(table.columns), 2)
            table[col1] = table[col2].values

    elif op == 'keep_minimal_cols':
        # keep only 1/4 of columns (minimum 1), extremely brutal
        if len(table.columns) > 1:
            n_keep = max(1, len(table.columns) // 4)
            kept_cols = random.sample(list(table.columns), n_keep)
            kept_cols = sorted(kept_cols, key=list(table.columns).index)
            table = table[kept_cols]

    elif op == 'corrupt_col_entirely':
        # fill ALL values of a random half of the columns with garbage strings;
        # cast to object dtype first to avoid dtype conflicts with numeric columns
        import string
        if len(table.columns) >= 1:
            n_corrupt = max(1, len(table.columns) // 2)
            cols_to_corrupt = random.sample(list(table.columns), n_corrupt)
            # low-memory: single copy, all cols → object
            table = table.astype(object)
            for col in cols_to_corrupt:
                col_loc = table.columns.get_loc(col)
                for i in range(len(table)):
                    garbage = ''.join(random.choices(
                        string.ascii_letters + string.digits + string.punctuation,
                        k=random.randint(6, 16)))
                    table.iat[i, col_loc] = garbage

    elif op == 'mask_cell_values':
        # Replace a random proportion of ALL cell values in the table with "###".
        # The proportion is read from env var NOISE_MASK_RATIO (default 0.15).
        import os as _os
        try:
            mask_ratio = float(_os.environ.get('NOISE_MASK_RATIO', '0.15'))
        except ValueError:
            mask_ratio = 0.15
        mask_ratio = max(0.0, min(1.0, mask_ratio))
        table = table.copy()
        for col in table.columns:
            col_loc = table.columns.get_loc(col)
            for i in range(len(table)):
                if random.random() < mask_ratio:
                    table.iat[i, col_loc] = '###'

    return table


def random_column_augment(table: pd.DataFrame):
    """Randomly apply a column-level augmentation operator on a table.

    Args:
        table (DataFrame): the input table

    Return:
        DataFrame: the augmented table with a randomly applied column-level operation
    """

    # List of all available column augmentation operations
    column_ops = [
        'shuffle_col',
        'drop_random_cols',
        'drop_single_col',
        'sample_col',
        'drop_num_col',
        'drop_text_col',
        'drop_nan_col',
        'keep_top_cols',
        'reverse_col',
        'clear_col_values',
        'swap_two_cols',
        # brutal ops
        'drop_half_cols',
        'null_half_values',
        'corrupt_half_values',
        'shuffle_col_values',
        'overwrite_col_with_other',
        'keep_minimal_cols',
        'corrupt_col_entirely',
        'mask_cell_values',
    ]

    # Randomly select an operation
    selected_op = random.choice(column_ops)

    # Apply the selected operation
    return column_augment(table, selected_op)
