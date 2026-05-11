"""
测试列级别对比学习训练脚本
使用示例：
python test_column_contrastive.py
"""
import sys
import os

# 确保可以导入项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_column_dataset():
    """测试ColumnContrastiveDataset"""
    from taqwen.contrastive_learning_profilling import ColumnContrastiveDataset
    import glob

    # 获取测试CSV文件
    csv_files = glob.glob('./datasets/opendata_SG/*.csv')[:5]  # 只使用前5个文件进行测试

    if not csv_files:
        print("No CSV files found in ./datasets/opendata_SG/")
        return False

    print(f"Found {len(csv_files)} CSV files for testing")

    # 创建数据集
    try:
        dataset = ColumnContrastiveDataset(
            csv_files=csv_files,
            profilling_path1='./output/profilling_result.json',
            profilling_path2='./output/profilling_result.json',
            sample_rows=3
        )

        print(f"Dataset created successfully with {len(dataset)} columns")

        # 测试获取一个样本
        if len(dataset) > 0:
            sample = dataset[0]
            print("\nSample data:")
            print(f"Column: {sample['column_name']}")
            print(f"Anchor length: {len(sample['anchor'])}")
            print(f"Positive length: {len(sample['positive'])}")
            print("\nFirst 200 chars of anchor:")
            print(sample['anchor'][:200])
            return True
        else:
            print("Dataset is empty")
            return False
    except Exception as e:
        print(f"Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_collate_fn():
    """测试column_collate_fn"""
    from taqwen.contrastive_learning_profilling import ColumnContrastiveDataset, column_collate_fn
    from torch.utils.data import DataLoader
    import glob

    # 获取测试CSV文件
    csv_files = glob.glob('./datasets/opendata_SG/*.csv')[:3]

    if not csv_files:
        print("No CSV files found")
        return False

    try:
        dataset = ColumnContrastiveDataset(
            csv_files=csv_files,
            profilling_path1='./output/profilling_result.json',
            profilling_path2='./output/profilling_result.json',
            sample_rows=3
        )

        # 创建DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=column_collate_fn
        )

        # 测试获取一个batch
        batch = next(iter(dataloader))
        print(f"\nBatch test:")
        print(f"Anchor batch size: {len(batch['anchor'])}")
        print(f"Positive batch size: {len(batch['positive'])}")
        print(f"Negative batch size: {len(batch['negative'])}")
        return True
    except Exception as e:
        print(f"Error testing collate_fn: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("="*60)
    print("Testing Column Contrastive Learning Components")
    print("="*60)

    print("\n1. Testing ColumnContrastiveDataset...")
    test1 = test_column_dataset()

    print("\n" + "="*60)
    print("2. Testing column_collate_fn...")
    test2 = test_collate_fn()

    print("\n" + "="*60)
    if test1 and test2:
        print("✓ All tests passed!")
        print("\nYou can now run the training with:")
        print("python taqwen/contrastive_learning_profilling.py --mode column --csv_dir ./datasets/opendata_SG --epoch_num 2 --batch_size 8")
    else:
        print("✗ Some tests failed. Please check the errors above.")
