#!/usr/bin/env python3
"""
Qwen Embedding 索引构建和查询的快速开始示例

演示如何：
1. 构建索引
2. 进行查询
3. 评估结果
"""

import sys
import os
import subprocess
import argparse


def run_command(cmd, description):
    """运行命令并打印进度"""
    print(f"\n{'='*60}")
    print(f"【{description}】")
    print(f"{'='*60}")
    print(f"执行命令: {' '.join(cmd)}")
    print()

    result = subprocess.run(
        cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"✗ 命令执行失败: {description}")
        return False

    print(f"✓ 命令执行成功: {description}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Qwen Embedding 索引构建和查询快速开始'
    )
    parser.add_argument(
        '--step',
        type=str,
        choices=['index', 'query', 'all'],
        default='all',
        help='执行步骤：index(构建索引) | query(查询) | all(全部)'
    )
    parser.add_argument(
        '--benchmark',
        type=str,
        default='SG',
        help='基准数据集名称'
    )
    parser.add_argument(
        '--qwen_model',
        type=str,
        default='qwen3-embedding-4b',
        help='Qwen 模型名称'
    )
    parser.add_argument(
        '--K',
        type=int,
        default=60,
        help='返回的最高匹配表数'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7,
        help='相似度阈值'
    )
    parser.add_argument(
        '--table_mapper',
        action='store_true',
        help='是否使用表名映射'
    )

    args = parser.parse_args()

    print(f"""
╔{'='*58}╗
║ {'Qwen Embedding 索引和查询快速开始':^56} ║
║ {'='*56} ║
║ 基准: {args.benchmark:<46} ║
║ 模型: {args.qwen_model:<46} ║
║ K值: {args.K:<46} ║
║ 阈值: {args.threshold:<46} ║
╚{'='*58}╝
    """)

    success = True

    # 步骤1：构建索引
    if args.step in ['index', 'all']:
        index_cmd = [
            sys.executable,
            'index_qwen.py',
            '--benchmark', args.benchmark,
            '--qwen_model', args.qwen_model,
        ]

        if args.table_mapper:
            index_cmd.append('--table_mapper')

        if not run_command(index_cmd, '构建 Qwen Embedding 索引'):
            success = False

    # 步骤2：执行查询
    if success and args.step in ['query', 'all']:
        query_cmd = [
            sys.executable,
            'query.py',
            '--benchmark', args.benchmark,
            '--encoder', 'tabert',  # encoder 名称保持不变（标识用）
            '--table_path', f'./results/{args.benchmark}/datalake.pkl',
            '--query_path', f'./results/{args.benchmark}/query.pkl',
            '--K', str(args.K),
            '--threshold', str(args.threshold),
        ]

        if args.table_mapper:
            query_cmd.extend([
                '--mapping_file',
                f'./results/{args.benchmark}/table_name_mapping.json'
            ])

        if not run_command(query_cmd, '执行查询'):
            success = False

    # 最终状态
    print(f"\n{'='*60}")
    if success:
        print("✓ 所有步骤执行成功！")
        print("\n生成的文件：")
        print(f"  - 数据湖索引: ./results/{args.benchmark}/datalake.pkl")
        print(f"  - 查询索引: ./results/{args.benchmark}/query.pkl")
        print(f"  - 查询结果: ./results/{args.benchmark}/")
    else:
        print("✗ 执行过程中出现错误，请检查日志")
        return 1

    print(f"{'='*60}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
