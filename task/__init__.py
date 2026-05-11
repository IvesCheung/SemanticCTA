"""
Task 模块

包含各种数据发现、模式匹配等任务的封装类。
"""

from .BaseTask import BaseTask, TaskResult
from .TableDDTask import TableIndexQueryTask
from .DeepJoinDDTask import DeepJoinDDTask
from .StarmieDDTask import StarmieDDTask

__all__ = [
    'BaseTask',
    'TaskResult',
    'TableIndexQueryTask',
    'DeepJoinDDTask',
    'StarmieDDTask',
]
