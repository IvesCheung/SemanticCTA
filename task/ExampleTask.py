import time
import random
from .BaseTask import BaseTask, TaskResult
from profilling import profilling_tables, profilling_table
# 具体profilling的结果形式可在output\profilling_result.json中查看
# 记得传递好相关参数
CONFIG = {
    "prompt_version": "base_profilling",            # 选择使用的 prompt 模板
    "sample_size": 64,                              # 每张表最多采样多少行进行剖析
    "sample_step": 64,                              # 每次处理多少列进行剖析
    "profilling_model": "qwen2.5-72b-instruct",     # 使用的 LLM 模型
    "drop_empty_rows": True,                        # 是否剔除空行后再进行剖析
    "description": "使用 LLM 对表格数据进行剖析，生成数据字典。",
}


class ExampleTask(BaseTask):
    """
    示例: 对指定表进行简单的剖析任务。
    """

    def prepare(self):
        super().prepare()
        assert "target_table_path" in self.args, "target_table_path is required."
        assert "target_table_paths" in self.args and type(
            self.args["target_table_paths"]) is list, "target_table_paths is required and must be a list."

    # 若无特殊需求,只需要实现 execute 方法, profilling信息均在self.get_arg 中获取
    def execute(self):
        # 模拟耗时
        time.sleep(0.1)
        table = self.get_arg("target_table_path")
        tables = self.get_arg("target_table_paths")
        print(f"开始对单表进行剖析: {table}")
        profiling_result = profilling_table(
            file_path=table,
            **CONFIG
        )
        print(f"单表剖析结果: {profiling_result}")
        profiling_results = profilling_tables(
            file_paths=tables,
            **CONFIG
        )
        print(f"多表剖析结果: {profiling_results}")
        # 这个数据会返回到执行结果中
        return profiling_results

    def validate(self, result: TaskResult):
        data = result.data
        if not isinstance(data, dict):
            raise ValueError("Result data must be dict.")

    def finalize(self, result: TaskResult):
        super().finalize(result)
        # 增补一些统计
        if result.success and isinstance(result.data, dict):
            result.extra["columns_profiled"] = random.randint(5, 20)


if __name__ == "__main__":
    target_table_path = './task/dataset_discovery/santos_small/datalake/HMRC_exceptions_to_spending_controls_October_to_December_2017_property.csv'
    task = ExampleTask(
        task_name="example_profilling_task",
        config={},
        target_table_path=target_table_path,
        target_table_paths=[target_table_path]
    )
    result = task.run()
    print("Final Task Result:")
    print(result.data)
