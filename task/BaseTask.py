import time
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable
import subprocess
from typing import List


@dataclass
class TaskResult:
    success: bool
    data: Any = None
    error: Optional[Exception] = None
    start_ts: float = 0.0
    end_ts: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    # 执行时间
    @property
    def elapsed(self) -> float:
        return self.end_ts - self.start_ts if self.end_ts and self.start_ts else 0.0

    # 执行是否成功
    @property
    def is_success(self) -> bool:
        return self.success and self.error is None

    # 获取执行结果
    @property
    def result(self) -> Any:
        if not self.is_success:
            raise RuntimeError(
                "Cannot get result data from a failed TaskResult.")
        return self.data


class BaseTask(ABC):
    """
    统一的表相关任务基类:
    生命周期:
      prepare()  -> 资源与参数检查
      execute()  -> 核心逻辑 (抽象方法, 子类实现)
      validate(result) -> 结果校验 (可选)
      finalize(result) -> 资源清理，补充统计
      run() -> 调度整个流程并返回 TaskResult

    线程相关:
      - 任务本身应避免共享可变状态
      - 内部提供 self._lock 供子类在需要时加锁
      - 提供 as_future(executor) 方便多线程提交
    """

    # --------- Initialization ----------
    def __init__(self, task_name: str, config: Dict[str, Any], **args):
        """
        初始化任务实例。
        :param task_name: 任务名称,可以快速定位到具体任务,最好设计成唯一的
        :param config: 核心配置
        :param args: 一些特定参数, profilling的数据在这里传入,方便克隆时更新
        """
        self.name = task_name
        self.config = config
        self.args = args
        self._lock = threading.RLock()
        self._logger = self._build_logger()
        self._prepared = False

    def _run_cmd(self, args: List[str], cwd: Optional[str] = None) -> None:
        """执行命令行指令"""
        cmd_str = " ".join(args)
        print(f"[{self.name}] Running: {cmd_str}")
        try:
            subprocess.run(args, cwd=cwd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[{self.name}] Command failed: {e}")
            raise

    # --------- Logger ----------
    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"Task.{self.name}")
        if not logger.handlers:
            h = logging.StreamHandler()
            fmt = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
            )
            h.setFormatter(fmt)
            logger.addHandler(h)
            logger.setLevel(logging.INFO)
        return logger

    # --------- Factory / Clone ----------
    def clone(self, **override_args) -> "BaseTask":
        """
        复制当前任务(保持 config)，用于快速为不同表/参数生成实例。
        """
        merged = {**self.args, **override_args}
        return self.__class__(task_name=self.name, config=self.config, **merged)

    # --------- Lifecycle Hooks ----------
    def prepare(self) -> None:
        """
        前置检查 & 轻量资源准备。可在子类扩展。
        """
        with self._lock:
            if self._prepared:
                return
            # 示例: 检查必须参数
            required = self.config.get("required_args", [])
            missing = [r for r in required if r not in self.args]
            if missing:
                raise ValueError(f"Missing required args: {missing}")
            self._prepared = True
            self._logger.debug("Prepared task.")

    # !!!! 核心实现在这里即可 !!!!
    # 若无特殊需求, 只需要实现 execute 方法
    @abstractmethod
    def execute(self) -> Any:
        """
        核心执行逻辑。必须由子类实现。
        返回任意结果数据。
        """
        raise NotImplementedError

    def validate(self, result: TaskResult) -> None:
        """
        结果校验，抛异常代表失败。可在子类覆盖。
        """
        pass

    def finalize(self, result: TaskResult) -> None:
        """
        清理或补充统计信息。异常不应再向外抛出。
        """
        try:
            result.extra.setdefault("task_name", self.name)
        except Exception as e:
            self._logger.warning(f"Finalize encountered error: {e}")

    # --------- Public Run ----------
    def run(self) -> TaskResult:
        """
        调度完整生命周期，返回 TaskResult。
        """
        tr = TaskResult(success=False)
        tr.start_ts = time.time()
        self._logger.info(f"Task [{self.name}] started.")
        try:
            self.prepare()
            data = self.execute()
            tr.data = data
            self.validate(tr)
            tr.success = True
            self._logger.info(
                f"Task [{self.name}] success. elapsed={time.time()-tr.start_ts:.3f}s"
            )
        except Exception as e:
            tr.error = e
            self._logger.error(
                f"Task [{self.name}] failed: {e}", exc_info=True)
        finally:
            tr.end_ts = time.time()
            self.finalize(tr)
        return tr

    # --------- Concurrency Helper ----------
    # 并行调用函数
    def as_callable(self) -> Callable[[], TaskResult]:
        """
        返回可直接在线程池中提交的零参可调用对象。
        """
        def _wrapper():
            return self.run()
        return _wrapper

    def as_future(self, executor) -> Any:
        """
        将任务提交到给定 executor(ThreadPoolExecutor / ProcessPoolExecutor)。
        """
        return executor.submit(self.as_callable())

    # --------- Utility for subclass ----------
    def get_arg(self, key: str, default: Any = None) -> Any:
        # print(key, self.args)
        # print(self.args.get(key, default))
        return self.args.get(key, default)

    def log(self, level: int, msg: str):
        self._logger.log(level, msg)
