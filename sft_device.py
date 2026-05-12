import argparse
import time
import sys
import math
import random
import threading

try:
    import pynvml
except ImportError:
    print("请先安装 pynvml: pip install pynvml")
    sys.exit(1)

# 尝试后端: torch > cupy
BACKEND = None
torch = None
cp = None
try:
    import torch
    if torch.cuda.is_available():
        BACKEND = "torch"
except Exception:
    torch = None


if BACKEND is None:
    print("需要安装并配置 PyTorch (GPU) 或 CuPy 任意一个.")
    sys.exit(1)


def init_nvml():
    pynvml.nvmlInit()


def shutdown_nvml():
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass


def get_gpu_handle(index):
    return pynvml.nvmlDeviceGetHandleByIndex(index)


def query_utilization(handle):
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return {
        "gpu_util": util.gpu,          # %
        "mem_util": util.memory,       # %
        "mem_used": meminfo.used,      # bytes
        "mem_total": meminfo.total     # bytes
    }


def human_bytes(n):
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while n >= 1024 and i < len(units)-1:
        n /= 1024
        i += 1
    return f"{n:.2f}{units[i]}"


def allocate_memory(memory_gb, device_index):
    # 申请指定显存 (近似)
    bytes_target = memory_gb * (1024**3)
    if BACKEND == "torch":
        torch.cuda.set_device(device_index)
        dtype_size = torch.tensor([], dtype=torch.float32).element_size()
        # float32
        elements = bytes_target // dtype_size
        # 降低一次性峰值风险: 分块
        chunk = 128 * 1024 * 1024 // dtype_size  # 128MB
        tensors = []
        remaining = elements
        while remaining > 0:
            take = min(chunk, remaining)
            tensors.append(torch.empty(
                int(take), dtype=torch.float32, device="cuda"))
            remaining -= take
        allocated_bytes = sum(t.numel() * t.element_size() for t in tensors)
        return tensors, allocated_bytes
    return [], 0


def compute_loop(matrix_size, seconds, device_index, handle=None,
                 preempt_mem_bytes=0, preempt_check_interval=2.0):
    start = time.time()
    if BACKEND == "torch":
        torch.cuda.set_device(device_index)
        a = torch.randn((matrix_size, matrix_size),
                        device="cuda", dtype=torch.float32)
        b = torch.randn((matrix_size, matrix_size),
                        device="cuda", dtype=torch.float32)
        c = a @ b  # warmup
        torch.cuda.synchronize()
        iters = 0
        preempted = False
        extra_mem_bytes = 0
        expected_mem_used = None
        last_preempt_check = start
        if handle is not None and preempt_mem_bytes > 0:
            expected_mem_used = query_utilization(handle)["mem_used"]
        while time.time() - start < seconds:
            c = a @ b
            # 简单防优化
            a, b = b, c
            iters += 1
            now = time.time()
            if expected_mem_used is not None and now - last_preempt_check >= preempt_check_interval:
                current_mem_used = query_utilization(handle)["mem_used"]
                extra_mem_bytes = max(0, current_mem_used - expected_mem_used)
                if extra_mem_bytes >= preempt_mem_bytes:
                    preempted = True
                    break
                last_preempt_check = now
        torch.cuda.synchronize()
        del a, b, c
        return iters, preempted, extra_mem_bytes
    return 0, False, 0


def workload(mem_gb, mat, duration, device_index, handle, preempt_mem_gb,
             preempt_check_interval):
    print(
        f"[INFO][GPU{device_index}] 启动 GPU 负载: 占用约 {mem_gb}GB 显存 + 持续矩阵乘 {duration}s (backend={BACKEND})")
    allocated, allocated_bytes = allocate_memory(mem_gb, device_index)
    print(
        f"[INFO][GPU{device_index}] 已分配块数量: {len(allocated)} (约 {human_bytes(allocated_bytes)})")
    try:
        iters, preempted, extra_mem_bytes = compute_loop(
            mat,
            duration,
            device_index,
            handle=handle,
            preempt_mem_bytes=int(preempt_mem_gb * (1024 ** 3)),
            preempt_check_interval=preempt_check_interval,
        )
        if preempted:
            print(
                f"[INFO][GPU{device_index}] 检测到额外显存占用约 {human_bytes(extra_mem_bytes)}, 主动结束当前负载")
        else:
            print(f"[INFO][GPU{device_index}] 计算迭代次数: {iters}")
    finally:
        # 显式释放占用的显存
        del allocated
        if BACKEND == "torch":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print(f"[INFO][GPU{device_index}] 已释放本次负载所占显存")
    return not preempted


def parse_gpu_list(gpus_arg, gpu_arg):
    if gpus_arg:
        gpu_ids = []
        for token in gpus_arg.split(","):
            token = token.strip()
            if not token:
                continue
            gpu_ids.append(int(token))
    elif gpu_arg is not None:
        gpu_ids = [gpu_arg]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

    if not gpu_ids:
        raise ValueError("未检测到可用 GPU")

    seen = set()
    result = []
    for gpu_id in gpu_ids:
        if gpu_id in seen:
            continue
        seen.add(gpu_id)
        result.append(gpu_id)
    return result


def monitor_gpu(args, device_index):
    handle = get_gpu_handle(device_index)
    idle_counter = 0
    free_counter = 0
    print(
        f"[INFO] 监控 GPU{device_index}. 条件: 连续 {args.idle_checks} 次 (gpu_util==0 且 mem<{args.mem_threshold}%) 触发负载.")
    while True:
        stats = query_utilization(handle)
        gpu_util = stats["gpu_util"]
        mem_used = stats["mem_used"]
        mem_total = stats["mem_total"]
        mem_pct = mem_used / mem_total * 100
        print(
            f"[STAT][GPU{device_index}] gpu_util={gpu_util:3d}% mem={mem_pct:5.1f}% (<{args.mem_threshold}% 判空) idle_counter={idle_counter}")
        if gpu_util == 0 and mem_pct < args.mem_threshold:
            idle_counter += 1
        else:
            idle_counter = 0
            free_counter += 1
            print(f"[INFO][GPU{device_index}] 目前 GPU 活动中")
        if idle_counter >= args.idle_checks:
            randomized_mem_gb = random.uniform(
                args.mem_gb_min, args.mem_gb_max)
            randomized_duration = random.randint(
                args.duration_min, args.duration_max)
            randomized_loop_time = random.randint(
                args.loop_time_min, args.loop_time_max)

            print(
                f"[RANDOM][GPU{device_index}] 本次负载: 显存={randomized_mem_gb:.1f}GB, 持续={randomized_duration}s, 下次间隔={randomized_loop_time}s")
            workload(
                randomized_mem_gb,
                args.mat,
                randomized_duration,
                device_index,
                handle,
                args.preempt_mem_gb,
                args.preempt_check_interval,
            )
            idle_counter = 0
            if not args.loop:
                print(f"[INFO][GPU{device_index}] 已执行一次负载, 退出.")
                break
            time.sleep(randomized_loop_time)
        if free_counter >= args.idle_checks:
            print(f"[INFO][GPU{device_index}] 有程序在使用 GPU, 不打搅, 等待下一次")
            free_counter = 0
            if not args.loop:
                print(f"[INFO][GPU{device_index}] 已执行一次负载, 退出.")
                break
            time.sleep(args.interval)
        time.sleep(args.interval)


def monitor(args):
    init_nvml()
    threads = []
    try:
        for device_index in args.gpus:
            thread = threading.Thread(
                target=monitor_gpu,
                args=(args, device_index),
                name=f"gpu-monitor-{device_index}",
            )
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
    finally:
        shutdown_nvml()


def parse():
    p = argparse.ArgumentParser(description="空闲时自动占用显存并进行 GPU 计算的监控程序")
    p.add_argument("--gpus", type=str, default=None,
                   help="要监控的 GPU 列表, 例如 0,1,3；默认监控全部 GPU")
    p.add_argument("--gpu", type=int,
                   default=None, help="单卡模式兼容参数")
    p.add_argument("--interval", type=float, default=10.0, help="轮询间隔秒")
    p.add_argument("--idle-checks", type=int,
                   default=5, help="连续多少次 0% 利用率判定为空闲")

    # 显存占用 - 随机化范围
    p.add_argument("--mem-gb-min", type=float,
                   default=5.0, help="负载显存占用最小值(GB)")
    p.add_argument("--mem-gb-max", type=float,
                   default=15.0, help="负载显存占用最大值(GB)")
    # 保留旧参数以兼容，设为已弃用
    p.add_argument("--mem-gb", type=float,
                   default=None, help=argparse.SUPPRESS)

    # 计算持续时间 - 随机化范围
    p.add_argument("--duration-min", type=int, default=60*5, help="计算持续最小秒数")
    p.add_argument("--duration-max", type=int, default=60*10, help="计算持续最大秒数")
    # 保留旧参数
    p.add_argument("--duration", type=int,
                   default=None, help=argparse.SUPPRESS)

    # 循环间隔时间 - 随机化范围 (建议1-4小时)
    p.add_argument("--loop-time-min", type=int, default=60*60*1,
                   help="循环触发最小间隔秒数(默认1h)")
    p.add_argument("--loop-time-max", type=int, default=60*60*4,
                   help="循环触发最大间隔秒数(默认4h)")
    # 保留旧参数
    p.add_argument("--loop-time", type=int,
                   default=None, help=argparse.SUPPRESS)

    p.add_argument("--mat", type=int, default=4096, help="矩阵大小 N (做 N x N 乘法)")
    p.add_argument("--loop", dest="loop", action="store_true",
                   help="多次循环触发 (默认开启)")
    p.add_argument("--no-loop", dest="loop", action="store_false",
                   help="只触发一次后退出")
    p.set_defaults(loop=True)
    p.add_argument("--mem-threshold", type=float, default=5.0,
                   help="判定空闲的显存占用百分比上限 (默认 5%)")
    p.add_argument("--preempt-mem-gb", type=float, default=10.0,
                   help="运行负载时若检测到新增显存占用超过该阈值(GB), 则主动结束")
    p.add_argument("--preempt-check-interval", type=float, default=5.0,
                   help="运行负载时检查额外显存占用的时间间隔(秒)")
    args = p.parse_args()
    args.gpus = parse_gpu_list(args.gpus, args.gpu)
    return args


if __name__ == "__main__":
    args = parse()
    monitor(args)
