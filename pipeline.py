from concurrent.futures import ThreadPoolExecutor, as_completed
from task.ExampleTask import ExampleTask

tables = ["user", "order", "product", "inventory"]
tasks = [ExampleTask("profile_task", config={}, table_name=t) for t in tables]

with ThreadPoolExecutor(max_workers=4) as pool:
    futures = {task.as_future(pool): task for task in tasks}
    for fut in as_completed(futures):
        r = fut.result()
        print(r.success, r.data, r.elapsed)
