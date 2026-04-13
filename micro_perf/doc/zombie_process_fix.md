# micro_perf 多进程僵尸进程问题修复分析

## 问题背景

程序卡死后会产生大量 `<defunct>` 僵尸进程无法自动清理。根因分布在
`core/engine.py` 的三处代码缺陷。

---

## 问题一：`stop()` 中 `kill()` 后缺少 `join()` → 直接产生僵尸进程

### 受影响位置

- `core/engine.py` — `ComputeEngine.stop()`
- `core/engine.py` — `XCCLEngine.stop()`

### 原始代码（两处相同模式）

```python
kill_flag = False
for subprocess in self.subprocess_procs.processes:
    subprocess.join(timeout=10)
    if subprocess.is_alive():
        kill_flag = True
        break                      # ← (A) 遇到第一个存活进程就跳出，后续进程未 join

if kill_flag:
    for subprocess in self.subprocess_procs.processes:
        subprocess.kill()          # ← (B) kill 后没有任何 join，进程立即变僵尸
```

### 缺陷分析

1. **(A) `break` 提前退出**：只对第一个"超时未退出"的进程做了标记，之后循环直接
   跳出，后续进程即使也超时也没有被 `join()`。
2. **(B) `kill()` 后无 `join()`**：Linux 内核规定，父进程必须通过 `wait()` /
   `waitpid()`（Python 层面是 `join()`）才能回收子进程的进程表条目。`kill()` 仅
   向目标进程发送 `SIGKILL`，进程终止后变为 `<defunct>` 状态，直到父进程 `join()`
   才彻底消除。程序长期运行时，每轮 `stop()` 调用都会留下若干僵尸，最终耗尽进程
   表资源。

### 修复方案

逐一对每个子进程执行 **join → 检查 → kill → 再 join** 的完整流程：

```python
def stop(self):
    if self.subprocess_procs:
        for _ in self.subprocess_procs.processes:
            self.input_queue.put(None)          # 发送退出信号

        for proc in self.subprocess_procs.processes:
            proc.join(timeout=10)               # 等待正常退出
            if proc.is_alive():
                logger.warning(f"Process {proc.pid} did not terminate gracefully, killing it")
                proc.kill()
                proc.join()                     # ← 必须在 kill 后 join，回收僵尸

        self.subprocess_procs = []
        self.subprocess_pids = []

    self.is_running = False
```

---

## 问题二：`dispatch()` 的 `output_queue.get()` 无超时 → 程序卡死

### 受影响位置

- `core/engine.py` — `BaseEngine.dispatch()`（`ComputeEngine` / `XCCLEngine` 共用）

### 原始代码

```python
all_results = {}
for _ in range(task_idx):
    result_idx, result_dict = self.output_queue.get()   # ← 无超时，永久阻塞
    all_results[result_idx] = result_dict
```

### 缺陷分析

子进程因异常崩溃后不会向 `output_queue` 放入结果，父进程在
`output_queue.get()` 处永久阻塞，整个程序"卡死"。此时子进程已经是僵尸（或已被
操作系统完全回收），却没有任何机制能唤醒父进程。

### 修复方案

加入 60 秒超时，超时后检测子进程存活状态，发现死亡进程则抛出 `RuntimeError`
以触发上层清理逻辑：

```python
all_results = {}
for _ in range(task_idx):
    result = None
    while result is None:
        try:
            result = self.output_queue.get(timeout=60)
        except queue.Empty:
            dead_procs = (
                [p for p in self.subprocess_procs.processes if not p.is_alive()]
                if self.subprocess_procs else []
            )
            if dead_procs:
                raise RuntimeError(
                    f"{len(dead_procs)} worker process(es) died unexpectedly "
                    f"(pids: {[p.pid for p in dead_procs]})"
                )
            logger.warning("Waiting for worker result (no response in 60s, retrying)...")
    result_idx, result_dict = result
    all_results[result_idx] = result_dict
```

---

## 问题三：`XCCLEngine.stop()` 心跳线程竞态条件 → shutdown 期间 dispatch 冲突

### 受影响位置

- `core/engine.py` — `XCCLEngine.stop()`

### 原始代码

```python
def stop(self):
    if self.subprocess_procs:
        if self.node_rank == 0:
            self.input_queue.put(None)
        # ... join / kill ...
        self.subprocess_procs = []
        self.subprocess_pids = []

    self.is_running = False     # ← 最后才置 False
```

`_heartbeat_monitor` 线程：

```python
def _heartbeat_monitor(self):
    while self.is_running:          # ← 依赖 is_running 退出
        ...
        self.dispatch(self.demo_test_case)   # ← 可能与 stop() 并发执行
```

### 缺陷分析

`is_running = False` 在 `stop()` 的**最末尾**才设置，在此之前心跳线程仍在运行，
可能调用 `dispatch()` 向子进程发送任务。而 `stop()` 同时向 `input_queue` 放入
`None` 终止信号，两者交错导致子进程收到乱序数据，出现死锁或数据错误。

### 修复方案

在 `stop()` **最开始**先将 `is_running` 置为 `False` 并等待心跳线程退出，再停止
子进程：

```python
def stop(self):
    # 先停心跳线程，避免 shutdown 期间并发 dispatch
    self.is_running = False
    if self.heartbeat_thread and self.heartbeat_thread.is_alive():
        self.heartbeat_thread.join(timeout=5)
        self.heartbeat_thread = None

    if self.subprocess_procs:
        if self.node_rank == 0:
            self.input_queue.put(None)

        for proc in self.subprocess_procs.processes:
            proc.join(timeout=10)
            if proc.is_alive():
                logger.warning(f"Process {proc.pid} did not terminate gracefully, killing it")
                proc.kill()
                proc.join()

        self.subprocess_procs = []
        self.subprocess_pids = []
```

---

## 修改文件汇总

| 文件 | 修改内容 |
|---|---|
| `core/engine.py` | 新增 `import queue` |
| `core/engine.py` | `BaseEngine.dispatch()`：`output_queue.get()` 加 60s 超时 + 死进程检测 |
| `core/engine.py` | `ComputeEngine.stop()`：移除 `kill_flag` 逻辑，改为逐进程 join→kill→join |
| `core/engine.py` | `XCCLEngine.stop()`：先停心跳线程，再逐进程 join→kill→join |
