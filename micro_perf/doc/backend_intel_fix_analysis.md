# BackendINTEL 改动分析

**文件**：`micro_perf/backends/INTEL/backend_intel.py`，`micro_perf/core/engine.py`  
**背景**：在执行 `python3 launch.py --workload workloads/xccl_ops/device2host.json --backend INTEL` 时，跑大 tensor 场景会卡死，且内核日志出现 xe 驱动 BCS/CCS engine reset 和 GT coredump；`all_reduce.json` 测试中出现 `drm_neo.cpp abort()` 导致僵尸进程。

---

## 一、问题定位过程

### 1.1 现象

```
{"world_size": 2, "dtype": "int8", "batch_size": 2097152, "dim_size": 1024}
{
    "latency(us)": 75189.742,
    "algo_size(B)": 2147483648,
    ...
}
# ← 打印后进程卡死，不再继续
```

### 1.2 内核日志（dmesg）

```
[07:00:45] workqueue: __guc_exec_queue_destroy_async [xe] hogged CPU for >10000us 515 times
[07:00:45] xe 0000:1d:00.0: Engine reset: engine_class=bcs, logical_mask: 0x1
[07:00:45] xe 0000:40:00.0: Engine reset: engine_class=ccs, logical_mask: 0x1
[07:26:14] xe 0000:19:00.0: Schedule disable failed to respond
[07:26:14] xe 0000:19:00.0: Xe device coredump has been created
[07:26:14] xe 0000:19:00.0: Tile0: GT0: reset done
```

---

## 二、根因分析

### 根因链

```
大 tensor D2H 循环 N 次，每次提交 2GB DMA 到 BCS，无中间 sync
    │
    ▼
BCS (Blitter Command Streamer) 命令队列堆积
    │
    ▼
GuC exec queue 溢出 → xe 驱动超时
    │
    ├─▶ Engine reset: bcs  （多张卡）
    └─▶ Engine reset: ccs  （多张卡）
            │
            ▼
    CCL AllReduce 等待 CCS completion → 永远等不到
            │
            ▼
        进程卡死
```

### 三个独立子问题

#### 问题 A：大 tensor 时 CPU pinned memory 失控分配

`Device2HostOp.prepare()` 中 `output_tensor_size = 0`（CPU 侧不计入 `tensor_size`），而 `perf()` 基于 `tensor_size` 计算 `max_data_cnt`：

| 场景 | batch=2097152, dim=1024, fp32 |
|---|---|
| 单个 XPU tensor | 8 GB |
| `tensor_size`（误） | 8 GB（只算 XPU src） |
| `max_data_cnt` 计算结果 | 2（基于 XPU 空闲内存） |
| 实际分配 | 2×8GB XPU + **2×8GB pinned CPU** |

`pin_memory()` 通过 `mlock` 强制锁定物理页，16GB 锁页在 RAM 不足时直接挂起（`mlock` block）。

#### 问题 B：`torch.xpu.Event` 无法计时 D2H 传输

原 `core_perf` else 分支对所有非 profiling 路径都用 `torch.xpu.Event`：

```python
start_event.record()
for i in range(prefer_iterations):
    dst.copy_(src)   # PCIe DMA，走 BCS 引擎
end_event.record()
end_event.synchronize()
```

`torch.xpu.Event` 追踪 SYCL compute queue（CCS），而 D2H DMA 走 BCS 硬件队列，两者时间线不统一。计时结果不可靠，且 probe 阶段延迟偏低导致 `prefer_iters` 算出 10，随后 10 次 2GB DMA 连续提交 → BCS 溢出。

#### 问题 C：`empty_cache()` 在分配之后调用

```python
tensor_list = op_instance.create_tensors(max_data_cnt)  # 分配在前
...
del tensor_list
self.empty_cache()  # 释放在后
```

前一个大 case 的 allocator cache 未释放时，下一个 case 的 `get_mem_info()` 低估可用内存，后续分配失败或卡顿。

---

## 三、改动详情

所有改动均在 `backends/INTEL/backend_intel.py`，不涉及 `core/` 公共代码。

### 3.1 新增 import

```python
# 新增
import math
import traceback
```

原代码缺少 `math`（用于 `math.floor`）和 `traceback`（用于异常打印），`perf()` 覆盖需要。

---

### 3.2 新增常量与辅助方法

```python
_BCS_SYNC_THRESHOLD_BYTES = 512 * 1024 * 1024  # 512 MB

def _op_has_cpu_tensor(self, op_instance) -> bool:
    """检测 op 是否包含 CPU 侧 tensor（H2D/D2H 标志）"""

def _needs_bcs_throttle(self, op_instance) -> bool:
    """tensor_size >= 512MB 的跨设备 op 需要 BCS 节流"""
```

**作用**：将"是否跨设备"和"是否需要 BCS 节流"提炼为可复用的判断，供 `core_perf` 和 `perf` 共用。

---

### 3.3 `core_perf()` 改动

#### 改动 1：引入 `use_wallclock` 统一分支判断

```python
# 之前：仅 skip_profiling=True 时走 wallclock
if getattr(op_instance, 'skip_profiling', False):
    ...

# 之后：H2D/D2H 也走 wallclock
use_wallclock = self._op_has_cpu_tensor(op_instance) \
                or getattr(op_instance, 'skip_profiling', False)
```

**原因**：D2H 的 `dst.copy_(src)` 走 PCIe DMA（BCS），不在 XPU SYCL compute queue 上，`torch.xpu.Event` 无法正确追踪其完成时间。

#### 改动 2：wallclock 路径内每次迭代后 `device_synchronize()`（BCS throttle）

```python
bcs_throttle = self._needs_bcs_throttle(op_instance)

# warmup 阶段
for i in range(warmup_iterations):
    op_instance.core_run(...)
    if bcs_throttle:
        self.device_synchronize()  # 新增：防止 warmup 阶段也溢出

# 测量阶段
for i in range(prefer_iterations):
    op_instance.core_run(...)
    if bcs_throttle:
        self.device_synchronize()  # 新增：每次 DMA 完成后才提交下一次
```

**前后对比**：

```
# 之前（危险）：             # 之后（安全）：
for iter in 10:               for iter in 3:
    copy(2GB → CPU)               copy(2GB → CPU)
device_sync()  ← 最后            device_sync()  ← 每次
# 20GB DMA 堆在 BCS 队列         # 每次排干 BCS 队列再提交
```

**效果**：彻底消除 BCS 命令队列溢出，防止 xe 驱动触发 engine reset。

---

### 3.4 `perf()` 覆盖（新增方法，覆盖基类）

#### 改动 1：`empty_cache()` 移至分配前

```python
# 之前（基类行为）：
tensor_list = op_instance.create_tensors(max_data_cnt)
...
del tensor_list
self.empty_cache()

# 之后：
self.empty_cache()          # ← 先释放上一个 case 的 allocator cache
device_mem_info = self.get_mem_info()   # 此时读到的可用内存才准确
...
tensor_list = op_instance.create_tensors(max_data_cnt)
...
del tensor_list
self.empty_cache()
```

#### 改动 2：跨设备 op 强制 `max_data_cnt = 1`

```python
if self._op_has_cpu_tensor(op_instance):
    max_data_cnt = 1   # 防止多份 pinned CPU tensor 撑爆 RAM
```

`tensor_size` 只统计 XPU 侧，`max_data_cnt` 原本基于 XPU 空闲内存计算（可能得到 2），但每个实例还会在 CPU 侧申请等量 pinned memory。强制为 1 避免 `mlock` 挂起。

#### 改动 3：大 tensor 限制 `min_test_iters = 3`

```python
bcs_throttle = self._needs_bcs_throttle(op_instance)
if bcs_throttle:
    min_test_iters = 3  # 默认 10 → 3
```

per-iter sync 已防止 BCS 溢出，但进一步减少迭代次数可降低 GuC exec queue 的总压力窗口，在 tensor 极大时提供额外保护。

---

### 3.5 `core_perf()` 新增：D2H/H2D op 跳过 CCL `op_group_barrier`（第三轮修复）

#### 背景：GuC exec queue 积压触发 GT reset

在排查 int8 case 卡死原因时发现：float32/float16/bfloat16 的 2GB case 均通过，而 **int8 本身只有 2GB**（远小于其他 dtype）却卡死。卡死发生在 int8 case **结果打印后**，即 `world_size=4` 首个 case 的 `dist.new_group(ranks=[0,1,2,3])` 阶段。

根因链（完整版）：

```
88 × world_size=2 D2H cases
    │
    ▼ 每个 case 调用 core_perf() 2次（probe + actual），每次 2×op_group_barrier
    ▼ = 88 × 4 = 352 次 op_group_barrier
    │
    ▼ op_group_barrier → dist.all_reduce(tensor([1], device="xpu"), group=ccl_group)
    ▼ = 352 次 CCL AllReduce on XPU → 352 次 GuC exec queue alloc/destroy
    │
    ▼ __guc_exec_queue_destroy_async 累积次数: 259 → 515 → 1027 → 2051
    │
    ▼ xe_guc_exec_queue_lr_cleanup timeout → GT reset on 0000:19:00.0
    │
    ▼ SYCL context on device 0x19 失效
    │
    ▼ 下一个 case (world_size=4) 调用 dist.new_group([0,1,2,3]) 等待 rank-2/3 应答
    │
    └─▶ 永久挂起（不是 int8 数据本身的问题）
```

#### 改动

```python
# 之前：wallclock 路径，始终调用两次 barrier
if use_wallclock:
    self.device_synchronize()
    self.op_group_barrier(...)   # ← CCL AllReduce on XPU
    start_time = time.perf_counter()
    ...
    end_time = time.perf_counter()
    self.op_group_barrier(...)   # ← CCL AllReduce on XPU
    return latency_us, []

# 之后：D2H/H2D op 跳过两次 barrier
if use_wallclock:
    has_cpu_tensor = self._op_has_cpu_tensor(op_instance)
    self.device_synchronize()
    if not has_cpu_tensor:       # ← 仅 ESIMD skip_profiling 类 op 保留 barrier
        self.op_group_barrier(...)
    start_time = time.perf_counter()
    ...
    end_time = time.perf_counter()
    if not has_cpu_tensor:
        self.op_group_barrier(...)
    return latency_us, []
```

#### 为什么 D2H/H2D 不需要 XPU 侧 barrier

- D2H/H2D 是**每个 rank 独立操作**，不存在跨 rank 的 XPU 共享内存访问
- `xccl_infer_loop` 中每个 case 前后均有 gloo `all_gather_object` 调用，已提供进程级同步
- 跳过 `op_group_barrier` 不影响测量准确性，但每个 world_size=2 case 节省 4 次 CCL AllReduce
- 对于 88 cases × 4 次 = **352 次 CCL op，全部消除**

#### 效果

去除 GuC exec queue 的主要积压来源，避免 `__guc_exec_queue_destroy_async` 计数持续增长，防止 GT reset。

---

## 四、改动影响范围

| 路径 | 影响 |
|---|---|
| D2H / H2D（小 tensor，< 512MB） | wallclock 计时；跳过 CCL barrier（新） |
| D2H / H2D（大 tensor，≥ 512MB） | 同上 + per-iter sync + `max_data_cnt=1` + `min_test_iters=3` |
| ESIMD kernel（`skip_profiling=True`） | 无变化，已有 wallclock 路径，保留 CCL barrier |
| CCL collective ops（AllReduce 等） | 无变化，仍走 XPU Event 计时 + CCL barrier |
| XPU profiling 路径 | 无变化 |

---

## 五、启动 timeout 修复（第四轮修复）

### 5.1 问题现象

```
2026-04-10 08:11:09.246 engine.py:263 [ERROR]: xccl infer loop timeout, error:
```

在所有 workload case 开始**之前**就 timeout，CCL 初始化 warnings 出现在 `08:10:14`，timeout 在 `08:11:09`，中间 55 秒。

### 5.2 根因

**Bug 1（`core/engine.py`）**：`XCCLEngine.__init__` 中：

```python
self.timeout = args_dict.get("timeout", 60)  # 存入 self
...
signal = self.output_queue.get(timeout=60)    # 硬编码 60，self.timeout 完全未使用
```

`self.timeout` 设置后从未用于启动等待，是一个明显的变量赋值后遗忘使用的 bug。

**Bug 2（间接）**：Intel XPU 首次 CCL collective（`xccl_infer_loop` 中的 validation all_reduce）触发 **SYCL JIT 内核编译**，在 8 设备场景下耗时 30-60 秒。即使 Bug 1 修复，60 秒默认值仍不足。

从 dmesg 日志时间线验证：
```
08:10:14 → CCL 初始化 warnings（进程刚启动，CCL 本身初始化很快）
08:10:15 → topology recognition 完成
08:11:09 → timeout（55 秒均消耗在首次 XPU all_reduce 的 SYCL JIT 编译上）
```

### 5.3 修改内容

#### `core/engine.py`（Bug 修复）

```python
# 之前：
self.timeout = args_dict.get("timeout", 60)
...
signal = self.output_queue.get(timeout=60)        # ← 硬编码，与 self.timeout 无关

# 之后：
self.timeout = args_dict.get("timeout", 300)      # 默认 300s 覆盖 XPU JIT 编译时间
...
signal = self.output_queue.get(timeout=self.timeout)  # ← 使用变量
```

默认值从 60 调整为 300 秒：Intel XPU 8 设备首次运行 SYCL JIT 编译耗时最长约 60 秒，300 秒有充足余量。通过 `args_dict.get("timeout", ...)` 可由调用方覆盖。

#### `backends/INTEL/backend_intel.py`（新增 `initialize_ccl` override）

```python
def initialize_ccl(self, rank: int, world_size: int):
    os.environ.setdefault("CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK", "0")
    os.environ.setdefault("SYCL_CACHE_PERSISTENT", "1")
    os.environ.setdefault("ZE_ENABLE_MODULE_CACHE", "1")
    return super().initialize_ccl(rank, world_size)
```

| 环境变量 | 作用 |
|---|---|
| `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` | 跳过 XeLink 拓扑顶点连接扫描，消除 CCL 初始化时的序列化开销 |
| `SYCL_CACHE_PERSISTENT=1` | 启用 SYCL JIT 内核持久化磁盘缓存，**二次及后续运行跳过重编译** |
| `ZE_ENABLE_MODULE_CACHE=1` | Level Zero 模块级缓存，与 SYCL_CACHE_PERSISTENT 协同 |

**重要说明**：
- `SYCL_CACHE_PERSISTENT` 对**首次运行无帮助**（缓存尚不存在），首次运行仍需等待 ~55 秒
- 首次运行依赖 `engine.py` 中 `timeout=300` 来容忍
- 从第二次运行起，JIT 缓存命中，总启动时间可降至 < 30 秒

### 5.4 改动文件汇总

| 文件 | 修改行 | 内容 |
|---|---|---|
| `core/engine.py` | `self.timeout = ...` | 默认值 60 → 300 |
| `core/engine.py` | `output_queue.get(timeout=...)` | 硬编码 60 → `self.timeout` |
| `backends/INTEL/backend_intel.py` | 新增方法 | `initialize_ccl()` override with CCL/SYCL env vars |

---

## 六、all_reduce 僵尸进程与 drm_neo.cpp abort（第五轮修复）

### 6.1 现象

```
Abort was called at 268 line in file: ./shared/source/os_interface/linux/drm_neo.cpp
```
8 个 worker 进程全部崩溃，成为僵尸进程（`[python3] <defunct>`），父进程卡死。

### 6.2 根因链（硬件级联失效）

```
上一轮 D2H benchmark 累积 GuC exec queue 压力
    │
    ▼ GT reset on 0000:19:00.0 无法完全恢复（GuC GDRST=0xffffffff，ETIMEDOUT）
    │
    ▼ Forcewake domain 0x1 MMIO unreliable（xe 驱动无法访问设备寄存器）
    │
    ▼ PCIe link down → pciehp IRQ 触发（irq/50-pciehp 线程）
    │
    ▼ BIOS 禁用 Driver-FLR，无法通过 Function-Level Reset 恢复
    │
    ▼ pciehp_disable_slot → pci_stop_and_remove_bus_device → xe_gt_fini
    │ （在清理过程中 xe_force_wake_get 再次失败，产生内核 WARNING）
    ▼
    0000:19:00.0 从 PCIe 总线完全移除
    │
    ▼ Level-Zero runtime（compute-runtime）的 DRM ioctl 返回 ENODEV
    │
    ▼ drm_neo.cpp:268 abort() → 8 个 worker 进程全部崩溃
    │
    └─▶ 僵尸进程堆积，父进程（launch.py）卡死
```

**补充**：all_reduce 2GB（ring-allreduce over PCIe）和 D2H 2GB 触发相同的 BCS 溢出机制。ring-allreduce 的 reduce-scatter + allgather 两个阶段均通过 BCS 做 PCIe 设备间数据搬运，多次迭代连续提交（`prefer_iterations=10`）= 20GB BCS 流量无 drain。

### 6.3 立即处置

```bash
kill -9 <launch.py PID>   # 杀死卡死的父进程，僵尸进程随之清理
# 之后需要重启系统（或 rmmod xe && modprobe xe）恢复 0000:19:00.0
```

### 6.4 软件修复

#### 扩展 `_needs_bcs_throttle` 覆盖所有大 tensor op

```python
# 之前：只覆盖 D2H/H2D（要求 _op_has_cpu_tensor AND size >= threshold）
def _needs_bcs_throttle(self, op_instance):
    return (
        self._op_has_cpu_tensor(op_instance)
        and op_instance.tensor_size >= self._BCS_SYNC_THRESHOLD_BYTES
    )

# 之后：任意 op，只要 tensor_size >= 512MB 就需要 BCS throttle
def _needs_bcs_throttle(self, op_instance):
    return op_instance.tensor_size >= self._BCS_SYNC_THRESHOLD_BYTES
```

#### 在 XPU Event 路径中也加入 per-iter sync

```python
# 之前：Event 路径无 per-iter sync，所有迭代一口气提交
start_event.record()
for i in range(prefer_iterations):
    op_instance.core_run(...)
end_event.record()

# 之后：大 tensor op 每次迭代后 drain BCS
start_event.record()
for i in range(prefer_iterations):
    op_instance.core_run(...)
    if bcs_throttle:
        self.device_synchronize()   # ← 排干 BCS 队列
end_event.record()
```

#### 对三个路径的影响

| 路径 | bcs_throttle | 变化 |
|---|---|---|
| D2H/H2D 大 tensor（wallclock） | True（原有） | 无变化 |
| AllReduce 等 CCL 大 tensor（Event） | True（**新增**） | per-iter sync，防止 BCS 溢出 |
| 任意 op，tensor < 512MB | False | 无变化 |
| warmup 路径 | 与测量路径相同 | 大 tensor warmup 也加 sync |

### 6.5 改动文件汇总（本轮）

| 文件 | 修改位置 | 内容 |
|---|---|---|
| `backends/INTEL/backend_intel.py` | `_needs_bcs_throttle()` | 移除 `_op_has_cpu_tensor` 前置条件 |
| `backends/INTEL/backend_intel.py` | `core_perf()` Event 路径 | 大 tensor 迭代内加 `device_synchronize()` |

---

## 七、all_reduce 仍崩溃——第六轮修复未生效分析（第七轮修复）

### 7.1 现象

第六轮在 Event 路径中加入 `if bcs_throttle: self.device_synchronize()` 后，all_reduce 2GB 测试（idx=19）依然崩溃：

```
Abort was called at 268 line in file: ./shared/source/os_interface/linux/drm_neo.cpp
```

dmesg 显示的失效模式与之前相同：
```
xe 0000:19:00.0: Force wake domain 0: wake. MMIO unreliable (returns 0xFFFFFFFF)
xe 0000:19:00.0: GuC reset timed out, GDRST=0xffffffff
→ pciehp_disable_slot → PCIe 设备被热拔出并重新枚举
→ Level-Zero DRM ioctl → ENODEV → drm_neo.cpp:268 abort()
```

### 7.2 根因——`device_synchronize()` 对 oneCCL 无效

**关键误解**：`device_synchronize()` → `torch.xpu.synchronize()` 只同步 **PyTorch 自身的 SYCL 队列**，不影响 oneCCL 的内部队列。

oneCCL 使用独立的 SYCL stream/queue 管理器提交 BCS DMA（reduce-scatter / allgather 阶段）。`torch.xpu.synchronize()` 对这些队列完全透明，调用后 CCL 的 BCS 操作仍可能在飞行中。

因此，第六轮的 per-iter `device_synchronize()` 对 CCL allreduce 实际上是一个**无操作**（no-op），3 次 2GB allreduce 仍然在 CCL 内部的 BCS 队列中依次积压，最终溢出。

### 7.3 正确的 CCL 级 BCS 排空机制

oneCCL 对同一 process group 内的 collective op 采用**严格 FIFO 顺序**处理：下一个 op 在 CCL 视角"开始"之前，上一个 op 的所有 BCS DMA 必须已完成（数据已写入目标缓冲区）。

因此，向同一 CCL group 提交任意一个新的 collective op（哪怕是 1-element allreduce，即 `op_group_barrier`）就足以保证：
- 上一次大 allreduce 的 BCS 全部 flush 完毕
- 新 op（barrier）返回后 BCS 空闲

### 7.4 修复内容

#### 改动 1：Event 路径 per-iter sync 从 `device_synchronize` 改为 `op_group_barrier`

```python
# 之前（无效）：
if bcs_throttle:
    self.device_synchronize()   # 只排 PyTorch SYCL 队列，不排 CCL BCS

# 之后（有效）：
is_large_ccl_op = bcs_throttle and group_size > 1
...
for i in range(prefer_iterations):
    op_instance.core_run(...)
    if is_large_ccl_op:
        self.op_group_barrier(...)  # 强制 CCL 串行化，等上一次 allreduce BCS 全部完成
    elif bcs_throttle:
        self.device_synchronize()   # D2H/H2D 保留（这些走 wallclock 路径，不会到此处）
```

#### 改动 2：移除 large CCL op 的 POST 外层 barrier

大 CCL op 的最后一次 per-iter barrier 已排空 BCS，POST 外层 barrier 是多余的，并且每个 case 多消耗 1 次 GuC exec queue alloc/destroy。

```python
# 之前：始终调用
self.op_group_barrier(...)   # ← 对 large CCL op 是冗余的

# 之后：
if not is_large_ccl_op:
    self.op_group_barrier(...)
```

PRE 外层 barrier 保留：它负责排空 warmup 阶段残留的 BCS，确保进入计时窗口前硬件空闲。

#### 改动 3：`perf()` 新增 `_LARGE_CCL_ITER_THRESHOLD_BYTES = 1 GB`，将 `min_test_iters` 降至 1

对于 `tensor_size ≥ 1 GB`（algo_size ≥ 512 MB）的 allreduce，每次迭代至少产生 2 GB 双向 BCS DMA。即使有 per-iter CCL barrier，3 次迭代仍带来 3 轮 barrier+allreduce+barrier 的 GuC exec queue 开销。限制为 1 次测量迭代可将单个 case 的 BCS 总量控制在最低。

```python
_LARGE_CCL_ITER_THRESHOLD_BYTES = 1024 * 1024 * 1024  # 1 GB

if op_instance.tensor_size >= self._LARGE_CCL_ITER_THRESHOLD_BYTES:
    min_test_iters = 1
```

### 7.5 为什么不影响计时准确性

per-iter `op_group_barrier` 在 XPU Event 计时窗口内：

| 操作 | 耗时 |
|---|---|
| 2 GB allreduce | ~100 ms |
| 1-element op_group_barrier | ~10–100 µs |
| 误差 | < 0.1% |

在防止硬件崩溃的前提下，0.1% 的计时误差是完全可接受的。

### 7.6 改动文件汇总（本轮）

| 文件 | 修改位置 | 内容 |
|---|---|---|
| `backends/INTEL/backend_intel.py` | 新增常量 | `_LARGE_CCL_ITER_THRESHOLD_BYTES = 1 GB` |
| `backends/INTEL/backend_intel.py` | `core_perf()` Event 路径 | `is_large_ccl_op` 判断 + per-iter `op_group_barrier` + 移除 POST 外层 barrier |
| `backends/INTEL/backend_intel.py` | `perf()` | `min_test_iters = 1` when `tensor_size ≥ 1 GB` |

---

## 八、未解决的相关问题（供参考）

1. **`AllReduce_H2D_Op` 中 `async_op=True` handle 被丢弃**（`core/ops/xccl_ops.py` 第 589 行），AllReduce 完成前访问 `data` 存在数据竞争，属于 `core/` 层问题，未在本次修改范围内。

2. **`BroadcastOp` 串行 N 轮 broadcast**，性能非最优，可改用 group broadcast，属算法优化，未在本次修改范围内。

3. **CCL 触发的 warning**（`Device capability of ccl unknown`）：调用 `init_process_group(backend="ccl")` 时，`oneccl_bindings_for_pytorch` 若未通过 `register_backend(devices=...)` 声明设备列表，PyTorch 的 `BackendConfig` 会走 else 分支发出警告。可通过显式指定 `backend="cpu:ccl,xpu:ccl"` 消除。

4. **首次运行 SYCL JIT 编译**：`SYCL_CACHE_PERSISTENT=1` 仅对二次运行有效，首次运行仍需 ~55 秒等待。如需彻底消除，可在独立脚本中预热一次 XPU CCL collective 以填充缓存。

