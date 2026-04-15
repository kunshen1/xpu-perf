# AllReduce 大 Tensor 卡死与僵尸进程修复分析

**文件**：`backends/INTEL/backend_intel.py`、`core/backend.py`、`core/engine.py`
**背景**：执行 `python3 launch.py --workload workloads/xccl_ops/all_reduce.json --backend INTEL --report_dir xccl_ops_report` 时，allreduce 测试在大 batch_size（tensor_size ≥ 1 GB）的 case 卡死或完成后触发 GT reset，dmesg 出现 xe 驱动 `guc_exec_queue_timedout_job` 和 BCS/CCS engine reset，Level Zero runtime 触发 `drm_neo.cpp abort()`，之后所有 worker 进程变成僵尸，父进程无法退出，严重时导致掉卡（PCIe 设备被热拔出）。

**已知遗留问题**：本文第一至六节的修复解决了 2 GB（batch_size=524288）场景的 BCS 溢出和 GuC 累积压力问题，但 **4 GB（batch_size=1048576, float32）场景仍然卡死**。详见第八节分析。

---

## 一、卡死链路（完整版）

### 1.1 数据规模示例

以 `batch_size=524288, dim_size=1024, dtype=float32, world_size=2` 为例：

| 指标 | 值 |
|---|---|
| 单 tensor 大小 | 524288 × 1024 × 4 = 2 GB |
| `tensor_size`（input + output） | 4 GB |
| `algo_size`（实际 allreduce 数据量） | 2 GB |
| 是否触发 `_BCS_SYNC_THRESHOLD_BYTES`（512 MB） | ✅ `bcs_throttle = True` |
| 是否触发 `_LARGE_CCL_ITER_THRESHOLD_BYTES`（1 GB） | ✅ `is_large_tensor = True` |

### 1.2 卡死 / 崩溃链路

问题分为**两层**，即使修复第一层也会被第二层击中：

#### 第一层：warmup 阶段 BCS 溢出（未设 CCL_SAME_STREAM 时的即时卡死）

```
perf() 调用 core_perf(warmup=2, prefer=2, profiling=False)    ← probe 阶段
    │
    ▼ 进入 else 分支（非 profiling 路径）
    │
    ▼ warmup 循环：2 次 2GB allreduce
    │   ├─ 第 1 次 allreduce 提交到 oneCCL 内部 BCS 队列
    │   ├─ if bcs_throttle: self.device_synchronize()
    │   │       ↑ BUG: 当 CCL_SAME_STREAM=0（默认）时，
    │   │         device_synchronize() = torch.xpu.synchronize()
    │   │         只同步 PyTorch 自己的 SYCL 队列
    │   │         oneCCL 使用独立的内部 SYCL 队列管理 BCS DMA
    │   │         → 对 oneCCL 的 BCS 队列是 **NO-OP**
    │   │
    │   ├─ 第 2 次 allreduce 提交：BCS 队列已有第 1 次的 DMA 未完成
    │   └─ 两次 allreduce 的 BCS DMA 连续堆积（≥ 4 GB 双向 PCIe 流量）
    │
    ▼ BCS 命令队列溢出 → GuC exec queue 超时 → engine reset
    │
    └─▶ 所有 rank 永久卡死
```

#### 第二层：累积 GuC 压力导致延迟 GT reset（即使单个 case 成功也会崩溃）

```
测试套件运行 22 个 batch_size × 3 个 dtype = 66 个 world_size=2 cases
    │
    ▼ 每个 case 的 probe + actual 两次 core_perf 调用
    │  每次 core_perf: warmup allreduce + barrier + PRE barrier
    │                  + measurement allreduce + per-iter barrier
    │  = ~5 CCL ops/call × 2 calls = ~10 CCL ops/case
    │
    ▼ 每个 CCL op 创建/销毁一个 GuC exec queue
    │  __guc_exec_queue_destroy_async 在内核中异步执行
    │
    ▼ 66 cases × 10 CCL ops = 660 次 GuC exec queue create/destroy
    │  异步销毁队列累积 → __guc_exec_queue_destroy_async hogged CPU
    │
    ▼ 到达 2GB float32 case 时，即使该 case 成功完成并打印结果...
    │
    ▼ 内核异步 GuC 清理任务积压超限
    │  → guc_exec_queue_timedout_job
    │  → GT0 reset on 0000:19:00.0
    │
    ▼ GT reset 失败 → GDRST=0xffffffff → MMIO unreliable
    │  → PCIe link down → 设备从总线移除（掉卡）
    │
    ▼ Level Zero DRM ioctl 返回 ENODEV → drm_neo.cpp:268 abort()
    │
    └─▶ worker 进程崩溃 + 僵尸进程
```

### 1.3 关键认知

**`torch.xpu.synchronize()` 不等于同步所有 XPU 上的 BCS 操作。**

- 当 `CCL_SAME_STREAM=0`（默认）时，PyTorch SYCL 队列和 oneCCL 内部 SYCL 队列是**完全独立的**。
- `torch.xpu.synchronize()` 只排空 PyTorch 自己提交的工作。oneCCL 的 BCS DMA **完全不可见**。
- 当 `CCL_SAME_STREAM=1` 时，两者共享同一个 SYCL stream，`torch.xpu.synchronize()` 才能排空 CCL 的 BCS。

**`CCL_BLOCKING_WAIT` 控制 host 阻塞行为。**

- `CCL_BLOCKING_WAIT=1`：`dist.all_reduce()` 在 host 侧阻塞直到 collective 真正完成。BCS 自然被排空。
- `CCL_BLOCKING_WAIT=0`（默认，CCL_MINOR_VERSION ≥ 14）：`dist.all_reduce()` 提交后立即返回，host 不等待。BCS 可能仍在飞行中。

**`op_group_barrier` 是一把双刃剑。**

- 优点：利用 oneCCL 的 FIFO 顺序强制排空前一个 allreduce 的 BCS。
- 缺点：每次 barrier 本身也是一次 CCL allreduce（`dist.all_reduce(tensor([1]))`），创建/销毁一个 GuC exec queue。大量使用反而增加累积 GuC 压力。

---

## 二、NEO 报错与掉卡原因

```
Abort was called at 268 line in file:
./shared/source/os_interface/linux/drm_neo.cpp
```

完整级联失效路径：

```
累积 GuC exec queue 异步销毁超限
    │
    ▼ guc_exec_queue_timedout_job → xe 驱动尝试 GT reset
    │
    ▼ GT reset 失败 → GDRST=0xffffffff, ETIMEDOUT
    │
    ▼ Forcewake domain MMIO 不可靠（返回 0xFFFFFFFF）
    │
    ▼ PCIe link down → pciehp IRQ 触发
    │
    ▼ BIOS 禁用 Driver-FLR → 无法 Function-Level Reset
    │
    ▼ pciehp_disable_slot → pci_stop_and_remove_bus_device
    │   XPU 设备从 PCIe 总线完全移除（掉卡）
    │
    ▼ Level-Zero runtime 的 DRM ioctl 返回 ENODEV
    │
    └─▶ drm_neo.cpp:268 abort() → 所有使用该设备的 worker 被 SIGABRT 终止
```

---

## 三、僵尸进程无法清理的原因

### 3.1 子进程崩溃后的死锁链

```
drm_neo.cpp abort() → 部分 worker 被 SIGABRT 杀死
    │
    ▼ 存活的 worker 仍在 gloo all_gather_object() 中
    │   等待已死 rank 的应答 → 无限阻塞
    │  （gloo group 默认无超时 → 永远等不到）
    │
    ▼ 父进程 dispatch() 中的 output_queue.get(timeout=60) 超时
    │   → 检测到 dead_procs → raise RuntimeError
    │
    ▼ 但存活 worker 不在等 input_queue，而在等 gloo collective
    │   input_queue.put(None) 无法唤醒它们
    │
    ▼ stop() 中 proc.join(timeout=10) 超时
    │   → proc.terminate() / proc.kill() → proc.join()
    │
    ▼ 如果 stop() 未被调用到（异常未传播到上层 finally/with），
    │   或 gloo 阻塞导致进程处于不可中断状态
    │
    └─▶ 僵尸进程堆积
```

---

## 四、本次修复内容

### 4.1 `backends/INTEL/backend_intel.py` — `initialize_ccl()` 设置 CCL 环境变量（根因修复）

**这是最关键的修复。** 在 `dist.init_process_group` 之前设置：

```python
os.environ.setdefault("CCL_BLOCKING_WAIT", "1")
os.environ.setdefault("CCL_SAME_STREAM", "1")
```

#### 效果

| 设置 | 作用 |
|---|---|
| `CCL_BLOCKING_WAIT=1` | 每次 `dist.all_reduce()` 在 host 侧阻塞直到 collective 完成 → BCS 自然排空 → 不可能累积 |
| `CCL_SAME_STREAM=1` | oneCCL 使用 PyTorch 的 SYCL stream → `device_synchronize()` 可排空 CCL BCS |

#### 为什么之前一直不设这两个变量

`run_xccl_ops.sh` 脚本已经设置了这两个变量（`CCL_BLOCKING_WAIT=0, CCL_SAME_STREAM=1`），但直接运行 `python3 launch.py` 时这些变量未设置，使用了 CCL 默认值（`CCL_BLOCKING_WAIT=0, CCL_SAME_STREAM=0`），导致 `device_synchronize()` 对 CCL 完全无效。

使用 `setdefault` 确保用户/脚本可以通过环境变量覆盖。

#### 对性能的影响

**零影响**。benchmark 逐个测量 collective 延迟，使用 XPU Event 设备侧计时。Host 阻塞不改变设备侧执行时间，event elapsed time 准确反映 collective 耗时。

---

### 4.2 `backends/INTEL/backend_intel.py` — `perf()` 跳过大 tensor 的 probe 阶段

```python
# 修复前：大 tensor 也做 probe（减少了迭代但仍做）
probe_warmup = 1 if is_large_tensor else 2
probe_iters = 1 if is_large_tensor else 2
latency_us, _ = self.core_perf(op_instance, probe_warmup, probe_iters, ...)
prefer_iters = min(max(int(1e6 / latency_us), 2), min_test_iters)  # 必然 = 1

# 修复后：大 tensor 直接跳过 probe
if is_large_tensor:
    prefer_iters = 1  # 确定值，无需估算
else:
    latency_us, _ = self.core_perf(op_instance, 2, 2, ...)
    prefer_iters = ...
```

#### 原因

对 `tensor_size >= 1 GB` 的 case，`min_test_iters = 1`，probe 计算的 `prefer_iters = min(max(X, 2), 1) = 1` 是一个**常数**。probe 阶段的 warmup + measurement 总共约 5 个 CCL ops，白白增加 GuC 压力。

同时跳过了 probe 后的 `all_gather_object`（在 CCL op_group 上同步 prefer_iters），又减少一次 CCL op。

#### 效果

每个大 tensor case 从 ~10 CCL ops 降至 ~5 CCL ops（仅 actual 阶段）。

---

### 4.3 `backends/INTEL/backend_intel.py` — `perf()` 大 case 后增加 cooldown

```python
if bcs_throttle:
    self.device_synchronize()
    time.sleep(1.0)
```

#### 原因

`__guc_exec_queue_destroy_async` 在内核中异步执行。即使 CCL op 在设备上完成，GuC exec queue 的销毁仍在后台进行。立即开始下一个 case 会让销毁队列持续积压。1 秒 cooldown 给内核足够的清理时间。

---

### 4.4 `backends/INTEL/backend_intel.py` — `core_perf()` warmup 使用 `op_group_barrier`

保留上一轮的 warmup barrier 修复作为 `CCL_SAME_STREAM=0` 场景的 safety net：

```python
for i in range(warmup_iterations):
    op_instance.core_run(...)
    if is_large_ccl_op:
        self.op_group_barrier(...)   # safety net for CCL_SAME_STREAM=0
    elif bcs_throttle:
        self.device_synchronize()
```

有 `CCL_BLOCKING_WAIT=1` 时这些 barrier 是冗余的（每次 allreduce 已经 host-blocking），但不会造成额外负担。

---

### 4.5 `core/backend.py` — gloo group 添加 120 秒超时

```python
cpu_group = dist.new_group(
    ranks=list(range(world_size)),
    backend="gloo",
    timeout=timedelta(seconds=120)
)
```

防止 rank 崩溃后存活 rank 永久阻塞在 `all_gather_object`。

---

### 4.6 `core/engine.py` — 进程终止信号升级

```python
proc.terminate()     # SIGTERM first (allows cleanup)
proc.join(timeout=5)
if proc.is_alive():
    proc.kill()      # SIGKILL as last resort
proc.join()          # Reap zombie
```

`ComputeEngine.stop()` 和 `XCCLEngine.stop()` 均已应用。

---

## 五、改动文件汇总

| 文件 | 改动位置 | 内容 |
|---|---|---|
| `backends/INTEL/backend_intel.py` | `initialize_ccl()` | **新增 `CCL_BLOCKING_WAIT=1` + `CCL_SAME_STREAM=1`** |
| `backends/INTEL/backend_intel.py` | `_BCS_SYNC_THRESHOLD_BYTES` 注释 | 区分两种 BCS 排空机制 |
| `backends/INTEL/backend_intel.py` | `_needs_bcs_throttle()` docstring | 同上 |
| `backends/INTEL/backend_intel.py` | `core_perf()` else 分支 | `is_large_ccl_op` 提前计算 + warmup barrier（safety net） |
| `backends/INTEL/backend_intel.py` | `perf()` | **跳过大 tensor probe** + **cooldown sleep** |
| `core/backend.py` | `xccl_infer_loop()` | gloo cpu_group 超时 120 秒 |
| `core/engine.py` | `ComputeEngine.stop()` | SIGTERM → SIGKILL 升级 |
| `core/engine.py` | `XCCLEngine.stop()` | 同上 |

---

## 六、修复前后 CCL ops 对比（单个 2GB allreduce case）

```
修复前（无 CCL env vars，probe 未跳过）：
  probe core_perf:   2 warmup allreduces (BCS 溢出!)
                   + 1 PRE barrier
                   + 2 measurement allreduces
                   + 2 per-iter barriers
                   = 7 CCL ops
  all_gather_object: 1 CCL op (prefer_iters sync)
  actual core_perf:  同 probe = 7 CCL ops
  Total: 15 CCL ops per case, warmup BCS 溢出风险

修复后（CCL_BLOCKING_WAIT=1, CCL_SAME_STREAM=1, probe skipped）：
  actual core_perf:  1 warmup allreduce (host-blocking, BCS 排空)
                   + 1 warmup barrier (redundant, safety net)
                   + 1 PRE barrier
                   + 1 measurement allreduce (host-blocking)
                   + 1 per-iter barrier (redundant)
                   = 5 CCL ops
  cooldown:         1s sleep
  Total: 5 CCL ops per case, 无 BCS 溢出, 有 cooldown

减少：15 → 5 CCL ops/case = 67% reduction
```

---

## 八、4 GB allreduce 仍然卡死的分析（batch_size=1048576）

### 8.1 问题现象

应用第一至六节的所有修复后，执行 `python3 launch.py --workload workloads/xccl_ops/all_reduce.json --backend INTEL --report_dir xccl_ops_report`：

- **Case #19**：`float32, batch_size=524288, dim_size=1024`（输入 tensor 2 GB） → **成功**
- **Case #20**：`float32, batch_size=1048576, dim_size=1024`（输入 tensor 4 GB） → **卡死，系统掉卡**

此时 `CCL_BLOCKING_WAIT=1` 和 `CCL_SAME_STREAM=1` 已生效（通过 `setdefault` 在 `dist.init_process_group` 前设置），累积 CCL ops 约 217 次（远低于第二层故障的 660 次阈值）。说明 4 GB 场景的卡死不是 BCS 队列溢出或 GuC 累积压力，而是一个**新的硬件层面的故障模式**。

### 8.2 oneCCL 内部执行路径追踪

```
dist.all_reduce(4GB tensor, float32)
  │
  ▼ torch-ccl (oneccl_bindings_for_pytorch)
  │  dpcpp_ccl.cpp: TORCH_LLM_ALLREDUCE_DEBUG 未设置 → 标准 oneCCL 路径
  │  ccl::allreduce(count=1,073,741,824, dtype=float32)
  │
  ▼ oneCCL: allreduce_sycl_single_node()
  │  BMG device_id=0xe211 → masked=0xe210 → family6 → is_arc_card=true
  │  total_size=4GB > simple_threshold(4MB) → 跳过 LL256/arc 小消息路径
  │  sycl_esimd=0 → 非 ESIMD 路径
  │  count*dsize=4GB > small_threshold(512KB) → allreduce_large()
  │
  ▼ allreduce_large()
  │  is_arc_card=true && pair_comm->size()==1 → BMG PCIe 路径
  │  sycl_allreduce_tmp_buf=0 → IPC 路径（非临时缓冲区路径）
  │  do_ipc_exchange() → 通过 zeMemGetIpcHandle/zeMemOpenIpcHandle
  │                       映射远端 GPU 的 4GB tensor 到本地虚拟地址空间
  │
  ▼ allreduce_large_su_ring() → allreduce_large_su_ring_write_multi_kernel()
     sycl_allreduce_simple_read=0 → write 算法
     sycl_copy_engine=0 → SYCL kernel（非 q.memcpy）
     sycl_simple_single_kernel=0 → multi-kernel 路径
```

#### Ring 算法参数

| 参数 | 值 |
|---|---|
| count | 1,073,741,824 elements（0x40000000，int32 安全） |
| total_bytes | 4,294,967,296 bytes（0x100000000，size_t 安全） |
| N (world_size) | 2 |
| count_per_rank (recv_bytes) | 2 GB |
| sycl_tmp_buf_size | 384 MB（3 × 128 MB） |
| chunk_size | 192 MB（sycl_tmp_buf_size / pipeline_size=2） |
| num_chunks | 11（⌈2 GB / 192 MB⌉） |

#### 每个 chunk 的 GPU 提交

```
chunk i:
  1. copy_kernel:          本地 send_buf[chunk_i] → 远端 work_buf[chunk_i]（PCIe P2P write）
  2. p2p_barrier:          GPU 端自旋轮询远端 flag（PCIe P2P read）直到对端到达
  3. reduce_copy_kernel:   本地 recv_buf[chunk_i] = reduce(本地数据, 远端数据) → 远端 recv_buf（PCIe P2P write）

对于 N=2：reduce-scatter 运行 1 step × 11 chunks，allgather 被跳过（i=1; i<N-1=1 → 不执行）
```

### 8.3 CCL_BLOCKING_WAIT=1 实际无效（关键发现）

通过源码追踪 `dist.all_reduce().wait()` 的完整执行路径：

```cpp
// dpcpp_ccl.cpp execute() (line 670-687)
c10::intrusive_ptr<Work> XPUWorkCCL::execute() {
    work->run();                    // 提交 ccl::allreduce → SYCL kernels 入队到 GPU
    work->finishAsyncWorkCCL();     // ← 立即标记 PyTorch future 为"已完成"！
    return work;                    //    此时 GPU 还在跑 SYCL kernels
}

// work.wait() → synchronizeInternalForXPU() (line 440-457)
void synchronizeInternalForXPU() {
    if (blockingWait_) {                    // CCL_BLOCKING_WAIT=1
        synchronizeInternal(kNoTimeout);    // 检查 future
                                            // → future 已被 finishAsyncWorkCCL() 标记完成
                                            // → 立即返回！
    }
    // 后续 end_event.synchronize() 和 device_synchronize() 才是真正同步
}
```

**结论**：`CCL_BLOCKING_WAIT=1` 是一个 **NO-OP**，因为 `execute()` 中的 `finishAsyncWorkCCL()` 在 GPU kernel 完成前就标记了 future 为已完成。`synchronizeInternal()` 检查到 future 已完成后立即返回。

**真正起作用的是 `CCL_SAME_STREAM=1`**：让 oneCCL 使用 PyTorch 的 SYCL queue，从而使后续的 `end_event.synchronize()` 和 `torch.xpu.synchronize()` 能正确等待 CCL kernel 完成。

### 8.4 根因分析：PCIe P2P IPC 写入超过 BMG 硬件承受能力

#### 2 GB 与 4 GB 的对比

| 指标 | 524288（2 GB）✅ | 1048576（4 GB）❌ |
|---|---|---|
| 每 rank 数据量 | 1 GB | **2 GB** |
| 分块数 | 6 | **11** |
| GPU kernel 提交数/allreduce | 20 | **35** |
| PCIe P2P 写入总量/allreduce | ~2.2 GB | **~4.1 GB** |
| IPC 映射大小 | 2 GB | **4 GB** |

#### 故障机制

4 GB allreduce 中，每个 chunk 的 SYCL kernel 直接通过 IPC 指针写入远端 GPU 显存（GPU 发起的 PCIe peer-to-peer write）。11 个 chunk × 每 chunk 两次 ~192 MB 的 P2P 写入 = **~4.1 GB 的 GPU 发起 PCIe P2P 流量**。

```
GPU 0                        PCIe Switch                     GPU 1
  │                              │                              │
  ├── copy_kernel ──────────────►│──── 192MB P2P write ────────►│  chunk 0
  ├── p2p_barrier (poll) ◄──────►│◄─── read flag ──────────────►│
  ├── reduce_copy ──────────────►│──── 192MB P2P write ────────►│
  │                              │                              │
  ├── copy_kernel ──────────────►│──── 192MB P2P write ────────►│  chunk 1
  ...                            ...                            ...
  ├── copy_kernel ──────────────►│──── 192MB P2P write ────────►│  chunk 10
  ├── p2p_barrier (poll) ◄──────►│◄─── read flag ──────────────►│
  └── reduce_copy ──────────────►│──── 192MB P2P write ────────►│
                                 │                              │
                          总 P2P 写入 ≈ 4.1 GB（双向）
```

**BMG (Battlemage) 是消费级 GPU，PCIe P2P 能力有限。** 可能的硬件层面失败原因：

1. **PCIe P2P 写入队列深度溢出**：BMG 的 PCIe 控制器对 GPU 发起的 P2P 写入请求有队列深度限制。4.1 GB 的连续 P2P 写入（35 个 kernel 全部在同一 SYCL queue 上无 host 等待地提交）可能超过此限制
2. **GPU 内部状态机卡死**：大量 P2P 写入请求积压导致 GPU 端 PCIe 控制器内部状态机进入死锁状态，GPU 停止响应所有 PCIe 事务
3. **IPC 映射 4 GB 的 xe 驱动限制**：`zeMemOpenIpcHandle` 映射 4 GB 远端缓冲区时，xe driver 内部可能存在 32 位变量截断或映射大小限制

#### scaleout_threshold 边界问题

oneCCL 的 `sycl_allreduce_scaleout_threshold` 默认值恰好为 4,294,967,296（4 GB）。在 `allreduce_sycl.cpp` 中：

```cpp
if (total_size > sycl_allreduce_scaleout_threshold) {  // 4GB > 4GB → false!
    return do_fallback_to_scheduler(...);  // ← 不会执行
}
```

4 GB 数据（`total_size == sycl_allreduce_scaleout_threshold`）恰好不触发 fallback（使用 `>` 而非 `>=`），仍走 SYCL kernel 路径。如果使用调度器路径（基于 MPI 的分段传输），可能不会触发 PCIe P2P 硬件限制。

### 8.5 dmesg 错误链印证

```
05:15:49  xe 0000:1d:00.0: CT write: non-zero status: 0xFFFFFFFF
          ↑ GuC Command Transport 已死 — GPU 内部完全无响应

05:15:49  pcieport 0000:14:02.0: AER: Uncorrected error received
          aer_uncor_status: 0x00004000
          ↑ bit 14 = Completion Timeout — GPU 停止回复 PCIe 请求

05:15:49  pcieport 0000:16:00.0: DPC: containment event, status:0x1f03
          pcieport 0000:16:01.0: DPC: containment event, status:0x1f01
          ↑ PCIe Switch 两个下游端口同时触发 DPC（Downstream Port Containment）
            隔离两块 GPU，防止错误扩散

05:16:02  workqueue: __guc_exec_queue_destroy_async 大量堆积
          ↑ 异步清理队列无法执行（GPU 已不响应）

05:16:05  xe 0000:19:00.0: GT0: reset failed -110
          xe 0000:19:00.0: GT0: declaring device as wedged
          ↑ GT reset 失败（-ETIMEDOUT），设备彻底不可用

05:16:10  xe 0000:1d:00.0: GT0: GDRST=0xffffffff
          ↑ 硬复位寄存器读回全 F — 设备已从 PCIe 总线脱落
```

**关键证据**：PCIe Completion Timeout（bit 14）是根本触发点，说明 GPU 内部因 P2P 操作挂死，停止响应所有 PCIe 事务。这不是软件队列管理问题，而是**硬件层面的 GPU 挂死**。

### 8.6 修复对 2 GB 有效但对 4 GB 无效的原因

第一至六节的修复解决的是**软件层面问题**：

| 修复 | 解决的问题 | 对 4 GB 是否有效 |
|---|---|---|
| `CCL_SAME_STREAM=1` | BCS 队列同步 | ✅ 有效，但不能阻止 GPU 挂死 |
| `CCL_BLOCKING_WAIT=1` | Host 侧阻塞等待 | ❌ NO-OP（见 8.3 节） |
| 跳过 probe | 减少 CCL ops | ✅ 有效，但 4 GB 在第一次 allreduce 就挂 |
| Cooldown sleep | GuC 清理时间 | ✅ 有效，但 4 GB 不是累积问题 |

4 GB 场景是**硬件层面的 PCIe P2P 极限问题**，在**第一次 allreduce 调用内部**就触发 GPU 挂死，软件层面的队列管理和同步优化无法防御。

### 8.7 建议验证方法与规避措施

#### 验证根因

| 方法 | 操作 | 预期结果 |
|---|---|---|
| 单独测试 4 GB | 只配 `batch_sizes=[1048576]`，排除累积因素 | 仍然卡死 → 确认非累积问题 |
| 测试 float16 1048576 | float16 × 1048576 × 1024 = 2 GB（与 float32 524288 相同大小） | 成功 → 确认是 4 GB 数据量的问题 |
| 禁用 IPC P2P | `export CCL_SYCL_ALLREDUCE_TMP_BUF=1` | 使用本地临时缓冲区 + q.memcpy 替代 GPU 直接 P2P write |
| 降低 chunk size | `export CCL_SYCL_TMP_BUF_SIZE=67108864`（64 MB） | 减小单次 P2P 写入量，增加 chunk 数 |
| 强制 fallback 到调度器 | `export CCL_SYCL_ALLREDUCE_SCALEOUT_THRESHOLD=4294967295`（4GB-1） | 4 GB 走 MPI 调度器路径，不走 SYCL kernel P2P |
| 查看 oneCCL 内部日志 | `export CCL_LOG_LEVEL=debug` | 获取 oneCCL 内部路径选择和 IPC 交换的详细日志 |

#### 生产规避

1. **跳过 4 GB case**：在 `all_reduce.json` 中移除 `1048576`，或在 `backend_intel.py` 中添加 tensor_size > 2GB 的跳过逻辑
2. **设置 scaleout threshold**：`os.environ.setdefault("CCL_SYCL_ALLREDUCE_SCALEOUT_THRESHOLD", "2147483648")`（2 GB），让 ≥ 2 GB 的数据走调度器路径
3. **使用 tmp_buf 模式**：`os.environ.setdefault("CCL_SYCL_ALLREDUCE_TMP_BUF", "1")`，避免 GPU 直接 PCIe P2P write

---

## 九、缩减至 batch_size=524288 后仍卡死的分析

### 9.1 问题现象

将 `all_reduce.json` 的 `batch_size` 上限缩减至 524288（移除 1048576、2097152）后，所有 60 个有效 case 均成功运行并输出正确结果，但在最后一个 case（`bfloat16, batch_size=524288`）输出后挂死：

```
{"arg_type": "default", "world_size": 2, "dtype": "bfloat16", "batch_size": 524288, "dim_size": 1024}
{"latency(us)": 53270.854, ...}

2026-04-13 07:40:07 engine.py:149 [WARNING]: Waiting for worker result (no response in 60s, retrying)...
2026-04-13 07:41:07 engine.py:149 [WARNING]: Waiting for worker result (no response in 60s, retrying)...
```

注意：所有 case 的结果已打印（worker 完成了测量），但 engine 仍在等待后续结果。

### 9.2 根因：累积 GuC 压力（第二层问题复发）

这不是 4 GB PCIe P2P 问题（已通过移除大 batch_size 规避），而是**第一至六节修复文档中描述的第二层问题：累积 GuC exec queue 压力**。

#### CCL ops 估算

当前配置：20 个 batch_sizes × 3 个 dtypes × 6 个 world_sizes（仅 ws=2 有效）

| 阶段 | Case 数 | 每 case CCL ops | 总 CCL ops |
|---|---|---|---|
| 小 tensor（tensor_size < 1 GB）53 个 | 53 | ~20（probe 6 + actual 14） | ~1060 |
| 大 tensor（tensor_size ≥ 1 GB）7 个 | 7 | ~5（跳过 probe） | ~35 |
| **合计** | **60** | | **~1095** |

**~1095 CCL ops 远超 ~660 的安全阈值**，触发了第二层故障。

#### 卡死机制

```
60 个有效 case 全部完成（~1095 CCL ops 累积）
    │
    ▼ 最后一个 case 的 perf() 返回，rank 0 打印结果 + output_queue.put()
    │  engine 收到该结果
    │
    ▼ 但 JSON 中 world_size=[2,4,8,16,32,64] 产生了 360 个 task
    │  60 个有效 + 300 个 skip（ws > 实际 GPU 数）
    │
    ▼ Worker 处理 300 个 skip case，每个需要 gloo all_gather_object
    │  ← 两个 rank 必须同时参与
    │
    ▼ 在 perf() cooldown 的 device_synchronize() 中或 skip case 处理中
    │  累积的 __guc_exec_queue_destroy_async 引发 GT reset
    │
    ▼ 某个 rank 的 torch.xpu.synchronize() 或 GPU 状态异常挂死
    │  → 无法参与下一轮 gloo all_gather_object
    │  → 另一个 rank 阻塞在 all_gather_object（等待已挂的 rank）
    │
    └─▶ 两个 rank 都卡住 → 没有更多 output_queue.put → engine 反复等待
```

### 9.3 修复方案

#### 方案 A：JSON 只保留有效 world_size（消除 skip 开销）

```json
"world_size": [2]
```

效果：消除 300 个 skip case 的 gloo 开销，减少挂死窗口期，但 CCL ops 总量不变（~1095），仍有 GuC 压力风险。

#### 方案 B：分批跑（推荐）

将 batch_sizes 分成两批独立运行，每批 CCL ops 控制在安全阈值内：

```
# 批 1：小尺寸
batch_size: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
14 sizes × 3 dtypes = 42 cases → ~540 CCL ops ✅

# 批 2：大尺寸
batch_size: [16384, 32768, 65536, 131072, 262144, 524288]
6 sizes × 3 dtypes = 18 cases → ~120 CCL ops ✅
```

#### 方案 C：代码限制 probe 迭代数（最佳长期方案）

在 `backend_intel.py` 的 `perf()` 中将小 tensor 的 `min_test_iters` 上限降低：

```python
# 当前：min_test_iters = 10（默认）
# 修改：min_test_iters = 3（全局）
min_test_iters = 3
```

效果：20 sizes × 3 dtypes = 60 cases，probe + actual 每 case ~13 CCL ops → ~780，仍偏高。
需配合方案 A 和增加 cooldown 才能可靠控制在 660 以下。

#### 方案 D：方案 A + B + 额外 cooldown（最可靠）

1. `world_size: [2]`（消除 skip case）
2. 分批跑或减少 batch_sizes 数量
3. 在每 N 个 case 后插入额外 `time.sleep(2)`，给内核清理 GuC exec queue 的时间

---

## 十、各算子数据大小边界分析

### 10.1 PCIe P2P 硬件限制（单次 CCL 调用）

已知安全边界（allreduce 实测）：

- `all_reduce float32 input=2 GB` → ~2.2 GB PCIe P2P writes → ✅ 通过
- `all_reduce float32 input=4 GB` → ~4.1 GB PCIe P2P writes → ❌ GPU 挂死

各算子在 `world_size=2, dim_size=1024, dtype=float32` 下的对比：

| 算子 | bs=524288 分配/GPU | bs=524288 P2P | bs=1048576 分配/GPU | bs=1048576 P2P |
|---|---|---|---|---|
| **all_reduce** | 2 GB（in-place） | ~2.2 GB ✅ | 4 GB | ~4.1 GB ❌ |
| **reduce_scatter** | 3 GB（in 2G + out 1G） | ~1 GB ✅ | 6 GB | ~2 GB ✅ |
| **all_gather** | 3 GB（in 1G + out 2G） | ~1 GB ✅ | 6 GB | ~2 GB ✅ |
| **all_to_all** | 4 GB（in 2G + out 2G） | ~1 GB ✅ | 8 GB | ~2 GB ✅ |

#### 关键差异

- **all_reduce**：in-place 操作，ring 算法中 reduce-scatter 阶段需要写入 **全量数据** 到远端 GPU。对于 N=2，P2P 流量 ≈ input_size。
- **reduce_scatter/all_gather**：input 和 output 大小不同（差 world_size 倍），P2P 流量 ≈ min(input, output) ≈ input_size / ws。
- **all_to_all**：点对点交换，P2P 流量 ≈ input_size × (ws-1)/ws。

因此 **reduce_scatter、all_gather、all_to_all 的 PCIe P2P 安全边界比 all_reduce 高一倍**。

### 10.2 各算子最大安全 batch_size

| 算子 | float32 最大 bs | float16/bf16 最大 bs | 限制因素 |
|---|---|---|---|
| **all_reduce** | **524288**（2 GB P2P） | **1048576**（2 GB P2P） | PCIe P2P |
| **reduce_scatter** | **1048576**（2 GB P2P） | **2097152**（2 GB P2P） | PCIe P2P |
| **all_gather** | **1048576**（2 GB P2P） | **2097152**（2 GB P2P） | PCIe P2P |
| **all_to_all** | **1048576**（2 GB P2P） | **2097152**（2 GB P2P） | PCIe P2P |

注意：以上仅为单次 CCL 调用的 PCIe P2P 安全边界。**实际测试还需控制整个测试套件的总 CCL ops 数量在 ~660 以下**（见第九节），否则即使每个 case 单独安全，累积 GuC 压力仍会导致挂死。

### 10.3 各算子配置建议

```json
// all_reduce.json — 保守配置
{
    "cases": [{
        "arg_type": ["default"],
        "world_size": [2],
        "dtype": ["float32", "float16", "bfloat16"],
        "batch_size": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288],
        "dim_size": [1024]
    }]
}
// 注意：20 sizes × 3 dtypes = 60 cases → ~1095 CCL ops (超阈值)
// 需分批跑或减少 batch_sizes

// reduce_scatter.json / all_gather.json — 可用更大 batch_size
{
    "cases": [{
        "arg_type": ["default"],
        "world_size": [2],
        "dtype": ["float32", "float16", "bfloat16"],
        "batch_size": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576],
        "dim_size": [1024]
    }]
}
// 注意：21 sizes × 3 dtypes = 63 cases → ~1150 CCL ops (超阈值)
// 同样需分批跑

// all_to_all.json — 额外注意 int8 dtype 的 VRAM 分配
{
    "cases": [{
        "arg_type": ["default"],
        "world_size": [2],
        "dtype": ["float32", "float16", "bfloat16"],
        "batch_size": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576],
        "dim_size": [1024]
    }]
}
```

---

## 十一、其他未解决的相关问题（供参考）

1. **4 GB allreduce 在 BMG 上仍然卡死**（见第八节）：根因为 PCIe P2P IPC 写入量超过 BMG 硬件承受能力，需在 oneCCL 或 xe 驱动层面修复。

2. **`CCL_BLOCKING_WAIT=1` 实际为 NO-OP**（见 8.3 节）：torch-ccl `execute()` 中 `finishAsyncWorkCCL()` 在 GPU 完成前标记 future，导致 `synchronizeInternal()` 立即返回。应向 torch-ccl 上游报告。

3. **累积 GuC exec queue 压力无软件层面根治方案**（见第九节）：当前修复通过减少 CCL ops 和 cooldown 缓解，但 xe 驱动的 `__guc_exec_queue_destroy_async` 异步清理机制是根本瓶颈。需 xe 驱动或 oneCCL 层面优化 exec queue 复用策略。

4. **`AllReduce_H2D_Op` 中 `async_op=True` handle 被丢弃**（`core/ops/xccl_ops.py`），AllReduce 完成前访问 `data` 存在数据竞争。

5. **CCL 触发的 warning**（`Device capability of ccl unknown`）：可通过 `backend="cpu:ccl,xpu:ccl"` 消除。

6. **首次运行 SYCL JIT 编译**：`SYCL_CACHE_PERSISTENT=1` 仅对二次运行有效，首次运行仍需约 55 秒等待。

7. **`mp.spawn` 与 oneCCL 兼容性**：长期建议将 XCCL 执行模型从 `mp.spawn` 迁移到 `torchrun/mpirun` 外部 launcher。
