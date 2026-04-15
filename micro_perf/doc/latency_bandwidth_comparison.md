# micro_perf 与 oneCCL benchmark 的 Latency / Bandwidth 对比分析

**背景**：在相同硬件、相同数据大小上跑同一个集合通信算子（如 allreduce），oneCCL 自带 benchmark
（`examples/benchmark/`）报告的 **latency 显著低于** micro_perf，但 **algo bandwidth 接近**。
本文分析差异根因。

---

## 一、两套计时方式对比

### 1.1 oneCCL benchmark（纯 C++）

源码位置：`examples/benchmark/src/benchmark.cpp` L122-167

```cpp
ccl::barrier(service_comm);                     // 跨 rank 同步
for (iter = 0; iter < warmup + measure; iter++) {
    t1 = when();                                // std::chrono::high_resolution_clock
    ccl::allreduce(send, recv, ...);            // 异步提交，立即返回
    t2 = when();
    req.wait();                                 // GPU 执行完成
    t3 = when();

    if (iter >= warmup) {
        coll_time += (t2 - t1);                 // submit 延迟（几十 μs）
        wait_time += (t3 - t2);                 // GPU 执行延迟
    }
}
total_time = coll_time + wait_time;
```

带宽计算（`examples/benchmark/include/benchmark.hpp` L442-487）：

```
t_avg = sum(total_time_all_ranks) / (iter_count × nranks)   // 跨所有 rank 平均
algbw = bytes / t_avg / 1000                                // GB/s
// allgather/alltoall/alltoallv 特殊: algbw *= nranks
```

特点：
- **直接调用** `ccl::allreduce()`，无中间层开销
- 迭代之间**无额外 barrier / device_synchronize**
- `when()` 使用 `std::chrono::high_resolution_clock`（主机侧计时）
- 最终 latency 是 **所有 rank 的平均值**（快 rank 拉低均值）

### 1.2 micro_perf（Python + torch-ccl）

源码位置：`backends/INTEL/backend_intel.py` L362-396

```python
start_event.record()                            # XPU Event 计时
for i in range(N):
    dist.all_reduce(tensor, ...)                # Python → dispatch_stub → torch-ccl → ccl::allreduce
    if is_large_ccl_op:
        op_group_barrier(...)                   # 额外 dist.all_reduce(int32[1]) 作为 barrier！
end_event.record()
end_event.synchronize()

latency_us = elapsed_time / N                   # 单 rank 端到端
algbw = algo_size / latency_us / 1e3
```

特点：
- 通过 PyTorch `dist.all_reduce()` 间接调用 oneCCL，中间经过完整软件栈：
  ```
  Python dist.all_reduce()
    → GIL 释放/重获
    → dispatch_stub.cpp verbose 打印
    → globalMutex 锁
    → ccl::allreduce()
    → execute() → runLoop 队列 → work->synchronize()
  ```
- `CCL_SAME_STREAM=1` 强制同步执行（同一 SYCL stream）
- 大 tensor 每次迭代插入 `op_group_barrier()`（额外 allreduce 用于 BCS 排空）

---

## 二、Latency 差异来源

按影响程度排序：

| # | 因素 | oneCCL benchmark | micro_perf | 影响量级 |
|---|------|-----------------|------------|---------|
| **1** | **op_group_barrier** | 无 | 每次迭代插入一个额外 `dist.all_reduce(int32[1])` 作为 BCS 排空 barrier | **+数百~数千 μs/iter** |
| **2** | **CCL_SAME_STREAM=1** | 默认不同 stream（异步） | 强制同一 stream，每个 op 完全串行 | 增加排队延迟 |
| **3** | **torch-ccl 中间层** | 直接 `ccl::allreduce()` | dispatch_stub → globalMutex → execute() → runLoop | **+100~500 μs/iter** |
| **4** | **Python 解释器开销** | C++ 紧凑循环 | Python for 循环 + tensor indexing + GIL + 函数调用 | **+数十 μs/iter** |
| **5** | **跨 rank 平均方式** | `sum / (iters × nranks)` — 快 rank 拉低均值 | 每 rank 独立报告后 merge_summary 再平均 | 差异较小 |

### 2.1 op_group_barrier 的额外开销（最大因素）

```python
# core/backend.py L321-328
def op_group_barrier(self, op_group=None, group_size=1):
    dist_module.all_reduce(
        torch.tensor([1], dtype=torch.int32, device='xpu'),
        op=dist_module.ReduceOp.SUM,
        group=op_group
    )
```

每次 `op_group_barrier()` 本质上是一次完整的 allreduce 调用（虽然数据量极小），包含：
- oneCCL 初始化 + 调度开销
- GuC exec queue 创建/销毁
- 跨卡同步等待

这个 barrier 在 oneCCL benchmark 中**完全不存在**，是 latency 差距的最主要来源。

> 注意：这个 barrier 不能随意移除——它是为了排空 BCS DMA 队列、防止大 tensor 场景下
> GuC 累积压力导致 GT reset。详见 `allreduce_2gb_hang_fix.md` 第九节。

---

## 三、为什么 algo bandwidth 却接近

### 3.1 公式相同

两者都使用：
```
algbw = algo_size / latency / 1000   （GB/s）
```

algo_size 定义一致：都是 per-rank 的数据量（`elem_count × dtype_size`）。

### 3.2 大 tensor 下固定开销占比趋零

```
大 tensor (≥ 512 MB):
  latency ≈ GPU传输时间(~50ms) + 固定开销(~1ms)
  固定开销 / GPU传输时间 ≈ 2%
  → algbw_micro ≈ algbw_oneccl

小 tensor (< 1 MB):
  latency ≈ 固定开销(~1ms) + GPU传输时间(~10μs)
  固定开销主导
  → algbw_micro ≪ algbw_oneccl
```

### 3.3 不同数据规模下的预期差距

| 数据规模 | latency 差距 | algbw 差距 |
|---------|-------------|-----------|
| 小 (< 1 MB) | **10x ~ 100x** | 显著偏低 |
| 中 (1 MB ~ 128 MB) | **2x ~ 5x** | 偏低 |
| 大 (≥ 512 MB) | **1.1x ~ 2x** | 接近一致 |

---

## 四、bus bandwidth 系数对比

两者的 busbw 修正系数**完全一致**（N = world_size）：

| 算子 | oneCCL benchmark | micro_perf | 是否一致 |
|------|-----------------|------------|---------|
| allreduce | `algbw × 2(N-1)/N` | `algo_size × 2(N-1)/N / latency` | ✅ |
| reduce_scatter | `algbw × (N-1)/N` | `algo_size × (N-1)/N / latency` | ✅ |
| allgather | `algbw × nranks × (N-1)/N` | `output_size × (N-1)/N / latency` | ✅（等价） |
| alltoall | `algbw × nranks × (N-1)/N` | `output_size × (N-1)/N / latency` | ✅（等价） |

> 说明：oneCCL benchmark 对 allgather/alltoall 先 `algbw *= nranks` 再 `busbw = algbw × (N-1)/N`，
> 等价于 `bytes × nranks × (N-1)/N / latency`；micro_perf 直接用 `output_tensor_size`（已含 nranks 因子），
> 结果相同。

---

## 五、micro_perf 带宽/延迟计算公式详解

源码位置：`core/op.py` L186-225

### 5.1 summary()（单 rank）

```python
algo_size = op_instance.algo_size(tensor_list)        # per-rank 算法数据量（字节）
bus_size  = op_instance.bus_size(tensor_list)          # per-rank 总线数据量（字节）
latency_us = ...                                      # XPU Event 或 wallclock 计时

algo_bw = algo_size / latency_us / 1e3                # GB/s
bus_bw  = bus_size  / latency_us / 1e3                # GB/s
```

### 5.2 各算子的 algo_size / bus_size

| 算子 | algo_size | bus_size | 备注 |
|------|-----------|----------|------|
| AllReduce | `input_tensor_size` | `2(N-1)/N × algo_size` | N=2 时 bus_size = algo_size |
| ReduceScatter | `input_tensor_size` | `(N-1)/N × algo_size` | |
| AllGather | `output_tensor_size` | `(N-1)/N × algo_size` | output = input × N |
| AlltoAll | `output_tensor_size` | `(N-1)/N × algo_size` | output = input |
| Broadcast | `tensor_size` | `algo_size` | 非标准系数 |

### 5.3 merge_summary()（跨 rank 合并）

```python
latency(us)    = avg(latency_list)                    # 各 rank latency 平均
algo_bw(GB/s)  = avg(algo_bw_list)                    # 各 rank algo_bw 平均
bus_bw(GB/s)   = avg(bus_bw_list)                     # 各 rank bus_bw 平均
algo_bw_sum    = sum(algo_bw_list)                    # 所有 rank algo_bw 之和
bus_bw_sum     = sum(bus_bw_list)                     # 所有 rank bus_bw 之和
```

### 5.4 实际数据验证

以 `batch_size=524288, dim_size=1024, dtype=bfloat16, world_size=2` 为例：

```
algo_size = 524288 × 1024 × 2 = 1,073,741,824 B (1 GB)
latency   = 53270.854 μs

algo_bw = 1,073,741,824 / 53270.854 / 1000 = 20.156 GB/s  ✅ 与实际输出吻合

bus_size = 2 × (2-1)/2 × 1,073,741,824 = 1,073,741,824 B
bus_bw  = 1,073,741,824 / 53270.854 / 1000 = 20.156 GB/s  ✅（N=2 时 bus_bw = algo_bw）
```

---

## 六、结论与建议

1. **micro_perf 的 latency 偏高是预期行为**：它测量的是 PyTorch 用户视角下的端到端延迟
   （包含框架开销和安全 barrier），而非纯 CCL 内核延迟。两者测量目标不同，都是正确的。

2. **对比带宽时应使用 algo bandwidth / bus bandwidth**：这两个指标在大 tensor 场景下
   两套测试趋于一致，可直接对比。

3. **若需在 micro_perf 中获取更接近 oneCCL benchmark 的原始延迟**，可考虑：
   - 移除 `op_group_barrier()` 并改用 `device_synchronize()` 作为轻量级同步
   - 减少 verbose 打印开销（`ONECCL_BINDINGS_FOR_PYTORCH_ENV_VERBOSE=0`）
   - ⚠️ **注意**：移除 barrier 可能导致大 tensor 场景下 BCS 溢出和 GuC 累积压力
     （详见 `allreduce_2gb_hang_fix.md` 第九节）

4. **Latency 对比参考表**（同一硬件、同一数据大小）：

   | 来源 | 测量对象 | 包含的开销 |
   |------|---------|-----------|
   | oneCCL benchmark | 纯 CCL 内核延迟 | `ccl::start()` + `req.wait()` |
   | micro_perf | PyTorch 端到端延迟 | CCL + torch-ccl + dispatch_stub + barrier + Python |
   | NCCL-tests | 纯 NCCL 内核延迟 | `ncclAllReduce()` + `cudaStreamSynchronize()` |
