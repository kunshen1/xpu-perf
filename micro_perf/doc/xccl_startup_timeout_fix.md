# `XCCLEngine` 启动 timeout 修复说明

本文记录 `micro_perf` 在容器内执行：

```bash
python3 launch.py --workload workloads/xccl_ops/all_reduce.json --device 0,1 --backend INTEL --report_dir xccl_ops_report
```

时，`xccl infer loop timeout` 的修复内容。

---

## 1. 现象

启动日志中可以看到：

- `device_id: ...`
- `xccl init done, rank: ...`

但主进程仍在 [`core/engine.py`](../core/engine.py) 中等待子进程回传 `"success"` 超时：

```python
signal = self.output_queue.get(timeout=60)
```

说明问题发生在 `xccl_infer_loop()` 启动自检阶段，而不是 workload 正式执行阶段。

---

## 2. 根因判断

`xccl_infer_loop()` 原来的关键顺序是：

```python
self.initialize_ccl(rank, world_size)
self.set_device(local_device_id)
dist.all_reduce(data, op=dist.ReduceOp.SUM)
cpu_group = dist.new_group(..., backend="gloo")
```

这里有两个风险点：

1. **设备绑定顺序过晚**  
   子进程先初始化 `ccl`，再设置 `torch.xpu` 当前设备。在容器内 `mp.spawn` 多进程场景下，这容易导致 rank 的设备上下文不稳定。

2. **启动挂点不可见**  
   即使日志里出现 `xccl init done`，也无法区分到底是：
   - 首次 XPU `all_reduce` 卡住
   - 还是后续 `gloo` `new_group()` 卡住

---

## 3. 已应用的修改

最终结构调整后，这些修复**不再落在 `core/`**，而是放在：

- [`backends/INTEL/backend_intel.py`](../backends/INTEL/backend_intel.py)
- [`backends/INTEL/engine_intel.py`](../backends/INTEL/engine_intel.py)

其中：

- `backend_intel.py` 负责 INTEL backend 的 `xccl_infer_loop()` 启动/执行/退出细节
- `engine_intel.py` 负责 INTEL 专用的 `XCCLEngine.stop()` 优雅退出逻辑
- `backend_intel.py` 在初始化时直接覆盖 `perf_engine.ENGINE_TYPE_MAPPING["XCCLEngine"]`

### 3.1 提前绑定本地 XPU

文件：[`backends/INTEL/backend_intel.py`](../backends/INTEL/backend_intel.py)

将：

```python
self.initialize_ccl(rank, world_size)
self.set_device(local_device_id)
```

改为：

```python
self.set_device(local_device_id)
self.initialize_ccl(rank, world_size)
```

这样每个 rank 会先绑定自己的本地设备，再初始化 `ccl` backend。

### 3.2 补充启动阶段定位日志

在启动自检路径新增了以下日志：

- `rank X: set_device done -> ...`
- `rank X: before startup all_reduce`
- `rank X: after startup all_reduce`
- `rank X: before gloo new_group`
- `rank X: after gloo new_group`

这样可以快速判断卡点落在：

- XPU collective
- 还是辅助 `gloo` group 初始化

### 3.3 调整默认 `master_addr`

文件：

- [`launch.py`](../launch.py)
- [`server.py`](../server.py)

将默认值从：

```python
localhost
```

改为：

```python
127.0.0.1
```

原因是单机容器场景下，`127.0.0.1` 通常比 `localhost` 更稳定，能减少 hostname / IPv4/IPv6 解析带来的干扰。

---

## 4. 修改后的排查结果

在应用上述修改并重新运行后，日志已经可以把挂点进一步收敛。

观察到：

- 两个 rank 都能完成 `set_device`
- 两个 rank 都能完成 `dist.init_process_group(backend="ccl")`
- 两个 rank 都打印了 `before startup all_reduce`
- 但没有任何一个 rank 打印 `after startup all_reduce`

这说明：

- 问题**不在** `gloo new_group`
- 问题也**不再是**设备绑定顺序
- 当前真正卡死的位置是第一发：

```python
dist.all_reduce(data, op=dist.ReduceOp.SUM)
```

也就是 `torch-ccl` / `oneCCL` 在容器内执行首个 XPU collective 时挂住。

---

## 5. 从 oneCCL 日志得到的新结论

进一步打开 oneCCL 日志后，可以看到：

- `did not find MPI-launcher specific variables, switch to ATL/OFI`
- `provider: tcp`
- 网卡被识别为 `tcp:br-...`
- `atl-ofi ... hmem: 0`
- `atl attrs ... shm: 0, hmem: 0`
- `ze fabric ports: 0 were able to be detected`
- `could not initialize umf api`

这些信息组合起来说明：

1. 当前运行方式没有使用 MPI launcher，oneCCL 自动退回到了 **ATL/OFI**
2. OFI 最终选择的是 **Docker bridge 上的 tcp provider**
3. 当前路径没有可用的：
   - GPU memory HMEM 支持
   - OFI SHM 支持
   - Level Zero fabric / XeLink 能力
   - UMF 路径

因此当前 hang 更像是：

> `init_process_group("ccl")` 完成了控制面初始化，但在第一发 XPU `all_reduce` 需要真正建立 device 数据路径时，容器内 oneCCL 实际落到了一条 `OFI/tcp + docker bridge + no hmem/shm` 的退化路径上，最终 collective 无法完成。

---

## 6. 当前结论

到这一步，已经可以把问题分成两层：

### 6.1 已经修复/确认的部分

- `XCCLEngine` 启动时的 `set_device` 顺序已修正
- 启动挂点日志已补齐
- `master_addr` 默认值已收紧为 `127.0.0.1`
- 问题**不是** `gloo new_group`

### 6.2 仍然存在的问题

- 容器内的 `mp.spawn + torch-ccl + oneCCL` 组合仍然在第一发 XPU collective 卡住
- 因此剩余问题更偏向 **运行环境 / launcher 模式**，而不再是 `micro_perf` 启动逻辑本身

---

## 7. 推荐的后续改动方向

基于当前定位结果，继续修复的重点不应再是 `CCL_BLOCKING_WAIT` 或 `CCL_SAME_STREAM`，而应转到 **运行方式** 和 **容器网络/IPC 环境**。

### 7.1 运行环境改动

推荐优先把容器改成更接近宿主机的模式：

```bash
docker run --rm -it \
  --net=host \
  --ipc=host \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  -v /dev/dri:/dev/dri \
  ...
```

并在运行前显式限制 OFI 的网络选择，例如：

```bash
export MASTER_ADDR=127.0.0.1
export FI_TCP_IFACE=lo
```

目标是避免 oneCCL 再次选中 `br-*` 这种 Docker bridge 接口。

### 7.2 验证最小脚本

在继续改 `micro_perf` 前，应先用最小脚本验证环境：

```bash
cd /home/kshen/xpu-perf/micro_perf

CCL_ZE_IPC_EXCHANGE=sockets \
FI_TCP_IFACE=lo \
torchrun --standalone --nproc_per_node=2 \
  backends/INTEL/projects/dist_profiler.py
```

如果这个脚本也挂，说明问题在容器 / oneCCL / XPU distributed 环境本身，不在 `micro_perf`。

### 7.3 代码层后续重构方向

如果最小脚本能通过，而 `micro_perf` 仍有问题，则推荐把 `micro_perf` 的 XCCL 执行模型从：

- 父进程内 `mp.spawn` 拉 rank

逐步改成：

- `torchrun` / `mpirun` 外部 launcher 管理 rank 生命周期

建议做法：

1. 在 [`launch.py`](../launch.py) 增加 `--xccl_launcher` 选项
2. 在 [`core/engine.py`](../core/engine.py) 中区分：
   - `spawn`
   - `torchrun`
   - `mpi`
3. 在 [`core/backend.py`](../core/backend.py) 中把当前 `xccl_infer_loop()` 拆成统一的 rank 主循环
4. 在外部 launcher 模式下，用 `broadcast_object_list` 替代当前基于 `Queue + all_gather_object` 的父子进程分发方式
5. 仅由 rank0 负责：
   - 解析 workload
   - 汇总结果
   - 导出 report

这样可以避免 `mp.spawn` 与 oneCCL 在容器里的兼容性问题。

---

## 8. 如何继续判断问题边界

当前建议的判断顺序如下：

1. **先看最小脚本**
   - 如果 `dist_profiler.py` 也挂：环境问题
   - 如果 `dist_profiler.py` 能过：`micro_perf` 执行模型问题

2. **再看 launcher**
   - `mp.spawn` 不通时，优先试 `torchrun`
   - 如果需要更接近官方/仓库主路径，再试 `mpirun`

3. **最后再看 micro_perf 重构**
   - 只有在环境可用的前提下，再推进 `--xccl_launcher` 之类的代码改造

---

## 9. 结论

这份修复目前完成了两件事：

- 修正了 `XCCLEngine` 启动顺序上的明显问题
- 把 timeout 的真实挂点从“启动超时”精确收敛到了“第一发 XPU `dist.all_reduce` 卡死”

后续如果继续推进，重点应放在：

- 容器网络 / IPC / `/dev/dri` / 权限配置
- oneCCL 在容器中的传输路径选择
- 将 `micro_perf` 从 `mp.spawn` 模式逐步迁移到 `torchrun/mpirun` 驱动模式

---

## 10. 新现象：结果已输出，但退出阶段触发 `drm_neo.cpp` abort

在后续运行中，benchmark 已经成功输出了结果，例如：

- `latency(us)`
- `algo_bw(GB/s)`
- `bus_bw(GB/s)`

随后才出现：

```text
Abort was called at 268 line in file:
./shared/source/os_interface/linux/drm_neo.cpp
```

这说明：

- collective 执行路径已经跑通
- 问题不再是 benchmark 本身失败
- 新问题更像发生在 **进程退出 / process group 销毁 / XPU runtime teardown** 阶段

---

## 11. 对退出路径的修复

针对退出期 abort，又补充了以下修改。

### 11.1 子进程先完成 XPU 与 distributed 清理，再回传 shutdown ack

文件：[`backends/INTEL/backend_intel.py`](../backends/INTEL/backend_intel.py)

在 `xccl_infer_loop()` 末尾，增加了更完整的退出路径：

1. `torch.xpu.synchronize()`
2. 显式销毁 `dist_group_mapping` 中动态创建的 group
3. 显式销毁 `cpu_group`
4. 最后销毁默认 process group
5. 由 `rank 0` 回传：

```python
("shutdown", "done")
```

这样父进程可以明确知道：子进程已经走完清理逻辑，而不是仅靠 `join(timeout=10)` 猜测。

### 11.2 父进程等待 shutdown ack，再回收子进程

文件：[`backends/INTEL/engine_intel.py`](../backends/INTEL/engine_intel.py)

原来的 `XCCLEngine.stop()` 会：

1. 往 `input_queue` 发 `None`
2. 只等 10 秒
3. 超时就直接 `kill()`

这很容易在子进程还没完成 XPU/CCL/Level Zero 清理时就强行结束，进而触发底层 `drm_neo.cpp` abort。

现在改成：

1. 先发送退出信号
2. 等待 `("shutdown", "done")`
3. 再用更长的时间 `join`
4. 只有仍未退出时才作为兜底 `kill()`

### 11.3 增加 stop 防重入

文件：[`backends/INTEL/engine_intel.py`](../backends/INTEL/engine_intel.py)

由于：

- `BaseEngine.__del__()` 会调用 `stop()`
- `XpuPerfServer.destroy()` 也会调用 `engine_instance.stop()`

退出阶段存在重复 stop 的风险。

因此新增了 `_stopped` 标志，避免多次回收同一批子进程和队列资源。

---

## 12. 继续收敛后的代码优化

在上述修复基础上，又补充了几项稳定性优化。

### 12.1 本地默认地址统一为 `127.0.0.1`

文件：

- [`launch.py`](../launch.py)
- [`server.py`](../server.py)
- [`client.py`](../client.py)

本地单机场景默认地址统一改为：

```python
127.0.0.1
```

这样直接运行：

```bash
python3 launch.py --workload workloads/xccl_ops/all_reduce.json --device 0,1 --backend INTEL --report_dir xccl_ops_report
```

时，不再默认依赖 `localhost` 的解析结果，更接近 `run_xccl_ops.sh` 的推荐运行方式。

### 12.2 XCCL heartbeat 在 stop 前先停掉

文件：[`core/engine.py`](../core/engine.py)

`XCCLEngine` 原本带有 heartbeat 线程，会周期性注入一个 demo `all_reduce` 来维持 worker 活性。

这条路径如果和退出流程重叠，会让父进程的 stop 与后台 demo case 抢同一套 queue / worker / process group 资源。

现在补充了：

1. heartbeat stop event
2. `stop()` 前先停 heartbeat
3. `stop()` 与 `dispatch()` 共用同一把锁
4. 正常 `dispatch()` 也会刷新 `last_dispatch_time`

这样可以避免：

- 已有真实 workload 时 heartbeat 误判空闲
- stop 与 heartbeat dispatch 交错
- teardown 阶段又被注入新的 XCCL collective

### 12.3 超出 active world size 的 XCCL case 显式跳过

文件：[`perf_engine.py`](../perf_engine.py)

以前如果 workload 中写了：

- `world_size=4/8/16/...`

但当前实际只启动了：

- `--device 0,1`

对应的 case 会在 worker 里被动返回空结果，问题是：

- 跳过发生得太晚
- 用户很难看出哪些 case 根本不可能执行

现在在 dispatch 前就会：

1. 按 active XCCL world size 过滤 case
2. 打印 skipped warning
3. 仍保持结果顺序与原 workload 对齐

这样 report/export 的结构不变，但无效 case 不再真的进入 XCCL 执行路径。

---

## 13. 当前状态

目前这组修改已经覆盖了两个阶段的问题：

1. **启动阶段**
   - 修复设备绑定顺序
   - 补充启动挂点日志
   - 收敛到首个 XPU collective

2. **退出阶段**
   - 增加更完整的 distributed/XPU 清理顺序
   - 避免父进程过早 `kill()`
   - 减少 `drm_neo.cpp` abort 这类 teardown 问题

3. **结构层面**
   - `INTEL` backend 继续提供专用 `xccl_infer_loop`
   - `core/engine.py` 额外承担 heartbeat/stop 协调
   - 由 INTEL backend 自己提供：
      - backend 专用 `xccl_infer_loop`
      - backend 专用 `XCCLEngine`
   - 由 `BackendINTEL` 在初始化时接管 `perf_engine` 中的 `XCCLEngine` 映射

如果后续还有退出期异常，新的清理逻辑已经能把问题进一步区分为：

- group 销毁顺序问题
- XPU runtime 收尾问题
- 或底层驱动/容器环境问题
