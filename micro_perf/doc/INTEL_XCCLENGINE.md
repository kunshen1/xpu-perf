# INTEL 后端下 `XCCLEngine` 的调用与执行路径

本文总结 `micro_perf` 中 `XCCLEngine` 在 **INTEL backend** 下的完整调用链，重点回答：

1. `XCCLEngine` 在什么条件下会被选中
2. 它实际调用的是哪个后端
3. INTEL backend 如何初始化 XCCL / CCL 通信环境
4. 通信 op 最终是如何落到 `torch.distributed` 和 `torch.xpu` 上执行的

---

## 1. 关键函数索引

| 关键函数 / 类 | 路径 |
| --- | --- |
| `OP_ENGINE_MAPPING` / `SUPPORTED_OPS` | [`core/ops/__init__.py`](../core/ops/__init__.py) |
| `XpuPerfServer` | [`perf_engine.py#L22`](../perf_engine.py#L22) |
| `XpuPerfServer.normal_bench()` | [`perf_engine.py#L108`](../perf_engine.py#L108) |
| `XCCLEngine` | [`core/engine.py#L208`](../core/engine.py#L208) |
| `XCCLEngine.start()` | [`core/engine.py#L235`](../core/engine.py#L235) |
| `Backend.initialize_ccl()` | [`core/backend.py#L304`](../core/backend.py#L304) |
| `Backend.xccl_infer_loop()` | [`core/backend.py#L505`](../core/backend.py#L505) |
| `Backend.perf()` | [`core/backend.py#L363`](../core/backend.py#L363) |
| `BackendINTEL` | [`backends/INTEL/backend_intel.py#L33`](../backends/INTEL/backend_intel.py#L33) |
| `BackendINTEL.get_dist_backend()` | [`backends/INTEL/backend_intel.py#L102`](../backends/INTEL/backend_intel.py#L102) |
| `BackendINTEL.core_perf()` | [`backends/INTEL/backend_intel.py#L106`](../backends/INTEL/backend_intel.py#L106) |
| `AllReduceOp` | [`core/ops/xccl_ops.py#L19`](../core/ops/xccl_ops.py#L19) |
| `ReduceScatterOp` | [`core/ops/xccl_ops.py#L98`](../core/ops/xccl_ops.py#L98) |
| `AllGatherOp` | [`core/ops/xccl_ops.py#L178`](../core/ops/xccl_ops.py#L178) |

---

## 2. `XCCLEngine` 在什么条件下会被选中

`XCCLEngine` 是否参与执行，取决于 workload 中的 op 是否映射到了它。

在 [`core/ops/__init__.py`](../core/ops/__init__.py) 里：

```python
"XCCLEngine": {
    "all_reduce": AllReduceOp,
    "reduce_scatter": ReduceScatterOp,
    "all_gather": AllGatherOp,
    "all_to_all": AlltoAllOp,
    "broadcast": BroadcastOp,
    "p2p": P2POp,
    "all_reduce_h2d": AllReduce_H2D_Op,
}
```

也就是说，下面这些 op 会进入 `XCCLEngine`：

- `all_reduce`
- `reduce_scatter`
- `all_gather`
- `all_to_all`
- `broadcast`
- `p2p`
- `all_reduce_h2d`

随后 [`XpuPerfServer.normal_bench()`](../perf_engine.py#L108) 会根据 `OP_ENGINE_MAPPING` 把 case 按 engine 分组，分给 `XCCLEngine`。

---

## 3. 什么时候 `XCCLEngine` 会真正启动

在 [`XpuPerfServer.__init__()`](../perf_engine.py#L22) 中：

```python
if engine_name == "XCCLEngine" and len(self.device_ids) * self.node_world_size <= 1:
    continue
```

这意味着：

- 如果总参与设备数 `<= 1`
- 即使 workload 中存在 `all_reduce` 之类的 op
- 也**不会**启动 `XCCLEngine`

所以 `XCCLEngine` 只在 **多卡** 或 **多节点** 场景下有意义。

---

## 4. `XCCLEngine` 实际调用的是哪个后端

如果命令行传的是：

```bash
--backend INTEL
```

那么 `launch.py` 会动态导入并实例化：

- [`backends/INTEL/backend_intel.py`](../backends/INTEL/backend_intel.py)
- 类：[`BackendINTEL`](../backends/INTEL/backend_intel.py#L33)

之后这个 `backend_instance` 会传进 `XpuPerfServer`，再传给 `XCCLEngine`。

因此在 INTEL 场景下：

```text
XCCLEngine
  -> backend_instance
  -> BackendINTEL
```

所以 `XCCLEngine` 自己只是调度层，真正的设备和通信后端都由 `BackendINTEL` 提供。

---

## 5. `XCCLEngine.start()` 实际启动的是什么

在 [`core/engine.py#L235`](../core/engine.py#L235)：

```python
self.subprocess_procs = mp.spawn(
    self.backend_instance.xccl_infer_loop,
    args=(
        self.process_mapping,
        self.input_queue,
        self.output_queue,
        self.master_addr,
        self.device_port,
        self.node_world_size,
        self.node_rank,
    ),
    nprocs=self.device_num,
    join=False,
    daemon=False
)
```

在 INTEL backend 下，`self.backend_instance` 是 `BackendINTEL`，所以真正被 `spawn` 的入口是：

- [`Backend.xccl_infer_loop()`](../core/backend.py#L505)

这个函数定义在 `Backend` 基类里，但运行时 `self` 是 `BackendINTEL` 实例，因此里面调用的抽象/虚方法会落到 INTEL 实现。

---

## 6. `BackendINTEL` 提供了哪些关键能力

### 6.1 设备能力：`torch.xpu`

在 [`backends/INTEL/backend_intel.py`](../backends/INTEL/backend_intel.py) 中：

```python
def get_torch_device_name(self):
    return "xpu"

def set_device(self, device_index: int):
    torch.xpu.set_device(device_index)

def device_synchronize(self):
    torch.xpu.synchronize()
```

这说明：

- tensor 创建在 `xpu`
- 当前设备切换用 `torch.xpu.set_device(...)`
- 同步用 `torch.xpu.synchronize()`

### 6.2 通信能力：`torch.distributed + ccl`

同一文件中：

```python
def get_dist_module(self):
    return dist

def get_dist_backend(self):
    return "ccl"
```

因此 INTEL backend 下的 XCCL 通信基础是：

- Python API：`torch.distributed`
- 进程组 backend：`ccl`

### 6.3 `torch.distributed + ccl` 底层是怎么接上的

这里的 `ccl` 不是 `xpu-perf` 自己实现的，也不是这个仓库中的 Python 代码手写实现的后端。

它的实现关系可以理解为：

1. `torch.distributed` 提供统一的 Python 分布式 API
2. `backend="ccl"` 表示创建一个 **CCL ProcessGroup**
3. 这个 CCL ProcessGroup 通常由 Intel 的 **oneCCL bindings for PyTorch**
   （`oneccl_bindings_for_pytorch` / `torch-ccl`）提供
4. 该 ProcessGroup 再把 collective 请求转发到底层 **Intel oneCCL** 动态库执行

也就是说：

```text
用户代码
  -> torch.distributed.init_process_group(backend="ccl")
  -> PyTorch C10D ProcessGroup 框架
  -> Intel oneCCL bindings 提供的 CCL ProcessGroup
  -> Intel oneCCL 通信库
```

从 Intel 的 `oneccl_bindings_for_pytorch` 项目可以看到，它的定位就是：

- 实现 PyTorch 的 **C10D ProcessGroup API**
- 作为外部 ProcessGroup 动态加载
- 底层加载 oneCCL 相关动态库

因此，`torch.distributed + ccl` 的真正含义是：

- **PyTorch 负责统一 API**
- **CCL ProcessGroup 负责接收 collective 请求**
- **oneCCL 负责真正执行跨 rank 通信**

### 6.4 在 `micro_perf` 中的具体落点

结合本项目，INTEL backend 下的通信链路是：

```text
BackendINTEL.get_dist_module()   -> torch.distributed
BackendINTEL.get_dist_backend()  -> "ccl"

torch.distributed.init_process_group(backend="ccl")
  -> 创建 CCL ProcessGroup

torch.distributed.all_reduce(..., group=self.op_group)
  -> 请求交给 CCL ProcessGroup
  -> 再由底层 oneCCL 执行
```

与此同时，设备侧仍由 `torch.xpu` 提供：

- `torch.xpu.set_device(...)`
- `torch.xpu.synchronize()`
- XPU tensor 分配与访问

所以在 INTEL 场景里，可以把执行层拆成两部分：

| 执行层 | 具体实现 |
| --- | --- |
| 设备执行 | `torch.xpu` |
| 分布式通信 | `torch.distributed` + `ccl` + oneCCL |

---

## 7. `xccl_infer_loop()` 如何初始化 INTEL 分布式环境

在 [`core/backend.py#L505`](../core/backend.py#L505) 的 `xccl_infer_loop()` 中，主要流程如下。

### 7.1 绑定 CPU 和设备

每个子进程先：

1. 绑定 CPU affinity
2. 计算本地 rank / 全局 rank / world_size
3. 设置环境变量：
   - `RANK`
   - `LOCAL_RANK`
   - `WORLD_SIZE`
   - `MASTER_ADDR`
   - `MASTER_PORT`

### 7.2 初始化 `ccl` 进程组

如果 `world_size > 1`，会调用：

```python
self.initialize_ccl(rank, world_size)
```

最终落到 [`core/backend.py#L304`](../core/backend.py#L304)：

```python
dist.init_process_group(
    backend=self.get_dist_backend(),
    init_method="env://",
    timeout=timedelta(seconds=1800)
)
```

由于 INTEL backend 的 `get_dist_backend()` 返回 `"ccl"`，所以这里实际执行的是：

```python
torch.distributed.init_process_group(backend="ccl", ...)
```

### 7.3 切换到当前 XPU 设备

在 `xccl_infer_loop()` 中紧接着执行：

```python
self.set_device(local_device_id)
```

在 INTEL backend 下，它等价于：

```python
torch.xpu.set_device(local_device_id)
```

### 7.4 创建 `cpu_group`

为了同步 Python 对象（task、结果、状态），代码额外创建：

```python
cpu_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
```

这里：

- `ccl` 用来跑真正的 XPU collective
- `gloo` 用来跑 `all_gather_object()` 这种 Python 对象同步

---

## 8. `XCCLEngine` 如何把任务送给所有 rank

在 [`xccl_infer_loop()`](../core/backend.py#L559) 的主循环中：

1. 只有 `rank == 0` 从 `input_queue` 取任务
2. 其余 rank 先拿到 `None`
3. 再通过 `cpu_group` 的 `all_gather_object()` 同步任务对象
4. 所有 rank 都得到同一份 task

所以这里的调度方式是：

- 主进程只投递一次任务
- rank 0 接收
- 再由所有 rank 同步展开

### 8.1 `all_gather_object()` 是在哪里定义的

这里调用的：

```python
dist.all_gather_object(...)
```

不是 `xpu-perf` 自己实现的，而是来自 PyTorch：

- 模块：`torch.distributed`
- 具体源码文件：`torch/distributed/distributed_c10d.py`
- 函数名：`all_gather_object(object_list, obj, group=None)`

在 [`core/backend.py`](../core/backend.py) 顶部有：

```python
import torch.distributed as dist
```

所以这里的 `dist.all_gather_object(...)`，本质上就是：

```python
torch.distributed.all_gather_object(...)
```

### 8.2 `all_gather_object()` 的实现思路

这个 API 并不是底层直接支持“任意 Python 对象通信”，它的实现思路是：

1. 先把 Python 对象 `obj` 用 `pickle` 序列化
2. 再把序列化结果转成 `uint8 tensor`
3. 先 `all_gather` 每个 rank 上对象的字节长度
4. 取最大长度，把每个 rank 的输入 tensor pad / resize 到统一大小
5. 再调用普通 tensor 版 `all_gather`
6. 最后把 gather 回来的字节流逐个反序列化回 Python 对象

可以概括成：

```text
Python object
  -> pickle
  -> uint8 tensor
  -> all_gather(size)
  -> all_gather(bytes tensor)
  -> unpickle
  -> Python object
```

### 8.3 对应到 PyTorch 实现中的关键步骤

PyTorch 的 `all_gather_object()` 核心流程大致是：

```python
current_device = _get_object_coll_device(group)
input_tensor, local_size = _object_to_tensor(obj, current_device, group)

all_gather(object_size_list, local_size, group=group)
max_object_size = int(max(object_size_list).item())

input_tensor.resize_(max_object_size)
coalesced_output_tensor = torch.empty(
    max_object_size * group_size, dtype=torch.uint8, device=current_device
)

all_gather(output_tensors, input_tensor, group=group)

for i, tensor in enumerate(output_tensors):
    object_list[i] = _tensor_to_object(tensor, tensor_size, group)
```

这里最重要的点是：

- 它内部实际还是调用普通的 `all_gather`
- 只是把 Python 对象包装成了 tensor 再通信

### 8.4 回到 `XCCLEngine` 里的这段代码

在 [`xccl_infer_loop()`](../core/backend.py#L559) 里：

```python
dist.all_gather_object(
    exchange_area,
    {"rank": rank, "data": data},
    group=cpu_group,
)
```

这里同步的是一个 Python `dict`：

```python
{"rank": rank, "data": data}
```

也就是说，这段代码实际上是在 `cpu_group` 上：

1. 把这个 dict pickle 成字节流
2. 转成 tensor
3. 用 `gloo` 执行 `all_gather`
4. 再在每个 rank 上恢复成 Python 对象

之所以这里使用 `cpu_group` 而不是 XPU 的 `ccl` group，是因为这一步的目的不是做高性能张量 collective，而是为了方便地同步：

- task 对象
- rank 状态
- 结果字典

这类 Python 层面的元数据。

---

## 9. 通信 op 是如何构造的

在 [`core/backend.py#L603`](../core/backend.py#L603)：

```python
op_instance = op_cls(
    task_case,
    self,
    op_group=dist_group_mapping.get(task_world_size, None),
    group_size=task_world_size
)
```

其中：

- `op_cls` 可能是 `AllReduceOp` / `ReduceScatterOp` / `AllGatherOp`
- `self` 是 `BackendINTEL`
- `op_group` 是给这个 world_size 创建的通信 group
- `group_size` 表示当前多少 rank 参与这个 collective

因此每个通信 op 都拿到了：

1. INTEL backend
2. 当前 collective 的 group
3. 当前 collective 的参与规模

---

## 10. 以 `AllReduceOp` 为例的完整执行链

### 10.1 `prepare()` 绑定运行函数

在 [`core/ops/xccl_ops.py#L23`](../core/ops/xccl_ops.py#L23) 开始的 `AllReduceOp.prepare()` 中，最后会执行：

```python
self._run_func = self.all_reduce_run
```

同时 tensor device 取自：

```python
device=self.backend.get_torch_device_name()
```

在 INTEL backend 下，这个 device 就是：

- `"xpu"`

### 10.2 `XCCLEngine` 最终调用 `BackendINTEL.perf()`

在 `xccl_infer_loop()` 中：

```python
target_dict = self.perf(op_instance)
```

这里的 `self` 是 `BackendINTEL`。  
虽然 `perf()` 定义在基类里，但内部会继续调用 INTEL backend 的实现，比如：

- `get_mem_info()` -> `torch.xpu.*`
- `core_perf()` -> `BackendINTEL.core_perf()`
- `device_synchronize()` -> `torch.xpu.synchronize()`

### 10.3 `BackendINTEL.core_perf()` 调 `op_instance.core_run()`

在 [`backends/INTEL/backend_intel.py#L106`](../backends/INTEL/backend_intel.py#L106) 中，性能循环会调用：

```python
op_instance.core_run(tensor_list[i % len(tensor_list)])
```

因为 `AllReduceOp.prepare()` 里已经把：

```python
self._run_func = self.all_reduce_run
```

绑定好了，所以 `core_run()` 最终会调用到：

- `AllReduceOp.all_reduce_run()`

### 10.4 `AllReduceOp.all_reduce_run()` 最终调用 `torch.distributed.all_reduce`

在 [`core/ops/xccl_ops.py#L85`](../core/ops/xccl_ops.py#L85)：

```python
dist_module = self.backend.get_dist_module()
dist_module.all_reduce(
    src,
    op=dist_module.ReduceOp.SUM,
    group=self.op_group
)
```

对 INTEL backend 而言：

- `self.backend.get_dist_module()` 返回 `torch.distributed`
- 进程组已经按 `backend="ccl"` 初始化完成

所以这里实际执行的是：

```python
torch.distributed.all_reduce(src, op=SUM, group=<ccl group>)
```

而 `src` tensor 是：

- XPU tensor

因此这一步就是：

- 在 INTEL XPU 上
- 使用 `ccl` 通信后端
- 执行 `all_reduce`

---

## 11. 其他 XCCL op 对应到的后端 API

同样的模式还会出现在其他 XCCL op 中：

| Op | 典型调用 |
| --- | --- |
| `all_reduce` | `torch.distributed.all_reduce(...)` |
| `reduce_scatter` | `torch.distributed.reduce_scatter_tensor(...)` |
| `all_gather` | `torch.distributed.all_gather_into_tensor(...)` 或相关 gather API |
| `broadcast` | `torch.distributed.broadcast(...)` |
| `all_to_all` | `torch.distributed.all_to_all_single(...)` 或相关 all-to-all API |
| `p2p` | `torch.distributed.send/recv` 或 batched P2P API |

因此可以把 `XCCLEngine` 理解为：

- **负责起多进程、分发任务、同步 rank**

而真正的执行则由：

- **`BackendINTEL` 提供 XPU 设备能力**
- **`torch.distributed(ccl)` 提供通信能力**
- **各个 XCCL Op 提供具体 collective 调用**

---

## 12. 最终调用链总结

可以把 INTEL 场景下的 `XCCLEngine` 总结为下面这条链：

```text
XpuPerfServer.normal_bench()
  -> 根据 OP_ENGINE_MAPPING 识别为 XCCLEngine
  -> XCCLEngine.start()
  -> mp.spawn(BackendINTEL.xccl_infer_loop)

BackendINTEL.xccl_infer_loop()
  -> 设置 rank / world_size / env
  -> torch.distributed.init_process_group(backend="ccl")
  -> torch.xpu.set_device(...)
  -> 构造通信 op（如 AllReduceOp）
  -> BackendINTEL.perf(op_instance)
  -> BackendINTEL.core_perf()
  -> op_instance.core_run(...)
  -> 具体 op 的 *_run()
  -> torch.distributed.<collective>(..., group=self.op_group)
  -> rank 0 merge_summary()
  -> output_queue 返回结果
```

---

## 13. 一句话结论

在 INTEL backend 下，`XCCLEngine` 本质上是一个 **多进程调度层**；  
真正的执行层是：

- 设备侧：`torch.xpu`
- 通信侧：`torch.distributed` + `ccl`
- 算子侧：`AllReduceOp` / `AllGatherOp` / `ReduceScatterOp` 等 `xccl_ops.py` 中的实现

所以 `XCCLEngine` 并不是自己“实现通信”，而是把任务组织起来，最终交给 **INTEL XPU + CCL backend** 去执行。
