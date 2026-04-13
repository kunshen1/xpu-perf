# `python3 launch.py --workload workloads/xccl_ops/device2device.json --device 0 --backend INTEL --report_dir xccl_ops_report` 执行调用分析

## 0. 关键函数索引

> 以下使用相对路径 + 行号，便于直接跳转到源码位置。

| 关键函数 / 类 | 路径 |
| --- | --- |
| `parse_args()` | [`launch.py#L22`](../launch.py#L22) |
| `parse_workload()` | [`client.py#L59`](../client.py#L59) |
| `export_reports()` | [`client.py#L190`](../client.py#L190) |
| `get_cartesian_product()` | [`core/utils.py#L269`](../core/utils.py#L269) |
| `parse_json_file()` | [`core/utils.py#L307`](../core/utils.py#L307) |
| `XpuPerfServer` | [`perf_engine.py#L22`](../perf_engine.py#L22) |
| `XpuPerfServer.normal_bench()` | [`perf_engine.py#L108`](../perf_engine.py#L108) |
| `BaseEngine.dispatch()` | [`core/engine.py#L116`](../core/engine.py#L116) |
| `ComputeEngine` | [`core/engine.py#L144`](../core/engine.py#L144) |
| `ComputeEngine.start()` | [`core/engine.py#L148`](../core/engine.py#L148) |
| `Backend.load_all_ops()` | [`core/backend.py#L159`](../core/backend.py#L159) |
| `Backend.perf()` | [`core/backend.py#L363`](../core/backend.py#L363) |
| `Backend.compute_infer_loop()` | [`core/backend.py#L432`](../core/backend.py#L432) |
| `BackendINTEL` | [`backends/INTEL/backend_intel.py#L33`](../backends/INTEL/backend_intel.py#L33) |
| `BackendINTEL.core_perf()` | [`backends/INTEL/backend_intel.py#L106`](../backends/INTEL/backend_intel.py#L106) |
| `Device2DeviceOp` | [`core/ops/xccl_ops.py#L717`](../core/ops/xccl_ops.py#L717) |
| `Device2DeviceOp.prepare()` | [`core/ops/xccl_ops.py#L721`](../core/ops/xccl_ops.py#L721) |
| `Device2DeviceOp.device2device_run()` | [`core/ops/xccl_ops.py#L770`](../core/ops/xccl_ops.py#L770) |
| `OP_ENGINE_MAPPING` | [`core/ops/__init__.py`](../core/ops/__init__.py) |

## 1. 命令的实际执行上下文

这条命令对应的入口文件和 workload 文件都位于 `micro_perf/` 目录下：

- 入口：[`launch.py`](../launch.py)
- workload：[`workloads/xccl_ops/device2device.json`](../workloads/xccl_ops/device2device.json)

因此，这条命令**应当在 `micro_perf/` 目录中执行**。否则相对路径：

- `launch.py`
- `workloads/xccl_ops/device2device.json`
- `xccl_ops_report`

都会相对“当前工作目录”解析，可能导致找不到文件或把报告输出到错误位置。

## 2. 这条命令的总体执行链

主流程在 [`launch.py`](../launch.py) 中，执行顺序可以概括为：

1. `parse_args()` 解析命令行参数
2. 动态加载 `INTEL` backend
3. `backend_instance.load_all_ops()` 加载当前 backend 可用算子/provider
4. `parse_workload()` 读取并展开 `device2device.json`
5. 根据 `OP_ENGINE_MAPPING` 判断每个 op 应该交给哪个 engine
6. 创建 `XpuPerfServer`
7. 启动需要的 engine 子进程
8. 调用 `normal_bench()` 分发测试
9. 每个 case 在子进程中构造 `Device2DeviceOp`
10. 通过 backend 的 `perf()` / `core_perf()` 执行实际测量
11. 调用 `export_reports()` 输出 `info.json`、`.jsonl`、`.csv`

对应的关键文件：

- [`launch.py`](../launch.py)
- [`client.py`](../client.py)
- [`perf_engine.py`](../perf_engine.py)
- [`core/backend.py`](../core/backend.py)
- [`core/engine.py`](../core/engine.py)
- [`core/utils.py`](../core/utils.py)
- [`core/ops/__init__.py`](../core/ops/__init__.py)
- [`core/ops/xccl_ops.py`](../core/ops/xccl_ops.py)
- [`backends/INTEL/backend_intel.py`](../backends/INTEL/backend_intel.py)

## 3. 参数解析阶段

### 3.1 入口函数

[`launch.py`](../launch.py) 的主入口：

```python
if __name__ == "__main__":
    engine_args_dict, args = parse_args()
```

`parse_args()` 做了几件关键事情：

1. `mp.set_start_method('spawn', force=True)`  
   强制多进程使用 `spawn`
2. `setup_logger("INFO")`  
   初始化日志
3. 扫描 `micro_perf/backends/`，生成 `--backend` 可选值
4. 用 `argparse` 解析参数
5. 动态导入 backend，并实例化
6. 解析 NUMA、device、端口等运行参数

### 3.2 与本命令直接相关的参数

这条命令传入的是：

- `--workload workloads/xccl_ops/device2device.json`
- `--device 0`
- `--backend INTEL`
- `--report_dir xccl_ops_report`

对应影响如下：

| 参数 | 作用 |
| --- | --- |
| `--backend INTEL` | 选择 `backends.INTEL.backend_intel.BackendINTEL` |
| `--device 0` | 只使用 0 号设备，最终 `device_ids = [0]` |
| `--workload ...device2device.json` | 不走 `task_dir/task` 扫描，直接解析指定 workload |
| `--report_dir xccl_ops_report` | 最终报告输出到当前目录下的 `xccl_ops_report/` |

### 3.3 backend 动态加载

`launch.py` 中会执行：

```python
backend_module = importlib.import_module(
    "backends." + args.backend + ".backend_" + args.backend.lower())
backend_class = getattr(backend_module, "Backend" + args.backend)
backend_instance = backend_class()
backend_instance.backend_type = args.backend
backend_instance.load_all_ops()
```

对当前命令，实际效果是：

- 导入：`backends.INTEL.backend_intel`
- 实例化：`BackendINTEL`
- 记录：`backend_type = "INTEL"`
- 加载当前 backend 支持的所有 op/provider

## 4. `INTEL` backend 初始化阶段

`BackendINTEL` 位于：

- `micro_perf/backends/INTEL/backend_intel.py`

其初始化依赖：

- `torch.xpu.get_device_name(0)`
- `torch.xpu.device_count()`
- `torch.xpu.get_device_properties(0)`

这说明命令运行时要求：

1. Python 环境中的 `torch` 已支持 `xpu`
2. 机器上存在可访问的 Intel XPU
3. `device 0` 必须有效

文件里还有：

```python
try:
    import intel_extension_for_pytorch
except:
    pass
```

这表示 IPEX 通常是预期依赖，但这里没有做强校验；真正失败更可能发生在访问 `torch.xpu` 时。

## 5. workload 文件如何被解析

### 5.1 入口

在 [`launch.py`](../launch.py) 中：

```python
if args.workload is not None:
    test_cases = parse_workload(args.workload)
else:
    test_cases = parse_tasks(args.task_dir, args.task)
```

当前命令传了 `--workload`，因此会进入：

- [`client.py::parse_workload()`](../client.py#L59)

### 5.2 `parse_workload()`

逻辑很直接：

1. 把路径转成绝对路径
2. 判断文件是否存在
3. 根据后缀选择：
   - `.json` -> `parse_json_file()`
   - `.csv` -> `parse_csv_file()`

当前文件是 JSON，因此进入：

- [`core/utils.py::parse_json_file()`](../core/utils.py#L307)

### 5.3 `device2device.json` 的结构

文件内容：

```json
{
    "cases": [
        {
            "arg_type": "default",
            "dtype": ["float32", "float16", "bfloat16", "int8"],
            "batch_size": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152],
            "dim_size": [1024]
        }
    ]
}
```

### 5.4 `parse_json_file()` 如何转换

由于 JSON 顶层存在 `"cases"`，所以：

```python
task_name = json_file.stem
task_dict[task_name] = get_cartesian_product(parsed_data["cases"])
```

这里：

- `json_file.stem == "device2device"`
- 所以最终生成：

```python
{
  "device2device": [...]
}
```

### 5.5 `get_cartesian_product()` 如何展开

`core/utils.py` 中会对所有 list 字段做笛卡尔积展开。

本 workload 的组合数为：

- `dtype`: 4 种
- `batch_size`: 22 种
- `dim_size`: 1 种

所以最终会生成：

**4 x 22 x 1 = 88 个测试 case**

每个 case 最终类似：

```python
{
  "arg_type": "default",
  "dtype": "float32",
  "batch_size": 1024,
  "dim_size": 1024
}
```

## 6. 这个 op 实际会交给哪个 engine

这是这条命令最容易看错的点。

虽然 workload 文件放在：

- `workloads/xccl_ops/device2device.json`

但真正决定执行路径的不是目录名，而是：

- [`core/ops/__init__.py`](../core/ops/__init__.py)

里面的映射：

```python
"ComputeEngine": {
    "device2device": Device2DeviceOp,
}
```

也就是说：

- `device2device` -> `ComputeEngine`
- **不是** `XCCLEngine`

因此：

1. 不会走 XCCL 通信测量路径
2. 不会启动分布式通信专用 engine
3. 在当前 `--device 0`、单卡场景下，本质上是单卡上的 device-to-device copy 性能测试

## 7. `XpuPerfServer` 如何启动执行环境

### 7.1 构造需要的 engine 集合

`launch.py` 中先遍历 workload 中的 op：

```python
engine_set = set()
for op_name, op_cases in test_cases.items():
    if op_name not in OP_ENGINE_MAPPING:
        continue
    engine_name = OP_ENGINE_MAPPING[op_name]
    engine_set.add(engine_name)
```

对当前 workload：

- 只有一个 op：`device2device`
- 对应 engine：`ComputeEngine`

所以最终只会启动：

- `ComputeEngine`

### 7.2 进入 `XpuPerfServer`

```python
with XpuPerfServer(engine_args_dict, required_engines=engine_set) as server_instance:
```

构造发生在：

- [`perf_engine.py`](../perf_engine.py#L22)

关键逻辑：

```python
self.started_engines[engine_name] = ENGINE_TYPE_MAPPING[engine_name](args_dict)
```

当前只会构造：

- `ComputeEngine(args_dict)`

### 7.3 `ComputeEngine.start()`

位于：

- [`core/engine.py`](../core/engine.py#L148)

它会执行：

```python
self.subprocess_procs = mp.spawn(
    self.backend_instance.compute_infer_loop,
    args=(process_mapping, input_queue, output_queue),
    nprocs=self.device_num,
    join=False,
    daemon=False
)
```

当前命令中：

- `device_ids = [0]`
- 所以 `self.device_num = 1`

因此最终效果是：

- 启动 **1 个子进程**
- 该子进程绑定到 **device 0**

## 8. 子进程里真正发生了什么

### 8.1 `compute_infer_loop()`

在 [`core/backend.py`](../core/backend.py#L432) 中，子进程执行：

1. 设置 CPU affinity
2. `self.set_device(local_device_id)`，即设置到 device 0
3. 向主进程回传 `"success"`
4. 进入循环，从 `input_queue` 中取任务

每个任务结构大致是：

```python
(case_idx, (op_name, op_provider, op_cls), task_case)
```

### 8.2 任务如何进入队列

主进程在 `BaseEngine.dispatch()` 中做这件事：

```python
for op_provider, op_provider_info in self.backend_instance.op_mapping[op_name].items():
    op_cls = op_provider_info["op_cls"]
    self.input_queue.put((task_idx, (op_name, op_provider, op_cls), case))
```

这里说明一件关键事实：

- **同一个 case 会对该 op 的每个 provider 都执行一次**

## 9. `device2device` 在 INTEL 下会用哪个 provider

### 9.1 `load_all_ops()` 的决策逻辑

[`core/backend.py::load_all_ops()`](../core/backend.py#L159) 的优先级是：

1. 先加载 `backends/INTEL/ops/**/*.py`
2. 再看 `ProviderRegistry.OP_MAPPING`
3. 如果某个 op 仍然没有 provider，就回退到 `DEFAULT_OP_IMPL_MAPPING`

回退逻辑：

```python
self.op_mapping[op_name] = {
    "torch": {
        "op_cls": default_op
    }
}
```

### 9.2 当前 `device2device` 的实际情况

在 `backends/INTEL/ops/` 下没有找到 `device2device` 的专用实现；  
`ProviderRegistry` 里也没有看到 `device2device` 的 vendor 注册。

因此对这个命令来说，`device2device` 最终大概率使用：

- provider: **`torch`**
- op class: **[`core/ops/xccl_ops.py::Device2DeviceOp`](../core/ops/xccl_ops.py#L717)**

也就是说，这是用默认实现执行的基准测试，不是 INTEL backend 的定制 vendor op。

## 10. `Device2DeviceOp` 本身做了什么

定义位于：

- [`core/ops/xccl_ops.py`](../core/ops/xccl_ops.py#L717)

### 10.1 `prepare()`

它会读取并校验参数：

- `arg_type` 必须是 `"default"`
- `dtype` 只能是：
  - `float32`
  - `float16`
  - `bfloat16`
  - `int8`

然后构造两个 tensor：

- 输入：`src`
- 输出：`dst`

shape 为：

```python
[batch_size * dim_size]
```

例如：

- `batch_size=1024`
- `dim_size=1024`

则 tensor 长度是：

- `1024 * 1024`

### 10.2 实际执行函数

核心执行函数：

```python
def device2device_run(self, tensor_mapping):
    src = tensor_mapping["src"]
    dst = tensor_mapping["dst"]
    dst.copy_(src)
    return dst
```

因此这个 op 的实质就是：

- 在 XPU 上执行 `dst.copy_(src)` 的 device-to-device 拷贝

### 10.3 `core_run()` 为什么会调用到 `device2device_run()`

这里的关键不是 `core_run()` 里写死了 `device2device_run()`，而是对象初始化阶段已经完成了函数绑定。

完整过程如下：

1. 子进程创建 op 实例时，会执行：

   ```python
   op_instance = op_cls(task_case, self)
   ```

   对当前 workload，这里的 `op_cls` 就是 [`Device2DeviceOp`](../core/ops/xccl_ops.py#L717)。

2. [`Device2DeviceOp.__init__()`](../core/ops/xccl_ops.py#L718) 本身没有额外逻辑，只是调用父类构造：

   ```python
   super().__init__(args_dict, backend, *args, **kwargs)
   ```

3. 在 [`BasicOp.__init__()`](../core/op.py#L57) 中，先把运行函数初始化为默认占位函数：

   ```python
   self._run_func = self._empty_run
   ```

   对应位置：[`core/op.py#L79`](../core/op.py#L79)

4. 然后 `BasicOp.__init__()` 会立刻调用：

   ```python
   self.prepare()
   ```

   对应位置：[`core/op.py#L118`](../core/op.py#L118)

5. 因为当前实际对象是 `Device2DeviceOp`，这里会动态分派到它自己的 [`prepare()`](../core/ops/xccl_ops.py#L721)，而不是父类空实现。

6. 在 [`Device2DeviceOp.prepare()`](../core/ops/xccl_ops.py#L721) 的最后，会明确执行：

   ```python
   self._run_func = self.device2device_run
   ```

   对应位置：[`core/ops/xccl_ops.py#L768`](../core/ops/xccl_ops.py#L768)

7. 之后在测量阶段调用 [`op_instance.core_run(...)`](../core/op.py#L170) 时，`core_run()` 实际只有一层统一转发：

   ```python
   return self._run_func(*args, **kwargs)
   ```

   对应位置：[`core/op.py#L171`](../core/op.py#L171)

8. 由于 `self._run_func` 之前已经被绑定成 `self.device2device_run`，所以最终自然会落到 [`Device2DeviceOp.device2device_run()`](../core/ops/xccl_ops.py#L770)。

可以把这段关系概括成：

```text
BasicOp.__init__()
  -> self._run_func = self._empty_run
  -> self.prepare()                      # 动态分派到 Device2DeviceOp.prepare()
  -> self._run_func = self.device2device_run

后续执行：
op_instance.core_run(tensor_mapping)
  -> return self._run_func(tensor_mapping)
  -> Device2DeviceOp.device2device_run(tensor_mapping)
```

## 11. 性能测量链路

### 11.1 子进程中构造 op

在 `compute_infer_loop()` 中：

```python
op_instance = op_cls(task_case, self)
target_dict = self.perf(op_instance)
```

这里的 `self` 就是 backend 实例，即：

- `BackendINTEL`

### 11.2 `perf()` 做什么

[`core/backend.py::perf()`](../core/backend.py#L363) 的关键步骤：

1. 读取 op 的 `tensor_size`
2. 查询设备剩余内存
3. 判断是否有足够显存
4. 创建测试 tensor
5. 先做一次短测获得粗略时延
6. 根据粗略时延估算正式迭代次数
7. 做正式性能测量
8. 返回汇总结果

其中一个直接的失败点是：

```python
if tensor_size > assume_avail_bytes:
    raise RuntimeError("Not enough memory to run the op")
```

所以大 batch 场景下，如果显存不足，会在这里失败。

### 11.3 `BackendINTEL.core_perf()`

[`BackendINTEL.core_perf()`](../backends/INTEL/backend_intel.py#L106) 重写了 `core_perf()`：

- 默认先 warmup
- 再用 `torch.xpu.Event(enable_timing=True)` 计时
- 每轮执行 `op_instance.core_run(...)`

最终 `core_run()` 会调到：

- `Device2DeviceOp.device2device_run()`

也就是最终的 `dst.copy_(src)`。

## 12. `--device 0` 的实际影响

这条参数的影响非常直接：

1. `launch.py` 将其解析成 `device_ids = [0]`
2. `ComputeEngine` 中 `device_num = 1`
3. `mp.spawn(..., nprocs=1)` 只创建一个子进程
4. 子进程调用 `self.set_device(0)`
5. 所有 case 都只在 **device 0** 上执行

因此这条命令是：

- **单设备**
- **单子进程**
- **非分布式**

的本地性能测试。

## 13. `--backend INTEL` 的实际影响

它决定了以下几件事：

1. 使用 `torch.xpu` 作为设备接口
2. 设备信息从 Intel XPU 获取
3. 分布式 backend 使用：
   - `ccl`
4. provider 信息来自：
   - `backends/INTEL/provider_intel.py`
5. backend 清理逻辑会删除当前目录下可能产生的 `profiling/`

不过对当前 `device2device` workload 来说，最关键的影响还是：

- 运行设备是 XPU
- 实际测量由 `BackendINTEL.core_perf()` 完成

## 14. `--report_dir xccl_ops_report` 最终如何落盘

在 [`client.py::export_reports()`](../client.py#L190) 中：

```python
report_dir = pathlib.Path(given_report_dir).absolute()
report_dir.mkdir(parents=True, exist_ok=True)
```

因为传入的是相对路径 `xccl_ops_report`，而命令预期在 `micro_perf/` 下执行，所以最终根目录通常会变成：

- `/home/kshen/xpu-perf/micro_perf/xccl_ops_report`

后续输出结构是：

```text
xccl_ops_report/
  INTEL/
    <device_name>/
      info.json
      device2device/
        <provider>/
          device2device-<provider>.jsonl
          device2device-<provider>.csv
```

如果当前 provider 是默认回退的 `torch`，则常见结果类似：

```text
xccl_ops_report/
  INTEL/
    <device_name>/
      info.json
      device2device/
        torch/
          device2device-torch.jsonl
          device2device-torch.csv
```

## 15. 导出文件分别记录什么

### 15.1 `info.json`

记录四类信息：

- `backend_type`
- `common`
- `provider`
- `backend`
- `runtime`

其中 `runtime` 里会包含：

- `device_mapping`
- `device_ids`
- `numa_num`
- `numa_order`
- `node_world_size`
- `node_rank`

### 15.2 `device2device-*.jsonl`

每一行是一条测试结果，包含：

- `sku_name`
- `op_name`
- `provider`
- `arguments`
- `targets`

### 15.3 `device2device-*.csv`

会把：

- 固定字段
- arguments
- targets

拼成一行，便于后续表格分析。

## 16. 这条命令的关键调用顺序（函数级）

可以把这条命令的主干调用链总结为：

```text
launch.py::__main__                           -> ../launch.py
  -> parse_args()                            -> ../launch.py#L22
  -> BackendINTEL()                          -> ../backends/INTEL/backend_intel.py#L33
  -> Backend.load_all_ops()                  -> ../core/backend.py#L159
  -> parse_workload()                        -> ../client.py#L59
  -> parse_json_file()                       -> ../core/utils.py#L307
  -> get_cartesian_product()                 -> ../core/utils.py#L269
  -> XpuPerfServer(...)                      -> ../perf_engine.py#L22
  -> XpuPerfServer.__enter__()               -> ../perf_engine.py
  -> XpuPerfServer.create()                  -> ../perf_engine.py
  -> ComputeEngine.start()                   -> ../core/engine.py#L148
  -> BackendINTEL.compute_infer_loop()       -> ../core/backend.py#L432
  -> XpuPerfServer.get_info()                -> ../perf_engine.py#L65
  -> XpuPerfServer.normal_bench()            -> ../perf_engine.py#L108
  -> BaseEngine.dispatch()                   -> ../core/engine.py#L116
  -> Device2DeviceOp(...)                    -> ../core/ops/xccl_ops.py#L717
  -> Device2DeviceOp.prepare()               -> ../core/ops/xccl_ops.py#L721
  -> Backend.perf()                          -> ../core/backend.py#L363
  -> BackendINTEL.core_perf()                -> ../backends/INTEL/backend_intel.py#L106
  -> Device2DeviceOp.device2device_run()     -> ../core/ops/xccl_ops.py#L770
  -> export_reports()                        -> ../client.py#L190
```

## 17. 运行前提与常见失败点

### 17.1 前提

这条命令至少依赖：

- 可用的 Intel XPU 设备
- 支持 `torch.xpu` 的 PyTorch 环境
- `micro_perf/backends/INTEL` 存在
- workload 文件存在
- 当前目录可写，便于输出 `xccl_ops_report`

### 17.2 常见失败点

1. **执行目录不对**  
   导致 `launch.py` 或 `workload` 相对路径失效

2. **`INTEL` backend 无法导入**  
   会在动态导入 backend 时失败

3. **`torch.xpu` 不可用**  
   会在 `BackendINTEL.get_backend_info()` 阶段失败

4. **`device 0` 不存在**  
   会在设备信息获取或 `set_device(0)` 时失败

5. **显存不足**  
   会在 `Backend.perf()` 中触发 `Not enough memory to run the op`

6. **provider 未加载成功**  
   对这个 workload 通常会回退到默认 `torch` provider；如果默认 op 也不可用，则无法执行

## 18. 结论

这条命令的本质不是 XCCL 通信 benchmark，而是：

- 在 `micro_perf` 框架中
- 使用 `INTEL` backend
- 在 **device 0**
- 对 `device2device` 默认实现
- 进行 **88 组参数组合**
- 执行 XPU 上的 `dst.copy_(src)` 性能测试
- 并将结果输出到 `xccl_ops_report/INTEL/<device_name>/...`

最关键的判断点有两个：

1. workload 文件虽然位于 `xccl_ops/` 目录，但 `device2device` 实际映射到 `ComputeEngine`
2. INTEL backend 下没有明显的 `device2device` vendor 专用实现时，会回退到默认 `torch` provider
