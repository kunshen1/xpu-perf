# `CCL_BLOCKING_WAIT` 作用分析

本文总结 `torch-ccl` 中 `CCL_BLOCKING_WAIT` 的真实作用、代码路径，以及它和 `CCL_SAME_STREAM` / `TORCH_LLM_ALLREDUCE` 的关系。

---

## 1. 结论

`CCL_BLOCKING_WAIT` 控制的是：

- **XPU collective 的 `wait()` / `synchronize()` 是不是在 host 侧真正阻塞到通信完成**
- 而不是 collective 本身是否“异步提交”

从实现看：

- `CCL_BLOCKING_WAIT=1` -> **host blocking**
- `CCL_BLOCKING_WAIT=0` -> **host non-blocking**

也就是说，这个开关决定的是：

> collective 提交之后，完成性是由 host 线程显式等待来保证，还是由设备侧 stream dependency 来保证。

---

## 2. 关键代码位置

| 位置 | 路径 |
| --- | --- |
| 环境变量定义 | [`torch-ccl/src/ProcessGroupCCL.hpp#L94`](../../../torch-ccl/src/ProcessGroupCCL.hpp#L94) |
| 默认值定义 | [`torch-ccl/src/ProcessGroupCCL.hpp#L286`](../../../torch-ccl/src/ProcessGroupCCL.hpp#L286) |
| 环境变量解析 | [`torch-ccl/src/ProcessGroupCCL.cpp#L785`](../../../torch-ccl/src/ProcessGroupCCL.cpp#L785) |
| 构造时读取变量 | [`torch-ccl/src/ProcessGroupCCL.cpp#L927`](../../../torch-ccl/src/ProcessGroupCCL.cpp#L927) |
| 复制到 work 对象 | [`torch-ccl/src/utils.h#L705`](../../../torch-ccl/src/utils.h#L705) |
| XPU wait 路径 | [`torch-ccl/src/gpu/dpcpp_ccl.cpp#L433`](../../../torch-ccl/src/gpu/dpcpp_ccl.cpp#L433) |
| XPU stream barrier | [`torch-ccl/src/gpu/dpcpp_ccl.cpp#L453`](../../../torch-ccl/src/gpu/dpcpp_ccl.cpp#L453) |
| XPU work 异步完成标记 | [`torch-ccl/src/gpu/dpcpp_ccl.cpp#L678`](../../../torch-ccl/src/gpu/dpcpp_ccl.cpp#L678) |

---

## 3. 环境变量是怎么解析的

在 [`ProcessGroupCCL.cpp`](../../../torch-ccl/src/ProcessGroupCCL.cpp#L785) 中：

```cpp
bool parseTorchCCLEnvVarFlag(const char* envVarName, bool default_val) {
    char* stringValue = std::getenv(envVarName);
    int val;
    if (stringValue != nullptr) {
      val = std::stoi(stringValue);
    } else {
      return default_val;
    }

    if (val == 1) return true; else return false;
}
```

实际语义：

- 未设置：使用默认值
- 设为 `1`：返回 `true`
- 设为 `0`：返回 `false`
- 设为其他整数：也会走 `false`
- 设为 `ON/OFF/true/false`：`stoi` 失败，直接报错

因此从实现看，`CCL_BLOCKING_WAIT` **只可靠支持整数值**。

---

## 4. 默认值不是固定的

在 [`ProcessGroupCCL.hpp`](../../../torch-ccl/src/ProcessGroupCCL.hpp#L286) 中：

```cpp
#if CCL_MINOR_VERSION < 14
  bool blockingWait_ = true;
#else
  bool blockingWait_ = false;
#endif
```

所以环境变量未设置时：

- `CCL_MINOR_VERSION < 14` -> 默认 **blocking**
- `CCL_MINOR_VERSION >= 14` -> 默认 **non-blocking**

这说明默认行为和 **编译时 oneCCL 版本**有关，不是一个永远固定的常量。

---

## 5. 它在 XPU 后端里是怎么生效的

核心逻辑在 [`dpcpp_ccl.cpp`](../../../torch-ccl/src/gpu/dpcpp_ccl.cpp#L433)：

```cpp
bool wait(std::chrono::milliseconds timeout) override {
  this->synchronizeInternalForXPU(timeout);
  this->checkAndThrowException();
  return true;
}

void synchronizeInternalForXPU(std::chrono::milliseconds timeout) {
    if (this->blockingWait_) {
        this->synchronizeInternal(kNoTimeout);
    } else if (!this->useSameStream_) {
        ...
        torch_queue.ext_oneapi_submit_barrier({req.get_native()});
    }
}
```

对应含义如下。

### 5.1 `CCL_BLOCKING_WAIT=1`

`wait()` / `synchronize()` 会进入：

```cpp
this->synchronizeInternal(kNoTimeout);
```

效果：

- host 线程会真正等待 collective 完成
- `wait()` 返回时，通信已经完成
- 更容易理解，也更适合调试 hang / 测端到端通信完成时间

### 5.2 `CCL_BLOCKING_WAIT=0`

当 `CCL_SAME_STREAM=0` 时，`wait()` 不会在 host 侧真等，而是往当前 torch stream 插入 barrier：

```cpp
torch_queue.ext_oneapi_submit_barrier({req.get_native()});
```

效果：

- CPU 线程尽快返回
- 后续 compute stream 会在设备侧等待通信事件完成
- 语义是 **host non-blocking, device ordered**

当 `CCL_SAME_STREAM=1` 时，通信和计算本来就在同一条 stream 上，`wait()` 的额外动作就更少。

---

## 6. 和 `CCL_SAME_STREAM` 一起看最清楚

| `CCL_BLOCKING_WAIT` | `CCL_SAME_STREAM` | 实际效果 |
| --- | --- | --- |
| `1` | `0/1` | host 真等到 collective 完成 |
| `0` | `0` | host 不等，只在设备侧补 stream dependency |
| `0` | `1` | host 基本不等，通信和计算天然按同一 stream 顺序执行 |

所以 `CCL_BLOCKING_WAIT=0` 的价值主要体现在：

- separate communication stream 模式下
- 尽量避免 host stall
- 通过 stream 依赖维持正确执行顺序

---

## 7. 一个关键实现细节：future 完成不等于设备完成

在 [`dpcpp_ccl.cpp`](../../../torch-ccl/src/gpu/dpcpp_ccl.cpp#L670) 中：

```cpp
work->run();
work->finishAsyncWorkCCL();
queue_.push_back(work);
```

也就是说：

1. 先提交 collective
2. 立刻把 work 标成 finished
3. 再交给后台线程去调用 `work->synchronize()`

因此在 XPU backend 里：

- `future` 完成更接近“提交完成”
- 不一定代表 collective 在设备上已经彻底执行完

这也是 `CCL_BLOCKING_WAIT` 很重要的原因：

- `=1` 时，后台同步会真正等待到底
- `=0` 时，后台同步更像是在建立设备侧依赖，而不是 host 侧阻塞等待

---

## 8. 和 `TORCH_LLM_ALLREDUCE` 的关系

在 [`ProcessGroupCCL.cpp`](../../../torch-ccl/src/ProcessGroupCCL.cpp#L915) 中：

- 打开 `TORCH_LLM_ALLREDUCE` 时，会优先把配置往
  - `blockingWait_ = false`
  - `useSameStream_ = true`
  这个方向推

然后再读取：

- `CCL_SAME_STREAM`
- `CCL_BLOCKING_WAIT`

所以：

- `TORCH_LLM_ALLREDUCE` 会给出偏异步的默认组合
- 但 `CCL_BLOCKING_WAIT` 仍然可以显式覆盖它

---

## 9. 文档与实现存在不一致

当前代码和外部文档相比，有两点要特别注意：

1. 一些文档写 `CCL_BLOCKING_WAIT=0` 表示 blocking，但**代码实现不是这样**
2. 一些文档写支持 `ON/OFF`，但**代码实现实际上只接受整数并用 `stoi` 解析**

因此排查问题时，应以源码行为为准：

- `1` -> blocking
- `0` -> non-blocking

---

## 10. 对 micro_perf 使用上的建议

如果关注的是：

- **调试通信问题 / 明确判断 collective 是否真正完成**  
  建议优先用 `CCL_BLOCKING_WAIT=1`

- **追求 host 不阻塞、增强通信与执行重叠**  
  建议优先看 `CCL_BLOCKING_WAIT=0`，并结合 `CCL_SAME_STREAM` 一起分析

对 `micro_perf/run_xccl_ops.sh` 来说，这个变量本质上影响的是：

- benchmark 过程中 `torch-ccl` 对 wait/synchronize 的处理方式
- 进而影响 host stall、stream 编排，以及观测到的端到端表现

---

## 11. 一句话总结

`CCL_BLOCKING_WAIT` 控制的不是 collective 是否异步提交，而是：

> **XPU collective 提交之后，`wait()` / `synchronize()` 是由 host 线程真正阻塞到完成，还是只建立设备侧依赖后尽快返回。**
