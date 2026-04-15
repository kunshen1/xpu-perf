import os
import sys
import csv
import json
import math
import pathlib
import random
import shutil
import subprocess
import traceback
from datetime import timedelta

import torch
# try:
#     import intel_extension_for_pytorch
# except:
#     pass
import torch.distributed as dist
import time
try:
    import oneccl_bindings_for_pytorch
except:
    pass

FILE_DIR = pathlib.Path(__file__).parent.absolute()
BACKEND_DIR = FILE_DIR.parent
MICRO_PERF_DIR = BACKEND_DIR.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.backend import Backend
from core.utils import suppress_stdout_stderr
try:
    from backends.INTEL.provider_intel import INTEL_PROVIDER
except:
    INTEL_PROVIDER = {}


class BackendINTEL(Backend):
    def __init__(self):
        super().__init__()

    def get_backend_info(self):
        info_dict = {}

        # device info
        info_dict["device_name"] = torch.xpu.get_device_name(0)
        info_dict["device_count"] = torch.xpu.device_count()

        device_properties = torch.xpu.get_device_properties(0)
        info_dict["device_memory_mb"] = device_properties.total_memory / (1024 ** 2)

        info_dict["torch_version"] = torch.__version__

        return info_dict

    def get_provider_info(self):
        return INTEL_PROVIDER

    def clean_extra_files(self):
        PROFILER_DIR = pathlib.Path.cwd().joinpath("profiling")
        if PROFILER_DIR.exists():
            shutil.rmtree(PROFILER_DIR)


    """
    device management related
    """
    def get_torch_device_name(self):
        return "xpu"
    
    def get_device_name(self, index = 0):
        return torch.xpu.get_device_name(index)
    
    def get_device_properties(self, index = 0):
        return torch.xpu.get_device_properties(index)
    
    def get_mem_info(self, index = 0):
        total_memory = torch.xpu.get_device_properties(index).total_memory
        allocated_memory = torch.xpu.memory_allocated(index)
        cached_memory = torch.xpu.memory_reserved(index)
        free_memory = (total_memory - allocated_memory)
        return (free_memory, total_memory)
    
    def get_device_count(self):
        device_count = torch.xpu.device_count()
        return device_count, list(range(device_count))
    
    def set_device(self, device_index : int):
        torch.xpu.set_device(device_index)

    def get_device(self):
        return torch.xpu.current_device()

    def device_synchronize(self):
        torch.xpu.synchronize()

    def empty_cache(self):
        torch.xpu.empty_cache()


    """
    ccl related
    """
    def get_dist_module(self):
        return dist
    
    def get_dist_backend(self):
        return "ccl"

    def initialize_ccl(self, rank: int, world_size: int):
        # Apply CCL/XPU environment tuning before dist.init_process_group so
        # that CCL picks them up at startup time.
        #
        # CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0: skip the XeLink topology
        #   vertex-connection scan.  On PCIe-only systems the scan is a no-op
        #   but still serialises startup; disabling it is safe and consistent
        #   with the CCL warning output ("consider disabling if not correct").
        #
        # SYCL_CACHE_PERSISTENT=1 / ZE_ENABLE_MODULE_CACHE=1: enable persistent
        #   SYCL JIT kernel caches.  The first CCL collective on XPU triggers
        #   SYCL kernel compilation which can take 30-60 s per device.  With a
        #   persistent cache subsequent runs skip recompilation.
        os.environ.setdefault("CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK", "0")
        os.environ.setdefault("SYCL_CACHE_PERSISTENT", "1")
        os.environ.setdefault("ZE_ENABLE_MODULE_CACHE", "1")

        # CCL_BLOCKING_WAIT=1: make the host thread block in work.wait()
        #   until the collective truly completes on the device.  This ensures
        #   all BCS DMA is fully drained after every CCL op, preventing the
        #   cumulative GuC exec queue pressure that leads to GT resets.
        #
        #   Without this, dist.all_reduce() returns while BCS DMA is still
        #   in flight (host non-blocking mode).  Consecutive CCL ops then
        #   pile up BCS submissions across the entire test suite, and the
        #   async __guc_exec_queue_destroy_async in the kernel accumulates
        #   until the GuC times out and triggers a device-level GT reset.
        #
        #   For a benchmark that measures one collective at a time, host
        #   blocking has zero performance impact — we measure device-side
        #   latency via XPU Events, not host wall-clock time.
        #
        # CCL_SAME_STREAM=1: oneCCL reuses PyTorch's SYCL compute stream
        #   instead of creating a separate internal stream.  This makes
        #   torch.xpu.synchronize() (device_synchronize) effective for
        #   draining CCL's BCS queue, because both PyTorch ops and CCL ops
        #   are on the same stream.  Without this, device_synchronize()
        #   only drains PyTorch's own queue while CCL's BCS continues
        #   running on a separate stream — the root cause of BCS overflow
        #   when device_synchronize() was used in warmup loops.
        #
        # Both use setdefault so users/scripts (e.g. run_xccl_ops.sh) can
        # override via environment variables if needed.
        os.environ.setdefault("CCL_BLOCKING_WAIT", "1")
        os.environ.setdefault("CCL_SAME_STREAM", "1")

        return super().initialize_ccl(rank, world_size)

    # DMA transfers above this size require per-iteration BCS drain to
    # prevent the xe BCS (Blitter Command Streamer) command queue from
    # overflowing.  The drain mechanism differs by op type:
    #   - D2H / H2D: device_synchronize() — BCS via PyTorch's SYCL queue.
    #   - CCL collectives: op_group_barrier() — BCS via oneCCL's internal
    #     queue, which is invisible to device_synchronize().
    # Queuing multiple large DMA submissions without draining causes xe
    # driver engine resets, which cascade to CCS resets and permanently
    # hang any in-flight CCL collectives.
    _BCS_SYNC_THRESHOLD_BYTES = 512 * 1024 * 1024   # 512 MB
    # tensor_size >= this value → reduce prefer_iterations to 1 for CCL ops.
    # At this size (input+output ≥ 1 GB, algo_size ≥ 512 MB), each allreduce
    # produces ≥ 2 GB of bi-directional PCIe BCS DMA.  Even with per-iteration
    # CCL barriers the total BCS volume over 3 iterations is large enough to
    # destabilise the GuC exec queue on Battlemage.  Capping at 1 iteration
    # keeps the per-case BCS traffic bounded while still producing a valid
    # latency measurement.
    _LARGE_CCL_ITER_THRESHOLD_BYTES = 1024 * 1024 * 1024  # 1 GB

    def _op_has_cpu_tensor(self, op_instance):
        """Return True if the op transfers data between device and host memory."""
        for info_dict in [
            op_instance.input_tensor_info or {},
            op_instance.output_tensor_info or {},
        ]:
            for info in info_dict.values():
                if getattr(info, "device", None) == "cpu":
                    return True
        return False

    def _needs_bcs_throttle(self, op_instance):
        """Return True when per-iteration BCS drain is needed.

        BCS (Blitter Command Streamer) throttle is required for any operation
        whose tensor is large enough to risk overflowing the xe GuC exec queue
        when multiple iterations are submitted back-to-back without draining.

        This covers two cases with DIFFERENT drain mechanisms:
          1. D2H / H2D: explicit PCIe DMA via the BCS engine.
             Drain: device_synchronize() — effective because PyTorch owns
             the SYCL queue that submits the DMA.
          2. Large CCL collectives (AllReduce, etc.) over PCIe: the
             ring-allreduce algorithm internally issues BCS DMA transfers
             for the reduce-scatter and all-gather phases.
             Drain: op_group_barrier() — submits a new CCL op to the same
             process group, which forces oneCCL's FIFO to drain the
             previous op's BCS before starting the barrier.
             device_synchronize() is a NO-OP for oneCCL's internal queues.
        """
        return op_instance.tensor_size >= self._BCS_SYNC_THRESHOLD_BYTES

    def core_perf(
        self, op_instance, 
        warmup_iterations, prefer_iterations, 
        tensor_list, 
        profiling=True
    ):
        op_group = op_instance.op_group
        group_size = op_instance.group_size

        # H2D / D2H copies run on the PCIe DMA engine, not the XPU compute
        # queue.  torch.xpu.Event only tracks SYCL kernel completions, so it
        # cannot time these transfers accurately.  Fall back to wall-clock
        # timing (same approach as the base Backend.core_perf) for any op
        # that touches CPU pinned memory.
        use_wallclock = self._op_has_cpu_tensor(op_instance) or getattr(op_instance, 'skip_profiling', False)

        if not op_instance.is_concurrent and profiling and not use_wallclock:
            process_id = os.getpid()
            PROFILER_DIR = pathlib.Path.cwd().joinpath("profiling", f"{process_id}")
            PROFILER_DIR.mkdir(parents=True, exist_ok=True)
            TRACE_FILE = PROFILER_DIR.joinpath("trace.json")

            # profiling
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.XPU], 
                schedule=torch.profiler.schedule(
                    wait=0, 
                    warmup=warmup_iterations, 
                    active=prefer_iterations, 
                    repeat=1
                ), 
                on_trace_ready=lambda prof: prof.export_chrome_trace(f"{TRACE_FILE}")
            ) as prof:
                for i in range(prefer_iterations + warmup_iterations):
                    op_instance.core_run(tensor_list[i % len(tensor_list)])
                    self.device_synchronize()
                    prof.step()

            # parse and delete profiling json file
            average_latency = 0.
            flash_attn_latency = 0.
            kernel_latency_list = {}
            if TRACE_FILE.exists():
                profiling_data = json.loads(TRACE_FILE.read_text())
                for event in profiling_data["traceEvents"]:
                    if event.get("cat", None) in ["kernel", "gpu_memcpy"]:
                        kernel_name = event["name"]
                        kernel_latency = event["dur"]
                        if kernel_name not in kernel_latency_list:
                            kernel_latency_list[kernel_name] = []
                        kernel_latency_list[kernel_name].append(kernel_latency)

                for kernel_name, latency_list in kernel_latency_list.items():
                    average_latency += sum(latency_list)
                    if "micro_sdpa" in kernel_name or "cute::MhaName" in kernel_name:
                        flash_attn_latency += sum(latency_list)

                average_latency /= prefer_iterations
                flash_attn_latency /= prefer_iterations

                shutil.rmtree(PROFILER_DIR)

            return flash_attn_latency if flash_attn_latency > 0 else average_latency, list(kernel_latency_list.keys())
        
        else:
            bcs_throttle = self._needs_bcs_throttle(op_instance)

            # --- Classify the op BEFORE warmup so both warmup and
            # measurement loops use the correct BCS drain strategy. ---
            #
            # With CCL_BLOCKING_WAIT=1 + CCL_SAME_STREAM=1 (set in
            # initialize_ccl as defaults):
            #   - Each dist.all_reduce() blocks the host until the
            #     collective fully completes on the device.
            #   - device_synchronize() IS effective because CCL uses the
            #     same SYCL stream as PyTorch.
            #   - The op_group_barrier calls below are redundant but
            #     harmless — they serve as a safety net in case the user
            #     overrides CCL_SAME_STREAM=0 via environment.
            #
            # Without these env vars (CCL_SAME_STREAM=0):
            #   - device_synchronize() only drains PyTorch's own SYCL
            #     queue; oneCCL's internal BCS queue keeps running.
            #   - op_group_barrier is the only reliable BCS drain:
            #     oneCCL's strict FIFO ordering ensures the barrier
            #     cannot start until the previous allreduce's BCS
            #     transfers have fully completed.
            #
            # "Large CCL op" = tensor_size >= 512 MB (bcs_throttle) AND
            # the op participates in a CCL group (group_size > 1) AND it
            # is NOT a cross-device copy (D2H/H2D use PyTorch's own BCS
            # queue, not oneCCL's).
            is_large_ccl_op = (
                bcs_throttle
                and group_size > 1
                and not self._op_has_cpu_tensor(op_instance)
            )

            # Warmup: drain BCS after each iteration.
            #
            # With CCL_BLOCKING_WAIT=1, each allreduce already blocks
            # until completion, so the barriers are redundant.  They are
            # kept as a safety net for CCL_SAME_STREAM=0 override.
            #
            # - Large CCL ops: op_group_barrier forces oneCCL FIFO drain.
            # - D2H / H2D: device_synchronize() drains PyTorch's queue.
            for i in range(warmup_iterations):
                op_instance.core_run(tensor_list[i % len(tensor_list)])
                if is_large_ccl_op:
                    self.op_group_barrier(op_group=op_group, group_size=group_size)
                elif bcs_throttle:
                    self.device_synchronize()
            
            self.device_synchronize()

            # Wall-clock path: used for ESIMD kernels (skip_profiling=True) and
            # all cross-device (H2D / D2H) transfers where XPU events cannot
            # track PCIe DMA latency correctly.
            if use_wallclock:
                # For ops that touch CPU pinned memory (D2H / H2D), per-rank
                # copies are independent – no XPU-side CCL barrier is needed.
                # Calling op_group_barrier here would trigger a CCL AllReduce
                # on the XPU op_group for EVERY case iteration, creating and
                # destroying GuC exec queues each time.  Over many cases this
                # accumulates (e.g. 88 world_size=2 D2H cases × 4 barriers =
                # 352 CCL ops), causing __guc_exec_queue_destroy_async to hog
                # the CPU, eventually triggering xe BCS/GT engine resets that
                # corrupt SYCL contexts and hang subsequent dist.new_group()
                # calls.  The gloo all_gather_object in xccl_infer_loop already
                # provides inter-rank synchronisation between cases, so the
                # XPU-level barrier is redundant for independent D2H/H2D ops.
                has_cpu_tensor = self._op_has_cpu_tensor(op_instance)
                self.device_synchronize()
                if not has_cpu_tensor:
                    self.op_group_barrier(op_group=op_group, group_size=group_size)
                start_time = time.perf_counter()
                for i in range(prefer_iterations):
                    op_instance.core_run(tensor_list[i % len(tensor_list)])
                    # For large transfers (>= 512 MB) synchronize after every
                    # copy to drain the xe BCS command queue.  Without this,
                    # back-to-back large DMA submissions overflow the GuC exec
                    # queue, triggering BCS engine resets that cascade to CCS
                    # resets and permanently hang in-flight CCL collectives.
                    if bcs_throttle:
                        self.device_synchronize()
                if not bcs_throttle:
                    self.device_synchronize()
                end_time = time.perf_counter()
                if not has_cpu_tensor:
                    self.op_group_barrier(op_group=op_group, group_size=group_size)
                latency_us = (end_time - start_time) * 1e6 / prefer_iterations
                return latency_us, []
            
            start_event = torch.xpu.Event(enable_timing=True)
            end_event = torch.xpu.Event(enable_timing=True)

            # is_large_ccl_op is already computed before the warmup loop.

            self.device_synchronize()
            # Pre-measurement barrier: drains any pending BCS from warmup
            # iterations before the timing window opens.
            self.op_group_barrier(op_group=op_group, group_size=group_size)
            start_event.record()
            for i in range(prefer_iterations):
                op_instance.core_run(tensor_list[i % len(tensor_list)])
                if is_large_ccl_op:
                    # CCL-level barrier: forces the previous allreduce's BCS
                    # DMA to fully complete before the next iteration starts.
                    # device_synchronize() is deliberately NOT used here; it
                    # does not drain oneCCL's internal BCS queues.
                    self.op_group_barrier(op_group=op_group, group_size=group_size)
                elif bcs_throttle:
                    # D2H/H2D large copies: wallclock path handles these, but
                    # belt-and-suspenders device sync for any edge case.
                    self.device_synchronize()
            end_event.record()
            end_event.synchronize()

            self.device_synchronize()
            # Post-measurement barrier: for large CCL ops the last per-iteration
            # op_group_barrier above already drained BCS, so a second barrier
            # here is redundant (and adds unnecessary GuC exec queue overhead).
            if not is_large_ccl_op:
                self.op_group_barrier(op_group=op_group, group_size=group_size)

            latency_us = start_event.elapsed_time(end_event) * 1e3 / prefer_iterations
            return latency_us, []

    def perf(self, op_instance):
        tensor_size = op_instance.tensor_size

        # Release cached device memory *before* querying free memory and
        # allocating tensors.  Without this, the allocator cache from the
        # previous (possibly large) case inflates apparent device memory
        # usage and may cause spurious OOM or hangs.
        self.empty_cache()

        device_mem_info = self.get_mem_info()
        avail_memory = device_mem_info[0]

        assume_avail_bytes = int(avail_memory * 0.9)
        assume_cache_size = 1 * (1024 ** 3)

        latency_us = 0.
        kernel_mapping = {}

        try:
            min_test_iters = 10
            sleep_time = 0.2
            max_test_time = 1e6
            max_data_cnt = 1

            # For large D2H/H2D ops we throttle iterations at the BCS threshold
            # to keep total DMA submission volume within safe bounds.
            # The per-iteration device_synchronize() in core_perf already drains
            # the BCS queue, but capping iterations reduces the window during
            # which GuC exec queue pressure builds up.
            bcs_throttle = self._needs_bcs_throttle(op_instance)
            if bcs_throttle:
                min_test_iters = 3
            # For very large CCL tensors (tensor_size ≥ 1 GB, i.e. algo_size
            # ≥ 512 MB) each iteration generates ≥ 2 GB of bi-directional PCIe
            # BCS DMA.  With per-iteration CCL barriers (op_group_barrier) the
            # BCS queue is drained between iterations, but accumulating even 3
            # iterations of barrier+allreduce+barrier triples the GuC exec queue
            # pressure.  Capping at 1 measured iteration keeps the per-case BCS
            # volume bounded while still producing a valid latency result.
            if op_instance.tensor_size >= self._LARGE_CCL_ITER_THRESHOLD_BYTES:
                min_test_iters = 1

            if not op_instance.is_concurrent:
                # Cross-device ops (H2D / D2H) allocate both a device tensor
                # and a pinned-CPU tensor, but tensor_size only reflects the
                # device side.  Multiple instances would multiply locked pages
                # in RAM and risk hanging on mlock.  Always use a single
                # instance for ops that touch CPU memory.
                if self._op_has_cpu_tensor(op_instance):
                    max_data_cnt = 1
                elif tensor_size > assume_avail_bytes:
                    raise RuntimeError("Not enough memory to run the op")
                elif 2 * tensor_size > assume_avail_bytes:
                    max_data_cnt = 1
                elif tensor_size > assume_cache_size:
                    max_data_cnt = 2
                else:
                    max_data_cnt = min(
                        math.floor(max(assume_avail_bytes, assume_cache_size) / tensor_size),
                        math.floor(assume_cache_size / tensor_size)
                    )

            tensor_list = op_instance.create_tensors(max_data_cnt)
            random.shuffle(tensor_list)

            is_large_tensor = op_instance.tensor_size >= self._LARGE_CCL_ITER_THRESHOLD_BYTES

            if is_large_tensor:
                # Skip probe phase entirely for very large tensors.
                # We already know min_test_iters=1, so the probe result
                # cannot change prefer_iters — it always computes
                # min(max(X, 2), 1) = 1.  But the probe itself runs
                # warmup+measurement allreduces + barriers = ~5 extra CCL
                # ops, each of which creates/destroys a GuC exec queue.
                # Over the full test suite this cumulative pressure causes
                # __guc_exec_queue_destroy_async to hog the CPU and
                # eventually trigger GT reset + drm_neo.cpp abort.
                prefer_iters = 1
            else:
                latency_us, _ = self.core_perf(op_instance, 2, 2, tensor_list, profiling=False)
                prefer_iters = min(max(int(max_test_time / latency_us), 2), min_test_iters)
                if op_instance.group_size > 1:
                    dist_module = self.get_dist_module()
                    prefer_iters_list = [None for _ in range(op_instance.group_size)]
                    dist_module.all_gather_object(prefer_iters_list, prefer_iters, group=op_instance.op_group)
                    prefer_iters = max(prefer_iters_list)
            time.sleep(sleep_time)

            actual_profiling = self.enable_profiling and op_instance.require_profiling
            actual_warmup = 1 if is_large_tensor else 2
            latency_us, kernel_mapping = self.core_perf(op_instance, actual_warmup, prefer_iters, tensor_list, profiling=actual_profiling)

            del tensor_list
            self.empty_cache()

            # Cooldown after large ops: give the xe driver time to process
            # pending async GuC exec queue destructions.  Large CCL ops
            # generate heavy BCS traffic whose cleanup
            # (__guc_exec_queue_destroy_async) runs asynchronously in the
            # kernel.  Without a pause the next case starts while the
            # driver is still cleaning up, compounding GuC pressure until
            # a GT reset is triggered.
            if bcs_throttle:
                self.device_synchronize()
                time.sleep(1.0)
        except Exception as e:
            traceback.print_exc()

        return op_instance.summary(latency_us, kernel_mapping)
