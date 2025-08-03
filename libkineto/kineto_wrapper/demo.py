"""
demo_profile_all_configs.py
==============================================================
Showcases the full config surface of the pykineto wrapper:

1.  default_activities.json         – no args → defaultActivityTypes()
2.  custom_metrics.json             – add kernel-level HW counters
3.  per_kernel_metrics.json         – per-kernel PerfWorks collection
4.  warmup_and_trace_id.json        – warm-up delay + custom trace-id tags
5.  minimal_custom_activities.json  – trace just kernels & runtime calls
----------------------------------------------------------------------
Each section launches a trivial CuPy kernel so the trace is non-empty.
Open the resulting JSON files in chrome://tracing or Perfetto.
"""

from time import sleep

import pykineto
from cupy_kernel import vector_add_gpu
import numpy as np
# a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)

# ------------------------------------------------------------
# Utility: a toy CUDA workload we can re-use
# ------------------------------------------------------------
# def toy_kernel():
#     # 1M-element elementwise add
#     c = a @ a
#     return c.sum()

def toy_kernel():
    a = np.array(1024, dtype=np.float32)
    out = vector_add_gpu(a, a)
    return out

prof = pykineto.Profiler()

# ------------------------------------------------------------
# 1.  Default activities, no counters -------------------------
# ------------------------------------------------------------

prof.configure()                        # <- takes all defaults
prof.start()
toy_kernel()
prof.stop_to_file("default_activities.json")

# # ------------------------------------------------------------
# # 2.  Default activities + two HW metrics --------------------
# # ------------------------------------------------------------
# prof.configure(
#     metrics=[
#         "sm__cycles_elapsed.avg",       # GPU cycles
#         "dram__bytes_read.sum"          # memory traffic
#     ])
# prof.start()
# toy_kernel()
# prof.stop_to_file("custom_metrics.json")

# # ------------------------------------------------------------
# # 3.  Per-kernel metric collection ---------------------------
# # ------------------------------------------------------------
# prof.configure(
#     metrics=["sm__throughput.avg.pct_of_peak_sustained_elapsed"],
#     per_kernel=True)                    # CUPTI_PROFILER_ENABLE_PER_KERNEL
# prof.start()
# toy_kernel()
# prof.stop_to_file("per_kernel_metrics.json")

# # ------------------------------------------------------------
# # 4.  Warm-up period + user trace-ID -------------------------
# # ------------------------------------------------------------
# prof.configure(
#     metrics=["sm__pipe_active.avg.pct_of_peak_sustained_elapsed"],
#     warmup_sec=2,                       # collect after 2 s delay
#     trace_id="DEMO-RUN-42")
# prof.start()
# sleep(1)                                # in warm-up, nothing recorded yet
# toy_kernel()                            # still warm-up
# sleep(2)                                # warm-up over
# toy_kernel()                            # ← first recorded kernel
# prof.stop_to_file("warmup_and_trace_id.json")

# ------------------------------------------------------------
# 5.  Minimal activity set -----------------------------------
# ------------------------------------------------------------
# prof.configure(
#     activities=[
#         # "kernel",                       # ActivityType::CONCURRENT_KERNEL
#         "cuda_runtime"],                # ActivityType::CUDA_RUNTIME
#     metrics=["sm__cycles_elapsed.avg"])
# prof.start()
# toy_kernel()
# prof.stop_to_file("minimal_custom_activities.json")

# print("✓ 5 traces written; open them with Perfetto or chrome://tracing")
