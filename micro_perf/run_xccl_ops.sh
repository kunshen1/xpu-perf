#!/usr/bin/env bash

# Run XCCL collective benchmarks through micro_perf/launch.py.
#
# Default behavior:
#   - Run the INTEL backend
#   - Use devices 0,1
#   - Limit visible devices with ZE_AFFINITY_MASK=0,1
#   - Use workloads/xccl_ops/all_reduce.json as workload input
#   - Write results into xccl_ops_report/
#
# Quick start:
#   ./run_ccl_ops.sh
#
# Override defaults with environment variables:
#   ZE_AFFINITY_MASK=0,1 DEVICE_IDS=0,1 ./run_ccl_ops.sh
#   WORKLOAD=workloads/xccl_ops/all_gather.json ./run_ccl_ops.sh
#
# Pass extra launch.py arguments through:
#   ./run_ccl_ops.sh --node_world_size 2 --node_rank 0
#   ./run_ccl_ops.sh --help

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly LAUNCH_SCRIPT="launch.py"

export LD_LIBRARY_PATH=/opt/venv/lib:${LD_LIBRARY_PATH}
export MASTER_ADDR=127.0.0.1
export FI_TCP_IFACE=lo

ZE_AFFINITY_MASK_VALUE="${ZE_AFFINITY_MASK:-"0,1"}"
WORKLOAD="${WORKLOAD:-"workloads/xccl_ops/all_reduce.json"}"
DEVICE_IDS="${DEVICE_IDS:-"0,1"}"
BACKEND="${BACKEND:-"INTEL"}"
REPORT_DIR="${REPORT_DIR:-"xccl_ops_report"}"

ONECCL_BINDINGS_FOR_PYTORCH_ENV_VERBOSE="${ONECCL_BINDINGS_FOR_PYTORCH_ENV_VERBOSE:-"0"}"
CCL_BLOCKING_WAIT="${CCL_BLOCKING_WAIT:-"0"}"
CCL_SAME_STREAM="${CCL_SAME_STREAM:-"1"}"

CMD=()

print_usage() {
  cat <<EOF
Usage:
  ./run_ccl_ops.sh [extra launch.py args]

Environment overrides:
  ZE_AFFINITY_MASK                           Visible XPU devices for Level Zero. Default: 0,1
  WORKLOAD                                   Workload json path. Default: workloads/xccl_ops/all_reduce.json
  DEVICE_IDS                                 launch.py --device value. Default: 0,1
  BACKEND                                    launch.py --backend value. Default: INTEL
  REPORT_DIR                                 launch.py --report_dir value. Default: xccl_ops_report
  ONECCL_BINDINGS_FOR_PYTORCH_ENV_VERBOSE    oneCCL verbose level. Default: 0
  CCL_BLOCKING_WAIT                          Enable blocking wait. Default: 0
  CCL_SAME_STREAM                            Reuse compute stream for communication. Default: 1

Examples:
  ./run_ccl_ops.sh
  WORKLOAD=workloads/xccl_ops/all_gather.json ./run_ccl_ops.sh
  DEVICE_IDS=0 ZE_AFFINITY_MASK=0 ./run_ccl_ops.sh
  ./run_ccl_ops.sh --node_world_size 2 --node_rank 0
EOF
}

build_cmd() {
  CMD=(
    python3
    "$LAUNCH_SCRIPT"
    --workload "$WORKLOAD"
    --device "$DEVICE_IDS"
    --backend "$BACKEND"
    --report_dir "$REPORT_DIR"
    "$@"
  )
}

print_cmd() {
  local env_vars=(
    "ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK_VALUE"
    "ONECCL_BINDINGS_FOR_PYTORCH_ENV_VERBOSE=$ONECCL_BINDINGS_FOR_PYTORCH_ENV_VERBOSE"
    "CCL_BLOCKING_WAIT=$CCL_BLOCKING_WAIT"
    "CCL_SAME_STREAM=$CCL_SAME_STREAM"
  )

  echo "Running command:"
  printf '  '
  printf '%q ' "${env_vars[@]}"
  printf '%q ' "${CMD[@]}"
  printf '\n\n'
}

run_cmd() {
  ZE_AFFINITY_MASK="$ZE_AFFINITY_MASK_VALUE" \
  ONECCL_BINDINGS_FOR_PYTORCH_ENV_VERBOSE="$ONECCL_BINDINGS_FOR_PYTORCH_ENV_VERBOSE" \
  CCL_BLOCKING_WAIT="$CCL_BLOCKING_WAIT" \
  CCL_SAME_STREAM="$CCL_SAME_STREAM" \
  "${CMD[@]}"
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    print_usage
    exit 0
  fi

  cd "$SCRIPT_DIR"
  build_cmd "$@"
  print_cmd
  run_cmd
}

main "$@"
