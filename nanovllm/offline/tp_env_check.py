"""
T0: Tensor Parallel Environment Check
======================================
Validates that the multi-GPU environment is correctly configured for
tensor parallelism. Each rank reports its identity and hardware, then
participates in an all_reduce correctness test.

Launch via:
    torchrun --nproc_per_node=2 -m nanovllm.offline.tp_env_check

Or use the companion shell script:
    bash scripts/check_tp_env.sh
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import json
import sys
import datetime
from pathlib import Path

import torch
import torch.distributed as dist


def setup_distributed() -> dict:
    """Initialize the NCCL process group from torchrun environment variables.

    Returns a dict with rank metadata for logging.
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Bind this process to the correct GPU
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=30),
    )

    device_name = torch.cuda.get_device_name(local_rank)
    torch_version = torch.__version__
    cuda_version = torch.version.cuda or "N/A"

    info = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device_name": device_name,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "nccl_available": dist.is_nccl_available(),
    }
    return info


def all_reduce_test(info: dict) -> dict:
    """Run a simple all_reduce (SUM) and verify correctness.

    Each rank contributes its rank number as a tensor.  After all_reduce
    the expected value is sum(0..world_size-1) = world_size*(world_size-1)/2.
    """
    rank = info["rank"]
    world_size = info["world_size"]

    tensor = torch.tensor([float(rank)], dtype=torch.float32, device=f"cuda:{info['local_rank']}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    expected = world_size * (world_size - 1) / 2.0
    actual = tensor.item()
    passed = abs(actual - expected) < 1e-5

    result = {
        "all_reduce_expected": expected,
        "all_reduce_actual": actual,
        "all_reduce_passed": passed,
    }

    status = "PASS" if passed else "FAIL"
    print(f"[Rank {rank}] all_reduce test: {status} (expected={expected}, got={actual})")
    return result


def broadcast_test(info: dict) -> dict:
    """Verify broadcast: rank 0 sends a known tensor, all others receive it."""
    rank = info["rank"]
    local_rank = info["local_rank"]
    device = f"cuda:{local_rank}"

    if rank == 0:
        tensor = torch.tensor([42.0, 99.0], dtype=torch.float32, device=device)
    else:
        tensor = torch.zeros(2, dtype=torch.float32, device=device)

    dist.broadcast(tensor, src=0)

    expected = torch.tensor([42.0, 99.0], dtype=torch.float32, device=device)
    passed = torch.allclose(tensor, expected)

    result = {
        "broadcast_passed": passed,
    }

    status = "PASS" if passed else "FAIL"
    print(f"[Rank {rank}] broadcast test: {status}")
    return result


def main():
    print("=" * 60)
    print("NanoServe Tensor Parallel Environment Check (T0)")
    print("=" * 60)

    # --- Setup ---
    info = setup_distributed()
    rank = info["rank"]
    local_rank = info["local_rank"]

    print(f"[Rank {rank}] local_rank={local_rank}, "
          f"world_size={info['world_size']}, "
          f"device={info['device_name']}, "
          f"torch={info['torch_version']}, "
          f"cuda={info['cuda_version']}, "
          f"nccl={info['nccl_available']}")

    # --- Tests ---
    ar_result = all_reduce_test(info)
    bc_result = broadcast_test(info)

    # --- Collect and save results ---
    results = {
        "environment": info,
        "all_reduce_test": ar_result,
        "broadcast_test": bc_result,
        "overall_passed": ar_result["all_reduce_passed"] and bc_result["broadcast_passed"],
    }

    # Each rank writes its own results file
    output_dir = Path("tp_env_check_results")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"rank_{rank}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[Rank {rank}] Results saved to {output_path}")

    # --- Cleanup ---
    dist.destroy_process_group()

    if not results["overall_passed"]:
        print(f"[Rank {rank}] WARNING: One or more tests FAILED!")
        sys.exit(1)
    else:
        print(f"[Rank {rank}] All tests PASSED. Environment is ready for TP.")


if __name__ == "__main__":
    main()
