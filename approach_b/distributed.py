"""Optional torch.distributed helpers. One process (no LOCAL_RANK) is rank 0 of 1."""

from __future__ import annotations

import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def gpu_count() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def maybe_relaunch() -> None:
    """Re-exec under torchrun so every visible GPU gets a rank. No-op if already launched."""
    if "LOCAL_RANK" in os.environ:
        return
    nproc = gpu_count()
    if nproc <= 1:
        return
    os.execvp(
        sys.executable,
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={nproc}",
            *sys.argv,
        ],
    )


def setup() -> tuple[int, int, torch.device]:
    """Return (rank, world, device). Initializes NCCL when launched with torchrun."""
    if "LOCAL_RANK" not in os.environ:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return 0, 1, device
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group(
            "nccl",
            device_id=torch.device(f"cuda:{rank}"),
            timeout=timedelta(hours=2),
        )
    if rank == 0:
        print(f"distributed: world={world} devices={[torch.cuda.get_device_name(i) for i in range(world)]}", flush=True)
    return rank, world, torch.device(f"cuda:{rank}")


def is_main(rank: int | None = None) -> bool:
    if rank is not None:
        return rank == 0
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def wait_for_file(path: Path, rank: int, poll_seconds: float = 2.0) -> None:
    """Wait for rank 0 to finish a long CPU job without holding an NCCL collective."""
    if rank == 0:
        return
    path = Path(path)
    while not path.exists():
        time.sleep(poll_seconds)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def broadcast_object(obj, rank: int, world: int, src: int = 0):
    """Share a picklable object from src to every rank. No-op when world == 1."""
    payload = [obj]
    if world > 1:
        dist.broadcast_object_list(payload, src=src)
    return payload[0]


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
