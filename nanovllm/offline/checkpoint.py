"""Checkpoint management for offline data parallel inference.

Saves and loads per-worker progress so interrupted jobs can resume.
"""
import json
import os
import time
from dataclasses import dataclass, field


@dataclass
class Checkpoint:
    """Worker checkpoint for resume support."""
    worker_id: int
    completed_ids: list[str] = field(default_factory=list)
    failed_ids: list[str] = field(default_factory=list)
    last_update_time: float = 0.0

    def mark_completed(self, req_id: str):
        self.completed_ids.append(req_id)
        self.last_update_time = time.time()

    def mark_failed(self, req_id: str):
        self.failed_ids.append(req_id)
        self.last_update_time = time.time()

    @property
    def done_ids(self) -> set[str]:
        """All IDs that don't need reprocessing (completed + failed)."""
        return set(self.completed_ids) | set(self.failed_ids)

    def save(self, path: str):
        """Save checkpoint to JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "worker_id": self.worker_id,
            "completed_ids": self.completed_ids,
            "failed_ids": self.failed_ids,
            "last_update_time": self.last_update_time,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Checkpoint":
        """Load checkpoint from JSON file. Returns empty checkpoint if file doesn't exist."""
        if not os.path.exists(path):
            return cls(worker_id=-1)
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            worker_id=data["worker_id"],
            completed_ids=data.get("completed_ids", []),
            failed_ids=data.get("failed_ids", []),
            last_update_time=data.get("last_update_time", 0.0),
        )


def load_all_checkpoints(output_dir: str, num_workers: int) -> dict[int, Checkpoint]:
    """Load all worker checkpoints from output directory."""
    checkpoints = {}
    for wid in range(num_workers):
        path = os.path.join(output_dir, f"checkpoint_worker_{wid}.json")
        ckpt = Checkpoint.load(path)
        ckpt.worker_id = wid
        checkpoints[wid] = ckpt
    return checkpoints


def get_all_completed_ids(checkpoints: dict[int, Checkpoint]) -> set[str]:
    """Get union of all completed IDs across workers."""
    all_ids = set()
    for ckpt in checkpoints.values():
        all_ids.update(ckpt.completed_ids)
    return all_ids
