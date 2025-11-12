import torch
import ray

from recipe.streamrl.distributed_util import stateless_init_process_group
from verl.utils.device import is_cuda_available, is_npu_available

MASTER_ADDRESS = "192.168.0.3"
MASTER_PORT = 40321
WORLD_SIZE = 4


def _resolve_device_type() -> str:
    if is_cuda_available:
        return "cuda"
    if is_npu_available:
        return "npu"
    raise RuntimeError("stateless_init_process_group test requires CUDA or NPU devices.")

def _set_device_from_ray(device_type: str, fallback_rank: int) -> torch.device:
    if device_type == "cuda":
        gpu_ids = ray.get_gpu_ids()
        if not gpu_ids:
            raise RuntimeError("Ray actor did not receive a CUDA device.")
        # Ray exposes physical GPU ids, but the actor's CUDA_VISIBLE_DEVICES is
        # remapped to a contiguous local range starting from 0. Use the local
        # index to avoid invalid device ordinals.
        torch.cuda.set_device(0)
        print(f"Ray assigned physical GPU id(s) {gpu_ids}; using logical cuda:0 inside the actor.", flush=True)
        return torch.device("cuda", 0)
    torch.npu.set_device(fallback_rank)
    return torch.device("npu", fallback_rank)


@ray.remote(num_gpus=1)
class _StatelessPGActor:
    def __init__(self, rank: int, device_type: str):
        self.rank = rank
        self.device_type = device_type
        self.device = _set_device_from_ray(device_type, rank)

    def run(self, master_address: str, master_port: int, world_size: int) -> float:
        print(f"[rank={self.rank}] launching communicator on {self.device}", flush=True)
        communicator = stateless_init_process_group(
            master_address=master_address,
            master_port=master_port,
            rank=self.rank,
            world_size=world_size,
            device=self.device,
        )
        payload = torch.ones(1, device=self.device, dtype=torch.float32) * self.rank
        reduced = communicator.all_reduce(payload)
        value = reduced.item() if reduced is not None else payload.item()
        print(f"[rank={self.rank}] all_reduce result={value}", flush=True)
        return value


def test_stateless_init_process_group() -> None:
    device_type = _resolve_device_type()

    ray.init(ignore_reinit_error=True)
    try:
        actors = [_StatelessPGActor.remote(rank, device_type) for rank in range(WORLD_SIZE)]
        futures = [actor.run.remote(MASTER_ADDRESS, MASTER_PORT, WORLD_SIZE) for actor in actors]
        results = ray.get(futures)
        print(f"All ranks completed. Results: {results}", flush=True)
    finally:
        ray.shutdown()


def main():
    test_stateless_init_process_group()


if __name__ == "__main__":
    main()
