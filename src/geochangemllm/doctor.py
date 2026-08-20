from __future__ import annotations

import platform
import sys


def main() -> None:
    print(f"python={sys.version.split()[0]} platform={platform.platform()}")
    try:
        import torch
    except ImportError:
        print("ERROR: torch is not installed")
        raise SystemExit(1)

    print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()} gpu_count={torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA is unavailable; check the PyTorch wheel and container GPU passthrough")
        raise SystemExit(2)
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        memory_gib = properties.total_memory / (1024**3)
        print(
            f"gpu[{index}]={properties.name} memory={memory_gib:.1f}GiB "
            f"bf16={torch.cuda.is_bf16_supported()}"
        )
    if torch.cuda.device_count() < 2:
        print("WARNING: fewer than two GPUs are visible; DDP smoke test will not use both cards")
    else:
        print("OK: environment is ready for the two-GPU smoke test")


if __name__ == "__main__":
    main()
