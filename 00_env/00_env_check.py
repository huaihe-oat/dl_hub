"""环境验收脚本：确认 PyTorch 能在 RTX 5070 (sm_120) 上真实跑起来。

运行：
    conda activate dl
    python 00_env/00_env_check.py
"""

import time

import torch

print("torch      :", torch.__version__)
print("cuda build :", torch.version.cuda)
print("available  :", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA 不可用，检查显卡驱动或 PyTorch 是否装成了 CPU 版"

print("device     :", torch.cuda.get_device_name(0))
major, minor = torch.cuda.get_device_capability(0)
print(f"capability : sm_{major}{minor}")

props = torch.cuda.get_device_properties(0)
print(f"mem total  : {props.total_memory / 1024**3:.1f} GB")

torch.manual_seed(0)
x = torch.randn(4096, 4096, device="cuda")
w = torch.randn(4096, 4096, device="cuda", requires_grad=True)

torch.cuda.synchronize()
t0 = time.time()
loss = (x @ w).relu().sum()
loss.backward()
torch.cuda.synchronize()
elapsed = time.time() - t0
print(f"matmul+backward ok  {elapsed:.3f}s  grad_norm={w.grad.norm().item():.3f}")

for dtype in (torch.float16, torch.bfloat16):
    a = torch.randn(2048, 2048, device="cuda", dtype=dtype)
    b = torch.randn(2048, 2048, device="cuda", dtype=dtype)
    c = a @ b
    torch.cuda.synchronize()
    print(f"{str(dtype):22s} matmul ok  max={c.abs().max().item():.2f}")

free, total = torch.cuda.mem_get_info()
print(f"mem free   : {free / 1024**3:.2f} / {total / 1024**3:.2f} GB")
print("\nENV OK")
