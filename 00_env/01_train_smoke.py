"""训练冒烟测试：在 GPU 上真实训练，确认整条链路可用。

运行：
    conda activate dl
    python 00_env/01_train_smoke.py
"""

import time

import torch
import torch.nn as nn

DEV = "cuda"
torch.manual_seed(0)


def section(title):
    print(f"\n{'=' * 56}\n{title}\n{'=' * 56}")


section("1. MLP 完整训练循环 (AdamW + BCE)")
N, D = 20000, 64
X = torch.randn(N, D, device=DEV)
true_w = torch.randn(D, 1, device=DEV)
y = ((X @ true_w).squeeze(1) > 0).float()

mlp = nn.Sequential(
    nn.Linear(D, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, 1),
).to(DEV)
opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

t0 = time.time()
for epoch in range(1, 31):
    opt.zero_grad()
    loss = loss_fn(mlp(X).squeeze(1), y)
    loss.backward()
    opt.step()
    if epoch % 10 == 0 or epoch == 1:
        with torch.no_grad():
            acc = ((mlp(X).squeeze(1) > 0).float() == y).float().mean().item()
        print(f"  epoch {epoch:>2}  loss={loss.item():.4f}  acc={acc:.4f}")
print(f"  30 epoch 用时 {time.time() - t0:.2f}s")
assert acc > 0.9, f"训练没收敛，acc={acc}"
print("  [OK] loss 下降且收敛")

section("2. Transformer + 注意力 kernel (SDPA)")
enc = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024,
                               batch_first=True, norm_first=True),
    num_layers=4,
).to(DEV)
x = torch.randn(16, 128, 256, device=DEV, requires_grad=True)
out = enc(x)
out.sum().backward()
torch.cuda.synchronize()
print(f"  out {tuple(out.shape)}  grad_norm={x.grad.norm().item():.3f}")
print("  [OK] 4 层 TransformerEncoder 前向+反向通过")

section("3. 混合精度 (bf16 autocast + GradScaler)")
amp_model = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 1)).to(DEV)
amp_opt = torch.optim.AdamW(amp_model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler("cuda")
start = None
for step in range(1, 31):
    amp_opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = loss_fn(amp_model(X).squeeze(1), y)
    scaler.scale(loss).backward()
    scaler.step(amp_opt)
    scaler.update()
    if start is None:
        start = loss.item()
print(f"  loss {start:.4f} -> {loss.item():.4f}")
assert loss.item() < start, "混合精度下 loss 没有下降"
print("  [OK] bf16 混合精度训练收敛")

section("4. 显存占用")
free, total = torch.cuda.mem_get_info()
print(f"  free {free / 1024**3:.2f} GB / total {total / 1024**3:.2f} GB")
print(f"  peak allocated this process: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

print("\n*** 全部通过：环境可用于训练 ***")
