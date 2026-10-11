"""Hello World：让神经网络学会 sin(x)。

不用下载任何数据，跑完大约十几秒。
运行：conda activate dl   然后   python 00_env/hello.py
"""

import time

import torch
import torch.nn as nn

print("PyTorch :", torch.__version__)
print("GPU     :", torch.cuda.get_device_name(0))
print("CUDA    :", torch.cuda.is_available())
print()

torch.manual_seed(0)

# 造任务：在 [-3, 3] 上取 1000 个点，让网络学 y = sin(x)
x = torch.linspace(-3, 3, 1000).unsqueeze(1)
y = torch.sin(x)
x, y = x.cuda(), y.cuda()

# 一个很小的网络：1 -> 32 -> 32 -> 1
model = nn.Sequential(
    nn.Linear(1, 32), nn.Tanh(),
    nn.Linear(32, 32), nn.Tanh(),
    nn.Linear(32, 1),
).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

t0 = time.time()
loss = None
for step in range(1, 2001):
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
    if step % 400 == 0:
        print(f"  第 {step:>4} 步    loss = {loss.item():.6f}")

print(f"\n训练用时 {time.time() - t0:.2f} 秒")
print(f"最终 loss = {loss.item():.6f}")

# 拿几个训练时没见过的点考它
with torch.no_grad():
    test_x = torch.tensor([[-2.5], [-0.7], [0.0], [1.5], [2.9]], device="cuda")
    pred = model(test_x).squeeze()
    truth = torch.sin(test_x).squeeze()

print("\n预测 vs 真值：")
for p, t in zip(pred, truth):
    print(f"  预测 {p.item():+.4f}    真值 {t.item():+.4f}")

print("\n跑通了，环境没问题。")
