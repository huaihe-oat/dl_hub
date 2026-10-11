# dl_hub

深度学习学习仓库。记录从传统机器学习到 Transformer / 大模型的学习过程。

目标：每个阶段结束时留下一个**能给别人看的东西**，而不是一份看过的视频列表。

## 环境

```powershell
conda activate dl
```

- Python 3.11 · torch 2.11.0+cu128 · RTX 5070 Laptop（8 GB，sm_120）
- 已配置 `HF_ENDPOINT=https://hf-mirror.com`（直连 huggingface.co 不通）
- 自检：`python 00_env/00_env_check.py`，训练链路自检：`python 00_env/01_train_smoke.py`

## 进度

| # | 内容 | 状态 | 完成日期 |
| --- | --- | --- | --- |
| 01-1 | 线性回归 / 逻辑回归 手写实现 | ☐ | |
| 01-2 | 决策树与集成方法 | ☐ | |
| 01-3 | 一个完整的 ML 项目（真实数据集） | ☐ | |
| 02-1 | 反向传播从零手写 | ☐ | |
| 02-2 | CNN / RNN | ☐ | |
| 02-3 | 从零实现 GPT（nanoGPT） | ☐ | |
| 02-4 | 复现一篇经典论文 | ☐ | |
| 03-1 | 精读 Attention Is All You Need | ☐ | |
| 03-2 | QLoRA 微调一个小模型 | ☐ | |
| 03-3 | 推理部署（vLLM 或 llama.cpp） | ☐ | |
| 04-1 | Spinning Up 基础 | ☐ | |
| 04-2 | RLHF 小实验 或 MuJoCo agent | ☐ | |

## 目录

```
00_env/    环境脚本与自检
01_ml/     传统机器学习
02_dl/     深度学习
03_llm/    Transformer / 大模型
04_rl/     强化学习
papers/    论文精读笔记
```

## 约定

- 每个练习一个子目录，里面放代码 + `notes.md`
- `notes.md` 只记**自己踩的坑**和结论，不抄教程
- 做完一件事 commit 一次，message 用英文短句
- 每周看一眼上面的进度表
- 数据、模型权重、缓存一律不进 git（见 `.gitignore`）
