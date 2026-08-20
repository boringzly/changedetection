# GeoChangeMLLM：变化检测 × 语言模型最小闭环

本仓库先解决一件事：在一台 `2 × RTX 4090 24GB` 服务器上走通“双时相影像 → 像素变化掩膜 → 变化 Token → 小型 Qwen → 联合损失 → DDP 训练 → 推理结果”的完整链路。

当前版本包含有监督双时相 MVP，以及两条 TIF 时序训练链路：无监督视觉预训练、使用整段多帧影像和 Claude 描述的 Qwen 弱监督训练。数据格式、原理和 Docker 测试命令见 [TIF 时序无监督与弱监督训练](docs/sequence_training.md)。

共享服务器默认采用 Docker + K3s Job，不在宿主机安装 Python 依赖。直接从 [K3s 运行说明](k8s/README.md) 开始；下面的裸命令仅用于独占开发机或容器内部调试。

## 结构

```text
T1/T2 → 共享孪生卷积编码器 → 多尺度时相融合 → 变化解码器 → mask
                                  └→ Change Token Projector → Tiny/Qwen → description
```

联合损失：

```text
L = BCE(mask) + Dice(mask) + text_loss_weight × CausalLM(description)
```

`tiny` 文本头不需要联网，用于验证训练链路；`Qwen/Qwen3-0.6B` 用于验证真实语言模型连接和 LoRA。

## 服务器环境

优先使用平台已有的 PyTorch 镜像。若环境为空：

```bash
cd /path/to/change-model
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

安装后先做环境检查：

```bash
python -m geochangemllm.doctor
nvidia-smi topo -m
```

需要 Qwen 时再安装可选依赖：

```bash
python -m pip install -e '.[qwen]'
```

## 1. 生成合成变化样本

```bash
python -m geochangemllm.make_smoke_data \
  --output-dir data/smoke \
  --num-samples 32 \
  --image-size 256
```

生成的 `data/smoke/manifest.jsonl` 与下周真实数据使用同一格式。

## 2. 完全离线双卡烟雾测试

```bash
torchrun --standalone --nproc_per_node=2 -m geochangemllm.train \
  --manifest data/smoke/manifest.jsonl \
  --output-dir outputs/tiny-smoke \
  --text-model tiny \
  --image-size 256 \
  --batch-size 4 \
  --max-steps 20 \
  --log-every 1
```

成功标准：两张 GPU 都有显存和利用率、loss 能反传、目录中出现 `checkpoint.pt`。

## 3. Qwen + LoRA 双卡联合训练

首次运行会下载模型。建议先用 0.6B 验证结构，确认后再换 1.7B/4B。

```bash
torchrun --standalone --nproc_per_node=2 -m geochangemllm.train \
  --manifest data/smoke/manifest.jsonl \
  --output-dir outputs/qwen-smoke \
  --text-model Qwen/Qwen3-0.6B \
  --lora-rank 16 \
  --image-size 256 \
  --batch-size 2 \
  --grad-accum 2 \
  --epochs 20 \
  --max-steps 50 \
  --precision bf16 \
  --log-every 1
```

如果只想冻结语言模型、训练变化网络和视觉连接器，增加 `--freeze-text` 并去掉 `--lora-rank`。

## 4. 推理

```bash
python -m geochangemllm.infer \
  --checkpoint outputs/tiny-smoke/checkpoint.pt \
  --manifest data/smoke/manifest.jsonl \
  --output-dir outputs/tiny-infer \
  --generate-text
```

输出：

- `change_probability.png`：变化概率图。
- `result.json`：样本、阈值、预测描述等结构化结果。

## 真实数据清单

每行一个 JSON：

```json
{"t1":"images/a_t1.tif","t2":"images/a_t2.tif","mask":"masks/a.png","description":"林地被清理为裸地","domain":"forest","gsd":0.8}
```

路径相对于清单文件所在目录。RGB/PNG/JPEG 可直接读取；安装 `.[geo]` 后可读取 GeoTIFF。当前 MVP 取前三个波段，后续会加入多光谱波段适配器、传感器编码和无监督时相损失。

## 2 × 4090 当天验收

训练期间另开一个终端运行 `nvidia-smi dmon -s pucm` 观察双卡利用率。

依次确认：

1. `tiny` 双卡 20 step 完成并保存断点。
2. Qwen 0.6B LoRA 双卡 50 step 完成。
3. 单样本推理生成概率图和 JSON。
4. 重启训练时可通过 `--resume outputs/qwen-smoke/checkpoint.pt` 继续。
