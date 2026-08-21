# TIF 时序无监督与 Claude-Qwen3-VL 弱监督训练

## 训练结构

无监督阶段不加载大语言模型：

```text
TIF 时序随机相邻帧
  → 共享孪生编码器
  → 高置信辐射差异伪标签
  → 伪变化 BCE/Dice + 稳定区特征一致性 + 时序对称一致性
  → vision_checkpoint.pt
```

弱监督阶段使用整段多帧和双视觉流，而不是只使用首尾帧。Qwen3-VL
负责原始时序的语义理解，变化网络负责遥感变化定位和方向特征：

```text
前三个显示波段 → Qwen3-VL Video Encoder → 原生语义 Token ─────────┐
                                                                ├→ Qwen3-VL LLM + LoRA
每一帧 → 共享变化编码器                                            │       → Claude 整段描述
相邻帧 → C12/C23/C34... → 有序 Change Token + 时间/GSD Token ─────┘
                    └→ mean/max 聚合 → 变化解码器 → 全时序伪变化图
```

语言分支保留 `C12 → C23 → C34` 的先后顺序，不再用 mean/max Token
生成文本；mean/max 仅用于输出总体空间变化图。这可以减少“清理被描述成扩张”一类
时间方向混淆。

联合损失：

```text
Lweak = λpseudo × (masked BCE + Dice) + λtext × CausalLM(description)
```

Claude 描述负责语义弱监督，辐射差异产生的高置信伪标签负责保持空间变化分支可训练。无像素级人工 mask。
原生 Qwen3-VL Vision Encoder 默认冻结；训练变化网络、视觉连接器和语言模型 LoRA。

## 数据前提

- 同一序列的 TIF 必须完成正射校正和像素级配准。
- CRS、范围、分辨率、波段顺序应一致。
- 当前版本读取指定的输入波段并逐景做 2%～98% 拉伸；Qwen3-VL 语义流使用前三个显示波段，变化分支可以继续扩展为多光谱输入。
- 文件名应能按时间排序，例如 `20240101.tif`、`20240401.tif`。
- Claude 描述应概括整段时序的变化过程，不要只描述单帧。

若影像未配准，边界错位会被伪标签误认为变化；此时不应启动无监督或弱监督训练。

## 目录与描述格式

推荐一个子目录代表一景时序：

```text
/data/raw/
  scene_001/
    20240101.tif
    20240401.tif
    20240701.tif
    20241001.tif
  scene_002/
    20240115.tif
    20240515.tif
    20240915.tif
```

Claude 描述使用 JSONL：

```json
{"id":"scene_001","description":"前期林地稳定，中期开始出现采伐，后期裸地范围继续扩大","domain":"forest","gsd":0.8}
{"id":"scene_002","description":"汛期河道水面扩大，随后逐步回落，沿岸未见新增建设活动","domain":"disaster","gsd":10.0}
```

生成训练清单：

```bash
python -m geochangemllm.prepare_sequence_manifest \
  --tif-root /data/raw \
  --descriptions /data/claude_descriptions.jsonl \
  --output /data/sequences/manifest.jsonl
```

生成后的每条记录如下。`frames` 也支持带 `path`、`timestamp` 的对象：

```json
{"id":"scene_001","frames":["../raw/scene_001/20240101.tif","../raw/scene_001/20240401.tif","../raw/scene_001/20240701.tif"],"description":"前期林地稳定，中后期采伐裸地持续扩大","domain":"forest","gsd":0.8}
```

## 合成 GeoTIFF 烟雾数据

镜像和 Qwen 本地缓存准备好后，可以从宿主机一键运行全部三阶段测试：

```bash
IMAGE=geochangemllm:0.3.0 \
QWEN_VL_PATH=/cache/models/Qwen3-VL-2B-Instruct \
bash scripts/run_sequence_docker_smoke.sh
```

运行前需将官方 `Qwen/Qwen3-VL-2B-Instruct` 完整下载到命名卷的
`/cache/models/Qwen3-VL-2B-Instruct`。它会顺序执行合成 TIF 生成、双卡无监督
10步、双卡 Qwen3-VL 弱监督10步和单卡序列推理。以下命令用于分阶段运行或调整参数。

若服务器访问 Hugging Face 不稳定，可继续使用 ModelScope 下载到现有缓存卷：

```bash
docker run --rm --user 0 \
  -v geochangemllm-cache:/cache \
  geochangemllm:0.3.0 \
  bash -lc "
    python -m pip install modelscope
    modelscope download \
      --model Qwen/Qwen3-VL-2B-Instruct \
      --local_dir /cache/models/Qwen3-VL-2B-Instruct
  "
```

```bash
python -m geochangemllm.make_sequence_smoke_data \
  --output-dir /data/sequence-smoke \
  --num-sequences 8 \
  --frames-per-sequence 4 \
  --image-size 256
```

## 双卡无监督烟雾训练

```bash
torchrun --standalone --nproc_per_node=2 \
  -m geochangemllm.train_unsupervised \
  --manifest /data/sequence-smoke/manifest.jsonl \
  --output-dir /data/outputs/unsupervised-smoke \
  --pair-mode adjacent \
  --image-size 256 \
  --batch-size 2 \
  --epochs 20 \
  --max-steps 20 \
  --precision bf16 \
  --log-every 1
```

输出：

```text
/data/outputs/unsupervised-smoke/vision_checkpoint.pt
```

## 全帧 Qwen3-VL 弱监督烟雾训练

`max-frames` 是进入模型的最大时相数。序列不超过该值时使用全部帧；超过时在完整时间跨度内均匀抽帧。4090 建议先从4～8帧开始。

```bash
torchrun --standalone --nproc_per_node=2 \
  -m geochangemllm.train_weak_sequence \
  --manifest /data/sequence-smoke/manifest.jsonl \
  --vision-checkpoint /data/outputs/unsupervised-smoke/vision_checkpoint.pt \
  --output-dir /data/outputs/weak-qwenvl-smoke \
  --vl-model /cache/models/Qwen3-VL-2B-Instruct \
  --lora-rank 16 \
  --max-frames 4 \
  --token-grid 2 \
  --max-text-tokens 256 \
  --image-size 256 \
  --batch-size 1 \
  --grad-accum 2 \
  --epochs 20 \
  --max-steps 20 \
  --precision bf16 \
  --learning-rate 1e-4 \
  --text-learning-rate 1e-5 \
  --log-every 1
```

`learning-rate` 用于变化网络、顺序 Change Adapter 和模态门控；
`text-learning-rate` 仅用于 Qwen3-VL 语言模型 LoRA。若只想回归验证旧的纯文本
Qwen 软前缀模式，仍可使用 `--text-model /cache/models/Qwen3-0.6B`。

如果暂时不运行无监督阶段，删掉 `--vision-checkpoint`，弱监督可以从随机视觉主干开始验证流程。

## 全帧序列推理

```bash
python -m geochangemllm.infer_sequence \
  --checkpoint /data/outputs/weak-qwenvl-smoke/checkpoint.pt \
  --manifest /data/sequence-smoke/manifest.jsonl \
  --output-dir /data/outputs/weak-qwenvl-infer \
  --generate-text
```

输出 `change_probability.png` 和 `result.json`，JSON 会记录原始帧数、实际进入模型的帧数、变化比例和生成描述。
