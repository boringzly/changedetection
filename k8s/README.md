# K3s / Kubernetes 运行说明

## 原则

- Python、PyTorch、Transformers、Rasterio 全部放入镜像，不安装到宿主机。
- 宿主机只需要已有的 NVIDIA 驱动、K3s/containerd 和 NVIDIA device plugin。
- 训练 Pod 以 UID 10001 非 root 运行，不挂载宿主目录、不挂载 ServiceAccount Token、丢弃 Linux capabilities。
- 数据、模型缓存和结果分别写入 PVC；Pod 删除后结果仍保留。
- 4090 不使用 time-slicing。双卡 DDP 申请两张独占 `nvidia.com/gpu`。

## 0. 只读预检

```bash
bash k8s/preflight.sh
```

`Allocatable NVIDIA GPUs` 必须显示 `2`，且 `nvidia.com/gpu.sharing-strategy` 应为空或 `none`。如果显示 `time-slicing`/`mps`，或者两张物理卡却暴露出大于2的 GPU 数量，不要提交双卡 Job；让管理员为训练时段切换为独占配置。time-slicing 的两个资源份额不保证对应两张物理卡，也不隔离显存。

如果 GPU 数量为空，说明集群管理员还没有配置 NVIDIA device plugin；不要在项目 Job 中绕过它挂载 `/dev/nvidia*`。

## 1. 构建并推送镜像

将镜像推送到 K3s 节点能够访问的私有仓库：

```bash
IMAGE=你的仓库/geochangemllm:0.1.0 bash scripts/build_push_image.sh
```

然后把两个 Job YAML 中的：

```yaml
image: geochangemllm:0.1.0
```

替换为刚刚推送的完整地址。不要把仓库密码写入 YAML；私有仓库使用管理员提供的 `imagePullSecret`。

如果没有仓库，可由集群管理员将 Docker 镜像导入 K3s containerd；普通租户不要直接操作宿主 containerd。

## 2. 创建隔离资源

以下操作会创建独立 namespace、资源配额和两个 PVC，但不会启动训练：

```bash
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/01-quota.yaml
kubectl apply -f k8s/02-storage.yaml
kubectl -n geochangemllm get resourcequota,limitrange,pvc
```

如果 PVC 一直 `Pending`，检查 K3s 默认 `local-path` StorageClass，或将管理员给出的 `storageClassName` 加到 `02-storage.yaml`。

## 3. 运行完全离线的双卡烟雾测试

```bash
kubectl create -f k8s/10-job-tiny-smoke.yaml
kubectl -n geochangemllm get pods -w
kubectl -n geochangemllm logs -f job/geochangemllm-tiny-smoke
```

应看到：

- `gpu_count=2`。
- `world_size=2`。
- 连续输出20个 `step=... loss=...`。
- `/data/outputs/tiny-smoke/checkpoint.pt` 被保存。

失败时查看：

```bash
kubectl -n geochangemllm describe job geochangemllm-tiny-smoke
kubectl -n geochangemllm describe pod -l app.kubernetes.io/component=smoke-train
```

重新运行固定名称的 Job 前先删除旧 Job；删除 Job 不会删除 PVC：

```bash
kubectl -n geochangemllm delete job geochangemllm-tiny-smoke
```

## 4. 运行 Qwen LoRA 闭环

`tiny` 成功后再提交：

```bash
kubectl create -f k8s/11-job-qwen-smoke.yaml
kubectl -n geochangemllm logs -f job/geochangemllm-qwen-smoke
```

模型会缓存到 `geochangemllm-cache` PVC，后续 Job 不重复下载。若集群禁止外网，先由管理员或独立下载 Job 将模型同步到该 PVC，再把模型路径改为 PVC 内本地路径。

## 5. 取回结果

Job 结束后通过一个不申请 GPU 的临时 Pod 读取 PVC。先把 `20-pod-data-shell.yaml` 中的镜像地址同步替换为私有仓库地址，然后运行：

```bash
kubectl create -f k8s/20-pod-data-shell.yaml
kubectl -n geochangemllm wait --for=condition=Ready pod/geochangemllm-data-shell --timeout=120s
kubectl cp geochangemllm/geochangemllm-data-shell:/data/outputs ./outputs
kubectl -n geochangemllm delete pod geochangemllm-data-shell
```

## 6. 停止与清理

停止正在运行的训练：

```bash
kubectl -n geochangemllm delete job geochangemllm-tiny-smoke geochangemllm-qwen-smoke --ignore-not-found
```

这不会删除数据。确认结果已备份后，才删除 PVC：

```bash
kubectl -n geochangemllm delete pvc geochangemllm-data geochangemllm-cache
```
